# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""End-to-end check that the Zeva serving path is actually live.

Phase 6 in one command: start ``action_policy_server_xhand`` against a stage-2
checkpoint plus the CTE and task-context bank, drive it with the openpi websocket
protocol the TactileTTT client speaks, and assert that what came back is real.

It exists because the failure modes here are silent.  ``infer``
(``action_policy_server_xhand.py``) will happily return actions while the CTE path
quietly falls back to ``_empty_causal_interaction_features`` — zeros for phase and a
zeroed effect history — whenever the boundary buffer cannot be built.  A client that
receives plausible-looking actions would never notice.  So this script checks four
independent things:

1. **The checkpoint loads** with whatever modules the experiment declares (a stage-2
   DCP carries the ``behavior_*`` tensors; a stage-1 one does not, so the two need
   different ``--experiment`` values — the script picks it for you).
2. **The protocol works**: each request returns ``[chunk, action_dim]``; ``--history-length``
   is pinned to 0 on this server (there is no state row to trim).
3. **The CTE path ran** (stage-2 runs only): the server logs ``boundary_frames=N`` per
   request and N grows request over request.  If it stalls at 0 or 1 the features are
   zeros and the policy is running unaided.
4. **Proprio is live**: two requests with identical pixels and sampler seed, differing
   only in ``observation/state``, must return different actions.  A dropped ``proprio``
   key is dropped *silently* — every other check here passes either way — so this is the
   only thing that distinguishes "the policy sees its arm state" from "it does not".

Both the stage-2 (full Zeva) and stage-1 (bare policy) paths can be checked; stage-1 is
just the same command with the two Zeva artifacts omitted::

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m cosmos_framework.zeva_training.verify_serving \\
        --checkpoint "$ZEVA_WORK/runs/zeva/zeva_xhand/action_policy_xhand_zeva/checkpoints/iter_00000500" \\
        --cte-checkpoint "$ZEVA_WORK/runs/zeva_cte/cte-v4-20260924/cte_step_003000.pt" \\
        --task-context-bank "$ZEVA_WORK/datasets/xhand_task_context_bank.pt" \\
        --task-context-instruction PressButton4Times

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m cosmos_framework.zeva_training.verify_serving \\
        --checkpoint "$ZEVA_WORK/runs/zeva/action_xhand/v3-joint18-20260925/checkpoints/iter_000004000"
"""

from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

READY_PATTERN = re.compile(r"ready|listening|Serving", re.IGNORECASE)
FRAMES_PATTERN = re.compile(r"boundary_frames=(\d+)")
ACTION_PATTERN = re.compile(r"action shape=\((\d+), (\d+)\)")


def synthetic_observation(rng: np.random.Generator, *, state_dim: int, prompt: str) -> dict:
    """One client request.

    Pixels are synthetic but statistically plausible (uint8, mid-range, some
    structure) and state is drawn from realistic joint ranges, because the point is
    to exercise the plumbing — dedup, buffers, normalization, shapes — not to judge
    policy quality.
    """
    height, width = 480, 640
    grad = np.linspace(40, 210, width, dtype=np.float32)[None, :, None]
    left = np.broadcast_to(grad, (height, width, 3)).copy()
    front = np.broadcast_to(grad[:, ::-1], (height, width, 3)).copy()
    left = np.clip(left + rng.normal(0, 8, left.shape), 0, 255).astype(np.uint8)
    front = np.clip(front + rng.normal(0, 8, front.shape), 0, 255).astype(np.uint8)

    # Fill the whole native layout, even the parts the server ignores, because that is
    # what a real client sends. Only the arm `[0:6]` and the *position* lanes of the
    # hand `[28:52:2]` are read (`_PROPRIO_INDICES` == the dataset's "joint18"), and the
    # server only length-checks -- a client with a wrong layout is silently believed.
    state = np.zeros(state_dim, dtype=np.float32)
    state[:6] = rng.uniform(-1.5, 1.5, 6)  # arm joints (READ)
    state[12:28] = rng.uniform(-0.3, 0.3, 16)  # ee pose (not read)
    state[28:52] = rng.uniform(-0.5, 0.5, 24)  # hand pos/torque interleaved (pos lanes READ)
    state[52:] = rng.uniform(0, 60, state_dim - 52)  # tactile (not read, but the client sends it)
    return {
        "prompt": prompt,
        "observation.images.cam_left": left,
        "observation.images.cam_front": front,
        "observation.images.cam_right": np.flip(front, axis=0).copy(),
        "observation/state": state,
    }


def wait_for_ready(proc: subprocess.Popen, log_path: Path, timeout: float) -> None:
    """Block until the server says it is serving, or fail with the tail of its log."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            tail = log_path.read_text()[-3000:] if log_path.is_file() else "(no log)"
            raise RuntimeError(f"server exited with code {proc.returncode} before becoming ready:\n{tail}")
        if log_path.is_file() and READY_PATTERN.search(log_path.read_text()):
            return
        time.sleep(5)
    raise TimeoutError(f"server not ready within {timeout:.0f}s; see {log_path}")


CONSOLE_LOG_PATTERN = re.compile(r"Console log saved to (\S+)")


def parse_log(log_path: Path) -> tuple[list[int], list[tuple[int, int]]]:
    """Progress lines from a single server log sink.

    The framework installs several loguru sinks, so the same message lands in the
    subprocess's stdout *and* in the ``console.log`` it announces at startup.
    Concatenating them duplicates every request and makes a perfectly monotonic
    sequence look like it restarts. Parse each candidate separately and keep the one
    that recorded the most requests.
    """
    candidates = [log_path]
    if log_path.is_file():
        for match in CONSOLE_LOG_PATTERN.findall(log_path.read_text(errors="replace")):
            candidates.append(Path(match))

    best: tuple[list[int], list[tuple[int, int]]] = ([], [])
    for path in candidates:
        if not path.is_file():
            continue
        text = path.read_text(errors="replace")
        frames = [int(m) for m in FRAMES_PATTERN.findall(text)]
        shapes = [(int(a), int(b)) for a, b in ACTION_PATTERN.findall(text)]
        if len(shapes) > len(best[1]):
            best = (frames, shapes)
    return best


def _bypass_proxy_for_localhost() -> None:
    """Keep the openpi client off the site HTTP proxy.

    This machine exports ``http_proxy``/``https_proxy`` globally with no
    ``no_proxy``, so `websockets` routes even a 127.0.0.1 connection through the
    proxy and it answers ``403``. Must run before the client connects — the env is
    read at connect time, not import time.
    """
    existing = {v.strip() for v in os.environ.get("no_proxy", "").split(",") if v.strip()}
    existing.update({"127.0.0.1", "localhost", "::1"})
    joined = ",".join(sorted(existing))
    os.environ["no_proxy"] = joined
    os.environ["NO_PROXY"] = joined


def main() -> int:
    _bypass_proxy_for_localhost()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--checkpoint",
        required=True,
        help="DCP iteration directory: a stage-2 one for the full Zeva path, a stage-1 one for the bare policy",
    )
    # Omit BOTH of these to check a stage-1 (nano) checkpoint: no CTE, no bank, and the
    # `--experiment` below is then picked as the nano recipe. Passing one but not the
    # other is an error -- the server needs both or neither.
    ap.add_argument("--cte-checkpoint", default=None)
    ap.add_argument("--task-context-bank", default=None)
    # The bank LOOKUP KEY. It matches the bank entry's `instruction` field, which
    # `build_task_context_bank.py` writes from the task-cluster name. This is NOT the
    # text the model reads.
    ap.add_argument("--task-context-instruction", default="PressButton4Times")
    # The text the model actually conditions on. It must be byte-identical to the
    # `ai_caption` the dataset fed during training, which comes from the LeRobot
    # `tasks` field (`xhand_lerobot_dataset.py:224`) -- NOT the task-cluster name.
    # Getting this wrong is silent: the policy still emits plausible actions, it is
    # just conditioning on an out-of-distribution string.
    ap.add_argument(
        "--prompt",
        default="press the button 4 times and put it into the box",
        help="client-side prompt; must equal the training caption (LeRobot `tasks` field)",
    )
    ap.add_argument(
        "--experiment",
        default=None,
        help="Hydra experiment to build the server from; defaults to the Zeva recipe when the "
        "CTE artifacts are given, else the nano recipe",
    )
    ap.add_argument("--action-stats-path", default=os.environ.get("XHAND_ACTION_STATS_PATH"))
    ap.add_argument("--wan-vae-path", default=os.environ.get("WAN_VAE_PATH"))
    ap.add_argument("--port", type=int, default=8998)
    ap.add_argument("--requests", type=int, default=6, help="boundary frames to feed (>4 exercises the CTE)")
    ap.add_argument("--action-dim", type=int, default=18)
    ap.add_argument("--state-dim", type=int, default=1972)
    ap.add_argument("--chunk-size", type=int, default=32)
    # Must equal len(_PROPRIO_INDICES) == len(_STATE_INDICES["joint18"]) == 18. The server
    # ACCEPTS this flag but the XHand path never reads it -- it slices `_client_proprio`
    # by its own constant. So it documents the contract; it cannot enforce it.
    ap.add_argument("--proprio-dim", type=int, default=18)
    # Must stay 0: the XHand server raises on anything else (no state row to trim).
    ap.add_argument("--history-length", type=int, default=0)
    ap.add_argument("--startup-timeout", type=float, default=900.0)
    ap.add_argument("--log", default="/tmp/zeva-verify-serving.log")
    args = ap.parse_args()

    # The two Zeva artifacts go together: the server builds the CTE path from the
    # checkpoint and looks up the bank by `--task-context-instruction`, so one without
    # the other either fails at startup or silently skips the injection entirely.
    zeva_enabled = bool(args.cte_checkpoint or args.task_context_bank)
    if bool(args.cte_checkpoint) != bool(args.task_context_bank):
        print("ERROR: --cte-checkpoint and --task-context-bank must be given together", file=sys.stderr)
        return 2
    experiment = args.experiment or ("action_policy_xhand_zeva" if zeva_enabled else "action_policy_xhand_nano")

    checks = [("checkpoint", Path(args.checkpoint) / "model" / ".metadata")]
    if args.cte_checkpoint:
        checks.append(("cte-checkpoint", Path(args.cte_checkpoint)))
    if args.task_context_bank:
        checks.append(("task-context-bank", Path(args.task_context_bank)))
    for name, path in checks:
        if not path.exists():
            print(f"ERROR: {name} missing at {path}", file=sys.stderr)
            if name == "checkpoint":
                print("       --checkpoint must be an iteration dir (…/checkpoints/iter_XXXXXXXX)", file=sys.stderr)
            return 2
    if not args.action_stats_path or not args.wan_vae_path:
        print("ERROR: --action-stats-path / --wan-vae-path (or XHAND_ACTION_STATS_PATH / WAN_VAE_PATH)", file=sys.stderr)
        return 2

    log_path = Path(args.log)
    log_path.write_text("")
    print(f"experiment: {experiment}   zeva(CTE+bank): {zeva_enabled}   proprio_dim: {args.proprio_dim}")
    cmd = [
        sys.executable, "-m", "cosmos_framework.scripts.action_policy_server_xhand",
        "--checkpoint-path", args.checkpoint, "--allow-dcp-checkpoint",
        "--experiment", experiment,
        "--experiment-overrides", f"model.config.tokenizer.vae_path={args.wan_vae_path}",
        "--action-stats-path", args.action_stats_path,
        "--domain-name", "ur7e-xhand",
        "--resolution", "256", "--action-dim", str(args.action_dim),
        "--conditioning-fps", "15", "--proprio-dim", str(args.proprio_dim),
        "--image-height", "256", "--image-width", "512",
        "--action-chunk-size", str(args.chunk_size),
        "--history-length", str(args.history_length),
        "--num-steps", "30", "--guidance", "3.0", "--shift", "5.0",
        "--host", "127.0.0.1", "--port", str(args.port),
        # The official server advances its RNG per request (`_next_seed`), which would
        # make the proprio ablation below meaningless: two requests would differ even
        # with the state held constant. Pinning the seed is what turns it into a
        # two-valued test. Verification only -- never a production setting.
        "--deterministic-seed",
    ]
    if zeva_enabled:
        cmd += [
            "--cte-checkpoint", args.cte_checkpoint,
            "--task-context-bank", args.task_context_bank,
            "--task-context-instruction", args.task_context_instruction,
        ]
    print("launching server ...")
    with log_path.open("w") as sink:
        proc = subprocess.Popen(cmd, stdout=sink, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        wait_for_ready(proc, log_path, args.startup_timeout)
        print("server ready")

        from openpi_client.websocket_client_policy import WebsocketClientPolicy

        client = WebsocketClientPolicy(host="127.0.0.1", port=args.port)
        rng = np.random.default_rng(0)
        for i in range(args.requests):
            obs = synthetic_observation(rng, state_dim=args.state_dim, prompt=args.prompt)
            if i == 0:
                obs["reset_tactile_memory"] = True  # start each run from a clean episode
            result = client.infer(obs)
            actions = np.asarray(result["actions"])
            print(f"  request {i}: actions {actions.shape} range [{actions.min():+.3f}, {actions.max():+.3f}]")

        # Let the server's log sinks flush before we scrape them; SIGTERM at teardown
        # discards buffered loguru output, which is how the first run of this script
        # came back with valid actions but an empty progress log.
        time.sleep(5)
        frames, shapes = parse_log(log_path)
        print()
        print(f"boundary_frames per request: {frames}")
        print(f"action shapes:               {shapes}")

        expected = (args.chunk_size - args.history_length, args.action_dim)
        problems = []
        if not shapes:
            problems.append("server never logged an action shape — inference did not run")
        elif any(s != expected for s in shapes):
            problems.append(f"action shape {shapes[0]} != expected {expected}")
        # The CTE needs >= 2 boundary frames with a matching transition to produce
        # features; below that it returns zeros and the policy runs unaided. A stage-1
        # run has no CTE at all, so it never logs the line and this must be skipped --
        # otherwise the absence of a Zeva module reads as a Zeva failure.
        if not zeva_enabled:
            pass
        elif not frames:
            problems.append("server never logged boundary_frames — the Zeva path was not exercised")
        elif max(frames) < 2:
            problems.append(f"boundary_frames peaked at {max(frames)}; CTE fell back to empty features")
        elif frames != sorted(frames):
            problems.append(f"boundary_frames did not grow monotonically: {frames}")

        # ------------------------------------------------------------ proprio ablation
        #
        # Does the state actually reach the output? Everything above can pass with proprio
        # switched off entirely, because a dropped `proprio` key is dropped silently:
        # `_attach_proprio_condition` returns early, no prefix slots are reserved, and the
        # actions look completely normal. This is the check that can tell the two apart.
        #
        # With `--deterministic-seed` every request reuses the same sampler seed, so two
        # requests differing ONLY in `observation/state` must return byte-identical
        # actions if the state is being discarded, and different actions if it is live.
        # `reset_tactile_memory` on both keeps the CTE buffer identical too.
        print()
        print("proprio ablation (identical pixels and seed, only state differs):")
        base = synthetic_observation(
            np.random.default_rng(7), state_dim=args.state_dim, prompt=args.prompt
        )
        probe = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in base.items()}
        # Perturb exactly the channels `_PROPRIO_INDICES` reads, so the probe state still
        # looks like a real one everywhere else: state[:6] is the arm joints, and the
        # hand positions are the even offsets inside [28:52] (`range(28, 52, 2)`) -- NOT
        # the whole block, half of which is torque and never read.
        probe["observation/state"][:6] += 0.5
        probe["observation/state"][28:52:2] += 0.3

        repeat_obs = dict(base, reset_tactile_memory=True)
        probe_obs = dict(probe, reset_tactile_memory=True)
        a_first = np.asarray(client.infer(repeat_obs)["actions"])
        a_repeat = np.asarray(client.infer(repeat_obs)["actions"])
        a_probe = np.asarray(client.infer(probe_obs)["actions"])

        d_repeat = float(np.abs(a_first - a_repeat).max())
        d_state = float(np.abs(a_first - a_probe).max())
        print(f"  same state twice : max|delta| = {d_repeat:.6f}  <- negative control, expect 0")
        print(f"  state perturbed  : max|delta| = {d_state:.6f}  <- expect > 0")

        if d_state == 0.0:
            problems.append(
                f"proprio ablation: perturbing the {args.proprio_dim}-dim state did not move the actions "
                "at all -- state is being dropped (proprio_condition not enabled, proprio_projector "
                "missing from keys_to_select, or the server slice mismatched)"
            )
        elif d_state <= d_repeat:
            problems.append(
                f"proprio ablation inconclusive: state-induced delta {d_state:.6f} does not exceed "
                f"the repeat noise {d_repeat:.6f}"
            )

        print()
        if problems:
            for p in problems:
                print(f"❌ {p}", file=sys.stderr)
            return 1
        print(f"✅ 服务端加载检查点、协议返回 {expected}、proprio 生效"
              + (f"、CTE 路径活跃（boundary_frames 到 {max(frames)}）" if zeva_enabled else "（阶段一，无 CTE）"))
        return 0
    finally:
        if proc.poll() is None:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        print(f"server log: {log_path}")


if __name__ == "__main__":
    raise SystemExit(main())
