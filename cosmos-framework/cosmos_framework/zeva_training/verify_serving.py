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
receives plausible-looking actions would never notice.  So this script checks three
independent things:

1. **The checkpoint loads** with the Zeva modules present (a stage-2 DCP, not a
   Phase-1 one — the latter has no ``behavior_*`` tensors and fails at load).
2. **The protocol works**: each request returns ``[chunk - history_length, action_dim]``.
3. **The CTE path ran**: the server logs ``boundary_frames=N`` per request and N grows
   request over request.  If it stalls at 0 or 1, the features are zeros and the
   policy is running unaided.

Usage::

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m cosmos_framework.zeva_training.verify_serving \\
        --checkpoint "$ZEVA_WORK/runs/zeva/zeva_xhand/action_policy_xhand_zeva/checkpoints/iter_00000500" \\
        --cte-checkpoint "$ZEVA_WORK/runs/zeva_cte/cte-v2-20260921/cte_step_003000.pt" \\
        --task-context-bank "$ZEVA_WORK/datasets/xhand_task_context_bank.pt" \\
        --task-context-instruction PressButton4Times
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

    state = np.zeros(state_dim, dtype=np.float32)
    state[:6] = rng.uniform(-1.5, 1.5, 6)  # arm joints
    state[12:28] = rng.uniform(-0.3, 0.3, 16)  # ee pose
    state[28:52] = rng.uniform(-0.5, 0.5, 24)  # hand joints
    state[52:] = rng.uniform(0, 60, state_dim - 52)  # tactile (unused, but the client sends it)
    return {
        "prompt": prompt,
        "observation.images.cam_left": left,
        "observation.images.cam_front": front,
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
    ap.add_argument("--checkpoint", required=True, help="stage-2 DCP iteration directory")
    ap.add_argument("--cte-checkpoint", required=True)
    ap.add_argument("--task-context-bank", required=True)
    ap.add_argument("--task-context-instruction", default="PressButton4Times")
    ap.add_argument("--experiment", default="action_policy_xhand_zeva")
    ap.add_argument("--action-stats-path", default=os.environ.get("XHAND_ACTION_STATS_PATH"))
    ap.add_argument("--wan-vae-path", default=os.environ.get("WAN_VAE_PATH"))
    ap.add_argument("--port", type=int, default=8998)
    ap.add_argument("--requests", type=int, default=6, help="boundary frames to feed (>4 exercises the CTE)")
    ap.add_argument("--action-dim", type=int, default=18)
    ap.add_argument("--state-dim", type=int, default=1972)
    ap.add_argument("--chunk-size", type=int, default=32)
    ap.add_argument("--history-length", type=int, default=1)
    ap.add_argument("--startup-timeout", type=float, default=900.0)
    ap.add_argument("--log", default="/tmp/zeva-verify-serving.log")
    args = ap.parse_args()

    for name, path in (
        ("checkpoint", Path(args.checkpoint) / "model" / ".metadata"),
        ("cte-checkpoint", Path(args.cte_checkpoint)),
        ("task-context-bank", Path(args.task_context_bank)),
    ):
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
    cmd = [
        sys.executable, "-m", "cosmos_framework.scripts.action_policy_server_xhand",
        "--checkpoint-path", args.checkpoint, "--allow-dcp-checkpoint",
        "--experiment", args.experiment,
        "--experiment-overrides", f"model.config.tokenizer.vae_path={args.wan_vae_path}",
        "--action-stats-path", args.action_stats_path,
        "--domain-name", "ur7e-xhand",
        "--resolution", "256", "--action-dim", str(args.action_dim),
        "--conditioning-fps", "15", "--proprio-dim", "22",
        "--image-height", "256", "--image-width", "512",
        "--action-chunk-size", str(args.chunk_size),
        "--history-length", str(args.history_length),
        "--num-steps", "30", "--guidance", "3.0", "--shift", "5.0",
        "--cte-checkpoint", args.cte_checkpoint,
        "--task-context-bank", args.task_context_bank,
        "--task-context-instruction", args.task_context_instruction,
        "--host", "127.0.0.1", "--port", str(args.port),
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
            obs = synthetic_observation(rng, state_dim=args.state_dim, prompt=args.task_context_instruction)
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
        # features; below that it returns zeros and the policy runs unaided.
        if not frames:
            problems.append("server never logged boundary_frames — the Zeva path was not exercised")
        elif max(frames) < 2:
            problems.append(f"boundary_frames peaked at {max(frames)}; CTE fell back to empty features")
        elif frames != sorted(frames):
            problems.append(f"boundary_frames did not grow monotonically: {frames}")

        print()
        if problems:
            for p in problems:
                print(f"❌ {p}", file=sys.stderr)
            return 1
        print(f"✅ 服务端加载 stage2 检查点、协议返回 {expected}、CTE 路径活跃"
              f"（boundary_frames 到 {max(frames)}）")
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
