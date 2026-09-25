# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""WebSocket policy server for the UR7e + XHand policy, speaking the openpi /
TactileTTT robot-client protocol.

The robot client reads the dataset's ``meta/info.json`` for state/action order.
The base observation schema is::

    observation/state            [state_dim] float32   (info.json state order)
    observation.images.<camera>  HWC uint8             (cam_front, cam_left, cam_right)
    prompt                       str
    current_action_step          int
    reset_tactile_memory         bool                  (first request of a run)

and expects back::

    {"actions": [T, action_dim] float32}               (raw joint positions)

Transport is identical to the rest of the stack (openpi ``WebsocketPolicyServer``,
msgpack + NumPy), so nothing below the message schema needs to change.

Three things this server does that the RoboCasa one does not:

1. **View composition.** All three cameras use the shared training compositor:
   wrist-mounted cam_right above external cam_front and cam_left.

2. **Action de-normalization.** Training applied ``minmax`` over the dataset's
   own joint-position ranges, so — unlike RoboCasa, whose arm7 channels happen
   to have ``-1/+1`` stats and whose normalization is therefore the identity —
   the model's output must be inverted before it goes back to the robot.

3. **Server-side CTE history.** Zeva's CTE expects one frame every
   ``cte_stride`` (=4) raw controls plus the four actions executed between them.
   The client queries once per chunk and executes the first ``query_frequency``
   actions, so running it with ``--query-frequency 4`` makes each request equal
   exactly one CTE transition. The server then reconstructs
   ``cte_boundary_images`` / ``cte_transition_actions`` itself and the client
   must execute four actions per query. See ``_BoundaryBuffer`` for the attribution rule.

   This assumes the robot executed the actions this server returned, in order.
   If the client clamps or drops actions (check for action-deadzone flags), the
   reconstructed transition differs from what physically happened — at that
   point this reconstruction cannot represent the executed history.

Tactile checkpoints additionally require a dense, client-sampled 15 Hz window:
``tactile_state``, ``tactile_valid``, ``tactile_frame_indices``, ``tactile_fps``,
``control_frame_index`` and ``episode_id``. See ``docs/xhand_tactile_serving.md``
and ``tools/run_xhand_tactile_client.py`` for the adapter. Query frames alone
cannot reconstruct this history.

Usage::

    PYTHONPATH=. python -m cosmos_framework.scripts.action_policy_server_xhand \\
      --checkpoint-path "$BASE_POLICY_CKPT" --allow-dcp-checkpoint \\
      --experiment action_policy_xhand_nano \\
      --experiment-overrides model.config.tokenizer.vae_path="$WAN_VAE_PATH" \\
      --action-stats-path "$XHAND_ACTION_STATS_PATH" \\
      --cte-checkpoint "$ZEVA_RELEASE/weights/stage1/zeva_cte.pt" \\
      --task-context-bank "$ZEVA_RELEASE/weights/stage1/train_memory_effect_v3.pt" \\
      --domain-name ur7e-xhand --action-dim 18 --conditioning-fps 15 \\
      --rescan-cte-checkpoint \\
      --host 0.0.0.0 --port 8990

Then on the robot, pointing the existing client at this server::

    python TactileTTT_client_multi.py --server-port 8990 --query-frequency 4 ...
"""

from __future__ import annotations

import collections
import json
import socket
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from cosmos_framework.data.generator.action.action_processing import resolve_action_normalization
from cosmos_framework.data.generator.action.domain_utils import get_domain_id
from cosmos_framework.data.generator.action.xhand_camera import (
    CAMERA_KEYS,
    VIEW_DESCRIPTION,
    compose_xhand_views,
    require_camera_contract,
)
from cosmos_framework.inference.xhand_pim import XHandPIMContext
from cosmos_framework.inference.xhand_tactile import (
    TACTILE_FPS,
    TACTILE_MEMORY_STEPS,
    TactileWindow,
    XHandTactileInput,
)
from cosmos_framework.scripts.action_policy_server_robocasa365_zeva import (
    RobolabPolicyService,
    RobolabServerArgs,
    _build_data_batch_from_sample,
    _ensure_rgb_uint8_image,
    _load_openpi_websocket_policy_server,
)

# The framework's loguru wrapper, NOT stdlib `logging`: stdlib drops INFO records
# unless a handler is configured, so every operational message from this module
# (startup, per-request action shape, boundary-frame count) would vanish silently —
# exactly the lines you need when debugging a robot deployment.
from cosmos_framework.utils import log

# One CTE timestep == one Wan-VAE latent frame == four raw controls.
# Keep in sync with CausalTransitionEncoderConfig.effect_window_transitions.
_CTE_STRIDE = 4

# Boundary frames kept per episode. **This must equal `window_latents` from CTE
# training** (`zeva_training/train_cte.py`, default 17) and the feature cache
# (`zeva_training/cte_features.py`), and it is a correctness knob, not a memory knob:
# the CTE's temporal mixers are GRUs with no positional embedding, so `phase[:, -1]`
# depends on how many frames preceded the current one. Feed 64 frames here and the
# model would see a state it was never trained on, silently shifting every injected
# phase/effect away from the cached features the stage-2 policy was fit against.
# (The reference RoboCasa server has this same freedom and no such guarantee, because
# its features are recomputed per request rather than looked up from a cache.)
_CTE_MAX_BOUNDARY_FRAMES = 17

# observation.state layout of the ur7e_xhand embodiment [1972]:
#   [0:6] arm joints pos, [6:12] arm joints vel, [12:28] ee pose,
#   [28:52] hand joints pos/torque interleaved, [52:1972] tactile.
# Proprio mirrors XHandLeRobotDataset's "joint18" -- the arm + hand joint *positions*,
# i.e. exactly the 18-D absolute-joint-position space the action lives in.
#
# Two traps here, both silent (see `_client_proprio` -- it only length-checks):
#   * `[28:52]` interleaves position and torque, so the hand positions are the EVEN
#     offsets only. Reading `range(28, 40)` would feed the policy torque readings as
#     joint angles.
#   * These indices must stay in lockstep with the dataset's `_STATE_INDICES["joint18"]`;
#     `action_policy_server_xhand_test.py` asserts the two are equal.
#
# `joint18` reads joint positions directly from the controllers on both the
# data-collection and deployment rigs.
_PROPRIO_INDICES = tuple(range(6)) + tuple(range(28, 52, 2))


class _BoundaryBuffer:
    """Rebuilds Zeva's CTE history from a chunk-cadence client.

    The client sends one observation per query and then executes the first
    ``stride`` actions of the chunk this server returned. So when request *t+1*
    arrives, the actions returned at request *t* are exactly the transition
    between boundary frame *t* and *t+1*. That attribution is what makes a
    client with ``query_frequency == stride`` equivalent to one that ships
    ``cte_boundary_images`` / ``cte_transition_actions`` explicitly.

    Latents are cached on append: the Wan tokenizer is causal, so encoding a
    frame once and keeping it equals re-encoding the whole prefix (which is what
    the RoboCasa server does per request), at a fraction of the cost.
    """

    def __init__(self, *, stride: int, action_dim: int, max_frames: int) -> None:
        if stride < 1:
            raise ValueError("stride must be >= 1")
        if max_frames < 2:
            raise ValueError("max_frames must be >= 2")
        self.stride = int(stride)
        self.action_dim = int(action_dim)
        self.max_frames = int(max_frames)
        self._latents: collections.deque[torch.Tensor] = collections.deque(maxlen=self.max_frames)
        self._actions: collections.deque[np.ndarray] = collections.deque(maxlen=self.max_frames - 1)
        self._pending: np.ndarray | None = None

    def reset(self) -> None:
        self._latents.clear()
        self._actions.clear()
        self._pending = None

    def observe(self, latent: torch.Tensor) -> None:
        """Append a boundary frame's latent, attributing the previous chunk's actions.

        A missing pending chunk *after* the first frame means a request was lost
        or failed mid-flight, so the attribution for everything buffered so far is
        no longer trustworthy. Drop the window and restart from this frame rather
        than let the transition count drift out of step with the frame count.
        """
        if self._pending is None and self._latents:
            self._latents.clear()
            self._actions.clear()
        if self._pending is not None:
            self._actions.append(self._pending)
            self._pending = None
        self._latents.append(latent)

    def record_returned_actions(self, actions: np.ndarray) -> None:
        """Remember the first ``stride`` actions of the chunk just generated."""
        self._pending = np.asarray(actions[: self.stride], dtype=np.float32)

    @property
    def num_frames(self) -> int:
        return len(self._latents)

    def as_cte_inputs(self) -> tuple[torch.Tensor, np.ndarray] | None:
        """Return ``(latents [1,T,C,H,W], transitions [1,T-1,stride,A])`` or None.

        The transition count must trail the frame count by exactly one; anything
        else means a request was dropped or the cadence changed, and guessing
        would silently mis-attribute actions.
        """
        if self.num_frames < 1 or len(self._actions) != self.num_frames - 1:
            return None
        latents = torch.stack(list(self._latents), dim=0).unsqueeze(0)
        transitions = (np.stack(list(self._actions), axis=0)[None] if self._actions
                       else np.empty((1, 0, self.stride, self.action_dim), dtype=np.float32))
        return latents, transitions


class XHandServerArgs(RobolabServerArgs):
    """``RobolabServerArgs`` plus the XHand / TactileTTT-protocol knobs."""

    action_stats_path: str | None = None
    """``meta/feature_stats_compact.json``; drives action de-normalization. Required when the policy was trained with normalization."""

    action_normalization: str | None = "minmax"
    """Normalization scheme used at training time. ``None`` disables de-normalization."""

    num_cameras_expected: int = 3
    camera_view_size: int = 256
    """Number of client cameras the composite consumes (front + left + wrist). Informational, used for validation."""

    resolution: str | None = "256"
    image_height: int = 576
    image_width: int = 512

    pim_demonstration: str | None = None
    """Optional completed demonstration features_*.npz to prefill each new scene."""

    cte_boundary_frames: int = _CTE_MAX_BOUNDARY_FRAMES
    """Max boundary frames retained per episode."""

    require_cte_history: bool = False
    """If set, refuse to serve until the CTE buffer holds at least one full effect window."""


class XHandPolicyService(RobolabPolicyService):
    """Zeva policy service with a TactileTTT-protocol observation adapter."""

    def __init__(self, args: XHandServerArgs) -> None:
        # Use the saved training configuration, not a rebuilt recipe whose defaults
        # may have changed since the checkpoint was trained.
        checkpoint = Path(args.checkpoint_path).expanduser().resolve()
        if checkpoint.name == "model":
            checkpoint = checkpoint.parent
        config_path = checkpoint.parent.parent / "config.yaml"
        if not config_path.is_file():
            raise ValueError(f"Three-view XHand serving requires saved training config: {config_path}")
        trained = yaml.safe_load(config_path.read_text())
        dataset = trained["dataloader_train"]["dataloader"]["datasets"]["xhand"]["dataset"]
        if dataset.get("camera_layout") != "three_view_grid":
            raise ValueError("This XHand server requires a three-view checkpoint; the old two-view policy is incompatible")
        if dataset.get("view_size") != args.camera_view_size or str(dataset.get("resolution")) != str(args.resolution):
            raise ValueError("Server camera_view_size/resolution must match the saved training configuration")
        if args.cte_checkpoint is not None:
            require_camera_contract(
                torch.load(args.cte_checkpoint, map_location="cpu", weights_only=False), str(args.cte_checkpoint)
            )
        super().__init__(args)
        self.xargs = args
        if self.cfg.action_dim != 18:
            log.warning(
                f"[xhand-policy-server] action_dim={self.cfg.action_dim}; the ur7e_xhand "
                "embodiment is 18-D (6 arm + 12 hand joint positions). Check --action-dim."
            )

        # XHand is "Case A" in the sequence planner (transforms.py:318-327): unlike DROID,
        # nothing is ever prepended to the action stream, so the model emits exactly
        # `action_chunk_size` *predicted* rows and `infer`'s `action[history_length:]`
        # would silently drop a real action -- shifting every executed command, and the
        # boundary actions `_BoundaryBuffer` hands to the CTE, one frame late relative to
        # training. DROID/RoboCasa can trim because their row 0 is the state anchor; here
        # there is no such row. Zeva's own deploy recipes run `--history-length 0`.
        if int(self.cfg.history_length) != 0:
            raise ValueError(
                f"--history-length={self.cfg.history_length} is not valid for the XHand server: "
                "no state row is prepended to the action stream (transforms.py Case A), so there "
                "is nothing to trim and a positive value silently discards a predicted action. "
                "Use --history-length 0 (and --use-state is a no-op here, leave it off)."
            )

        self._normalizer = self._build_normalizer(args)
        self._cte_buffer = _BoundaryBuffer(
            stride=_CTE_STRIDE,
            action_dim=self.cfg.action_dim,
            max_frames=int(args.cte_boundary_frames),
        )
        self._episode_reset_pending = True
        self._init_tactile_input()
        behavior = self.model.config.behavior_stage2
        if behavior.pim_memory_enabled and not self._zeva_enabled:
            raise ValueError("PIM requires a three-view CTE checkpoint")
        self._pim_context = (
            XHandPIMContext(top_k=behavior.pim_persistent_length, demonstration=args.pim_demonstration)
            if behavior.pim_memory_enabled else None
        )

        if args.require_cte_history and not self._zeva_enabled:
            raise ValueError("--require-cte-history needs Zeva enabled (--cte-checkpoint)")

        log.info(
            f"[xhand-policy-server] ready cameras={CAMERA_KEYS} wrist=cam_right "
            f"proprio_dim={len(_PROPRIO_INDICES)} cte_stride={_CTE_STRIDE} zeva={self._zeva_enabled} "
            f"tactile={self._tactile_input is not None}"
        )

    def _init_tactile_input(self) -> None:
        behavior = self.model.config.behavior_stage2
        self._tactile_input = None
        if not behavior.tactile_enabled:
            return
        if not self._zeva_enabled:
            raise ValueError("Tactile stage-2 serving requires the CTE checkpoint and task-context bank")
        if behavior.tactile_memory_steps != TACTILE_MEMORY_STEPS or self.cfg.conditioning_fps != TACTILE_FPS:
            raise ValueError("XHand tactile serving requires the trained 30-frame window at 15 Hz")
        self._tactile_input = XHandTactileInput(query_stride=_CTE_STRIDE)

    # ---------------------------------------------------------------- normalizer
    def _build_normalizer(self, args: XHandServerArgs) -> Any:
        """Rebuild the training-time action normalizer so output can be inverted."""
        if not args.action_normalization:
            log.warning("[xhand-policy-server] no action_normalization configured; returning raw model output")
            return None
        if not args.action_stats_path:
            raise ValueError(
                "--action-stats-path is required when --action-normalization is set. Point it at the "
                "dataset's meta/feature_stats_compact.json (tools/convert_xhand_dataset.py writes it)."
            )
        raw = json.loads(open(args.action_stats_path).read())
        raw = raw.get("action", raw)
        missing = [k for k in ("min", "max") if k not in raw]
        if missing:
            raise KeyError(f"action stats {args.action_stats_path} lacks {missing}")
        index = torch.tensor(range(self.cfg.action_dim), dtype=torch.long)
        stats = {k: torch.tensor(raw[k], dtype=torch.float32).index_select(0, index) for k in ("min", "max")}
        log.info(f"[xhand-policy-server] action normalizer={args.action_normalization} from {args.action_stats_path}")
        return resolve_action_normalization(args.action_normalization, stats)

    def _denormalize(self, action: np.ndarray) -> np.ndarray:
        if self._normalizer is None:
            return action
        t = torch.from_numpy(np.ascontiguousarray(action)).float()
        return self._normalizer.denormalize_action(t).numpy()

    # ------------------------------------------------------------------ adapter
    def _client_image(self, obs: dict[str, Any], key: str) -> np.ndarray | None:
        value = obs.get(key)
        if value is None:
            return None
        return _ensure_rgb_uint8_image(value, key)

    def _compose_client_view(self, obs: dict[str, Any]) -> np.ndarray:
        """Use exactly the same three-view compositor as the training loader."""
        views = {}
        for name, key in CAMERA_KEYS.items():
            image = self._client_image(obs, key)
            if image is None:
                raise ValueError(f"Client must send {key!r}")
            views[name] = torch.from_numpy(image.copy()).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        composite = compose_xhand_views(views, view_size=self.xargs.camera_view_size)
        return (composite[0].permute(1, 2, 0) * 255).clamp(0, 255).to(torch.uint8).numpy()

    def _client_proprio(self, obs: dict[str, Any]) -> np.ndarray:
        state = np.asarray(obs.get("observation/state"), dtype=np.float32)
        if state.ndim > 1:  # structured client modes send a [K, state_dim] window
            state = state[-1]
        if state.ndim != 1:
            raise ValueError(f"'observation/state' must be 1-D or [K,state_dim], got {state.shape}")
        if state.shape[0] <= max(_PROPRIO_INDICES):
            raise ValueError(
                f"'observation/state' has {state.shape[0]} entries but proprio needs index "
                f"{max(_PROPRIO_INDICES)}; is this the ur7e_xhand dataset's state order?"
            )
        return np.asarray(state[list(_PROPRIO_INDICES)], dtype=np.float32)

    def _build_client_sample(
        self, obs: dict[str, Any], *, tactile_window: TactileWindow | None = None
    ) -> dict[str, Any]:
        """Same sample contract as training, built from the client's message."""
        prompt = obs.get("prompt")
        if not isinstance(prompt, str):
            raise ValueError("'prompt' must be a string")
        image = self._compose_client_view(obs)
        h, w = image.shape[:2]
        video = torch.zeros((3, self.cfg.action_chunk_size + 1, h, w), dtype=torch.uint8)
        video[:, 0] = torch.from_numpy(image.copy()).permute(2, 0, 1)
        sample: dict[str, Any] = {
            "ai_caption": prompt,
            "video": video,
            "action": torch.zeros((self.cfg.action_chunk_size, self.cfg.action_dim), dtype=torch.float32),
            "conditioning_fps": torch.tensor(int(self.cfg.conditioning_fps), dtype=torch.long),
            "mode": "wam",
            "domain_id": torch.tensor(get_domain_id(self.cfg.domain_name), dtype=torch.long),
            "viewpoint": "concat_view",
            "additional_view_description": (
                VIEW_DESCRIPTION
            ),
            "proprio": torch.from_numpy(self._client_proprio(obs)),
        }
        sample = self._transform(sample, self.cfg.resolution)
        if self._tactile_input is not None:
            if tactile_window is None:
                reset = (
                    bool(obs.get("reset_tactile_memory")) or bool(obs.get("cte_reset")) or self._episode_reset_pending
                )
                tactile_window = self._tactile_input.prepare(obs, reset=reset)
            # Attach after the visual transform: preserve raw sensor units and mask.
            sample["tactile_state"] = tactile_window.state
            sample["tactile_valid"] = tactile_window.valid
        if isinstance(sample.get("ai_caption"), dict):
            sample["ai_caption"] = json.dumps(sample["ai_caption"])
        return sample

    # -------------------------------------------------------------- cte history
    def _update_cte_buffer(self, obs: dict[str, Any], image: np.ndarray) -> None:
        """Feed one boundary frame into the rolling CTE window."""
        if not self._zeva_enabled:
            return
        latent = self._encode_cte_frame(image).unsqueeze(0)
        self._cte_buffer.observe(latent[0])

    def _causal_interaction_features_from_buffer(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """CTE features from the server-side buffer (replaces the obs-supplied path)."""
        prepared = self._cte_buffer.as_cte_inputs()
        if prepared is None:
            return self._empty_causal_interaction_features()
        latents, transitions = prepared
        device = next(self.model.parameters()).device
        frames = latents.to(device=device, dtype=torch.float32)
        actions = torch.from_numpy(np.ascontiguousarray(transitions)).to(device=device, dtype=torch.float32)
        valid = torch.ones((1, frames.shape[1]), dtype=torch.bool, device=device)
        transition_valid = torch.ones(actions.shape[:-1], dtype=torch.bool, device=device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            encoded = self._cte(frames, actions, valid, transition_valid)
        completed = encoded["effect_post"][0][encoded["effect_complete"][0]].float()
        history = torch.zeros((1, 4, self._cte.cfg.effect_dim), dtype=torch.float32, device=device)
        history_valid = torch.zeros((1, 4), dtype=torch.bool, device=device)
        take = min(4, completed.shape[0])
        if take:
            history[0, -take:] = completed[-take:]
            history_valid[0, -take:] = True
        global_feature = self._task_context_from_initial_observation(self._last_composite, str(self._last_prompt))
        return global_feature, encoded["phase"][:, -1].float(), history, history_valid

    def _empty_causal_interaction_features(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = next(self.model.parameters()).device
        # `self._task_context` is already [1, 256] (base server: `torch.stack(...).mean(dim=0,
        # keepdim=True)`), and `infer` indexes [0] before handing it to the batch shim, which
        # adds exactly one leading axis. Unsqueezing here yields [1, 1, 256] -> the shim makes
        # it [1, 1, 1, 256] and `as_batch` rejects it.
        global_feature = self._task_context.to(device=device, dtype=torch.float32)
        phase = torch.zeros((1, self._cte.cfg.phase_dim), dtype=torch.float32, device=device)
        history = torch.zeros((1, 4, self._cte.cfg.effect_dim), dtype=torch.float32, device=device)
        history_valid = torch.zeros((1, 4), dtype=torch.bool, device=device)
        return global_feature, phase, history, history_valid

    # -------------------------------------------------------------------- infer
    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            reset = bool(obs.get("reset_tactile_memory")) or bool(obs.get("cte_reset")) or self._episode_reset_pending
            if self._pim_context is not None:
                reset = self._pim_context.prepare(obs, reset=reset)
            tactile_window = None
            if self._tactile_input is not None:
                tactile_window = self._tactile_input.prepare(obs, reset=reset)
            image = self._compose_client_view(obs)
            sample = self._build_client_sample(obs, tactile_window=tactile_window)
            if reset:
                self._cte_buffer.reset()
            self._last_composite = image
            self._last_prompt = obs.get("prompt")
            with torch.inference_mode():
                if self._zeva_enabled:
                    self._update_cte_buffer(obs, image)
                    if self.xargs.require_cte_history and self._cte_buffer.num_frames < _CTE_STRIDE + 1:
                        raise RuntimeError(
                            f"CTE history has {self._cte_buffer.num_frames} boundary frames; "
                            f"need >= {_CTE_STRIDE + 1} (run the client with --query-frequency {_CTE_STRIDE})."
                        )
                    global_feature, phase, effect, effect_valid = self._causal_interaction_features_from_buffer()
                    sample["behavior_global"] = global_feature[0].cpu()
                    sample["behavior_phase"] = phase[0].cpu()
                    sample["behavior_effect"] = effect[0].cpu()
                    sample["behavior_effect_valid"] = effect_valid[0].cpu()
                    if self._pim_context is not None:
                        pp, pe, pv = self._pim_context.observe_and_query(phase[0], effect[0], effect_valid[0])
                        sample.update(behavior_pim_phase=pp, behavior_pim_effect=pe, behavior_pim_valid=pv)

                data_batch = _build_data_batch_from_sample(sample)
                seed = self._next_seed()
                samples = self.model.generate_samples_from_batch(
                    data_batch,
                    guidance=self.cfg.guidance,
                    seed=[seed],
                    num_steps=self.cfg.num_steps,
                    shift=self.cfg.shift,
                )

            action = samples["action"][0][:, : self.cfg.action_dim]
            action = action[self.cfg.history_length :]
            action_np = action.detach().cpu().numpy().astype(np.float32, copy=False)
            action_np = self._denormalize(action_np)

            if self._zeva_enabled:
                self._cte_buffer.record_returned_actions(action_np)
            if tactile_window is not None:
                self._tactile_input.commit(tactile_window)
            self._episode_reset_pending = False

        log.info(
            f"[xhand-policy-server] action shape={action_np.shape} "
            f"min={action_np.min():.4f} max={action_np.max():.4f} "
            f"boundary_frames={self._cte_buffer.num_frames}"
        )
        return {"actions": action_np}


def serve(args: XHandServerArgs) -> None:
    hostname = socket.gethostname()
    log.info(f"[xhand-policy-server] starting host={hostname} bind={args.host}:{int(args.port)}")
    service = XHandPolicyService(args)
    server_cls = _load_openpi_websocket_policy_server()
    # Pass the service *object*: openpi's handler calls ``self._policy.infer(obs)``, so
    # handing it the bound method fails at request time with
    # "'function' object has no attribute 'infer'". Matches the reference servers
    # (action_policy_server_robolab.py:630, action_policy_server_robocasa365_zeva.py:935).
    metadata = {"server": "xhand-zeva", "cte_query_stride": _CTE_STRIDE,
                "cameras": CAMERA_KEYS, "pim_enabled": service._pim_context is not None}
    if service._tactile_input is not None:
        metadata.update(
            tactile_protocol="xhand_dense_v1", tactile_fps=TACTILE_FPS, tactile_memory_steps=TACTILE_MEMORY_STEPS
        )
    server = server_cls(policy=service, host=args.host, port=int(args.port), metadata=metadata)
    log.info(f"[xhand-policy-server] listening on {hostname}:{int(args.port)}")
    server.serve_forever()


def main() -> None:
    import tyro

    args = tyro.cli(XHandServerArgs, description=__doc__)
    serve(args)


if __name__ == "__main__":
    main()
