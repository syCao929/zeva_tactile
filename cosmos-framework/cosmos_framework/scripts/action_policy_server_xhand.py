# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""WebSocket policy server for the UR7e + XHand policy, speaking the openpi /
TactileTTT robot-client protocol.

The robot client (``TactileTTT_client_multi.py``) is **unmodified**. It is
data-driven off the dataset's ``meta/info.json``, so it sends whichever cameras
and action names that file declares::

    observation/state            [state_dim] float32   (info.json state order)
    observation/<camera>_image   HWC uint8             (one key per camera)
    prompt                       str
    current_action_step          int
    reset_tactile_memory         bool                  (first request of a run)

and expects back::

    {"actions": [T, action_dim] float32}               (raw joint positions)

Transport is identical to the rest of the stack (openpi ``WebsocketPolicyServer``,
msgpack + NumPy), so nothing below the message schema needs to change.

Three things this server does that the RoboCasa one does not:

1. **View composition.** The client sends cameras separately; training fed the
   loader a ``left | wrist`` horizontal composite with ``wrist = cam_front``
   (the source has no wrist camera, see ``xhand_lerobot_dataset``). This mirrors
   ``XHandLeRobotDataset._compose_video`` exactly so the policy sees
   training-distribution input.

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
   needs no code change. See ``_BoundaryBuffer`` for the attribution rule.

   This assumes the robot executed the actions this server returned, in order.
   If the client clamps or drops actions (check for action-deadzone flags), the
   reconstructed transition differs from what physically happened — at that
   point, send the history from the client instead (it already does exactly this
   for tactile state via its ``StateHistoryBuffer``).

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
import logging as log
import socket
from typing import Any

import numpy as np
import torch

from cosmos_framework.data.generator.action.action_processing import resolve_action_normalization
from cosmos_framework.data.generator.action.domain_utils import get_domain_id
from cosmos_framework.scripts.action_policy_server_robocasa365_zeva import (
    RobolabPolicyService,
    RobolabServerArgs,
    _build_data_batch_from_sample,
    _ensure_rgb_uint8_image,
    _load_openpi_websocket_policy_server,
)

# One CTE timestep == one Wan-VAE latent frame == four raw controls.
# Keep in sync with CausalTransitionEncoderConfig.effect_window_transitions.
_CTE_STRIDE = 4

# Cap on boundary frames kept per episode. The reference evaluator sends the whole
# episode history; because the tokenizer is causal we can keep every latent anyway,
# so this only bounds transformer memory, not correctness. 64 frames = 256 controls.
_CTE_MAX_BOUNDARY_FRAMES = 64

# Left | wrist composite: the source data has no wrist camera, cam_front doubles
# for it (see xhand_lerobot_dataset._CAMERAS).
_COMPOSITE_LEFT_CAMERA = "observation.images.cam_left"
_COMPOSITE_WRIST_CAMERA = "observation.images.cam_front"

# observation.state layout of the ur7e_xhand embodiment [1972]:
#   [0:6] arm joints pos, [6:12] arm joints vel, [12:28] ee pose,
#   [28:52] hand joints pos/torque interleaved, [52:1972] tactile.
# Proprio mirrors XHandLeRobotDataset's "arm22": arm pos + ee pose.
_PROPRIO_INDICES = tuple(range(6)) + tuple(range(12, 28))


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
        if self.num_frames < 2 or len(self._actions) != self.num_frames - 1:
            return None
        latents = torch.stack(list(self._latents), dim=0).unsqueeze(0)
        transitions = np.stack(list(self._actions), axis=0)[None]
        return latents, transitions


class XHandServerArgs(RobolabServerArgs):
    """``RobolabServerArgs`` plus the XHand / TactileTTT-protocol knobs."""

    action_stats_path: str | None = None
    """``meta/feature_stats_compact.json``; drives action de-normalization. Required when the policy was trained with normalization."""

    action_normalization: str | None = "minmax"
    """Normalization scheme used at training time. ``None`` disables de-normalization."""

    num_cameras_expected: int = 2
    """Number of client cameras the composite consumes (left + wrist). Informational, used for validation."""

    cte_boundary_frames: int = _CTE_MAX_BOUNDARY_FRAMES
    """Max boundary frames retained per episode."""

    require_cte_history: bool = False
    """If set, refuse to serve until the CTE buffer holds at least one full effect window."""


class XHandPolicyService(RobolabPolicyService):
    """Zeva policy service with a TactileTTT-protocol observation adapter."""

    def __init__(self, args: XHandServerArgs) -> None:
        super().__init__(args)
        self.xargs = args
        if self.cfg.action_dim != 18:
            log.warning(
                f"[xhand-policy-server] action_dim={self.cfg.action_dim}; the ur7e_xhand "
                "embodiment is 18-D (6 arm + 12 hand joint positions). Check --action-dim."
            )

        self._normalizer = self._build_normalizer(args)
        self._cte_buffer = _BoundaryBuffer(
            stride=_CTE_STRIDE,
            action_dim=self.cfg.action_dim,
            max_frames=int(args.cte_boundary_frames),
        )
        self._episode_reset_pending = True

        if args.require_cte_history and not self._zeva_enabled:
            raise ValueError("--require-cte-history needs Zeva enabled (--cte-checkpoint)")

        log.info(
            f"[xhand-policy-server] ready cameras(composite)={_COMPOSITE_LEFT_CAMERA}+{_COMPOSITE_WRIST_CAMERA} "
            f"proprio_dim={len(_PROPRIO_INDICES)} cte_stride={_CTE_STRIDE} zeva={self._zeva_enabled}"
        )

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
        """Build the ``left | wrist`` composite the training loader produced."""
        left = self._client_image(obs, _COMPOSITE_LEFT_CAMERA)
        wrist = self._client_image(obs, _COMPOSITE_WRIST_CAMERA)
        if left is None or wrist is None:
            present = sorted(k for k in obs if k.endswith("_image"))
            raise ValueError(
                f"Client must send {_COMPOSITE_LEFT_CAMERA!r} and {_COMPOSITE_WRIST_CAMERA!r}; "
                f"got {present or 'no *_image keys'}"
            )
        h, w = self.cfg.image_height, self.cfg.image_width // 2
        if left.shape[:2] != (h, w):
            left = _resize_rgb_uint8(left, (h, w))
        if wrist.shape[:2] != (h, w):
            wrist = _resize_rgb_uint8(wrist, (h, w))
        return np.concatenate([left, wrist], axis=1)

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

    def _build_client_sample(self, obs: dict[str, Any]) -> dict[str, Any]:
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
                "The left panel is the left agent view and the right panel is the front camera."
            ),
            "proprio": torch.from_numpy(self._client_proprio(obs)),
        }
        sample = self._transform(sample, self.cfg.resolution)
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
        global_feature = self._task_context_from_initial_observation(
            self._last_composite, str(self._last_prompt)
        )
        return global_feature, encoded["phase"][:, -1].float(), history, history_valid

    def _empty_causal_interaction_features(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = next(self.model.parameters()).device
        global_feature = self._task_context.unsqueeze(0).to(device=device, dtype=torch.float32)
        phase = torch.zeros((1, self._cte.cfg.phase_dim), dtype=torch.float32, device=device)
        history = torch.zeros((1, 4, self._cte.cfg.effect_dim), dtype=torch.float32, device=device)
        history_valid = torch.zeros((1, 4), dtype=torch.bool, device=device)
        return global_feature, phase, history, history_valid

    # -------------------------------------------------------------------- infer
    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        reset = bool(obs.get("reset_tactile_memory")) or bool(obs.get("cte_reset")) or self._episode_reset_pending
        if reset:
            self._cte_buffer.reset()
            self._episode_reset_pending = False

        image = self._compose_client_view(obs)
        self._last_composite = image
        self._last_prompt = obs.get("prompt")
        sample = self._build_client_sample(obs)

        with self._lock:
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
    server = server_cls(service.infer, host=args.host, port=int(args.port), metadata={"server": "xhand-zeva"})
    log.info(f"[xhand-policy-server] listening on {hostname}:{int(args.port)}")
    server.serve_forever()


def main() -> None:
    import tyro

    args = tyro.cli(XHandServerArgs, description=__doc__)
    serve(args)


if __name__ == "__main__":
    main()
