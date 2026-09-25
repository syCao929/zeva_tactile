# XHand tactile inference

The tactile policy consumes the same raw 30-frame, 15 Hz window as training. The
robot client collects **every control frame** and queries the policy every four
controls. A server receiving only query observations cannot reconstruct the
intermediate tactile frames.

## Server

The current recipes require newly trained three-view policy and CTE checkpoints.
`cam_right` is wrist-mounted; `cam_front` and `cam_left` are external cameras.
The shared compositor places the wrist above the two external views. The old
`v3-joint18` / CTE v4 checkpoints used two external views and are not inputs for
this version. Use `datasets/xhand_cte_cache_threeview` and
`datasets/xhand_cte_features_threeview`; caches carry a checked camera contract.

Set `THREEVIEW_TACTILE_CHECKPOINT` and `THREEVIEW_CTE_CHECKPOINT` to the newly
trained artifacts. Export the local frozen encoder path before starting the server; it is required
even when loading the policy checkpoint:

```bash
source /path/to/zeva-work/env.sh
export TACTILE_ENCODER_CHECKPOINT=/path/to/zeva-work/models/zeva/tactile_patch_encoder_19999.pt
: "${THREEVIEW_TACTILE_CHECKPOINT:?Select a saved three-view tactile/PIM checkpoint}"
test -f "$THREEVIEW_TACTILE_CHECKPOINT/model/.metadata" || exit 1
cd /path/to/zeva-work/cosmos-framework
PYTHONPATH=. python -m cosmos_framework.scripts.action_policy_server_xhand \
  --checkpoint-path "$THREEVIEW_TACTILE_CHECKPOINT" \
  --allow-dcp-checkpoint --experiment action_policy_xhand_zeva_tactile \
  --experiment-overrides \
    "model.config.tokenizer.vae_path=$WAN_VAE_PATH" \
    "model.config.vlm_config.tokenizer.pretrained_model_name=$ZEVA_WORK/models/Qwen3-VL-8B-Instruct" \
  --action-stats-path "$XHAND_ACTION_STATS_PATH" \
  --cte-checkpoint "$THREEVIEW_CTE_CHECKPOINT" \
  --task-context-bank "$ZEVA_WORK/datasets/xhand_task_context_bank.pt" \
  --task-context-instruction PressButton4Times \
  --domain-name ur7e-xhand --resolution 256 --action-dim 18 --proprio-dim 18 \
  --conditioning-fps 15 --image-height 576 --image-width 512 \
  --action-chunk-size 32 --history-length 0 --host 0.0.0.0 --port 8990
```

Choose the CTE checkpoint and bank used for this policy's training features. The
server enables tactile input from the **loaded model configuration**; missing
history on a tactile checkpoint is an error. Baseline checkpoints do not require
the extra fields. `--use-state` is unrelated to the independent proprio prefix
and should remain off; `--history-length` must be zero for XHand.

## Existing robot client adapter

The repository contains a NumPy-only wrapper. Run it in the existing robot/client
environment. It imports the original client without editing that file:

```bash
python /path/to/zeva-work/tools/run_xhand_tactile_client.py \
  --client-source /path/to/FactileLDM/ur7e_xhand_deploy_pi0_client.py -- \
  --dataset-dir /path/to/the/training/lerobot \
  --server-ip SERVER_IP --server-port 8990 --task PressButton4Times \
  --check-config
```

`--check-config` exits before connecting to the server or robot. Remove it for an
authorized robot run and supply the rig's normal hardware options. The wrapper
fixes 15 Hz, four controls per query, 30 consecutive tactile frames, synchronous
queries, smoothing=1 and action scale=1. It maps the client's camera keys to the
server's `observation.images.cam_front`, `observation.images.cam_left`, and
`observation.images.cam_right` keys. All three are required.

The original `StateHistoryBuffer` uses configurable sparse offsets and repeats
the earliest observation before sufficient history exists. The wrapper replaces
that buffer with actual consecutive frame samples and invalid zero padding.
An inference failure terminates the attempt, because continuing with hold
commands would invalidate the server's inferred CTE action history.

The original robot control loop waits for policy inference. The 15 Hz setting
defines control-frame sampling; inference latency can lengthen the interval at
a query. Measure control timing on the deployment machine before claiming
continuous wall-clock 15 Hz performance. Hardware safety limits remain active;
if the robot changes returned commands, the current CTE reconstruction needs an
executed-command feedback interface before its action attribution is exact.

## Request contract

In addition to the normal images, prompt and current `observation/state[1972]`,
each tactile request supplies:

| Field | Meaning |
|---|---|
| `tactile_state` | Float32 `[T,1972]`, `1 <= T <= 30`, oldest to newest, raw dataset channel order and units |
| `tactile_valid` | Boolean `[T]`; only an episode-start left prefix may be invalid |
| `tactile_frame_indices` | Integer `[T]`; consecutive episode control-frame indices, padding is `-1` |
| `control_frame_index` | Current zero-based control tick within the episode, **not** query number |
| `tactile_fps` | `15` |
| `episode_id` | New nonempty string for each attempt |
| `cte_reset` or `reset_tactile_memory` | True on a new attempt or explicit reconnection |

The valid suffix contains all `min(control_frame_index + 1, 30)` available
frames. At tick 0 there is one valid frame; at tick 4 there are five, even though
only two policy queries occurred. The server pads short windows to 30, zeros
invalid rows, checks finite values and verifies that the last raw state equals
the current observation. Sparse, future, missing or duplicate frame indices are
rejected. Successful queries advance by exactly four control ticks.

A fresh attempt starts at tick 0 with a fresh client buffer and episode ID.
An explicit reconnection may carry a complete valid window at a later tick;
the server clears its CTE buffer and warms up visual history again. Tactile
history is supplied independently on every request, so the server never mixes
sensor windows from different attempts. The helper
`cosmos_framework.inference.xhand_tactile_client.TactileClientWindow` is available
for clients with another control loop.

## PIM demonstration conditioning and retries

Stage 3 trains the PIM prompt encoder/projector/gate using an independent,
same-task training demonstration, with 20% context dropout. Query trajectories
and validation trajectories are excluded from support. No failure/retry labels
are inferred from independent successful demonstrations.

To match demonstration-conditioned training at deployment, pass
`--pim-demonstration /path/to/features_000000.npz` to the server. Select a separate
completed demonstration encoded with the same three-view CTE. Without a demo,
PIM starts empty and fills only with observed completed effects.

For retries, add `--pim-episode-id scene-001 --pim-attempt-id 0` **before** the
client adapter's `--` separator. After restoring the same initial scene, rerun
with the same scene ID and attempt 1, then 2, etc. A new scene uses a new ID and
attempt 0. The server keeps PIM across attempts but resets CTE/BIT and tactile
history; it clears PIM at a new scene. Without these IDs, every client reset
starts a new PIM episode, preventing unrelated scenes from sharing memory.

Direct clients send `pim_episode_id` and `pim_attempt_id` on each request. These
are distinct from tactile `episode_id`, which identifies one continuous attempt.
The server must remain running between retries. Memory survives attempts, not
server restarts. PIM currently stores visual CTE effects; tactile still enters
through the current effect residual branch.
