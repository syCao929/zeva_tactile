# XHand tactile inference

The tactile policy consumes the same raw 30-frame, 15 Hz window as training. The
robot client collects **every control frame** and queries the policy every four
controls. A server receiving only query observations cannot reconstruct the
intermediate tactile frames.

## Server

The current base policy is `runs/zeva/action_xhand/v3-joint18-20260925`, with
`joint18` proprio. Select an actually saved checkpoint from its `checkpoints/`
directory for both baseline and tactile stage-2 training. A new V3 tactile
stage-2 checkpoint is **not yet listed as available** here; complete that training
before deploying. A base-policy checkpoint cannot replace a tactile stage-2
checkpoint. Retired V2 policies and their stage-2 descendants are not deployment
inputs.

The shared resources remain CTE v4
(`runs/zeva_cte/cte-v4-20260924/cte_step_003000.pt`),
`datasets/xhand_cte_features_v4` and `datasets/xhand_task_context_bank.pt`.
The shared VAE latent cache `datasets/xhand_cte_cache_v2` belongs to this CTE
pipeline; its name does not refer to the retired V2 policy.

After training, set `V3_TACTILE_CHECKPOINT` to its saved `iter_XXXXXXXX` directory.
Export the local frozen encoder path before starting the server; it is required
even when loading the policy checkpoint:

```bash
source /path/to/zeva-work/env.sh
export TACTILE_ENCODER_CHECKPOINT=/path/to/zeva-work/models/zeva/tactile_patch_encoder_19999.pt
: "${V3_TACTILE_CHECKPOINT:?Select a saved tactile stage-2 checkpoint trained from V3 joint18}"
test -f "$V3_TACTILE_CHECKPOINT/model/.metadata" || exit 1
cd /path/to/zeva-work/cosmos-framework
PYTHONPATH=. python -m cosmos_framework.scripts.action_policy_server_xhand \
  --checkpoint-path "$V3_TACTILE_CHECKPOINT" \
  --allow-dcp-checkpoint --experiment action_policy_xhand_zeva_tactile \
  --experiment-overrides \
    "model.config.tokenizer.vae_path=$WAN_VAE_PATH" \
    "model.config.vlm_config.tokenizer.pretrained_model_name=$ZEVA_WORK/models/Qwen3-VL-8B-Instruct" \
  --action-stats-path "$XHAND_ACTION_STATS_PATH" \
  --cte-checkpoint "$ZEVA_WORK/runs/zeva_cte/cte-v4-20260924/cte_step_003000.pt" \
  --task-context-bank "$ZEVA_WORK/datasets/xhand_task_context_bank.pt" \
  --task-context-instruction PressButton4Times \
  --domain-name ur7e-xhand --resolution 256 --action-dim 18 --proprio-dim 18 \
  --conditioning-fps 15 --image-height 256 --image-width 512 \
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
server's `observation.images.cam_left` and `observation.images.cam_front` keys.

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
