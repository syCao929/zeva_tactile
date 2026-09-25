"""Serial 8-GPU Cosmos pipeline, with stage logs, explicit resume and artifact checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import signal
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="Unique pipeline/run name")
    parser.add_argument(
        "--gpus", default="0,1,2,3,4,5,6,7", help="Exactly eight distinct GPU indices"
    )
    parser.add_argument("--stage1-steps", type=int, default=5000)
    parser.add_argument("--cte-steps", type=int, default=3000)
    parser.add_argument("--stage3-steps", type=int, default=2000)
    parser.add_argument(
        "--stage3", choices=("baseline", "tactile", "both"), default="baseline"
    )
    parser.add_argument(
        "--global-batch", type=int, default=112, help="Stage 1/3 effective batch"
    )
    parser.add_argument(
        "--cte-batch-per-gpu",
        type=int,
        default=8,
        help="CTE global batch = this value * 8",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="DataLoader workers per GPU"
    )
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        default=None,
        help="Original Cosmos DCP for Stage 1",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip completed stages; resume incomplete training",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print all commands without writing files or using GPUs",
    )
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.name):
        parser.error(
            "name must contain only letters, digits, dot, underscore or hyphen"
        )
    gpus = args.gpus.split(",")
    if len(gpus) != 8 or len(set(gpus)) != 8 or any(not gpu.isdigit() for gpu in gpus):
        parser.error("--gpus requires exactly eight distinct GPU indices")
    for name in (
        "stage1_steps",
        "cte_steps",
        "stage3_steps",
        "global_batch",
        "cte_batch_per_gpu",
        "workers",
        "save_every",
    ):
        if getattr(args, name) < 1:
            parser.error(f"{name} must be positive")
    if args.seed < 0 or args.global_batch % 8:
        parser.error("seed must be nonnegative and global-batch divisible by eight")
    if args.stage1_steps % args.save_every or args.stage3_steps % args.save_every:
        parser.error(
            "Stage 1/3 steps must be multiples of save-every so final DCP checkpoints exist"
        )
    return args


def build_plan(args, root=ROOT):
    root = Path(root)
    python = os.environ.get("COSMOS_PYTHON", str(root / "envs/zeva/bin/python"))
    base = str(
        args.base_checkpoint
        or os.environ.get("BASE_CHECKPOINT_PATH", root / "models/cosmos3-nano")
    )
    data = os.environ.get(
        "XHAND_DATA_ROOT", str(root / "datasets/press_button_4_times_merged_filtered")
    )
    vae = os.environ.get(
        "WAN_VAE_PATH", str(root / "models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth")
    )
    pipeline = root / "runs/pipelines" / args.name
    latent = pipeline / "vae_cache"
    features = pipeline / "cte_features"
    cte_run = pipeline / "cte"
    cte_checkpoint = cte_run / f"cte_step_{args.cte_steps:06d}.pt"
    policy_root = root / "runs/zeva/comparison_cosmos"
    stage1_run = policy_root / f"{args.name}-stage1"
    stage1_checkpoint = stage1_run / "checkpoints" / f"iter_{args.stage1_steps:09d}"
    torchrun = [
        python,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc_per_node=8",
        "-m",
    ]
    common = [
        "--gpus",
        args.gpus,
        "--global-batch",
        str(args.global_batch),
        "--workers",
        str(args.workers),
        "--save-every",
        str(args.save_every),
        "--seed",
        str(args.seed),
    ]
    vae_args = ["--vae-path", vae, "--dataset-root", data, "--output", str(latent)]
    feature_args = [
        "--cte-checkpoint",
        str(cte_checkpoint),
        "--latent-cache",
        str(latent),
        "--output",
        str(features),
    ]
    stages = [
        {
            "name": "01-stage1",
            "commands": [
                [
                    "bash",
                    str(root / "tools/train-cosmos-base.sh"),
                    *common,
                    "--base-checkpoint",
                    base,
                    "--run-name",
                    f"{args.name}-stage1",
                    "--steps",
                    str(args.stage1_steps),
                ]
            ],
            "artifact": str(stage1_checkpoint / "model/.metadata"),
            "kind": "policy",
            "run": str(stage1_run),
        },
        {
            "name": "02-vae-cache",
            "commands": [
                [*torchrun, "cosmos_framework.zeva_training.vae_cache", *vae_args],
                [
                    python,
                    "-m",
                    "cosmos_framework.zeva_training.vae_cache",
                    *vae_args,
                    "--rebuild-manifest",
                ],
            ],
            "artifact": str(latent / "manifest.json"),
            "kind": "latent",
        },
        {
            "name": "03-cte",
            "commands": [
                [
                    *torchrun,
                    "cosmos_framework.zeva_training.train_cte",
                    "--cache-dir",
                    str(latent),
                    "--output",
                    str(cte_run),
                    "--steps",
                    str(args.cte_steps),
                    "--batch-size",
                    str(args.cte_batch_per_gpu),
                    "--num-workers",
                    str(args.workers),
                    "--save-every",
                    str(args.save_every),
                    "--seed",
                    str(args.seed),
                ]
            ],
            "artifact": str(cte_checkpoint),
            "kind": "cte",
            "run": str(cte_run),
        },
        {
            "name": "04-cte-features",
            "commands": [
                [
                    *torchrun,
                    "cosmos_framework.zeva_training.cte_features",
                    *feature_args,
                ],
                [
                    python,
                    "-m",
                    "cosmos_framework.zeva_training.cte_features",
                    *feature_args,
                    "--rebuild-manifest",
                ],
            ],
            "artifact": str(features / "manifest.json"),
            "kind": "features",
            "latent_manifest": str(latent / "manifest.json"),
        },
    ]
    modes = ("baseline", "tactile") if args.stage3 == "both" else (args.stage3,)
    for index, mode in enumerate(modes, 5):
        run = policy_root / f"{args.name}-stage3-{mode}"
        stages.append(
            {
                "name": f"{index:02d}-stage3-{mode}",
                "commands": [
                    [
                        "bash",
                        str(root / f"tools/train-cosmos-{mode}.sh"),
                        *common,
                        "--base-checkpoint",
                        str(stage1_checkpoint),
                        "--cte-cache",
                        str(features),
                        "--pair-name",
                        f"{args.name}-pim",
                        "--run-name",
                        run.name,
                        "--steps",
                        str(args.stage3_steps),
                    ]
                ],
                "artifact": str(
                    run
                    / "checkpoints"
                    / f"iter_{args.stage3_steps:09d}"
                    / "model/.metadata"
                ),
                "kind": "policy",
                "run": str(run),
            }
        )
    return pipeline, stages


def validate_artifact(stage):
    path = Path(stage["artifact"])
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Stage did not produce its final artifact: {path}")
    if stage["kind"] == "policy":
        if not list(path.parent.glob("*.distcp")):
            raise RuntimeError(f"No policy weight shards: {path.parent}")
    elif stage["kind"] in ("latent", "features"):
        manifest = json.loads(path.read_text())
        if manifest.get("camera_contract") != "xhand_three_view_v2_wrist_right":
            raise RuntimeError(f"Wrong camera contract: {path}")
        entries = manifest["episodes"]
        if len(entries) < 8 or not all(
            (path.parent / row["file"]).is_file() for row in entries
        ):
            raise RuntimeError(f"Incomplete cache or fewer than eight episodes: {path}")
        if stage["kind"] == "features":
            latent = json.loads(Path(stage["latent_manifest"]).read_text())
            if {row["episode_id"] for row in entries} != {
                row["episode_id"] for row in latent["episodes"]
            }:
                raise RuntimeError(
                    "Feature cache does not cover every latent-cache episode"
                )


def main(argv=None):
    args = parse_args(argv)
    pipeline, stages = build_plan(args, ROOT)
    if args.dry_run:
        for stage in stages:
            print(f"\n[{stage['name']}] -> {stage['artifact']}")
            for command in stage["commands"]:
                print(f"CUDA_VISIBLE_DEVICES={args.gpus} " + shlex.join(command))
        return 0
    # Never let inherited torchrun rank variables turn manifest aggregation into a worker.
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE"):
        os.environ.pop(key, None)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    import torch

    if torch.cuda.device_count() != 8:
        raise RuntimeError(
            f"Expected eight visible CUDA GPUs, found {torch.cuda.device_count()}"
        )
    if args.stage3 in ("tactile", "both"):
        encoder = Path(
            os.environ.get(
                "TACTILE_ENCODER_CHECKPOINT",
                ROOT / "models/zeva/tactile_patch_encoder_19999.pt",
            )
        )
        if not encoder.is_file() or encoder.suffix != ".pt":
            raise FileNotFoundError(f"Missing converted tactile encoder: {encoder}")
    sources = [
        Path(__file__),
        ROOT / "tools/train-cosmos-three-stage.sh",
        ROOT / "tools/cosmos-comparison-common.sh",
    ]
    sources += sorted(
        (ROOT / "cosmos-framework/cosmos_framework/zeva_training").glob("*.py")
    )
    contract = {
        "stages": stages,
        "options": {
            k: str(v) if isinstance(v, Path) else v
            for k, v in vars(args).items()
            if k not in ("resume", "dry_run")
        },
        "sources": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
        },
        "environment": {
            key: os.environ.get(key, "")
            for key in (
                "XHAND_DATA_ROOT",
                "XHAND_ACTION_STATS_PATH",
                "WAN_VAE_PATH",
                "QWEN_VLM_PATH",
                "TACTILE_ENCODER_CHECKPOINT",
                "COSMOS_PYTHON",
            )
        },
    }
    plan_path = pipeline / "plan.json"
    if args.resume:
        if not plan_path.is_file() or json.loads(plan_path.read_text()) != contract:
            raise RuntimeError(
                "Resume requires the same existing name, options, resources and training source files"
            )
    else:
        if pipeline.exists() or any(
            Path(s["run"]).exists() for s in stages if "run" in s
        ):
            raise RuntimeError(
                "Output already exists; choose a new --name or use --resume"
            )
        pipeline.mkdir(parents=True)
        plan_path.write_text(json.dumps(contract, indent=2) + "\n")
    log_dir = ROOT / "logs" / args.name
    log_dir.mkdir(parents=True, exist_ok=True)
    child = None

    def stop(signum, _frame):
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    for stage in stages:
        done = pipeline / f"{stage['name']}.done"
        if done.exists():
            validate_artifact(stage)
            print(f"SKIP completed {stage['name']}", flush=True)
            continue
        print(
            f"START {stage['name']} | logs: {log_dir / (stage['name'] + '.log')}",
            flush=True,
        )
        with (log_dir / f"{stage['name']}.log").open("a") as log:
            for command in stage["commands"]:
                command = list(command)
                if (
                    args.resume
                    and stage["kind"] == "policy"
                    and (Path(stage["run"]) / "launch_manifest.json").is_file()
                ):
                    command.append("--resume")
                if (
                    args.resume
                    and stage["kind"] == "cte"
                    and (Path(stage["run"]) / "cte_latest.pt").is_file()
                ):
                    command.append("--resume")
                log.write("\n$ " + shlex.join(command) + "\n")
                log.flush()
                child = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                code = child.wait()
                child = None
                if code:
                    raise RuntimeError(
                        f"{stage['name']} failed (exit {code}); see {log_dir / (stage['name'] + '.log')}"
                    )
        validate_artifact(stage)
        done.write_text(stage["artifact"] + "\n")
        print(f"DONE {stage['name']}", flush=True)
    print(f"All stages complete. Logs: {log_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
