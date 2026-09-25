"""CPU-only evidence checks for the real π0 smoke training and exact resume.

Run with the independent π0 Python. This tool only reads training artifacts and
writes its JSON report; it never moves, prunes, or overwrites a checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


def chunks(tensor, elements=2_000_000):
    flat = tensor.reshape(-1)
    for start in range(0, flat.numel(), elements):
        yield flat[start : start + elements]


def group(key):
    for name in (
        "action_in_proj",
        "action_out_proj",
        "state_proj",
        "action_time_mlp_in",
        "action_time_mlp_out",
    ):
        if key.startswith(name + "."):
            return name
    if ".gemma_expert." in key:
        return "action_expert"
    if ".vision_tower." in key:
        return "vision_encoder"
    return "vision_language_backbone"


def compare_weights(before_path, after_path, *, require_change):
    result = {
        "groups": {},
        "keys_equal": False,
        "all_finite": True,
        "dtype_changes": [],
    }
    with (
        safe_open(before_path, framework="pt", device="cpu") as before,
        safe_open(after_path, framework="pt", device="cpu") as after,
    ):
        before_keys, after_keys = set(before.keys()), set(after.keys())
        result["keys_equal"] = before_keys == after_keys
        result["missing"] = sorted(before_keys - after_keys)
        result["unexpected"] = sorted(after_keys - before_keys)
        for key in sorted(before_keys & after_keys):
            x, y = before.get_tensor(key), after.get_tensor(key)
            stats = result["groups"].setdefault(
                group(key),
                {
                    "tensors": 0,
                    "changed_tensors": 0,
                    "elements": 0,
                    "changed_elements": 0,
                    "max_abs_delta": 0.0,
                },
            )
            if x.shape != y.shape:
                raise ValueError(f"Weight shape changed: {key}: {x.shape} -> {y.shape}")
            if x.dtype != y.dtype:
                result["dtype_changes"].append(key)
            changed = 0
            maximum = 0.0
            for a, b in zip(chunks(x), chunks(y)):
                # A strict restore casts the source into the model's mixed dtypes.
                # Exclude that cast from evidence of optimizer updates.
                a = a.to(b.dtype)
                result["all_finite"] &= bool(torch.isfinite(b).all())
                changed += int(torch.count_nonzero(a != b))
                maximum = max(maximum, float((a.float() - b.float()).abs().max()))
            stats["tensors"] += 1
            stats["changed_tensors"] += int(changed > 0)
            stats["elements"] += y.numel()
            stats["changed_elements"] += changed
            stats["max_abs_delta"] = max(stats["max_abs_delta"], maximum)
            del x, y
    result["changed_elements"] = sum(
        value["changed_elements"] for value in result["groups"].values()
    )
    result["exact_equal"] = (
        result["keys_equal"]
        and not result["changed_elements"]
        and not result["dtype_changes"]
    )
    required = (
        "action_in_proj",
        "action_out_proj",
        "state_proj",
        "action_time_mlp_in",
        "action_time_mlp_out",
        "action_expert",
    )
    result["required_groups_updated"] = all(
        result["groups"].get(name, {}).get("changed_elements", 0) > 0
        for name in required
    )
    result["ok"] = (
        result["keys_equal"]
        and result["all_finite"]
        and (
            result["required_groups_updated"]
            if require_change
            else result["exact_equal"]
        )
    )
    return result


def read_training(checkpoint):
    return torch.load(
        checkpoint / "training.pt", map_location="cpu", weights_only=False, mmap=True
    )


def inspect_training(checkpoint):
    manifest = json.loads((checkpoint / "manifest.json").read_text())
    state = read_training(checkpoint)
    optimizer = state["optimizer"]
    steps, nonzero_moments, finite = [], 0, True
    for value in optimizer["state"].values():
        steps.append(float(value["step"]))
        for name in ("exp_avg", "exp_avg_sq"):
            tensor = value[name]
            has_nonzero = False
            for part in chunks(tensor):
                finite &= bool(torch.isfinite(part).all())
                has_nonzero |= bool(torch.any(part != 0))
            nonzero_moments += int(has_nonzero)
    config = manifest["config"]
    batches_per_epoch = (
        # The smoke currently stays in epoch zero. The expected offset below
        # deliberately avoids inventing a dataset length for longer runs.
        manifest["step"] * config["grad_accum"]
    )
    loader_ok = state["loader"]["epoch"] >= 0 and state["loader"]["offset"] >= 0
    if state["loader"]["epoch"] == 0:
        loader_ok &= state["loader"]["offset"] == batches_per_epoch
    rng = state["rng_by_rank"]
    rng_ok = len(rng) == manifest["world_size"] and all(
        set(item) == {"python", "numpy", "torch", "cuda"}
        and isinstance(item["python"], tuple)
        and isinstance(item["numpy"], tuple)
        and item["torch"].dtype == torch.uint8
        and item["torch"].numel() > 0
        and item["cuda"] is not None
        and item["cuda"].dtype == torch.uint8
        and item["cuda"].numel() > 0
        for item in rng
    )
    result = {
        "step": manifest["step"],
        "world_size": manifest["world_size"],
        "optimizer_parameter_states": len(steps),
        "optimizer_min_step": min(steps, default=None),
        "optimizer_max_step": max(steps, default=None),
        "optimizer_nonzero_moment_tensors": nonzero_moments,
        "optimizer_all_finite": finite,
        "loader": state["loader"],
        "loader_ok": loader_ok,
        "rng_by_rank_present": rng_ok,
    }
    result["ok"] = bool(
        steps
        and min(steps) >= 1
        and max(steps) == manifest["step"]
        and nonzero_moments > 0
        and finite
        and loader_ok
        and rng_ok
    )
    del state
    return result


def compare_training(reference, actual):
    before, after = read_training(reference), read_training(actual)
    result = {"mismatches": [], "tensors_compared": 0, "tensor_elements": 0}

    def mismatch(path):
        if len(result["mismatches"]) < 30:
            result["mismatches"].append(path)

    def visit(a, b, path):
        if type(a) is not type(b):
            mismatch(path + ":type")
        elif isinstance(a, torch.Tensor):
            result["tensors_compared"] += 1
            result["tensor_elements"] += a.numel()
            if a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b):
                mismatch(path)
        elif isinstance(a, np.ndarray):
            if a.dtype != b.dtype or not np.array_equal(a, b):
                mismatch(path)
        elif isinstance(a, dict):
            if a.keys() != b.keys():
                mismatch(path + ":keys")
            for key in a.keys() & b.keys():
                visit(a[key], b[key], f"{path}.{key}")
        elif isinstance(a, (list, tuple)):
            if len(a) != len(b):
                mismatch(path + ":length")
            for index, (x, y) in enumerate(zip(a, b)):
                visit(x, y, f"{path}[{index}]")
        elif a != b:
            mismatch(path)

    visit(before, after, "training")
    result["exact_equal"] = not result["mismatches"]
    result["ok"] = result["exact_equal"]
    del before, after
    return result


def training_segment(path):
    events = [
        json.loads(line) for line in path.read_text().splitlines() if line.strip()
    ]
    starts = [i for i, event in enumerate(events) if event.get("event") == "start"]
    if not starts:
        raise ValueError(f"No start event in {path}")
    return events[starts[-1] :]


def inspect_metrics(path, reference=None):
    segment = training_segment(path)
    records = {item["step"]: item for item in segment if item["event"] == "train"}
    result = {"start_step": segment[0]["step"], "train_steps": sorted(records)}
    fields = ("loss", "action_flow_loss", "prior_nll", "lr", "grad_norm")
    result["finite_positive_gradient"] = bool(records) and all(
        all(math.isfinite(item[field]) for field in fields)
        and item["grad_norm"] > 0
        and item["lr"] > 0
        for item in records.values()
    )
    result["peak_memory_allocated_gib"] = max(
        (item.get("peak_memory_allocated_gib", 0) for item in records.values()),
        default=0,
    )
    result["ok"] = result["finite_positive_gradient"]
    if reference is not None:
        expected = {
            item["step"]: item
            for item in training_segment(reference)
            if item["event"] == "train"
        }
        differences = []
        for step, item in records.items():
            for field in fields:
                if step not in expected or item[field] != expected[step][field]:
                    differences.append(f"step={step}:{field}")
        result["reference_differences"] = differences
        result["exact_equal"] = not differences
        result["ok"] &= result["exact_equal"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--reference-metrics", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.reference) != bool(args.reference_metrics):
        parser.error("--reference and --reference-metrics must be provided together")
    torch.set_num_threads(4)
    checkpoint = args.checkpoint.resolve()
    report = {
        "checkpoint": str(checkpoint),
        "initial": str(args.initial.resolve()),
        "initial_to_trained": compare_weights(
            str(args.initial),
            str(checkpoint / "backbone.safetensors"),
            require_change=True,
        ),
        "training_state": inspect_training(checkpoint),
        "metrics": inspect_metrics(args.metrics, args.reference_metrics),
    }
    if args.reference:
        report["reference"] = str(args.reference.resolve())
        report["resume_weights"] = compare_weights(
            str(args.reference / "backbone.safetensors"),
            str(checkpoint / "backbone.safetensors"),
            require_change=False,
        )
        report["resume_training_state"] = compare_training(args.reference, checkpoint)
    report["ok"] = all(
        value["ok"]
        for value in report.values()
        if isinstance(value, dict) and "ok" in value
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
