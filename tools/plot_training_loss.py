#!/usr/bin/env python3
"""Plot rank-0 training loss from Cosmos logs, including startup records."""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="Output prefix for PNG, SVG, CSV, and JSON")
    parser.add_argument("--title", default="V3 joint18 training loss")
    parser.add_argument("--smooth-points", type=int, default=20)
    args = parser.parse_args()
    if args.smooth_points < 1:
        parser.error("--smooth-points must be positive")

    text = args.log.read_text()
    prefix = r"\[(?P<timestamp>\d{2}-\d{2} \d{2}:\d{2}:\d{2})\|[^\n]*?\[RANK 0\]\s+"
    regular = re.compile(prefix + r"(?P<step>\d+)\s+: iter_speed .*?\| Loss:\s*(?P<loss>\S+)")
    startup = re.compile(prefix + r"Iteration (?P<step>\d+): Hit counter:.*?\| Loss:\s*(?P<loss>\S+)")
    by_step = {}
    for line_number, line in enumerate(text.splitlines(), 1):
        match = regular.search(line) or startup.search(line)
        if match:
            row = match.groupdict()
            step, loss = int(row["step"]), float(row["loss"])
            if not np.isfinite(loss) or loss <= 0:
                raise ValueError(f"Cannot plot nonpositive/nonfinite training loss at {args.log}:{line_number}")
            by_step[step] = {"step": step, "loss": loss, "timestamp": row["timestamp"], "source_line": line_number}
    rows = [by_step[step] for step in sorted(by_step)]
    if not rows:
        raise ValueError("No rank-0 training loss records found")
    steps = np.array([row["step"] for row in rows])
    losses = np.array([row["loss"] for row in rows])
    smooth = np.array([losses[max(0, i - args.smooth_points + 1) : i + 1].mean() for i in range(len(losses))])

    validation = []
    for match in re.finditer(r"Validation loss \(iteration (\d+)\):\s*(\S+)", text):
        value = float(match.group(2))
        validation.append({"step": int(match.group(1)), "loss": value if np.isfinite(value) else None})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "loss", "trailing_mean", "timestamp", "source_line"])
        writer.writeheader()
        for row, mean in zip(rows, smooth, strict=True):
            writer.writerow({**row, "trailing_mean": float(mean)})

    summary = {
        "source": str(args.log.resolve()),
        "snapshot_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "last_logged_at": rows[-1]["timestamp"] + " Asia/Shanghai",
        "records": len(rows),
        "first_step": int(steps[0]),
        "last_step": int(steps[-1]),
        "first_loss": float(losses[0]),
        "last_loss": float(losses[-1]),
        "trailing_mean_points": args.smooth_points,
        "last_trailing_mean": float(smooth[-1]),
        "definition": "Rank-0 current-batch total training loss, not action-only loss or a multi-rank mean",
        "sampling": "Startup steps 1-50 logged individually; subsequent records usually every 10 steps",
        "validation_records": validation,
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.5), gridspec_kw={"width_ratios": [1.2, 1]})
    fig.subplots_adjust(left=0.07, right=0.975, bottom=0.21, top=0.76, wspace=0.24)
    fig.suptitle(args.title, x=0.07, y=0.96, ha="left", fontsize=20, fontweight="bold", color="#18334b")
    fig.text(0.07, 0.885, f"Step {steps[-1]:,} / 5,000  |  Last loss {losses[-1]:.4f}  |  Smoothed {smooth[-1]:.4f}", fontsize=12)
    fig.text(0.07, 0.832, f"Log snapshot: {summary['last_logged_at']}  |  {len(rows)} training records", fontsize=10, color="#596774")
    recent = steps >= max(steps[0], steps[-1] - 1000)
    for ax, selected, title in zip(axes, [np.ones(len(steps), dtype=bool), recent], ["Full recorded history (log scale)", "Last 1,000 training steps"], strict=True):
        ax.plot(steps[selected], losses[selected], color="#759dbf", alpha=0.55, lw=1, label="Logged batch loss")
        ax.plot(steps[selected], smooth[selected], color="#d66a24", lw=2.5, label=f"{args.smooth_points}-record trailing mean")
        ax.set_title(title, loc="left", fontsize=12, pad=12)
        ax.set_xlabel("Training step")
        ax.grid(True, color="#d8e0e7", alpha=0.6, lw=0.7)
        ax.set_axisbelow(True)
        ax.margins(x=0.015)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Total training loss (rank 0)")
    axes[1].set_ylim(bottom=0)
    axes[1].scatter([steps[-1]], [losses[-1]], s=35, color="#365e85", zorder=5)
    axes[0].legend(frameon=False, loc="upper right", fontsize=9)
    fig.text(0.07, 0.075, "Loss includes visual + action objectives. The mean spans 20 records (usually ~200 steps after startup).", fontsize=9, color="#596774")
    fig.text(0.07, 0.035, "Validation is not plotted: only step 0 is valid; later empty passes report NaN / 0.", fontsize=9, color="#596774")
    fig.savefig(args.output.with_suffix(".png"), dpi=180, facecolor="white")
    fig.savefig(args.output.with_suffix(".svg"), facecolor="white")
    plt.close(fig)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
