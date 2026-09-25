"""Small standard-library helpers shared by the comparison launchers."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import tempfile


def enforce_pair_contract(path: Path, payload: dict, *, create: bool) -> None:
    """Check a pair's immutable common settings; publish new contracts atomically.

    Dry runs use create=False and never create directories. Concurrent visual
    and tactile launches cannot silently select different initial checkpoints.
    """
    path = Path(path)

    def compare():
        existing = json.loads(path.read_text())
        if existing != payload:
            changed = sorted(
                key for key in existing.keys() | payload.keys()
                if existing.get(key) != payload.get(key)
            )
            raise ValueError(
                f"Pair contract mismatch in {path}: {changed}. "
                "Use the same settings for both branches, or choose a new --pair-name."
            )

    if path.exists():
        compare()
        return
    if not create:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        # Linking a completed file is both no-overwrite and atomic on this FS.
        try:
            os.link(temporary, path)
        except FileExistsError:
            compare()
    finally:
        temporary.unlink(missing_ok=True)


def ensure_gpus_available(gpus: str, allow_busy: bool = False) -> None:
    """Query the driver without creating CUDA contexts or terminating processes."""
    selected = [part.strip() for part in gpus.split(",")]
    if not selected or any(not part for part in selected) or len(set(selected)) != len(selected):
        raise ValueError("--gpus must contain distinct comma-separated GPU indices or UUIDs")

    def query(arguments):
        result = subprocess.run(
            ["nvidia-smi", *arguments, "--format=csv,noheader,nounits"],
            text=True, capture_output=True, check=True, timeout=15,
        )
        return [[part.strip() for part in row] for row in csv.reader(result.stdout.splitlines())]

    devices = query(["--query-gpu=index,uuid"])
    mapping = {key: uuid for index, uuid in devices for key in (index, uuid)}
    missing = [device for device in selected if device not in mapping]
    if missing:
        raise ValueError(f"Requested GPUs do not exist: {missing}")
    if allow_busy:
        return
    used = query(["--query-compute-apps=gpu_uuid,pid"])
    targets = {mapping[device] for device in selected}
    conflicts = [f"{uuid}:pid={pid}" for uuid, pid in used if uuid in targets]
    if conflicts:
        raise ValueError(
            "Selected GPUs already run compute processes: " + ", ".join(conflicts)
            + ". Select free --gpus; --allow-busy-gpus explicitly permits sharing."
        )
