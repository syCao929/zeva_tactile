"""CPU checks for pipeline wiring, failure propagation and resume."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

spec = importlib.util.spec_from_file_location(
    "pipeline", Path(__file__).with_name("cosmos_three_stage.py")
)
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


def test_plan_connects_checkpoints_and_uses_eight_gpus(tmp_path):
    args = pipeline.parse_args(["--name", "test", "--stage3", "both"])
    _, stages = pipeline.build_plan(args, tmp_path)
    assert len(stages) == 6
    for stage in stages:
        command = stage["commands"][0]
        if command[0] == "bash":
            assert command[command.index("--gpus") + 1] == "0,1,2,3,4,5,6,7"
        else:
            assert "--nproc_per_node=8" in command
    for stage in stages[4:]:
        command = stage["commands"][0]
        base = Path(command[command.index("--base-checkpoint") + 1])
        assert base / "model/.metadata" == Path(stages[0]["artifact"])
    command = stages[3]["commands"][0]
    assert command[command.index("--cte-checkpoint") + 1] == stages[2]["artifact"]


def test_failure_stops_pipeline_and_resume_skips_completed(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)
    monkeypatch.setattr(pipeline.signal, "signal", lambda *args: None)
    (tmp_path / "tools").mkdir()
    for name in ("train-cosmos-three-stage.sh", "cosmos-comparison-common.sh"):
        (tmp_path / "tools" / name).write_text("# fixture\n")
    flag = tmp_path / "fail"
    flag.touch()
    stages = []
    for index in range(3):
        artifact = tmp_path / f"artifact{index}"
        code = (
            "from pathlib import Path; "
            f"assert not ({index} == 1 and Path({str(flag)!r}).exists()); "
            f"Path({str(artifact)!r}).write_text('saved')"
        )
        stages.append(
            {
                "name": f"stage{index}",
                "kind": "cte",
                "artifact": str(artifact),
                "run": str(tmp_path / f"run{index}"),
                "commands": [[sys.executable, "-c", code]],
            }
        )
    monkeypatch.setattr(
        pipeline, "build_plan", lambda *args: (tmp_path / "pipeline", stages)
    )
    with pytest.raises(RuntimeError, match="stage1 failed"):
        pipeline.main(["--name", "test"])
    assert (tmp_path / "artifact0").exists()
    assert not (tmp_path / "artifact2").exists()
    saved_time = (tmp_path / "artifact0").stat().st_mtime_ns
    flag.unlink()
    assert pipeline.main(["--name", "test", "--resume"]) == 0
    assert (tmp_path / "artifact0").stat().st_mtime_ns == saved_time
    assert (tmp_path / "artifact2").exists()
