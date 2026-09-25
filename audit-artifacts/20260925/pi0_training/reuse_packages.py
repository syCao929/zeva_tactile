"""Copy exact resolved distributions from the existing Python 3.11 environment.

Run with the source interpreter. Copies have independent inodes, and only
packages with the exact locked version are selected. The target remains a
fresh venv; uv installs the remaining packages and verifies dependencies.
"""

import importlib.metadata
import json
from pathlib import Path
import re
import shutil
import sys

workspace = Path(__file__).resolve().parents[3]
target = workspace / "envs/pi0"
source = Path(sys.prefix).resolve()
assert (target / ".zeva-pi0-environment").is_file()
assert source != target.resolve() and sys.version_info[:2] == (3, 11)
lock = Path(__file__).with_name("requirements-resolved.txt")
if Path(__file__).with_name("reused_packages.json").exists():
    raise FileExistsError("This one-time environment seed has already completed")


def normalize(name):
    return re.sub(r"[-_.]+", "-", name).lower()


resolved = dict(re.findall(r"^([\w.-]+)==([^\s]+)$", lock.read_text(), re.MULTILINE))
installed = {
    normalize(d.metadata["Name"]): d for d in importlib.metadata.distributions()
}
report = {"source": str(source), "target": str(target), "copied": [], "remaining": []}
for name, version in resolved.items():
    dist = installed.get(normalize(name))
    if dist is None or dist.version != version:
        report["remaining"].append(f"{name}=={version}")
        continue
    count, size = 0, 0
    for entry in dist.files or ():
        original = Path(dist.locate_file(entry)).resolve()
        relative = original.relative_to(source)
        if original.suffix == ".pyc":
            continue
        if not original.is_file():
            raise FileNotFoundError(original)
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, destination)
        if relative.parts[0] == "bin":
            content = destination.read_bytes()
            if content.startswith(b"#!") and b"python" in content.split(b"\n", 1)[0]:
                destination.write_bytes(
                    b"#!"
                    + str(target / "bin/python").encode()
                    + b"\n"
                    + content.split(b"\n", 1)[1]
                )
        assert (
            original.stat().st_ino != destination.stat().st_ino
            or original.stat().st_dev != destination.stat().st_dev
        )
        count += 1
        size += original.stat().st_size
    report["copied"].append(
        {"name": name, "version": version, "files": count, "bytes": size}
    )
    print(name, version, count, size, flush=True)
Path(__file__).with_name("reused_packages.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
