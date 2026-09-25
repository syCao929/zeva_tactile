#!/usr/bin/env bash
# Creates only this task's environment; existing Cosmos/OpenPI environments are read-only.
set -euo pipefail
PI0_WORKSPACE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PI0_ENV_DIR=${PI0_ENV_DIR:-"$PI0_WORKSPACE/envs/pi0"}
OPENPI_ROOT=${OPENPI_ROOT:-"$PI0_WORKSPACE/../openpi-3d-tactile"}
PI0_UV=${PI0_UV:-"$PI0_WORKSPACE/tools/uv/uv"}
if [[ ! -x "$PI0_UV" ]]; then
  PI0_UV=$(command -v uv)
fi
if [[ ! -f "$OPENPI_ROOT/src/openpi/models_pytorch/pi0_pytorch.py" ]]; then
  echo "OPENPI_ROOT must point to the documented OpenPI checkout" >&2
  exit 1
fi
if [[ -e "$PI0_ENV_DIR" && ! -f "$PI0_ENV_DIR/.zeva-pi0-environment" ]]; then
  echo "Refusing to modify an existing environment without the π0 ownership marker: $PI0_ENV_DIR" >&2
  exit 1
fi
if [[ ! -x "$PI0_ENV_DIR/bin/python" ]]; then
  "$PI0_UV" venv --python 3.11 "$PI0_ENV_DIR"
  touch "$PI0_ENV_DIR/.zeva-pi0-environment"
fi
"$PI0_UV" pip install --link-mode copy --python "$PI0_ENV_DIR/bin/python" -r "$PI0_WORKSPACE/pi0_zeva/requirements.txt"
export PYTHONPATH="$PI0_WORKSPACE:$PI0_WORKSPACE/cosmos-framework:$OPENPI_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH=""
export JAX_PLATFORMS=cpu
"$PI0_ENV_DIR/bin/python" - "$OPENPI_ROOT" "$PI0_ENV_DIR" <<'PY'
import pathlib
import shutil
import sys
import sysconfig
import tempfile
from pi0_zeva.runtime import configure_openpi, inspect_dependency_environment

root = pathlib.Path(sys.argv[1]).resolve()
environment = pathlib.Path(sys.argv[2]).resolve()
configure_openpi(str(root))
destination = pathlib.Path(sysconfig.get_paths()["purelib"]) / "transformers"
if not destination.resolve().is_relative_to(environment):
    raise RuntimeError("Transformer patch destination must be inside the dedicated π0 environment")
source = root / "src/openpi/models_pytorch/transformers_replace"
if not (source / "models/siglip/check.py").is_file():
    raise FileNotFoundError("The selected OpenPI checkout lacks its transformers patch")
# Replace each file with a fresh inode: an older uv installation may have
# hardlinked this environment to its cache or another environment.
for original in source.rglob("*"):
    if not original.is_file() or "__pycache__" in original.parts:
        continue
    target = destination / original.relative_to(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
        temporary = pathlib.Path(handle.name)
    try:
        shutil.copy2(original, temporary)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
report = inspect_dependency_environment()
if not report["ready"]:
    raise RuntimeError(report["errors"])
print("π0 runtime imports and transformers patch verified; weights and dataset are checked separately.")
PY
