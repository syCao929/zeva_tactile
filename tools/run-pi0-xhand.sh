#!/usr/bin/env bash
set -euo pipefail
PI0_WORKSPACE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PI0_PYTHON=${PI0_PYTHON:-"$PI0_WORKSPACE/envs/pi0/bin/python"}
PI0_NPROC=${PI0_NPROC:-1}
OPENPI_ROOT=${OPENPI_ROOT:-"$PI0_WORKSPACE/../openpi-3d-tactile"}
export PYTHONPATH="$PI0_WORKSPACE:$PI0_WORKSPACE/cosmos-framework:$OPENPI_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH=""
# OpenPI imports JAX for configuration/types, but the policy executes in PyTorch.
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TOKENIZERS_PARALLELISM=false
if [[ ! -x "$PI0_PYTHON" ]]; then
  echo "Missing π0 environment. Run bash tools/setup-pi0-env.sh first." >&2
  exit 1
fi
if [[ "$PI0_NPROC" == 1 ]]; then
  exec "$PI0_PYTHON" -m pi0_zeva.train --set "openpi_root=$OPENPI_ROOT" "$@"
fi
exec "$PI0_PYTHON" -m torch.distributed.run --standalone --nnodes=1 \
  --nproc-per-node="$PI0_NPROC" -m pi0_zeva.train --set "openpi_root=$OPENPI_ROOT" "$@"
