#!/usr/bin/env bash
# Stage 2 baseline: frozen Cosmos base + visual CTE / Zeva, without tactile.
set -euo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/cosmos-comparison-common.sh" baseline "$@"
