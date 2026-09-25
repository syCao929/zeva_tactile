#!/usr/bin/env bash
# Stage 2 tactile: same frozen Cosmos base + visual Zeva + causal tactile BIT.
set -euo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/cosmos-comparison-common.sh" tactile "$@"
