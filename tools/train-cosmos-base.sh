#!/usr/bin/env bash
# Stage 1: Cosmos XHand joint18 base, independent of the existing V3 run.
set -euo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/cosmos-comparison-common.sh" base "$@"
