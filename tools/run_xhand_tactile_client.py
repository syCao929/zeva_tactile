#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Run the existing Factile robot client with Zeva's dense tactile protocol."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cosmos-framework"))

from cosmos_framework.inference.xhand_tactile_client import main

if __name__ == "__main__":
    raise SystemExit(main())
