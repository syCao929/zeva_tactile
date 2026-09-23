# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Training-side utilities for reproducing Zeva on a new embodiment.

The shipped Zeva release is an *inference* release: it contains the model
definitions and the loss functions, but none of the training loops, and none of
the offline artifact builders (CTE feature cache, task-context bank, PIM
training bank) that the training recipes expect. Everything in this package is
therefore written from scratch against the contracts the inference code pins
down — see ``README_ZEVA_ENV.md`` for the full picture.

Layout:

- ``probe_vae``      — confirm the Wan VAE's latent geometry before building a CTE
- ``vae_cache``      — batch-encode frames to latents and persist them
- ``cte_dataset``    — CTE training windows over the cached latents
- ``train_cte``      — the CTE training loop (stage 1)
"""
