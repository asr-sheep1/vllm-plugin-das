# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Architecture gates for the HCU DFly speculative-decoding adapter."""

from __future__ import annotations


DFLY_ARCHITECTURE = "Qwen3DFlyModel"


def is_dfly_hf_config(hf_config: object | None) -> bool:
    if hf_config is None:
        return False
    architectures = getattr(hf_config, "architectures", None) or []
    return DFLY_ARCHITECTURE in architectures or str(
        getattr(hf_config, "model_arch", "")
    ).lower() == "dfly"


def is_dfly_draft_model_config(draft_model_config: object | None) -> bool:
    if draft_model_config is None:
        return False
    architectures = getattr(draft_model_config, "architectures", None) or []
    return DFLY_ARCHITECTURE in architectures or is_dfly_hf_config(
        getattr(draft_model_config, "hf_config", None)
    )


def is_dfly_speculative_config(speculative_config: object | None) -> bool:
    return is_dfly_draft_model_config(
        getattr(speculative_config, "draft_model_config", None)
    )


__all__ = [
    "DFLY_ARCHITECTURE",
    "is_dfly_draft_model_config",
    "is_dfly_hf_config",
    "is_dfly_speculative_config",
]
