# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Teach the v0.25.1 speculative config about Qwen3 DFly checkpoints."""

from __future__ import annotations

import functools
import inspect
from types import ModuleType

from vllm_hcu.v1.spec_decode.dfly_gate import is_dfly_hf_config

from ._common import PatchCompatibilityError, load_exact_module

TARGET_MODULE = "vllm.config.speculative"
PATCH_ID = "platform.core_fix.spec_decode.dfly_config"
TARGETS = (
    f"{TARGET_MODULE}.SpeculativeConfig.hf_config_override",
    f"{TARGET_MODULE}.SpeculativeConfig.__post_init__",
)
_MARKER = "_vllm_hcu_dfly_config_patch_applied"
_WRAPPER = "_vllm_hcu_dfly_config_wrapper"


def apply_to_module(module: ModuleType) -> bool:
    speculative_module = load_exact_module(TARGET_MODULE, module)
    config_class = getattr(speculative_module, "SpeculativeConfig", None)
    if not isinstance(config_class, type):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGET_MODULE}.SpeculativeConfig is missing"
        )
    if getattr(config_class, _MARKER, False):
        return False

    original_post_init = vars(config_class).get("__post_init__")
    original_hf_override = getattr(config_class, "hf_config_override", None)
    if not callable(original_hf_override) or tuple(
        inspect.signature(original_hf_override).parameters
    ) != ("hf_config",):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGETS[0]} has an incompatible signature"
        )
    if not callable(original_post_init) or tuple(
        inspect.signature(original_post_init).parameters
    ) != ("self",):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGETS[1]} has an incompatible signature"
        )

    @functools.wraps(original_hf_override)
    def hcu_dfly_hf_config_override(hf_config):
        hf_config = original_hf_override(hf_config)
        if is_dfly_hf_config(hf_config):
            # The published Hy3-DFly checkpoint identifies itself only by
            # architecture. Preserve a semantic marker and add the v0.25.1
            # Qwen DSpark compatibility alias while its post-init runs. The
            # alias prevents v0.25.1 from rewriting the config to DeepSeek-V4's
            # in-checkpoint DSparkDraftModel; post-init removes it again.
            hf_config.model_arch = "dfly"
            hf_config.architectures = ["Qwen3DFlyModel", "Qwen3DSparkModel"]
        return hf_config

    @functools.wraps(original_post_init)
    def hcu_dfly_post_init(self):
        original_post_init(self)
        draft_config = getattr(self, "draft_model_config", None)
        hf_config = getattr(draft_config, "hf_config", None)
        if hf_config is None or not is_dfly_hf_config(hf_config):
            return

        # v0.25.1 knows DSpark but assumes every non-Qwen3DSpark architecture
        # is the in-checkpoint DeepSeek-V4 drafter. Restore the standalone DFly
        # architecture after upstream validation and route it through DSpark's
        # sequential block runtime.
        hf_config.model_type = "qwen3"
        hf_config.architectures = ["Qwen3DFlyModel"]
        self.method = "dspark"
        self.parallel_drafting = True
        self.update_arch_()

        if getattr(self, "num_speculative_tokens_per_batch_size", None) is not None:
            raise ValueError(
                "Qwen3DFlyModel does not yet support dynamic speculative K "
                "in the HCU V1 runner; remove "
                "num_speculative_tokens_per_batch_size"
            )

    setattr(hcu_dfly_post_init, _WRAPPER, True)
    setattr(
        config_class,
        "_vllm_hcu_original_dfly_hf_config_override",
        original_hf_override,
    )
    setattr(
        config_class,
        "hf_config_override",
        staticmethod(hcu_dfly_hf_config_override),
    )
    setattr(config_class, "_vllm_hcu_original_dfly_post_init", original_post_init)
    setattr(config_class, "__post_init__", hcu_dfly_post_init)
    setattr(config_class, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
