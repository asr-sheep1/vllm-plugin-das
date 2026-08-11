# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Backport the D-Cut configuration surface to vLLM v0.25.1.

Upstream D-Cut was proposed after v0.25.1 and therefore its
``SpeculativeConfig`` dataclass does not accept ``dflash_dcut``.  Rebuilding
the pydantic dataclass at runtime would invalidate class references already
imported by vLLM, so this adapter extends the generated constructor in place.
Instances are not slotted; the normalized value remains pickle-safe in the
instance dictionary and is included in the compilation hash below.
"""

from __future__ import annotations

import functools
import math
import numbers
from types import ModuleType

from vllm_hcu.v1.spec_decode.dfly_gate import is_dfly_speculative_config

from ._common import PatchCompatibilityError, load_exact_module

TARGET_MODULE = "vllm.config.speculative"
PATCH_ID = "platform.core_fix.spec_decode.dcut_config"
TARGETS = (
    f"{TARGET_MODULE}.SpeculativeConfig.__init__",
    f"{TARGET_MODULE}.SpeculativeConfig.compute_hash",
    f"{TARGET_MODULE}.SpeculativeConfig.dflash_dcut_mode",
    f"{TARGET_MODULE}.SpeculativeConfig.uses_dflash_dcut",
)
_MARKER = "_vllm_hcu_dcut_config_patch_applied"


def _normalize_dcut(value: object) -> float | str:
    if isinstance(value, str):
        if value == "auto":
            return value
        try:
            value = float(value)
        except ValueError as exc:
            raise ValueError(
                'dflash_dcut must be a float in [0, 1] or "auto", '
                f"got {value!r}."
            ) from exc
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(
            'dflash_dcut must be a float in [0, 1] or "auto", '
            f"got {value!r}."
        )
    normalized = float(value)
    if not math.isfinite(normalized) or not 0 <= normalized <= 1:
        raise ValueError(
            'dflash_dcut must be a float in [0, 1] or "auto", '
            f"got {value!r}."
        )
    return normalized


def apply_to_module(module: ModuleType) -> bool:
    speculative_module = load_exact_module(TARGET_MODULE, module)
    config_class = getattr(speculative_module, "SpeculativeConfig", None)
    if not isinstance(config_class, type):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGET_MODULE}.SpeculativeConfig is missing"
        )
    if getattr(config_class, _MARKER, False):
        return False

    original_init = vars(config_class).get("__init__")
    original_compute_hash = vars(config_class).get("compute_hash")
    if not callable(original_init) or not callable(original_compute_hash):
        raise PatchCompatibilityError(
            "vLLM SpeculativeConfig constructor/hash contract is incompatible"
        )
    logger = getattr(speculative_module, "logger", None)
    safe_hash = getattr(speculative_module, "safe_hash", None)
    if not callable(safe_hash):
        raise PatchCompatibilityError("vLLM speculative safe_hash helper is missing")

    @functools.wraps(original_init)
    def hcu_dcut_init(self, *args, dflash_dcut=0.0, **kwargs):
        # Let v0.25.1 resolve the draft architecture/method first.  The DFly
        # adapter changes the published Qwen3DFlyModel method to ``dspark`` in
        # its post-init, which is the authoritative point for our method gate.
        original_init(self, *args, **kwargs)
        normalized = _normalize_dcut(dflash_dcut)
        method = getattr(self, "method", None)
        supported = method == "dspark" and is_dfly_speculative_config(self)
        if normalized != 0.0 and not supported:
            if logger is not None:
                logger.warning(
                    "HCU D-Cut currently supports only the Qwen3DFlyModel "
                    "drafter; disabling dflash_dcut for method='%s'.",
                    method,
                )
            normalized = 0.0
        self.dflash_dcut = normalized

    @functools.wraps(original_compute_hash)
    def hcu_dcut_compute_hash(self):
        upstream_hash = original_compute_hash(self)
        draft_hash = None
        layer_ids = None
        dfly_graph_factors = None
        if is_dfly_speculative_config(self):
            dfly_graph_factors = (
                getattr(self, "method", None),
                getattr(self, "num_speculative_tokens", None),
                getattr(self, "draft_sample_method", None),
                str(getattr(self, "attention_backend", None)),
            )
            draft_config = getattr(self, "draft_model_config", None)
            compute_draft_hash = getattr(draft_config, "compute_hash", None)
            if callable(compute_draft_hash):
                draft_hash = compute_draft_hash()
            hf_config = getattr(draft_config, "hf_config", None)
            if hf_config is not None:
                layer_ids = getattr(
                    hf_config,
                    "eagle_aux_hidden_state_layer_ids",
                    None,
                )
                if layer_ids is None:
                    drafter_config = (
                        getattr(hf_config, "dflash_config", None)
                        or getattr(hf_config, "dflare_config", None)
                        or {}
                    )
                    layer_ids = drafter_config.get(
                        "target_layer_ids"
                    ) or getattr(hf_config, "target_layer_ids", None)
        factors = (
            upstream_hash,
            dfly_graph_factors,
            draft_hash,
            tuple(layer_ids) if layer_ids is not None else None,
            getattr(self, "dflash_dcut", 0.0),
        )
        return safe_hash(str(factors).encode(), usedforsecurity=False).hexdigest()

    def dflash_dcut_mode(self):
        value = getattr(self, "dflash_dcut", 0.0)
        if value == "auto":
            return "selector"
        if value == 0:
            return "off"
        return "fixed_ratio"

    def uses_dflash_dcut(self):
        method = getattr(self, "method", None)
        supported = method == "dspark" and is_dfly_speculative_config(self)
        return supported and self.dflash_dcut_mode != "off"

    # Expose a class default so legacy/programmatically constructed objects
    # also read as D-Cut-off without requiring constructor participation.
    setattr(config_class, "dflash_dcut", 0.0)
    setattr(config_class, "_vllm_hcu_original_dcut_init", original_init)
    setattr(config_class, "__init__", hcu_dcut_init)
    setattr(
        config_class,
        "_vllm_hcu_original_dcut_compute_hash",
        original_compute_hash,
    )
    setattr(config_class, "compute_hash", hcu_dcut_compute_hash)
    setattr(config_class, "dflash_dcut_mode", property(dflash_dcut_mode))
    setattr(config_class, "uses_dflash_dcut", uses_dflash_dcut)
    setattr(config_class, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
