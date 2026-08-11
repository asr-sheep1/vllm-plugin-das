# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""VllmConfig adapters for HCU sidecar validation and graph buckets."""

from __future__ import annotations

import functools
import inspect
import logging
from types import ModuleType
from typing import Any

from vllm_hcu.patch.config import HcuFeatureConfig, get_hcu_config, set_hcu_config
from vllm_hcu.v1.spec_decode.dfly_gate import is_dfly_speculative_config

from ._common import PatchCompatibilityError, apply_once, load_exact_module
from .patch_compilation_config import bind_hcu_config

TARGET_MODULE = "vllm.config.vllm"
PATCH_ID = "platform.core_fix.hcu_config.vllm"
TARGETS = (
    f"{TARGET_MODULE}.VllmConfig.with_hf_config",
    f"{TARGET_MODULE}.VllmConfig._set_cudagraph_sizes",
    f"{TARGET_MODULE}.VllmConfig.use_v2_model_runner",
    "vllm.config.model.ModelConfig.get_model_arch_config",
    "vllm_hcu.platforms.hcu.HCUPlatform.check_and_update_config",
)
_MARKER = "_vllm_hcu_feature_config_patch_applied"
_REQUEST_CAPTURE_SIZES = (
    *range(1, 9),
    *range(10, 33, 2),
    *range(40, 65, 4),
    *range(72, 257, 8),
)
logger = logging.getLogger(__name__)


def validate_and_update_hcu_config(vllm_config: object) -> HcuFeatureConfig:
    """Validate cross-config invariants and bind the compilation adapter."""

    feature_config = get_hcu_config(vllm_config)
    updates: dict[str, str] = {}
    if feature_config.hcu_flash_attn_mode is None:
        # Persist the resolved sub-mode before vLLM computes compilation cache
        # hashes. Classic, CUTLASS, and CUSTOM do not share a KV-cache ABI.
        from vllm_hcu.platforms import envs as hcu_envs

        updates["hcu_flash_attn_mode"] = hcu_envs.resolve_hcu_flash_attn_mode(None)
    if updates:
        feature_config = feature_config.with_updates(**updates)
    # Persist the resolved mode so it enters vLLM's compilation hash.
    set_hcu_config(vllm_config, feature_config)

    feature_config = bind_hcu_config(vllm_config)
    parallel_config = getattr(vllm_config, "parallel_config", None)
    model_config = getattr(vllm_config, "model_config", None)
    kernel_config = getattr(vllm_config, "kernel_config", None)
    scheduler_config = getattr(vllm_config, "scheduler_config", None)
    compilation_config = getattr(vllm_config, "compilation_config", None)
    speculative_config = getattr(vllm_config, "speculative_config", None)
    uses_dcut_fn = getattr(speculative_config, "uses_dflash_dcut", None)
    uses_dcut = bool(callable(uses_dcut_fn) and uses_dcut_fn())

    if uses_dcut:
        if scheduler_config is None or compilation_config is None:
            raise PatchCompatibilityError(
                "D-Cut requires scheduler_config and compilation_config"
            )
        if getattr(scheduler_config, "async_scheduling", False):
            logger.warning(
                "Asynchronous scheduling is not compatible with D-Cut's "
                "variable verification width; disabling it."
            )
            scheduler_config.async_scheduling = False
            if parallel_config is not None and hasattr(
                parallel_config, "disable_nccl_for_dp_synchronization"
            ):
                parallel_config.disable_nccl_for_dp_synchronization = False

        cudagraph_mode = getattr(compilation_config, "cudagraph_mode", None)
        has_full_cudagraphs = getattr(cudagraph_mode, "has_full_cudagraphs", None)
        if callable(has_full_cudagraphs) and has_full_cudagraphs():
            from vllm.config.compilation import CUDAGraphMode

            logger.warning(
                "D-Cut changes target verification width at runtime; "
                "overriding cudagraph_mode from %s to PIECEWISE.",
                getattr(cudagraph_mode, "name", cudagraph_mode),
            )
            compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE

    if (
        parallel_config is not None
        and model_config is not None
        and _uses_dfly_draft(vllm_config)
        and not getattr(model_config, "enforce_eager", False)
        and not getattr(parallel_config, "disable_custom_all_reduce", False)
    ):
        # HCU custom all-reduce records graph-pool addresses in one native
        # communicator.  DFly captures both target and drafter graphs through
        # that communicator; the second address set invalidates the first and
        # faults at the capture synchronization boundary.  PyNCCL is graph
        # safe for this two-model topology, so select it until custom AR owns
        # independent target/drafter registrations.
        parallel_config.disable_custom_all_reduce = True
        logger.warning(
            "Disabling HCU custom all-reduce for DFly CUDAGraphs; the shared "
            "target/drafter graph-address registry is not yet safe. Falling "
            "back to PyNCCL all-reduce."
        )

    if parallel_config is not None:
        setattr(
            parallel_config,
            "_vllm_hcu_deepep_auto",
            feature_config.deepep_auto,
        )
    if feature_config.deepep_auto:
        if parallel_config is None:
            raise PatchCompatibilityError(
                "deepep_auto requires VllmConfig.parallel_config"
            )
        if getattr(parallel_config, "all2all_backend", None) != "deepep_low_latency":
            raise ValueError(
                "HCU deepep_auto must be normalized to the vLLM 0.25 "
                "deepep_low_latency configuration contract"
            )
        if feature_config.moe_backend not in ("auto", "dpsk_deep_gemm"):
            raise ValueError(
                "deepep_auto requires HCU moe_backend='auto' or "
                "'dpsk_deep_gemm'"
            )

    if feature_config.enable_lightly_cp:
        if model_config is None:
            raise PatchCompatibilityError(
                "Lightly-CP requires VllmConfig.model_config"
            )
        if not getattr(model_config, "enforce_eager", False):
            raise ValueError(
                "Lightly context parallel currently only supports eager mode."
            )
        if parallel_config is None:
            raise PatchCompatibilityError(
                "Lightly-CP requires VllmConfig.parallel_config"
            )
        if getattr(parallel_config, "decode_context_parallel_size", 1) > 1:
            raise ValueError(
                "Lightly context parallel and DCP cannot be enabled simultaneously."
            )

    if feature_config.moe_backend == "dpsk_deep_gemm":
        if kernel_config is None:
            raise PatchCompatibilityError(
                "dpsk_deep_gemm requires VllmConfig.kernel_config"
            )
        upstream_backend = getattr(kernel_config, "moe_backend", None)
        if upstream_backend == "dpsk_deep_gemm":
            # Defensive normalization for programmatic objects that bypassed
            # EngineArgs.  Pydantic's official Literal must never see this.
            setattr(kernel_config, "moe_backend", "auto")
        elif upstream_backend != "auto":
            raise ValueError(
                "HCU sidecar selects dpsk_deep_gemm but upstream "
                f"KernelConfig.moe_backend selects {upstream_backend!r}"
            )
    return feature_config


def _request_cudagraph_buckets_enabled() -> bool:
    from vllm_hcu.platforms import envs as hcu_envs

    return bool(
        hcu_envs.VLLM_HCU_USE_CUSTOM_OPS
        and hcu_envs.VLLM_HCU_ENABLE_REQUEST_CUDAGRAPH_BUCKETS
    )


def _uses_dfly_draft(vllm_config: object) -> bool:
    return is_dfly_speculative_config(
        getattr(vllm_config, "speculative_config", None)
    )


def _replace_with_request_cudagraph_buckets(
    vllm_config: object,
    *,
    compile_sizes_template: list[int | str] | None,
) -> None:
    compilation_config = getattr(vllm_config, "compilation_config", None)
    scheduler_config = getattr(vllm_config, "scheduler_config", None)
    if compilation_config is None or scheduler_config is None:
        raise PatchCompatibilityError(
            "request cudagraph buckets require compilation_config and scheduler_config"
        )
    current_sizes = getattr(compilation_config, "cudagraph_capture_sizes", None)
    max_size = getattr(compilation_config, "max_cudagraph_capture_size", None)
    if not current_sizes or not isinstance(max_size, int) or max_size < 1:
        return

    speculative_config = getattr(vllm_config, "speculative_config", None)
    num_speculative_tokens = getattr(speculative_config, "num_speculative_tokens", 0)
    decode_query_len = 1 + (num_speculative_tokens or 0)
    sizes = [
        request_size * decode_query_len
        for request_size in _REQUEST_CAPTURE_SIZES
        if request_size * decode_query_len <= max_size
    ]

    max_num_tokens = getattr(scheduler_config, "max_num_batched_tokens", None)
    if (
        isinstance(max_num_tokens, int)
        and max_num_tokens <= max_size
        and max_num_tokens not in sizes
    ):
        sizes.append(max_num_tokens)
    sizes = sorted(set(sizes))
    if not sizes:
        raise ValueError(
            "No valid request-oriented cudagraph bucket fits within "
            f"max_cudagraph_capture_size={max_size}"
        )

    compilation_config.cudagraph_capture_sizes = sizes
    compilation_config.max_cudagraph_capture_size = sizes[-1]
    # Upstream consumes the symbolic cudagraph sentinel during its first
    # post-init.  Restore the template and recompute after changing the list.
    compilation_config.compile_sizes = compile_sizes_template
    post_init = getattr(compilation_config, "post_init_cudagraph_sizes", None)
    if not callable(post_init):
        raise PatchCompatibilityError(
            "CompilationConfig.post_init_cudagraph_sizes is missing"
        )
    post_init()


def apply_to_module(module: ModuleType) -> bool:
    vllm_module = load_exact_module(TARGET_MODULE, module)
    vllm_config = getattr(vllm_module, "VllmConfig", None)
    model_config_class = getattr(vllm_module, "ModelConfig", None)
    if not isinstance(vllm_config, type):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGET_MODULE}.VllmConfig is missing"
        )
    if not isinstance(model_config_class, type):
        raise PatchCompatibilityError(
            "required HCU patch target vllm.config.model.ModelConfig is missing"
        )
    if getattr(vllm_config, _MARKER, False):
        return False

    with_hf_config = vars(vllm_config).get("with_hf_config")
    set_cudagraph_sizes = vars(vllm_config).get("_set_cudagraph_sizes")
    use_v2_model_runner = vars(vllm_config).get("use_v2_model_runner")
    get_model_arch_config = vars(model_config_class).get("get_model_arch_config")
    if (
        not callable(with_hf_config)
        or not callable(set_cudagraph_sizes)
        or not isinstance(use_v2_model_runner, property)
        or not callable(use_v2_model_runner.fget)
        or not callable(get_model_arch_config)
    ):
        raise PatchCompatibilityError(
            "required HCU VllmConfig compatibility methods are missing"
        )
    model_arch_signature = inspect.signature(get_model_arch_config)
    if tuple(model_arch_signature.parameters) != ("self",):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGETS[3]} has incompatible "
            f"signature {model_arch_signature}"
        )
    with_hf_signature = inspect.signature(with_hf_config)
    if tuple(with_hf_signature.parameters) != (
        "self",
        "hf_config",
        "architectures",
    ):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGETS[0]} has incompatible "
            f"signature {with_hf_signature}"
        )
    cudagraph_signature = inspect.signature(set_cudagraph_sizes)
    if tuple(cudagraph_signature.parameters) != ("self",):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGETS[1]} has incompatible "
            f"signature {cudagraph_signature}"
        )
    use_v2_signature = inspect.signature(use_v2_model_runner.fget)
    if tuple(use_v2_signature.parameters) != ("self",):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGETS[2]} has incompatible "
            f"signature {use_v2_signature}"
        )

    @functools.wraps(get_model_arch_config)
    def hcu_get_model_arch_config(self):
        hf_config = getattr(self, "hf_config", None)
        get_text_config = getattr(hf_config, "get_text_config", None)
        if not callable(get_text_config):
            raise PatchCompatibilityError(
                "ModelConfig.hf_config.get_text_config is missing"
            )
        self.hf_text_config = get_text_config()
        return get_model_arch_config(self)

    setattr(
        model_config_class,
        "_vllm_hcu_original_get_model_arch_config",
        get_model_arch_config,
    )
    setattr(
        model_config_class,
        "get_model_arch_config",
        hcu_get_model_arch_config,
    )

    @functools.wraps(with_hf_config)
    def hcu_with_hf_config(self, hf_config: object, architectures=None):
        updated = with_hf_config(self, hf_config, architectures)
        model_config = getattr(updated, "model_config", None)
        if model_config is None:
            return updated
        installed_hf_config = getattr(model_config, "hf_config", None)
        get_text_config = getattr(installed_hf_config, "get_text_config", None)
        if not callable(get_text_config):
            raise PatchCompatibilityError(
                "updated HuggingFace config does not expose get_text_config"
            )
        model_config.hf_text_config = get_text_config()
        refresh_model_arch_config = getattr(
            model_config, "get_model_arch_config", None
        )
        if not callable(refresh_model_arch_config):
            raise PatchCompatibilityError(
                "ModelConfig.get_model_arch_config is missing"
            )
        model_config.model_arch_config = refresh_model_arch_config()
        return updated

    @functools.wraps(set_cudagraph_sizes)
    def hcu_set_cudagraph_sizes(self) -> Any:
        # VllmConfig.__post_init__ performs its first cudagraph-size pass
        # before current_platform.check_and_update_config().  Bind directly
        # from the authoritative sidecar at this earlier boundary so the
        # CompilationConfig custom-SP wrapper observes the requested feature
        # during that first pass.  The later platform validation intentionally
        # keeps rebinding after spawn/unpickle.
        bind_hcu_config(self)
        compilation_config = getattr(self, "compilation_config", None)
        if compilation_config is None:
            raise PatchCompatibilityError("VllmConfig.compilation_config is missing")
        explicit_sizes = compilation_config.cudagraph_capture_sizes is not None
        compile_sizes = getattr(compilation_config, "compile_sizes", None)
        compile_sizes_template = (
            list(compile_sizes) if compile_sizes is not None else None
        )

        result = set_cudagraph_sizes(self)
        if explicit_sizes or not _request_cudagraph_buckets_enabled():
            return result
        _replace_with_request_cudagraph_buckets(
            self,
            compile_sizes_template=compile_sizes_template,
        )
        return result

    @functools.wraps(use_v2_model_runner.fget)
    def hcu_use_v2_model_runner(self) -> bool:
        # Upstream v0.25.1 forces all DSpark drafts into its V2 GPU runner.
        # The HCU DFly implementation is integrated with the mature HCU V1
        # runner, so this architecture must stay on that path even when the
        # generic DSpark rule (or VLLM_USE_V2_MODEL_RUNNER=1) would select V2.
        if _uses_dfly_draft(self):
            return False
        return use_v2_model_runner.fget(self)

    setattr(vllm_config, "_vllm_hcu_original_with_hf_config", with_hf_config)
    setattr(vllm_config, "with_hf_config", hcu_with_hf_config)
    setattr(
        vllm_config,
        "_vllm_hcu_original_set_cudagraph_sizes",
        set_cudagraph_sizes,
    )
    setattr(
        vllm_config,
        "_vllm_hcu_original_use_v2_model_runner",
        use_v2_model_runner,
    )
    setattr(
        vllm_config,
        "use_v2_model_runner",
        property(
            hcu_use_v2_model_runner,
            use_v2_model_runner.fset,
            use_v2_model_runner.fdel,
            use_v2_model_runner.__doc__,
        ),
    )
    setattr(vllm_config, "_set_cudagraph_sizes", hcu_set_cudagraph_sizes)
    setattr(vllm_config, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    vllm_module = load_exact_module(TARGET_MODULE, module)
    vllm_config = getattr(vllm_module, "VllmConfig", None)
    if not isinstance(vllm_config, type):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGET_MODULE}.VllmConfig is missing"
        )
    return apply_once(
        patch_id=PATCH_ID,
        targets=TARGETS,
        marker_owner=vllm_config,
        marker=_MARKER,
        callback=lambda: apply_to_module(vllm_module),
    )


__all__ = [
    "PATCH_ID",
    "TARGET_MODULE",
    "TARGETS",
    "apply",
    "apply_to_module",
    "validate_and_update_hcu_config",
]
