# AMD 实机源码检查（2026-10-01）

这里只保存与指定量化模型相关的包内源码和返回值。依赖源码真实路径已简化；没有读取其他项目数据。未执行模型推理或训练。

## vLLM 源码与算子注册

```text
VLLM_ORIGIN <vllm-source>/vllm/__init__.py
FILE <vllm-source>/vllm/model_executor/kernels/linear/mixed_precision/rdna3_w4a16.py EXISTS True
SHA256 037c54e54dc2ab2892dac3b10da5a6a9dae3e3a87373b2c3625a688dd589ecc6
FUNCTION can_implement LINE 35
def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        if not current_platform.is_rocm():
            return False, "RDNA3 W4A16 kernel is ROCm-only"

        from vllm.platforms.rocm import on_gfx1100

        if not on_gfx1100():
            return False, "RDNA3 W4A16 kernel requires gfx1100"

        # The HIP op is registered by the C++ extension; if a user is running
        # against a vLLM build that doesn't include it (e.g. partial rebuild),
        # fall through gracefully to the next kernel in the registry.
        if not (
            hasattr(torch.ops, "_rocm_C")
            and hasattr(torch.ops._rocm_C, "gptq_gemm_rdna3")
        ):
            return (
                False,
                "torch.ops._rocm_C.gptq_gemm_rdna3 missing — rebuild C++ extension",
            )

        if c.act_type not in (torch.float16, torch.bfloat16):
            return False, "RDNA3 W4A16 kernel only supports fp16 and bf16"

        if c.weight_type not in cls.SUPPORTED_QUANT_TYPES:
            return (
                False,
                f"Quant type ({c.weight_type}) not supported by "
                f"RDNA3 W4A16 kernel; supported: {cls.SUPPORTED_QUANT_TYPES}",
            )

        if c.group_size <= 0:
            return (
                False,
                "RDNA3 W4A16 kernel does not support channelwise quantization",
            )

        if c.full_weight_shape[0] % c.group_size != 0:
            return (
                False,
                f"Group size ({c.group_size}) does not evenly divide K "
                f"({c.full_weight_shape[0]})",
            )

        # Output features must be a multiple of the pack factor (8 nibbles per
        # int32) and of 8 so that qzeros (packed 4-bit per col) align cleanly
        # against the BLOCK_KN_SIZE*4 = 512 N-stride and per-thread 4 columns.
        if c.partition_weight_shape[1] % 8 != 0:
            return (
                False,
                "Output features must be a multiple of 8 for the RDNA3 "
                "W4A16 kernel (qzeros packing)",
            )

        if c.has_g_idx and c.partition_weight_shape[0] != c.full_weight_shape[0]:
            return (
                False,
                "Act-order with TP-partitioned input features is not "
                "supported by the RDNA3 W4A16 kernel",
            )

        return True, None
FILE <vllm-source>/vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16.py EXISTS True
SHA256 15550ed16f9931937fc662d69f2e190401b7c9ef9ce6c6ba6001bbd03d527562
FUNCTION __init__ LINE 54
def __init__(
        self,
        strategy: str,
        num_bits: int,
        group_size: int | None = None,
        symmetric: bool | None = True,
        actorder: ActivationOrdering | None = None,
        layer_name: str | None = None,
    ):
        self.num_bits = num_bits
        self.pack_factor = Fraction(32, num_bits)
        self.strategy = strategy
        self.symmetric = symmetric
        self.group_size = -1 if group_size is None else group_size
        self.has_g_idx = actorder == ActivationOrdering.GROUP
        self.layer_name = layer_name

        if self.group_size == -1 and self.strategy != "channel":
            raise ValueError(
                "Pack-quantized format requires group quantization "
                "or channelwise quantization, but found no group "
                "size and strategy is not channelwise."
            )

        if num_bits not in WNA16_SUPPORTED_TYPES_MAP:
            raise ValueError(
                f"Unsupported num_bits = {num_bits}. "
                f"Supported num_bits = {list(WNA16_SUPPORTED_TYPES_MAP)}"
            )

        if not self.symmetric and num_bits not in WNA16_ZP_SUPPORTED_TYPES_MAP:
            raise ValueError(
                f"Asymmetric quantization not supported for "
                f"num_bits = {num_bits}. Supported: "
                f"{list(WNA16_ZP_SUPPORTED_TYPES_MAP)}"
            )

        self.quant_type = (
            WNA16_ZP_SUPPORTED_TYPES_MAP[num_bits]
            if not self.symmetric
            else WNA16_SUPPORTED_TYPES_MAP[num_bits]
        )
ROCM_DISPATCH_LINE 370 
    PlatformEnum.ROCM: [AiterInt8ScaledMMLinearKernel, TritonInt8ScaledMMLinearKernel],
    PlatformEnum.XPU: [TritonInt8ScaledMMLinearKernel],
}

# in priority/performance order (when available)
_POSSIBLE_FP8_KERNELS: dict[PlatformEnum, list[type[FP8ScaledMMLinearKernel]]] = {
    PlatformEnum.CUDA: [
        MarlinFP8ScaledMMLinearKernel,
        FlashInferFP8ScaledMMLinearKernel,
        CutlassFP8ScaledMMLinearKernel,
        PerTensorTorchFP8ScaledMMLinearKernel,
        ChannelWiseTorchFP8ScaledMMLinearKernel,
        HummingFP8ScaledMMLinearKernel,
    ],
    PlatformEnum.ROCM: [
        AiterHipbMMPerTokenFp8ScaledMMLinearKernel,
        AiterPreshuffledPerTokenFp8ScaledMMLinearKernel,
        AiterPerTokenFp8ScaledMMLinearKernel,
        ROCmFP8ScaledMMLinearKernel,
ROCM_DISPATCH_LINE 384 
    PlatformEnum.ROCM: [
        AiterHipbMMPerTokenFp8ScaledMMLinearKernel,
        AiterPreshuffledPerTokenFp8ScaledMMLinearKernel,
        AiterPerTokenFp8ScaledMMLinearKernel,
        ROCmFP8ScaledMMLinearKernel,
        PerTensorTorchFP8ScaledMMLinearKernel,
        RowWiseTorchFP8ScaledMMLinearKernel,
        ChannelWiseTorchFP8ScaledMMLinearKernel,
    ],
    PlatformEnum.CPU: [
        PerTensorTorchFP8ScaledMMLinearKernel,
        ChannelWiseTorchFP8ScaledMMLinearKernel,
    ],
    PlatformEnum.XPU: [
        XPUW8A16FP8LinearKernel,
        XPUW8A8FP8LinearKernel,
        PerTensorTorchFP8ScaledMMLinearKernel,
        ChannelWiseTorchFP8ScaledMMLinearKernel,
    ],
ROCM_DISPATCH_LINE 419 
    PlatformEnum.ROCM: [
        AiterFp8BlockScaledMMKernel,
        TritonFp8BlockScaledMMKernel,
    ],
    PlatformEnum.CPU: [
        CPUFp8BlockScaledMMKernel,
    ],
    PlatformEnum.XPU: [
        XPUFp8BlockScaledMMKernel,
        TritonFp8BlockScaledMMKernel,
    ],
}

_POSSIBLE_WFP8A16_KERNELS: dict[PlatformEnum, list[type[FP8ScaledMMLinearKernel]]] = {
    PlatformEnum.CUDA: [
        HummingFP8ScaledMMLinearKernel,
        MarlinFP8ScaledMMLinearKernel,
    ],
    PlatformEnum.ROCM: [
ROCM_DISPATCH_LINE 437 
    PlatformEnum.ROCM: [
        # To be added
    ],
    PlatformEnum.CPU: [
        # To be added
    ],
    PlatformEnum.XPU: [
        XPUW8A16FP8LinearKernel,
    ],
}

# in priority/performance order (when available)
_POSSIBLE_KERNELS: dict[PlatformEnum, list[type[MPLinearKernel]]] = {
    PlatformEnum.CUDA: [
        CutlassW4A8LinearKernel,
        MacheteLinearKernel,
        AllSparkLinearKernel,
        MarlinLinearKernel,
        ConchLinearKernel,
ROCM_DISPATCH_LINE 460 
    PlatformEnum.ROCM: [
        RDNA3W4A16LinearKernel,
        RDNAHybridW4A16LinearKernel,
        TritonW4A16LinearKernel,
        ConchLinearKernel,
        ExllamaLinearKernel,
    ],
    PlatformEnum.XPU: [
        XPUW4A8IntLinearKernel,
        XPUwNa16LinearKernel,
    ],
    PlatformEnum.CPU: [
        Dynamic4bitLinearKernel,
        ZentorchWNA16LinearKernel,
        CPUWNA16LinearKernel,
    ],
}

# in priority/performance order (when available)
ROCM_DISPATCH_LINE 487 
    PlatformEnum.ROCM: [
        # Native CDNA4 (gfx950) MX linear; is_supported() gates to gfx95x and
        # falls through to BF16 emulation (hipBLASLt) elsewhere / on regression.
        RocmDotScaledMxfp8LinearKernel,
        EmulationMxfp8LinearKernel,
    ],
    PlatformEnum.XPU: [
        XPUMxFp8LinearKernel,
        EmulationMxfp8LinearKernel,
    ],
}

_POSSIBLE_NVFP4_KERNELS: dict[PlatformEnum, list[type[NvFp4LinearKernel]]] = {
    PlatformEnum.CUDA: [
        FlashInferCuteDslNvFp4LinearKernel,
        FlashInferCutlassNvFp4LinearKernel,
        FlashInferB12xNvFp4LinearKernel,
        CutlassNvFp4LinearKernel,
        MarlinNvFp4LinearKernel,
ROCM_DISPATCH_LINE 512 
    PlatformEnum.ROCM: [
        EmulationNvFp4LinearKernel,
    ],
}

_POSSIBLE_MXFP6_KERNELS: dict[PlatformEnum, list[type[MxFp6LinearKernel]]] = {
    PlatformEnum.CUDA: [
        EmulationMxfp6LinearKernel,
    ],
    PlatformEnum.ROCM: [
        EmulationMxfp6LinearKernel,
    ],
}

_POSSIBLE_MXFP4_KERNELS: dict[PlatformEnum, list[type[MxFp4LinearKernel]]] = {
    PlatformEnum.CUDA: [
        FlashInferMxFp4LinearKernel,
        MarlinMxFp4LinearKernel,
        HummingMxFp4LinearKernel,
ROCM_DISPATCH_LINE 521 
    PlatformEnum.ROCM: [
        EmulationMxfp6LinearKernel,
    ],
}

_POSSIBLE_MXFP4_KERNELS: dict[PlatformEnum, list[type[MxFp4LinearKernel]]] = {
    PlatformEnum.CUDA: [
        FlashInferMxFp4LinearKernel,
        MarlinMxFp4LinearKernel,
        HummingMxFp4LinearKernel,
        EmulationMxfp4LinearKernel,
    ],
    PlatformEnum.ROCM: [
        AiterMxfp4LinearKernel,
        EmulationMxfp4LinearKernel,
    ],
    PlatformEnum.XPU: [
        XPUMxFp4LinearKernel,
    ],
ROCM_DISPATCH_LINE 533 
    PlatformEnum.ROCM: [
        AiterMxfp4LinearKernel,
        EmulationMxfp4LinearKernel,
    ],
    PlatformEnum.XPU: [
        XPUMxFp4LinearKernel,
    ],
}

# TODO make all kernels inherit from MMLinearKernel
# then bound _KernelT only to MMLinearKernel
_KernelT = TypeVar("_KernelT", bound=ScaledMMLinearKernel | MMLinearKernel)
_KernelConfigT = TypeVar("_KernelConfigT", bound=MMLinearLayerConfig)


def is_supported_and_can_implement_kernel(
    kernel: type[_KernelT], config: _KernelConfigT, compute_capability: int | None
) -> tuple[bool, str]:
    if kernel.__name__ in envs.VLLM_DISABLED_KERNELS:
PARSER tool_parsers/__init__.py 205     "gemma4": (
PARSER tool_parsers/__init__.py 206         "gemma4_engine_tool_parser",
PARSER tool_parsers/__init__.py 207         "Gemma4EngineToolParser",
PARSER reasoning/__init__.py 51     "gemma4": (
PARSER reasoning/__init__.py 52         "gemma4_engine_reasoning_parser",
PARSER reasoning/__init__.py 53         "Gemma4ParserReasoningAdapter",
EXTENSION_IMPORT vllm._C ok
EXTENSION_IMPORT vllm._rocm_C ok
RDNA3_OP_REGISTERED True
```

## Transformers CT 加载器

第一次检查通过distribution目录定位vLLM失败，随后通过模块spec取得真实源码路径；该失败不影响下面已读取的Transformers内容。第二次检查已完成vLLM确认。

```text
FILE <site-packages>/vllm/model_executor/kernels/linear/mixed_precision/rdna3_w4a16.py EXISTS False
FILE <site-packages>/vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16.py EXISTS False
FILE <site-packages>/transformers/quantizers/quantizer_compressed_tensors.py EXISTS True
SHA256 e50131e5e6fca3ac8a4d0fdc997e6cfa897c9b8926d9b05aec198a37e789166d
FUNCTION validate_environment LINE 68
def validate_environment(self, *args, **kwargs):
        if not is_compressed_tensors_available():
            raise ImportError(
                "Using `compressed_tensors` quantized models requires compressed-tensors>=0.15.0: "
                "`pip install compressed-tensors>=0.15.0`"
            )

        # `use_optimized_inference` has been resolved against the checkpoint by
        # `CompressedTensorsConfig.post_init`; what is left to check is whether the hardware can run
        # the kernels. When it cannot, the model goes through the regular compressed-tensors route.
        self.use_fp8_kernel = self.quantization_config.use_optimized_inference
        if self.use_fp8_kernel and not (
            torch.xpu.is_available() or (torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9))
        ):
            logger.warning_once(
                "Ignoring `use_optimized_inference=True`: FP8 matmul kernels need a CUDA GPU with compute capability "
                ">= 8.9 (e.g. 4090/H100) or an Intel XPU, and none was found."
            )
            self.use_fp8_kernel = False
FUNCTION _process_model_before_weight_loading LINE 88
def _process_model_before_weight_loading(self, model, **kwargs):
        ct_config = self.compressor.quantization_config
        remaining_groups = dict(ct_config.config_groups)

        if self.use_fp8_kernel:
            from ..integrations.compressed_tensors import replace_with_compressed_tensors_fp8_linear

            # Quantization may only target a subset of the layers: each config group scopes
            # its scheme with `targets` (class names or `re:` regexes) minus `ignore`.
            fp8_groups = {name: group for name, group in remaining_groups.items() if _is_fp8_scheme(group)}
            remaining_groups = {name: group for name, group in remaining_groups.items() if name not in fp8_groups}

            self.modules_to_not_convert = self.get_modules_to_not_convert(model, None, model._keep_in_fp32_modules)
            targets = [target for group in fp8_groups.values() for target in group.targets]
            replace_with_compressed_tensors_fp8_linear(
                model,
                targets=targets,
                ignore=ct_config.ignore,
                modules_to_not_convert=self.modules_to_not_convert,
            )

        if not remaining_groups:
            return

        from compressed_tensors.quantization import apply_quantization_config

        remaining_config = deepcopy(ct_config)
        remaining_config.config_groups = remaining_groups

        apply_quantization_config(model, remaining_config, run_compressed=False)
        # Packed formats (e.g. int4 `weight_packed`) need the compressed module layout to
        # receive the checkpoint tensors.
        if self.quantization_config.is_quantization_compressed:
            self.compressor.compress_model(model=model)
FUNCTION _process_model_after_weight_loading LINE 123
def _process_model_after_weight_loading(self, model, **kwargs):
        """Dequantize the layers loaded through the compressed-tensors wrappers, but only when asked
        for: with `dequantize=False` the weights are left compressed, and the hook `compress_model`
        registered decompresses them on the first forward pass instead.

        FP8-kernel modules were never wrapped (their weights loaded directly in FP8) and hold no
        compressed-tensors scheme, so neither the call below nor that hook touches them."""
        if self.quantization_config.dequantize and self.quantization_config.is_quantization_compressed:
            self.compressor.decompress_model(model=model)
FUNCTION is_trainable LINE 149
def is_trainable(self):
        # The FP8 kernel path is inference-only; load with `dequantize=True` to fine-tune.
        return not self.use_fp8_kernel
```

