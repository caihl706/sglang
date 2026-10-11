"""W4A8-INT8 MoE (GPU) functional fallback.

Loads pack-quantized INT4 expert weights with per-channel scales and dynamic
per-token INT8 activations, unpacks the INT4 weights to INT8 after loading and
runs the existing Triton INT8 W8A8 MoE path. Numerically identical to a real
W4A8 kernel (INT4 values are exactly representable in INT8, scales unchanged),
but weights occupy INT8 memory and no INT4 bandwidth saving is obtained.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from compressed_tensors import CompressionFormat
from compressed_tensors.quantization import QuantizationStrategy
from torch.nn import Parameter

from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a8_int8 import (
    unpack_int32_to_int8,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_int8_moe import (
    CompressedTensorsW8A8Int8DynamicMoE,
)
from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo
from sglang.srt.utils import set_weight_attrs

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )

logger = logging.getLogger(__name__)

__all__ = ["CompressedTensorsW4A8Int8DynamicMoE"]

_PACK_FACTOR = 8  # int32 holds 8 x int4


_UNPACK_BUFFERS: dict = {}


def _unpack_into(packed: torch.Tensor, tag: str) -> torch.Tensor:
    """[E, N, K // 8] int32 -> [E, N, K] int8 into a process-wide buffer shared
    by all layers (layers run sequentially on one stream), one expert at a
    time to bound the temporary int32 buffer."""
    shape = (*packed.shape[:-1], packed.shape[-1] * _PACK_FACTOR)
    key = (tag, shape, packed.device)
    out = _UNPACK_BUFFERS.get(key)
    if out is None:
        out = torch.empty(shape, dtype=torch.int8, device=packed.device)
        _UNPACK_BUFFERS[key] = out
    for e in range(packed.shape[0]):
        out[e] = unpack_int32_to_int8(packed[e])
    return out


def _unpack_experts(packed: torch.Tensor) -> torch.Tensor:
    """[E, N, K // 8] int32 -> [E, N, K] int8 (fresh allocation)."""
    out = torch.empty(
        (*packed.shape[:-1], packed.shape[-1] * _PACK_FACTOR),
        dtype=torch.int8,
        device=packed.device,
    )
    for e in range(packed.shape[0]):
        out[e] = unpack_int32_to_int8(packed[e])
    return out


class CompressedTensorsW4A8Int8DynamicMoE(CompressedTensorsW8A8Int8DynamicMoE):
    def __init__(self, quant_config, weight_quant, input_quant):
        # Parent validates channelwise weights + dynamic per-token activations.
        super().__init__(weight_quant, input_quant)
        assert weight_quant.num_bits == 4 and weight_quant.symmetric
        assert (
            quant_config.quant_format == CompressionFormat.pack_quantized.value
        ), f"W4A8Int8 MoE requires pack-quantized format, got {quant_config.quant_format}"

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        assert hidden_size % _PACK_FACTOR == 0
        assert intermediate_size_per_partition % _PACK_FACTOR == 0

        w13_weight = Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // _PACK_FACTOR,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_packed", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // _PACK_FACTOR,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_packed", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        assert self.weight_quant.strategy == QuantizationStrategy.CHANNEL
        w13_scale = Parameter(
            torch.ones(
                num_experts, 2 * intermediate_size_per_partition, 1, dtype=torch.float32
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_scale)
        w2_scale = Parameter(
            torch.ones(num_experts, hidden_size, 1, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_scale)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value}
        )
        set_weight_attrs(w13_scale, extra_weight_attrs)
        set_weight_attrs(w2_scale, extra_weight_attrs)

        # Checkpoint carries a [out, in] shape tensor per projection; unused.
        for name in ("w13_weight_shape", "w2_weight_shape"):
            p = Parameter(
                torch.empty(num_experts, 2, dtype=torch.int32), requires_grad=False
            )
            layer.register_parameter(name, p)
            set_weight_attrs(p, extra_weight_attrs)

        layer.w13_input_scale = None
        layer.w2_input_scale = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Keep INT4 packed weights resident; unpack per forward in apply_weights.
        layer.w13_weight_packed = Parameter(
            layer.w13_weight_packed.data, requires_grad=False
        )
        layer.w2_weight_packed = Parameter(
            layer.w2_weight_packed.data, requires_grad=False
        )
        layer.w13_weight_scale = Parameter(
            layer.w13_weight_scale.data, requires_grad=False
        )
        layer.w2_weight_scale = Parameter(
            layer.w2_weight_scale.data, requires_grad=False
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        w13 = _unpack_into(layer.w13_weight_packed.data, "w13")
        w2 = _unpack_into(layer.w2_weight_packed.data, "w2")
        quant_info = TritonMoeQuantInfo(
            w13_weight=w13,
            w2_weight=w2,
            use_int8_w8a8=True,
            per_channel_quant=True,
            w13_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            a13_scale=None,
            a2_scale=None,
        )
        return self.runner.run(dispatch_output, quant_info)
