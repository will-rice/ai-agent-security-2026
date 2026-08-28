"""Inject gpt-oss MoE expert + attention-sink weights from the Q4_K_M GGUF.

transformers loads a gpt-oss GGUF's attention/norm/embedding tensors correctly, but
its ``GptOssTensorProcessor`` never matches this GGUF's expert tensor names
(``ffn_gate_exps``/``ffn_up_exps``/``ffn_down_exps``) nor the model's real parameter
layout, so ``mlp.experts.*`` and ``self_attn.sinks`` come up randomly initialised and
the model emits gibberish. :func:`inject_experts` MXFP4-dequantises the GGUF expert
tensors, reshapes them into the HF ``GptOssExperts`` layout, and overwrites the random
params in place -- yielding a ``GptOssForCausalLM`` numerically faithful to the same
weights llama.cpp serves (verified: greedy output matches llama.cpp).

Layout mapping (per layer, per expert ``e``):
  * GGUF ``ffn_gate_exps``/``ffn_up_exps`` dequantise to ``[32, n_ff, n_embd]`` =
    ``[expert, out, in]`` (llama.cpp stores ``ne = {n_embd, n_ff, n_expert}``; numpy
    reverses it, and MXFP4 packs the input dim ``n_embd`` last). HF
    ``gate_up_proj[e]`` is ``[hidden=in, 2*intermediate=out]`` with gate at even output
    columns and up at odd (``gate_up[..., ::2]``/``[..., 1::2]``), so each expert is
    transposed to ``[in, out]`` then interleaved.
  * GGUF ``ffn_down_exps`` dequantises to ``[32, n_embd, n_ff]`` = ``[expert, hidden,
    intermediate]`` = ``[out, in]``; HF ``down_proj[e]`` is ``[intermediate=in,
    hidden=out]``, so it is transposed.
  * Biases are F32; gate/up biases interleave the same even/odd way, down direct.
  * ``attn_sinks.weight`` (F32 ``[num_heads]``) fills ``self_attn.sinks``.
"""

import logging
from pathlib import Path

import torch
from torch import nn

_log = logging.getLogger(__name__)


def inject_experts(model: nn.Module, gguf_path: Path) -> None:
    """Fill the missing gpt-oss expert + sink params from the GGUF, in place.

    Args:
        model: A ``GptOssForCausalLM`` freshly loaded via
            ``from_pretrained(gguf_file=...)`` (experts/sinks randomly initialised).
        gguf_path: Path to the gpt-oss Q4_K_M GGUF whose experts to inject.

    Raises:
        KeyError: If an expected GGUF expert/sink tensor is absent.
    """
    from gguf import GGUFReader

    reader = GGUFReader(str(gguf_path))
    tensors = {t.name: t for t in reader.tensors}

    sample = model.get_parameter("model.layers.0.input_layernorm.weight")
    device, dtype = sample.device, sample.dtype
    # Count layers from the GGUF block indices present (blk.<i>.ffn_gate_exps.weight).
    n_layers = 1 + max(
        int(name.split(".")[1])
        for name in tensors
        if name.startswith("blk.") and name.endswith(".ffn_gate_exps.weight")
    )

    def dequant(name: str) -> torch.Tensor:
        """MXFP4-dequantise a GGUF weight to a float32 CPU tensor (logical shape)."""
        from gguf.quants import dequantize

        if name not in tensors:
            raise KeyError(f"missing GGUF tensor: {name}")
        t = tensors[name]
        return torch.from_numpy(dequantize(t.data, t.tensor_type))

    def field(name: str) -> torch.Tensor:
        """Read an F32 GGUF tensor (bias/sink) as a float32 CPU tensor."""
        if name not in tensors:
            raise KeyError(f"missing GGUF tensor: {name}")
        return torch.from_numpy(tensors[name].data.copy())

    def to_param(t: torch.Tensor) -> nn.Parameter:
        """Move a CPU tensor to the model device/dtype as a frozen Parameter."""
        return nn.Parameter(t.contiguous().to(device, dtype), requires_grad=False)

    for i in range(n_layers):
        experts = model.get_submodule(f"model.layers.{i}.mlp.experts")
        attn = model.get_submodule(f"model.layers.{i}.self_attn")

        # [expert, out=n_ff, in=n_embd] -> [expert, in, out]
        gate = dequant(f"blk.{i}.ffn_gate_exps.weight").transpose(1, 2)
        up = dequant(f"blk.{i}.ffn_up_exps.weight").transpose(1, 2)
        # [expert, out=n_embd, in=n_ff] -> [expert, in, out]
        down = dequant(f"blk.{i}.ffn_down_exps.weight").transpose(1, 2)

        gate_b = field(f"blk.{i}.ffn_gate_exps.bias")  # [expert, n_ff]
        up_b = field(f"blk.{i}.ffn_up_exps.bias")
        down_b = field(f"blk.{i}.ffn_down_exps.bias")  # [expert, n_embd]

        n_experts, hidden, inter = gate.shape
        gate_up = torch.empty(n_experts, hidden, 2 * inter, dtype=torch.float32)
        gate_up[..., ::2] = gate
        gate_up[..., 1::2] = up
        gate_up_b = torch.empty(n_experts, 2 * inter, dtype=torch.float32)
        gate_up_b[..., ::2] = gate_b
        gate_up_b[..., 1::2] = up_b

        with torch.no_grad():
            experts.register_parameter("gate_up_proj", to_param(gate_up))
            experts.register_parameter("gate_up_proj_bias", to_param(gate_up_b))
            experts.register_parameter("down_proj", to_param(down))
            experts.register_parameter("down_proj_bias", to_param(down_b))
            sinks = field(f"blk.{i}.attn_sinks.weight")  # [num_heads]
            attn.register_parameter("sinks", to_param(sinks))
    _log.info("injected experts + sinks for %d layers from %s", n_layers, gguf_path)
