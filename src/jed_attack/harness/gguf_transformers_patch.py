"""Patch transformers' gpt-oss GGUF config parser (unfixed bug, transformers 5.16).

``load_gguf_checkpoint`` reads ``reader.fields[key].parts[0]`` for the
``gpt-oss.rope.scaling.*`` metadata, but ``parts[0]`` is the key-length metadata (a
1-element array), not the value ``parts[field.data[0]]``; ``float(value)`` raises a
0-dim-array ``TypeError`` and no gpt-oss GGUF loads. The module ships the correct
``read_field`` helper above; this rebinds ``load_gguf_checkpoint`` to a copy
that uses it (verified: rope_scaling -> yarn factor=32.0).

Call :func:`install` before a gpt-oss ``from_pretrained(gguf_file=...)``. Idempotent.
Upstream bug unreported (PR #45506 shipped it). Lets GCG optimize the SAME Q4_K_M weight
values llama.cpp serves (no MXFP4 transfer gap).
"""

import inspect

_installed = False


def install() -> None:
    """Rebind ``load_gguf_checkpoint`` to the read_field-corrected copy. Idempotent."""
    global _installed
    if _installed:
        return
    import transformers.configuration_utils as configuration_utils
    import transformers.modeling_gguf_pytorch_utils as gguf_utils

    src = inspect.getsource(gguf_utils.load_gguf_checkpoint)
    if "reader.fields[key].parts[0]" not in src:
        _installed = True  # a future transformers release already fixed it
        return
    fixed = src.replace("reader.fields[key].parts[0]", "read_field(reader, key)[0]")
    namespace = dict(gguf_utils.__dict__)
    exec(compile(fixed, gguf_utils.__file__, "exec"), namespace)  # noqa: S102
    patched = namespace["load_gguf_checkpoint"]
    gguf_utils.load_gguf_checkpoint = patched
    # configuration_utils does a top-level `from ... import load_gguf_checkpoint`
    configuration_utils.load_gguf_checkpoint = patched
    _installed = True
