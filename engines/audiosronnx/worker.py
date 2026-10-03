"""ONNX ports via audiosronnx (TigreGotico). spec = denoise:ENGINE | bwe:ENGINE
denoise engines used: mpsenet, cmgan; bandwidth extension (bwe): lavasr, novasr."""

import numpy as np

import sorbio


def load(spec):
    kind, _, engine = spec.partition(":")
    from audiosronnx import load_denoise, load_sr

    rate = 16_000
    if kind == "denoise":
        model = load_denoise(engine)
        # MP-SENet / CMGAN attend over time in every frequency bin (memory ~ length^2):
        # 4 s chunks with a 0.5 s cross-fade keep a worker near 1-2 GB.
        return sorbio.chunked(lambda x: _back(*model.denoise(x, rate), rate), rate, 4.0, 0.5), rate
    model = load_sr(engine)
    return (lambda x: _back(*model.upscale(x, rate), rate)), rate


def _back(y, out_rate, rate):
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    return y if out_rate == rate else sorbio.soxr.resample(y, out_rate, rate).astype(np.float32)


sorbio.run(load)
