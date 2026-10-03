"""Official DPDFNet (ceva-ip/DPDFNet, ONNX via the `dpdfnet` package). spec = MODEL[:ATTN_LIMIT_DB]"""

from dpdfnet.api import _enhance_with_runtime
from dpdfnet.models import resolve_model
from dpdfnet.onnx_backend import build_runtime_model

import sorbio


def load(spec):
    name, _, limit = spec.partition(":")
    resolved = resolve_model(model=name, onnx_path=None, auto_download=True, verbose=False)
    runtime = build_runtime_model(resolved.onnx_path)
    rate = resolved.info.sample_rate
    attn = float(limit) if limit else None
    return (lambda x: _enhance_with_runtime(x, rate, runtime=runtime, model_sample_rate=rate, attn_limit_db=attn)), rate


sorbio.run(load)
