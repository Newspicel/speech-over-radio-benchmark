"""Runs inside 32-bit Windows Python under Wine. Loads SDR#'s shark.dll and applies NINR.

    python.exe host.py SHARK_DLL SPEC_JSON JOBS_JSON

JOBS_JSON lists [in_raw, out_raw]: little-endian float32 interleaved complex samples at
spec["rate"]. Calls happen in `block`-sample chunks, like SDR#'s audio stream."""

import ctypes
import json
import sys

V, F, D, I, B = ctypes.c_void_p, ctypes.c_float, ctypes.c_double, ctypes.c_int, ctypes.c_ubyte
lib = ctypes.WinDLL(sys.argv[1])
spec = json.load(open(sys.argv[2]))
jobs = json.load(open(sys.argv[3]))
prefix = "noise_filter_lp_" if spec["engine"] == "lp" else "noise_filter_"


def fn(name, *args):
    f = getattr(lib, prefix + name)
    f.argtypes = [V, *args]
    f.restype = None
    return f


create = getattr(lib, prefix + "create")
create.restype = V
destroy = fn("destroy")
process = fn("process", V, I)
setters = {
    "sample_rate": (D,), "fft_size": (I,), "depth": (F,), "time_smoothing": (F,),
    "frequency_smoothing": (F,), "boost_snr": (B,), "skip_dc": (B,),
    # LP (current NINR) only
    "margin": (F,), "sensitivity": (F,), "use_vad": (B,),
    # HQ (classic NINR) only
    "noise_threshold": (F,), "slope": (F,),
}

for src, dst in jobs:
    data = bytearray(open(src, "rb").read())
    count = len(data) // 8
    buf = (ctypes.c_float * (2 * count)).from_buffer(data)
    h = create()
    for key, value in spec["params"].items():
        fn("set_" + key, *setters[key])(h, value)
    block = spec.get("block", 1024)
    base = ctypes.addressof(buf)
    for start in range(0, count, block):
        process(h, ctypes.c_void_p(base + 8 * start), min(block, count - start))
    destroy(h)
    open(dst, "wb").write(bytes(data))
