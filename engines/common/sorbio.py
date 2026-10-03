"""Tiny shared I/O for engine workers (imported via PYTHONPATH, needs numpy, soundfile, soxr).

A worker is called as `python worker.py --spec SPEC --jobs jobs.json`, where jobs.json is a
list of [input_path, output_path]. It loads its model once, then for each job reads mono
audio at the model's rate, processes it, and writes the result (any rate; the harness
resamples). Output length should match input length; the harness trims/pads and removes
any constant delay.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf
import soxr

STORE_RATE = 16_000
THREADS = int(os.environ.get("SORB_THREADS", "2"))


def _limit_threads() -> None:
    """Many workers share the machine: keep each one to a couple of threads instead of
    letting every ONNX Runtime / torch pool spin up one thread per core."""
    try:
        import onnxruntime as ort

        original = ort.InferenceSession.__init__

        def init(self, path, sess_options=None, *args, **kwargs):
            sess_options = sess_options or ort.SessionOptions()
            sess_options.intra_op_num_threads = THREADS
            sess_options.inter_op_num_threads = 1
            # Inputs change length every clip; the default arena keeps a buffer set per
            # shape and never shrinks (MP-SENet reached 19 GB). Allocate per run instead.
            sess_options.enable_cpu_mem_arena = False
            sess_options.enable_mem_pattern = False
            original(self, path, sess_options, *args, **kwargs)

        ort.InferenceSession.__init__ = init
    except ImportError:
        pass
    try:
        import torch

        torch.set_num_threads(THREADS)
    except ImportError:
        pass


for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, str(THREADS))


def read(path: str, rate: int) -> np.ndarray:
    audio, native = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    return audio if native == rate else soxr.resample(audio, native, rate, quality="VHQ").astype(np.float32)


def write(path: str, audio: np.ndarray, rate: int) -> None:
    audio = np.nan_to_num(np.asarray(audio, dtype=np.float32).reshape(-1))
    if rate != STORE_RATE:
        audio = soxr.resample(audio, rate, STORE_RATE, quality="VHQ").astype(np.float32)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    sf.write(str(tmp), np.clip(audio, -1, 1), STORE_RATE, format="FLAC", subtype="PCM_24")
    tmp.replace(out)


def chunked(process: Callable[[np.ndarray], np.ndarray], rate: int, seconds: float = 10.0,
            overlap: float = 1.0) -> Callable[[np.ndarray], np.ndarray]:
    """Wraps an offline model whose memory grows with input length: runs it on `seconds`-long
    windows and cross-fades `overlap` seconds between neighbours."""
    win, fade = int(seconds * rate), int(overlap * rate)

    def run(x: np.ndarray) -> np.ndarray:
        if len(x) <= win:
            return process(x)
        out = np.zeros(len(x), np.float32)
        weight = np.zeros(len(x), np.float32)
        ramp = np.ones(win, np.float32)
        ramp[:fade] = np.linspace(0, 1, fade, dtype=np.float32)
        ramp[-fade:] = np.linspace(1, 0, fade, dtype=np.float32)
        start = 0
        while True:
            seg = x[start : start + win]
            y = np.asarray(process(seg), np.float32).reshape(-1)[: len(seg)]
            y = np.pad(y, (0, len(seg) - len(y)))
            w = ramp[: len(seg)].copy()
            if start == 0:
                w[:fade] = 1
            if start + win >= len(x):
                w[-min(fade, len(seg)):] = 1
            out[start : start + len(seg)] += y * w
            weight[start : start + len(seg)] += w
            if start + win >= len(x):
                break
            start += win - fade
        return out / np.maximum(weight, 1e-6)

    return run


def run(load: Callable[[str], tuple[Callable[[np.ndarray], np.ndarray], int]]) -> None:
    """`load(spec)` returns (process(audio) -> audio, sample_rate)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--jobs", required=True)
    args = parser.parse_args()
    _limit_threads()
    jobs = json.loads(Path(args.jobs).read_text())
    process, rate = load(args.spec)
    started = time.perf_counter()
    busy = audio_s = 0.0
    for i, (src, dst) in enumerate(jobs):
        x = read(src, rate)
        t0 = time.perf_counter()
        y = np.asarray(process(x), dtype=np.float32).reshape(-1)
        busy += time.perf_counter() - t0
        audio_s += len(x) / rate
        if len(y) < len(x):
            y = np.pad(y, (0, len(x) - len(y)))
        write(dst, y[: len(x)], rate)
        if (i + 1) % 25 == 0:
            print(f"  {args.spec}: {i + 1}/{len(jobs)} ({time.perf_counter() - started:.0f}s)", file=sys.stderr, flush=True)
    timing = os.environ.get("SORB_TIMING")
    if timing:  # model inference time only, excluding start-up and file I/O
        Path(timing).write_text(json.dumps({"seconds": busy, "audio_seconds": audio_s}))
