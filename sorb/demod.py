"""Demodulates real IQ recordings with the same receivers the simulator uses."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

from . import channel as C

FS = C.FS


def read_iq(src: Path, entry: dict) -> tuple[np.ndarray, float]:
    """Returns complex samples of the requested window and their sample rate."""
    fmt = entry.get("format", src.suffix.lstrip("."))
    start, duration = float(entry.get("start", 0)), entry.get("duration", 60)
    if fmt == "wav":
        info = sf.info(str(src))
        rate = info.samplerate
        frames = int(duration * rate) if duration else -1
        data, _ = sf.read(str(src), start=int(start * rate), frames=frames, dtype="float32", always_2d=True)
        iq = data[:, 0] + 1j * data[:, 1]
        if entry.get("swap_iq"):
            iq = np.conj(iq)
        return iq, float(rate)
    if fmt == "sigmf-data" or fmt == "sigmf":
        meta = json.loads(src.with_suffix(".sigmf-meta").read_text())
        rate = float(meta["global"]["core:sample_rate"])
        fmt = meta["global"]["core:datatype"]
    else:
        rate = float(entry["rate"])
    dtype, scale, offset = {
        "cu8": (np.uint8, 1 / 127.5, 127.5), "ci8": (np.int8, 1 / 128, 0), "cs8": (np.int8, 1 / 128, 0),
        "ci16_le": (np.int16, 1 / 32768, 0), "cs16": (np.int16, 1 / 32768, 0),
        "cf32_le": (np.float32, 1.0, 0), "cf32": (np.float32, 1.0, 0),
    }[fmt]
    item = np.dtype(dtype).itemsize * 2
    first = int(start * rate)
    count = int(duration * rate) * 2 if duration else -1
    raw = np.fromfile(src, dtype=dtype, count=count, offset=first * item).astype(np.float32)
    raw = (raw - offset) * scale
    return raw[0::2] + 1j * raw[1::2], rate


def to_audio_rate(iq: np.ndarray, rate: float, offset_hz: float) -> np.ndarray:
    if offset_hz:
        iq = iq * np.exp(-2j * np.pi * offset_hz * np.arange(len(iq)) / rate)
    if abs(rate - FS) < 1:
        return iq
    pair = np.stack([iq.real, iq.imag], axis=1).astype(np.float32)
    out = soxr.resample(pair, rate, FS, quality="HQ")
    return out[:, 0] + 1j * out[:, 1]


def demodulate_iq(src: Path, entry: dict, demod: str) -> np.ndarray:
    iq, rate = read_iq(src, entry)
    bb = to_audio_rate(iq, rate, float(entry.get("offset_hz", 0)))
    if demod == "nfm":
        audio = C._nfm_rx(bb)
        low, high = 300, 3_000
    elif demod in ("am", "am-envelope"):
        audio = C._am_rx(bb, "envelope")
        low, high = 250, 3_000
    elif demod == "am-sync":
        audio = C._am_rx(bb, "sync")
        low, high = 250, 3_000
    elif demod in ("usb", "lsb"):
        audio = C._usb_detect(bb if demod == "usb" else np.conj(bb))
        audio = audio * C.agc_gain(audio)
        low, high = C.SSB_LOW, C.SSB_HIGH
    else:
        raise ValueError(f"unknown demodulator {demod}")
    audio = C.band(np.asarray(audio, np.float64), low, high, 4)
    return audio.astype(np.float32)
