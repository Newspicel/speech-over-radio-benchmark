"""Audio file helpers shared by the harness: one rate, one channel, float32 everywhere."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

RATE = 48_000  # processing rate, in memory
STORE_RATE = 16_000  # radio speech has nothing above 4 kHz; files are 24-bit FLAC at 16 kHz


def load(path: str | Path, rate: int = RATE) -> np.ndarray:
    audio, native = sf.read(str(path), dtype="float32", always_2d=True)
    return resample(audio.mean(axis=1), native, rate)


def save(path: str | Path, audio: np.ndarray, rate: int = RATE) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    stored = np.clip(resample(np.nan_to_num(audio), rate, STORE_RATE), -1.0, 1.0)
    sf.write(str(tmp), stored, STORE_RATE, format="FLAC", subtype="PCM_24")
    tmp.replace(path)


def resample(audio: np.ndarray, src: int, dst: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if src == dst:
        return audio
    return soxr.resample(audio, src, dst, quality="VHQ").astype(np.float32)


def fit(audio: np.ndarray, length: int) -> np.ndarray:
    """Trim or zero-pad to exactly `length` samples."""
    if len(audio) >= length:
        return audio[:length]
    return np.pad(audio, (0, length - len(audio)))


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)) + 1e-20))


def db(power_ratio: float) -> float:
    return float(10 * np.log10(max(power_ratio, 1e-20)))


def peak_normalize(audio: np.ndarray, peak: float = 0.5) -> np.ndarray:
    top = float(np.max(np.abs(audio))) if len(audio) else 0.0
    return audio if top <= 0 else (audio * (peak / top)).astype(np.float32)


def best_lag(reference: np.ndarray, estimate: np.ndarray, max_lag: int) -> int:
    """Lag (in samples) that best aligns `estimate` to `reference`, by FFT cross-correlation."""
    n = len(reference) + len(estimate)
    size = 1 << (n - 1).bit_length()
    spec = np.fft.rfft(estimate, size) * np.conj(np.fft.rfft(reference, size))
    corr = np.fft.irfft(spec, size)
    lags = np.concatenate([corr[: max_lag + 1], corr[-max_lag:]])
    index = int(np.argmax(np.abs(lags)))
    return index if index <= max_lag else index - 2 * max_lag - 1


def shift(audio: np.ndarray, lag: int) -> np.ndarray:
    """Undo a positive lag (output late) or a negative one (output early), keeping length."""
    if lag > 0:
        return np.concatenate([audio[lag:], np.zeros(lag, np.float32)])
    if lag < 0:
        return np.concatenate([np.zeros(-lag, np.float32), audio[:lag]])
    return audio
