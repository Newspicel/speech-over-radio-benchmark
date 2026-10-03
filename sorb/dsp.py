"""Classic DSP speech enhancement, implemented here so every step is visible.

All estimators share one STFT (16 kHz, 32 ms Hann, 75 % overlap) and one noise tracker
(per-bin quantile of the smoothed periodogram over a sliding window, Stahl et al. 2000),
so the comparison is between gain rules rather than between noise estimators.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage, signal
from scipy.special import exp1, i0e, i1e

from .audio import resample

RATE = 16_000
NFFT = 512
HOP = 128


def _stft(x: np.ndarray) -> np.ndarray:
    _, _, spec = signal.stft(x, RATE, "hann", NFFT, NFFT - HOP, boundary="even", padded=True)
    return spec


def _istft(spec: np.ndarray, length: int) -> np.ndarray:
    _, x = signal.istft(spec, RATE, "hann", NFFT, NFFT - HOP, boundary=True)
    return x[:length].astype(np.float32)


def noise_psd(power: np.ndarray, window_s: float = 1.5, quantile: float = 15) -> np.ndarray:
    """Quantile-based noise estimate, tracking slow changes (AGC, fading noise floor)."""
    frames = max(3, int(window_s * RATE / HOP))
    smooth = ndimage.uniform_filter1d(power, 3, axis=1)
    est = ndimage.percentile_filter(smooth, quantile, size=(1, frames), mode="nearest")
    # Quantiles under-read the mean of an exponential (chi-square 2) variable.
    bias = -np.log(1 - quantile / 100)
    return np.maximum(est / bias, 1e-12)


def _enhance(audio: np.ndarray, rate: int, rule: str, floor_db: float, **kw) -> np.ndarray:
    x = resample(audio, rate, RATE)
    spec = _stft(x)
    power = np.abs(spec) ** 2
    noise = noise_psd(power)
    gain = GAIN_RULES[rule](power, noise, **kw)
    gain = np.maximum(gain, 10 ** (floor_db / 20))
    y = _istft(spec * gain, len(x))
    return resample(y, RATE, rate)[: len(audio)]


def _decision_directed(power: np.ndarray, noise: np.ndarray, gain_fn, alpha: float = 0.98):
    post = np.minimum(power / noise, 1e3)
    gains = np.ones_like(power)
    prev = np.ones(power.shape[0])
    for t in range(power.shape[1]):
        prior = alpha * prev**2 * post[:, t - 1] if t else np.zeros(power.shape[0])
        prior = np.maximum(prior + (1 - alpha) * np.maximum(post[:, t] - 1, 0), 10 ** (-25 / 10))
        g = gain_fn(prior, post[:, t])
        gains[:, t] = g
        prev = g
    return gains


def _wiener(power, noise):
    return _decision_directed(power, noise, lambda xi, gamma: xi / (1 + xi))


def _mmse_stsa(power, noise):
    def rule(xi, gamma):
        v = np.clip(xi / (1 + xi) * gamma, 1e-6, 500)
        g = (np.sqrt(np.pi * v) / (2 * gamma)) * ((1 + v) * i0e(v / 2) + v * i1e(v / 2))
        return np.minimum(g, 1.0)

    return _decision_directed(power, noise, rule)


def _mmse_lsa(power, noise):
    def rule(xi, gamma):
        v = np.clip(xi / (1 + xi) * gamma, 1e-6, 500)
        return np.minimum(xi / (1 + xi) * np.exp(0.5 * exp1(v)), 1.0)

    return _decision_directed(power, noise, rule)


def _subtraction(power, noise, over: float = 2.0):
    """Berouti power subtraction with SNR-dependent over-subtraction."""
    snr_db = 10 * np.log10(np.maximum(power / noise, 1e-6))
    alpha = np.clip(over + 3 - snr_db * 3 / 20, 1, over + 3)
    clean = np.maximum(power - alpha * noise, 0)
    return np.sqrt(clean / np.maximum(power, 1e-12))


GAIN_RULES = {
    "subtraction": _subtraction,
    "wiener": _wiener,
    "mmse-stsa": _mmse_stsa,
    "mmse-lsa": _mmse_lsa,
}


def spectral(audio: np.ndarray, rate: int, rule: str = "mmse-lsa", floor_db: float = -20) -> np.ndarray:
    return _enhance(audio, rate, rule, floor_db)


def bandpass(audio: np.ndarray, rate: int, low: float = 300, high: float = 3_000) -> np.ndarray:
    sos = signal.butter(6, [low, high], "bandpass", fs=rate, output="sos")
    return signal.sosfiltfilt(sos, audio).astype(np.float32)


def gate(audio: np.ndarray, rate: int, depth_db: float = -20, open_db: float = 6.0) -> np.ndarray:
    """Downward expander: frames below the noise floor + `open_db` drop by up to `depth_db`."""
    frame = rate // 100
    n = len(audio) // frame + 1
    padded = np.pad(audio, (0, n * frame - len(audio)))
    level = 10 * np.log10(np.mean(padded.reshape(n, frame) ** 2, axis=1) + 1e-12)
    floor = np.percentile(level, 10)
    opened = (level > floor + open_db).astype(float)
    opened = ndimage.maximum_filter1d(opened, 15)  # 150 ms hold
    gain_db = depth_db * (1 - ndimage.uniform_filter1d(opened, 3))
    gain = np.repeat(10 ** (gain_db / 20), frame)[: len(audio)]
    return (audio * gain).astype(np.float32)


def noisereduce(audio: np.ndarray, rate: int, stationary: bool) -> np.ndarray:
    import noisereduce as nr

    x = resample(audio, rate, RATE)
    y = nr.reduce_noise(y=x, sr=RATE, stationary=stationary, prop_decrease=0.9, n_jobs=1)
    return resample(y.astype(np.float32), RATE, rate)[: len(audio)]


def lms_notch(audio: np.ndarray, rate: int, taps: int = 32, delay_ms: float = 2.0, mu: float = 0.003,
              work_rate: int = 8_000) -> np.ndarray:
    """Adaptive line enhancer used as an automatic notch: predicts the input from itself
    `delay_ms` ago and outputs the prediction error. A delay of a few ms decorrelates speech
    (pitch varies, formants move) but not a steady heterodyne, so only tones are cancelled.
    Runs at 8 kHz, enough for 300-3000 Hz radio audio."""
    x = resample(audio, rate, work_rate).astype(np.float64)
    delay = int(round(delay_ms * work_rate / 1000))
    w = np.zeros(taps)
    out = np.empty_like(x)
    history = np.concatenate([np.zeros(delay + taps), x])
    for n in range(len(x)):
        ref = history[n : n + taps][::-1]  # x[n-delay-taps+1 .. n-delay], newest first
        e = x[n] - w @ ref
        w += mu * e * ref / (ref @ ref + 1e-6)
        out[n] = e
    return resample(out.astype(np.float32), work_rate, rate)[: len(audio)]
