"""Radio link simulator: transmitter audio chain, modulation, propagation, noise and receiver.

Everything runs as complex baseband at 48 kHz. Each `Link` produces two signals from the
same random realisation:

* `noisy`     what the receiver's speaker would play, noise and interference included
* `reference` what the same receiver would play with the additive noise and interference
              switched off (fading, mistuning, band-limiting and AGC still apply)

The reference is therefore the best a pure denoiser could possibly hand back, which keeps
the intrusive metrics about noise removal instead of about bandwidth the radio never carried.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal

FS = 48_000


# --------------------------------------------------------------------------- helpers


def _sos_band(low: float | None, high: float | None, order: int = 6) -> np.ndarray:
    if low and high:
        return signal.butter(order, [low, high], "bandpass", fs=FS, output="sos")
    if high:
        return signal.butter(order, high, "lowpass", fs=FS, output="sos")
    return signal.butter(order, low, "highpass", fs=FS, output="sos")


def band(x: np.ndarray, low: float | None, high: float | None, order: int = 6) -> np.ndarray:
    return signal.sosfiltfilt(_sos_band(low, high, order), x).astype(x.dtype)


def complex_lowpass(x: np.ndarray, cutoff: float, taps: int = 511) -> np.ndarray:
    """Linear-phase channel filter with zero group delay (centered 'same' convolution)."""
    h = signal.firwin(taps, cutoff, fs=FS, window=("kaiser", 8.0))
    return signal.oaconvolve(x, h, mode="same")


def complex_bandpass(x: np.ndarray, low: float, high: float, taps: int = 1023) -> np.ndarray:
    centre = 0.5 * (low + high)
    rot = np.exp(2j * np.pi * centre * np.arange(len(x)) / FS)
    return complex_lowpass(x * np.conj(rot), 0.5 * (high - low), taps) * rot


def emphasis(tau: float, inverse: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """First-order 6 dB/octave (de-)emphasis with time constant `tau`, shelved at 8 kHz."""
    shelf = 1.0 / (2 * np.pi * 8_000)
    b, a = signal.bilinear([tau, 1.0], [shelf, 1.0], fs=FS)
    return (a, b) if inverse else (b, a)


def active_rms(x: np.ndarray, frame: int = 960) -> float:
    """RMS over the loudest frames (a simple active-speech level, ITU P.56 in spirit)."""
    n = len(x) // frame
    if n == 0:
        return float(np.sqrt(np.mean(x**2)) + 1e-12)
    power = np.mean(x[: n * frame].reshape(n, frame) ** 2, axis=1)
    top = power.max()
    active = power[power > top * 10 ** (-35 / 10)]
    return float(np.sqrt(active.mean()) + 1e-12)


def compressor(x: np.ndarray, threshold_db: float, ratio: float, tau: float = 0.01) -> np.ndarray:
    """Feed-forward RMS compressor (speech processor), zero-phase smoothed envelope."""
    alpha = np.exp(-1.0 / (tau * FS))
    env = signal.filtfilt([1 - alpha], [1, -alpha], x.astype(np.float64) ** 2)
    level_db = 10 * np.log10(np.maximum(env, 1e-12))
    over = np.maximum(level_db - threshold_db, 0.0)
    gain = 10 ** (-over * (1 - 1 / ratio) / 20)
    return (x * gain).astype(x.dtype)


def agc_gain(x: np.ndarray, target: float = 0.3, attack_ms: float = 2.0,
             decay_db_s: float = 20.0, hang_ms: float = 400.0, max_gain_db: float = 70.0) -> np.ndarray:
    """Receiver audio AGC gain: peak detector with fast attack, hang, then a dB-linear decay."""
    block = int(FS * attack_ms / 1000)
    n = len(x) // block + 1
    padded = np.pad(np.abs(x), (0, n * block - len(x)))
    peaks = padded.reshape(n, block).max(axis=1)
    level = np.empty(n)
    held, hang = 1e-6, 0.0
    decay = 10 ** (-decay_db_s * attack_ms / 1000 / 20)
    hang_blocks = hang_ms / attack_ms
    for i, p in enumerate(peaks):
        if p >= held:
            held, hang = p, hang_blocks
        elif hang > 0:
            hang -= 1
        else:
            held = max(held * decay, p)
        level[i] = held
    gain = np.minimum(target / np.maximum(level, 1e-9), 10 ** (max_gain_db / 20))
    t_block = (np.arange(n) + 0.5) * block
    return np.interp(np.arange(len(x)), t_block, gain)


def complex_noise(rng: np.random.Generator, n: int, variance: float) -> np.ndarray:
    return (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * np.sqrt(variance / 2)


def rayleigh_taps(rng: np.random.Generator, n: int, spread_hz: float, rate: float = 200.0) -> np.ndarray:
    """Complex Gaussian tap with a Gaussian Doppler spectrum (ITU-R F.1487 / Watterson)."""
    slow = int(np.ceil(n / FS * rate)) + 8
    white = complex_noise(rng, slow + 400, 1.0)
    sigma = spread_hz / 2  # F.1487 defines the spread as twice the standard deviation
    t = np.arange(-200, 201) / rate
    kernel = np.exp(-2 * (np.pi * sigma * t) ** 2)
    tap = signal.fftconvolve(white, kernel / np.sqrt(np.sum(kernel**2)), mode="valid")[:slow]
    t_fast = np.arange(n) / FS * rate
    return np.interp(t_fast, np.arange(slow), tap.real) + 1j * np.interp(t_fast, np.arange(slow), tap.imag)


def watterson(rng: np.random.Generator, x: np.ndarray, delay_ms: float, spread_hz: float) -> np.ndarray:
    """Two equal-power independently fading paths, `delay_ms` apart."""
    delay = int(round(delay_ms * FS / 1000))
    a = rayleigh_taps(rng, len(x), spread_hz)
    b = rayleigh_taps(rng, len(x), spread_hz)
    late = np.concatenate([np.zeros(delay, x.dtype), x[: len(x) - delay]])
    return np.sqrt(0.5) * (a * x + b * late)


def qrn(rng: np.random.Generator, n: int, rate_hz: float, level: float) -> np.ndarray:
    """Atmospheric crashes: Poisson-timed bursts of decaying broadband noise."""
    out = np.zeros(n, complex)
    count = rng.poisson(rate_hz * n / FS)
    for start in rng.integers(0, n, count):
        length = int(FS * rng.uniform(0.005, 0.06))
        stop = min(n, start + 6 * length)
        envelope = np.exp(-np.arange(stop - start) / length)
        amp = level * rng.lognormal(0.0, 0.8)
        out[start:stop] += amp * envelope * complex_noise(rng, stop - start, 1.0)
    return out


# --------------------------------------------------------------------------- links


@dataclass(frozen=True)
class Link:
    """One radio condition. `snr_db` is carrier-to-noise for FM/AM and signal-to-noise for SSB."""

    name: str
    mode: str  # nfm | am | usb
    snr_db: float
    demod: str = "default"  # am: envelope | sync
    fading: tuple[float, float] | None = None  # (delay_ms, doppler spread Hz)
    qrn_rate: float = 0.0
    heterodyne: tuple[float, float] | None = None  # (offset Hz, level re carrier)
    mistune_hz: float = 0.0
    description: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)

    def run(self, speech: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
        speech = speech.astype(np.float64)
        speech = speech / active_rms(speech) * 0.1  # -20 dBFS active level into the mic
        return {"nfm": _nfm, "am": _am, "usb": _usb}[self.mode](self, speech, seed)


def _finish(noisy: np.ndarray, reference: np.ndarray, low: float, high: float) -> tuple[np.ndarray, np.ndarray]:
    noisy = band(noisy, low, high, 4)
    reference = band(reference, low, high, 4)
    # Noise biases AGC and envelope levels, so match the reference to the speech that is
    # actually inside the noisy stream (least-squares scale).
    reference = reference * np.dot(noisy, reference) / (np.dot(reference, reference) + 1e-12)
    # Speaker level: same gain on both, set so the noisy stream sits at -26 dBFS RMS.
    gain = 0.05 / (np.sqrt(np.mean(noisy**2)) + 1e-12)
    noisy, reference = noisy * gain, reference * gain
    top = max(np.max(np.abs(noisy)), 1e-9)
    if top > 0.99:
        noisy, reference = noisy * 0.99 / top, reference * 0.99 / top
    return noisy.astype(np.float32), reference.astype(np.float32)


# NFM, 12.5 kHz channel: 750 us pre-emphasis, deviation limiter, 2.5 kHz peak deviation.
NFM_DEVIATION = 2_500.0
NFM_IF_BW = 12_500.0
NFM_TAU = 750e-6


def _nfm_tx(speech: np.ndarray) -> np.ndarray:
    x = band(speech, 300, 3_000)
    x = signal.lfilter(*emphasis(NFM_TAU), x)
    x = compressor(x, -24, 2.0)
    x = x / (np.percentile(np.abs(x), 99.5) + 1e-12)
    x = np.clip(x, -1.0, 1.0)  # deviation limiter
    return band(x, None, 3_000)  # splatter filter


def _nfm_rx(iq: np.ndarray) -> np.ndarray:
    iq = complex_lowpass(iq, NFM_IF_BW / 2)
    audio = np.angle(iq[1:] * np.conj(iq[:-1])) * FS / (2 * np.pi * NFM_DEVIATION)
    audio = np.concatenate([[0.0], audio])
    audio = band(audio, None, 3_400)
    return signal.lfilter(*emphasis(NFM_TAU, inverse=True), audio)


def _nfm(link: Link, speech: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    m = _nfm_tx(speech)
    phase = 2 * np.pi * NFM_DEVIATION * np.cumsum(m) / FS
    carrier = np.exp(1j * phase)
    variance = 10 ** (-link.snr_db / 10) * FS / NFM_IF_BW
    noisy = _nfm_rx(carrier + complex_noise(rng, len(carrier), variance))
    clean = _nfm_rx(carrier)
    return _finish(noisy, clean, 300, 3_000)


# AM (airband / shortwave): 6 kHz receiver bandwidth, 85 % peak modulation.
AM_IF_BW = 6_000.0


def _am_rx(iq: np.ndarray, demod: str) -> np.ndarray:
    iq = complex_lowpass(iq, AM_IF_BW / 2)
    if demod == "sync":
        # Idealised synchronous detector: phase of a narrow (+/-15 Hz) carrier estimate.
        carrier = complex_lowpass(iq, 15.0, taps=8_191)
        audio = np.real(iq * np.exp(-1j * np.angle(carrier)))
    else:
        audio = np.abs(iq)
        carrier = audio
    # Carrier-driven AGC: divide by the slowly tracked carrier level.
    level = signal.filtfilt(*signal.butter(2, 2.0, fs=FS), np.abs(carrier))
    return (audio - level) / np.maximum(level, 1e-3)


def _am(link: Link, speech: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = band(speech, 300, 2_800)
    x = compressor(x, -24, 3.0)
    x = np.clip(x / (np.percentile(np.abs(x), 99.5) + 1e-12), -1, 1)
    x = band(x, None, 2_800)
    envelope = 1.0 + 0.85 * x
    t = np.arange(len(envelope)) / FS
    clean_iq = envelope.astype(complex)
    if link.fading:
        clean_iq = watterson(rng, clean_iq, *link.fading)
    interference = np.zeros_like(clean_iq)
    if link.heterodyne:
        offset, level = link.heterodyne
        drift = offset + 15 * np.sin(2 * np.pi * 0.05 * t)
        interference += level * np.exp(2j * np.pi * np.cumsum(drift) / FS)
    variance = 10 ** (-link.snr_db / 10) * FS / AM_IF_BW
    noise = complex_noise(rng, len(clean_iq), variance)
    noisy = _am_rx(clean_iq + noise + interference, link.demod)
    clean = _am_rx(clean_iq, link.demod)
    return _finish(noisy, clean, 250, 3_000)


# SSB (USB), 300-2700 Hz receiver filter, speech processor on the transmitter.
SSB_LOW, SSB_HIGH = 300.0, 2_700.0


def _usb_detect(iq: np.ndarray) -> np.ndarray:
    return np.real(complex_bandpass(iq, SSB_LOW, SSB_HIGH))


def _usb(link: Link, speech: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = band(speech, SSB_LOW, SSB_HIGH)
    x = compressor(x, -30, 4.0)
    x = np.clip(x / (np.percentile(np.abs(x), 99.8) + 1e-12), -1, 1)
    x = band(x, SSB_LOW, SSB_HIGH)
    tx = signal.hilbert(x)  # upper sideband: x + jH{x}
    if link.fading:
        tx = watterson(rng, tx, *link.fading)
    if link.mistune_hz:
        tx = tx * np.exp(2j * np.pi * link.mistune_hz * np.arange(len(tx)) / FS)
    power = active_rms(np.abs(tx)) ** 2
    variance = power * 10 ** (-link.snr_db / 10) * FS / (SSB_HIGH - SSB_LOW)
    noise = complex_noise(rng, len(tx), variance)
    if link.qrn_rate:
        noise = noise + qrn(rng, len(tx), link.qrn_rate, 12 * np.sqrt(variance))
    noisy = _usb_detect(tx + noise)
    # The reference hears the same AGC gain trajectory, so only the noise differs.
    gain = agc_gain(noisy)
    return _finish(noisy * gain, _usb_detect(tx) * gain, SSB_LOW, SSB_HIGH)


# --------------------------------------------------------------------------- catalogue

MODERATE = (1.0, 0.5)  # ITU-R F.1487 mid-latitude moderate: 1 ms delay, 0.5 Hz spread
DISTURBED = (2.0, 1.0)  # mid-latitude disturbed: 2 ms, 1 Hz

LINKS: list[Link] = [
    Link("nfm-cnr4", "nfm", 4, description="NFM below threshold: clicks and popcorn", tags=("nfm",)),
    Link("nfm-cnr7", "nfm", 7, description="NFM at threshold: hiss with occasional clicks", tags=("nfm",)),
    Link("nfm-cnr10", "nfm", 10, description="NFM just above threshold: steady hiss", tags=("nfm",)),
    Link("nfm-cnr16", "nfm", 16, description="NFM fair signal: light hiss", tags=("nfm",)),
    Link("am-cnr3", "am", 3, demod="envelope", description="AM very weak, envelope detector", tags=("am",)),
    Link("am-cnr8", "am", 8, demod="envelope", description="AM weak, envelope detector", tags=("am",)),
    Link("am-cnr14", "am", 14, demod="envelope", description="AM fair, envelope detector", tags=("am",)),
    Link("am-cnr20", "am", 20, demod="envelope", description="AM good, envelope detector", tags=("am",)),
    Link("am-sync-cnr3", "am", 3, demod="sync", description="AM very weak, synchronous detector", tags=("am", "demod")),
    Link("am-sync-cnr8", "am", 8, demod="sync", description="AM weak, synchronous detector", tags=("am", "demod")),
    Link("am-het-cnr14", "am", 14, heterodyne=(1_100.0, 0.25), description="AM with a drifting heterodyne whistle", tags=("am", "tone")),
    Link("am-fade-cnr14", "am", 14, fading=MODERATE, description="Shortwave AM, selective fading, envelope detector", tags=("am", "fading")),
    Link("am-fade-sync-cnr14", "am", 14, demod="sync", fading=MODERATE, description="Shortwave AM, selective fading, synchronous detector", tags=("am", "fading", "demod")),
    Link("usb-snr3", "usb", 3, description="SSB in white noise, very weak", tags=("ssb",)),
    Link("usb-snr10", "usb", 10, description="SSB in white noise, workable", tags=("ssb",)),
    Link("usb-fade-snr6", "usb", 6, fading=MODERATE, description="HF SSB, moderate Watterson fading, weak", tags=("ssb", "fading")),
    Link("usb-fade-snr15", "usb", 15, fading=DISTURBED, description="HF SSB, disturbed Watterson fading", tags=("ssb", "fading")),
    Link("usb-qrn-snr10", "usb", 10, qrn_rate=4.0, description="HF SSB with atmospheric crashes (QRN)", tags=("ssb", "impulse")),
    Link("usb-mistune-snr10", "usb", 10, mistune_hz=60.0, description="SSB tuned 60 Hz off (Donald Duck)", tags=("ssb",)),
]

LINKS_BY_NAME = {link.name: link for link in LINKS}
