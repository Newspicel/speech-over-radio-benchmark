"""NINR from AIRSPY's SDR# (closed source), run through its own shark.dll under Wine.

spec = lp:<profile>[:depth]  — current NINR engine with an SDR# profile, e.g. lp:comm-narrow
     | hq                     — classic NINR engine (threshold/slope)

Needs SORB_SDRSHARP=<SDR# folder containing shark.dll> (not redistributed here) and the
portable Wine + 32-bit Python from tools/setup_ninr.sh. Like SDR#, audio is resampled to
2.4 x the profile bandwidth, fed as complex (L = R) pairs, and resampled back. The engine's
fixed delay is measured once on a calibration signal and removed."""

import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soxr

import sorbio

ROOT = Path(__file__).resolve().parents[2]
WINE = ROOT / "tools/wine/Wine Staging.app/Contents/Resources/wine/bin/wine"
PYTHON = ROOT / "tools/winpython/python.exe"
RATE = 48_000

# SDR# 'next' (June 2026) DNR profiles: BaseDNR..cctor. fft = 1024 << index.
PROFILES = {
    "hifi-soft": dict(boost_snr=0, use_vad=0, sensitivity=12.0, margin=5.0, frequency_smoothing=40.0,
                      time_smoothing=30.0, fft=4, bandwidth=15_000.0),
    "hifi-hard": dict(boost_snr=0, use_vad=0, sensitivity=6.0, margin=10.0, frequency_smoothing=25.0,
                      time_smoothing=30.0, fft=3, bandwidth=15_000.0),
    "lofi-soft": dict(boost_snr=0, use_vad=0, sensitivity=8.0, margin=5.0, frequency_smoothing=20.0,
                      time_smoothing=30.0, fft=2, bandwidth=6_000.0),
    "lofi-hard": dict(boost_snr=0, use_vad=0, sensitivity=5.0, margin=6.0, frequency_smoothing=15.0,
                      time_smoothing=30.0, fft=2, bandwidth=6_000.0),
    "bc-dx": dict(boost_snr=0, use_vad=0, sensitivity=3.0, margin=5.0, frequency_smoothing=15.0,
                  time_smoothing=30.0, fft=2, bandwidth=6_000.0),
    "comm-narrow": dict(boost_snr=0, use_vad=1, sensitivity=5.0, margin=8.0, frequency_smoothing=50.0,
                        time_smoothing=35.0, fft=1, bandwidth=3_000.0),
    "comm-wide": dict(boost_snr=0, use_vad=1, sensitivity=7.0, margin=8.0, frequency_smoothing=50.0,
                      time_smoothing=35.0, fft=2, bandwidth=5_000.0),
}
DEFAULT_DEPTH = 10.0  # ProcessorPanel.LoadSettings default


def native_spec(spec: str) -> tuple[dict, int]:
    parts = spec.split(":")
    if parts[0] == "lp":
        p = PROFILES[parts[1]]
        rate = int(round(p["bandwidth"] * 2.4))
        params = dict(sample_rate=float(rate), fft_size=1024 << p["fft"],
                      depth=float(parts[2]) if len(parts) > 2 else DEFAULT_DEPTH,
                      sensitivity=p["sensitivity"], margin=p["margin"],
                      frequency_smoothing=p["frequency_smoothing"], time_smoothing=p["time_smoothing"],
                      boost_snr=p["boost_snr"], use_vad=p["use_vad"], skip_dc=0)
        return dict(engine="lp", params=params), rate
    rate = 7_200  # Comm (Narrow) rate
    params = dict(sample_rate=float(rate), fft_size=2048, depth=30.0, noise_threshold=-30.0, slope=50.0,
                  frequency_smoothing=50.0, time_smoothing=20.0, boost_snr=0, skip_dc=0)
    return dict(engine="hq", params=params), rate


def winpath(path: Path) -> str:
    return "Z:" + str(path).replace("/", "\\")


def run_native(spec: dict, signals: list[np.ndarray]) -> list[np.ndarray]:
    shark = Path(os.environ.get("SORB_SDRSHARP", Path.home() / "Downloads/sdrsharp-x86-next")) / "shark.dll"
    if not shark.exists():
        raise SystemExit(f"NINR needs SDR#'s shark.dll: set SORB_SDRSHARP (looked for {shark})")
    with tempfile.TemporaryDirectory(dir=ROOT / "data") as tmp:
        tmp = Path(tmp)
        jobs = []
        for i, x in enumerate(signals):
            pairs = np.repeat(x.astype("<f4"), 2)  # complex (L, R) with L = R
            (tmp / f"{i}.in").write_bytes(pairs.tobytes())
            jobs.append([winpath(tmp / f"{i}.in"), winpath(tmp / f"{i}.out")])
        (tmp / "spec.json").write_text(json.dumps(spec))
        (tmp / "jobs.json").write_text(json.dumps(jobs))
        env = dict(os.environ, WINEPREFIX=str(ROOT / "data/wineprefix"), WINEDEBUG="-all")
        subprocess.run([str(WINE), str(PYTHON), winpath(Path(__file__).with_name("host.py")), winpath(shark),
                        winpath(tmp / "spec.json"), winpath(tmp / "jobs.json")],
                       check=True, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return [np.frombuffer((tmp / f"{i}.out").read_bytes(), "<f4")[0::2].copy() for i in range(len(signals))]


def measure_delay(spec: dict, rate: int) -> int:
    """Delay in samples at `rate`: chirp bursts at high SNR, cross-correlated with the input."""
    rng = np.random.default_rng(0)
    t = np.arange(rate * 12) / rate
    burst = (np.sin(2 * np.pi * (300 + 100 * t) * t) * (np.sin(2 * np.pi * 0.5 * t) > 0)).astype(np.float32)
    x = (0.2 * burst + 0.003 * rng.standard_normal(len(t))).astype(np.float32)
    y = run_native(spec, [x])[0]
    a, b = x[rate * 4 :], y[rate * 4 :]
    corr = np.fft.irfft(np.fft.rfft(b, 2 * len(a)) * np.conj(np.fft.rfft(a, 2 * len(a))))
    lag = int(np.argmax(corr[: rate]))
    return lag


def load(spec_text):
    spec, rate = native_spec(spec_text)
    delay = measure_delay(spec, rate)
    def process(x):
        # Batching happens in run(); keep the per-clip path for completeness.
        low = soxr.resample(x, RATE, rate).astype(np.float32)
        y = run_native(spec, [np.concatenate([low, np.zeros(delay, np.float32)])])[0][delay:]
        return soxr.resample(y, rate, RATE).astype(np.float32)[: len(x)]

    process.batch = lambda xs: [
        soxr.resample(y[delay:], rate, RATE).astype(np.float32)[: len(x)]
        for x, y in zip(xs, run_native(spec, [np.concatenate([soxr.resample(x, RATE, rate).astype(np.float32),
                                                               np.zeros(delay, np.float32)]) for x in xs]))
    ]
    return process, RATE


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--jobs", required=True)
    args = parser.parse_args()
    jobs = json.loads(Path(args.jobs).read_text())
    import time

    process, rate = load(args.spec)
    busy = audio_s = 0.0
    for start in range(0, len(jobs), 64):  # one Wine start per 64 clips
        chunk = jobs[start : start + 64]
        xs = [sorbio.read(src, rate) for src, _ in chunk]
        t0 = time.perf_counter()
        ys = process.batch(xs)
        busy += time.perf_counter() - t0
        audio_s += sum(len(x) for x in xs) / rate
        for (src, dst), y in zip(chunk, ys):
            sorbio.write(dst, y, rate)
    if os.environ.get("SORB_TIMING"):  # includes Wine start-up per batch of 64 clips
        Path(os.environ["SORB_TIMING"]).write_text(json.dumps({"seconds": busy, "audio_seconds": audio_s}))
