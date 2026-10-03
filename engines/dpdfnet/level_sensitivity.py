"""Does input level explain DPDFNet deleting radio speech? Runs official DPDFNet variants on
the same synthetic clips at several input gains and writes speech/noise level changes to
results/level_sensitivity.csv (picked up by `sorb report`)."""

import csv
import glob
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
from dpdfnet.api import _enhance_with_runtime
from dpdfnet.models import resolve_model
from dpdfnet.onnx_backend import build_runtime_model

ROOT = Path(__file__).resolve().parents[2]
CONDITIONS = ("nfm-cnr10", "nfm-cnr16", "am-cnr14", "usb-snr10")
MODELS = ("baseline", "dpdfnet2", "dpdfnet8", "dpdfnet2_8khz", "dpdfnet8_8khz", "dpdfnet2_48khz_hr")
GAINS_DB = (-12, 0, 12, 20)  # corpus clips sit at -26 dBFS RMS


def clips():
    paths = sorted(p for c in CONDITIONS for p in glob.glob(str(ROOT / f"data/corpus/synthetic/{c}/*.flac")) if ".ref" not in p)
    return paths[::3]


def changes(x, y, labels, hop):
    n = min(len(labels), len(x) // hop)
    px = np.mean(x[: n * hop].reshape(n, hop) ** 2, 1)
    py = np.mean(y[: n * hop].reshape(n, hop) ** 2, 1)
    m = labels[:n]
    db = lambda a, b: 10 * np.log10((a.mean() + 1e-10) / (b.mean() + 1e-10))
    return db(py[m], px[m]), db(py[~m], px[~m])


rows = []
for model in MODELS:
    resolved = resolve_model(model=model, onnx_path=None, auto_download=True, verbose=False)
    runtime, rate = build_runtime_model(resolved.onnx_path), resolved.info.sample_rate
    for gain_db in GAINS_DB:
        speech, noise = [], []
        for path in clips():
            x, fs = sf.read(path, dtype="float32")
            x = soxr.resample(x, fs, rate).astype(np.float32)
            g = 10 ** (gain_db / 20)
            y = _enhance_with_runtime(x * g, rate, runtime=runtime, model_sample_rate=rate) / g
            s, n = changes(x, y, np.load(path.replace(".flac", ".vad.npy")), rate // 100)
            speech.append(s)
            noise.append(n)
        rows.append(dict(model=model, input_dbfs=-26 + gain_db, speech_db=round(float(np.median(speech)), 2),
                         noise_db=round(float(np.median(noise)), 2), clips=len(speech)))
        print(rows[-1], flush=True)

out = ROOT / "results" / "level_sensitivity.csv"
with open(out, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
