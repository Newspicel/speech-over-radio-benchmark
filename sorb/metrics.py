"""Per-clip scores.

Intrusive (synthetic set, against the clean-channel reference)
    pesq_nb   ITU-T P.862 narrowband MOS-LQO at 8 kHz (the band radio actually carries)
    pesq_wb   P.862.2 wideband at 16 kHz
    estoi     extended short-time objective intelligibility
    si_sdr    scale-invariant signal-to-distortion ratio, dB
    wer       Whisper large-v3-turbo word error rate against the LibriSpeech transcript

Non-intrusive (both sets)
    dnsmos_sig / dnsmos_bak / dnsmos_ovr   DNSMOS P.835 (Reddy et al. 2022)
    noise_db      level change in non-speech frames, output vs input (negative = suppression)
    speech_db     level change in speech frames (0 = speech kept at its level)
    snr_gain_db   speech_db - noise_db
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from . import audio as A
from .corpus import VAD_HOP

ROOT = Path(__file__).resolve().parent.parent
DNSMOS = ROOT / "models" / "dnsmos" / "sig_bak_ovr.onnx"
WHISPER = "mlx-community/whisper-large-v3-turbo"


def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    reference = reference - reference.mean()
    estimate = estimate - estimate.mean()
    scale = np.dot(estimate, reference) / (np.dot(reference, reference) + 1e-12)
    target = scale * reference
    return A.db(np.sum(target**2) / (np.sum((estimate - target) ** 2) + 1e-12))


def level_changes(noisy: np.ndarray, output: np.ndarray, labels: np.ndarray) -> dict:
    n = min(len(labels), len(noisy) // VAD_HOP, len(output) // VAD_HOP)
    labels = labels[:n]
    pin = np.mean(noisy[: n * VAD_HOP].reshape(n, VAD_HOP) ** 2, axis=1)
    pout = np.mean(output[: n * VAD_HOP].reshape(n, VAD_HOP) ** 2, axis=1)
    out = {}
    for key, mask in (("speech_db", labels), ("noise_db", ~labels)):
        if mask.sum() < 10:
            out[key] = np.nan
        else:
            # The epsilon (-100 dBFS) keeps digitally silent frames from reading as -200 dB.
            out[key] = A.db((pout[mask].mean() + 1e-10) / (pin[mask].mean() + 1e-10))
    out["snr_gain_db"] = out["speech_db"] - out["noise_db"]
    out["output_dbfs"] = A.db(np.mean(output**2) + 1e-20)
    return out


def intrusive(reference: np.ndarray, output: np.ndarray) -> dict:
    from pesq import pesq
    from pystoi import stoi

    ref8, out8 = A.resample(reference, A.RATE, 8_000), A.resample(output, A.RATE, 8_000)
    ref16, out16 = A.resample(reference, A.RATE, 16_000), A.resample(output, A.RATE, 16_000)
    scores = {"si_sdr": si_sdr(ref16, out16)}
    for key, fn in (
        ("pesq_nb", lambda: pesq(8_000, ref8, out8, "nb")),
        ("pesq_wb", lambda: pesq(16_000, ref16, out16, "wb")),
        ("estoi", lambda: stoi(ref16, out16, 16_000, extended=True)),
    ):
        try:
            scores[key] = float(fn())
        except Exception:  # PESQ refuses silent output ("no utterances detected")
            scores[key] = {"pesq_nb": 1.0, "pesq_wb": 1.0}.get(key, 0.0)
    return scores


_dnsmos = None


def dnsmos(x: np.ndarray) -> dict:
    """Official DNSMOS P.835 inference (dnsmos_local.py), non-personalised polynomial."""
    global _dnsmos
    import onnxruntime as ort

    if _dnsmos is None:
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        _dnsmos = ort.InferenceSession(str(DNSMOS), options)
    audio = A.resample(x, A.RATE, 16_000)
    win = 144_160
    while len(audio) < win:
        audio = np.concatenate([audio, audio])
    hops = int(np.floor(len(audio) / 16_000) - 9.01) + 1
    raw = []
    for i in range(max(hops, 1)):
        seg = audio[i * 16_000 : i * 16_000 + win]
        if len(seg) < win:
            break
        raw.append(_dnsmos.run(None, {"input_1": seg[None].astype(np.float32)})[0][0])
    sig, bak, ovr = np.mean(raw, axis=0)
    return {
        "dnsmos_sig": float(np.poly1d([-0.08397278, 1.22083953, 0.0052439])(sig)),
        "dnsmos_bak": float(np.poly1d([-0.13166888, 1.60915514, -0.39604546])(bak)),
        "dnsmos_ovr": float(np.poly1d([-0.06766283, 1.11546468, 0.04602535])(ovr)),
    }


def score_clip(job: tuple[str, str, str, str]) -> dict:
    clip, noisy_path, output_path, reference_path = job
    warnings.filterwarnings("ignore")
    noisy = A.load(ROOT / noisy_path)
    output = A.fit(A.load(output_path), len(noisy))
    labels = np.load((ROOT / noisy_path).with_suffix(".vad.npy"))
    row = {"clip": clip, **level_changes(noisy, output, labels), **dnsmos(output)}
    if reference_path:
        row.update(intrusive(A.load(ROOT / reference_path), output))
    return row


def score(jobs: list[tuple[str, str, str, str]], cache: Path) -> list[dict]:
    """Scores clips, caching rows per output directory in `cache` (a JSON lines file)."""
    done = {}
    if cache.exists():
        for line in cache.read_text().splitlines():
            row = json.loads(line)
            done[row["clip"]] = row
    todo = [job for job in jobs if job[0] not in done]
    if todo:
        workers = int(os.environ.get("SORB_SCORE_JOBS", "4"))  # ~350 MB each
        with ProcessPoolExecutor(workers) as pool, open(cache, "a") as f:
            for row in pool.map(score_clip, todo, chunksize=2):
                done[row["clip"]] = row
                f.write(json.dumps(row) + "\n")
    return [done[job[0]] for job in jobs]


# --------------------------------------------------------------------------- ASR


def transcribe(paths: list[Path]) -> list[str]:
    import mlx_whisper

    texts = []
    for path in paths:
        x = A.load(path, 16_000)
        result = mlx_whisper.transcribe(
            x, path_or_hf_repo=WHISPER, language="en", temperature=0.0,
            condition_on_previous_text=False, verbose=None,
            # Bound runaway hallucination loops on noise-only stretches.
            compression_ratio_threshold=2.4, no_speech_threshold=0.6,
        )
        texts.append(result["text"].strip())
    return texts


_normalizer = None


def normalize(text: str) -> str:
    """Whisper's English normaliser, plus radio number conventions: "3.1", "31" and
    "three one" all become "3 1", so ICAO digit-by-digit read-outs are not counted as errors."""
    import re

    global _normalizer
    if _normalizer is None:
        from whisper_normalizer.english import EnglishTextNormalizer

        _normalizer = EnglishTextNormalizer()
    text = _normalizer(re.sub(r"(?<=\d)[.,](?=\d)", "", text))
    text = re.sub(r"\d+", lambda m: " ".join(m.group()), text)
    return re.sub(r"\s+", " ", text).strip()


def word_errors(reference: str, hypothesis: str) -> tuple[int, int]:
    import jiwer

    ref, hyp = normalize(reference), normalize(hypothesis)
    if not ref.split():
        return 0, 0
    out = jiwer.process_words(ref, hyp if hyp.strip() else "<empty>")
    return out.substitutions + out.deletions + out.insertions, len(ref.split())


def read_cache(cache: Path) -> dict[str, dict]:
    done = {}
    if cache.exists():
        for line in cache.read_text().splitlines():
            row = json.loads(line)
            done[row["clip"]] = row
    return done


def asr_many(groups: list[tuple[Path, list[tuple[str, Path, str]]]], workers: int = 3,
             stall_s: float = 180.0) -> None:
    """Transcribes every uncached clip of every (cache, jobs) group with `workers` parallel
    Whisper processes. A worker silent for `stall_s` is killed; the clip it was on is
    recorded as an empty transcript (Whisper looping on noise counts as all deletions)."""
    import subprocess
    import tempfile
    import time

    truth = {}
    pending = []
    for cache, jobs in groups:
        done = read_cache(cache)
        for clip, path, text in jobs:
            truth[(str(cache), clip)] = text
            if clip not in done:
                pending.append([str(cache), clip, str(path)])
    if not pending:
        return
    tmp = Path(tempfile.mkdtemp(prefix="sorb-asr-"))
    shards = [pending[i::workers] for i in range(workers)]
    procs = {}

    def start(i):
        shard, out = tmp / f"shard{i}.json", tmp / f"out{i}.jsonl"
        shard.write_text(json.dumps(shards[i]))
        procs[i] = (subprocess.Popen([sys.executable, "-m", "sorb.asr_worker", str(shard), str(out)],
                                     cwd=ROOT, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL),
                    out, time.time(), 0)

    def flush(out: Path, start_line: int) -> int:
        lines = out.read_text().splitlines() if out.exists() else []
        for line in lines[start_line:]:
            row = json.loads(line)
            errors, words = word_errors(truth[(row["cache"], row["clip"])], row["text"])
            with open(row["cache"], "a") as f:
                f.write(json.dumps({"clip": row["clip"], "text": row["text"], "errors": errors,
                                    "words": words, **({"timeout": True} if row.get("timeout") else {})}) + "\n")
        return len(lines)

    import sys

    for i in range(workers):
        if shards[i]:
            start(i)
    while procs:
        time.sleep(5)
        for i, (proc, out, last, seen) in list(procs.items()):
            count = flush(out, seen)
            if count != seen:
                procs[i] = (proc, out, time.time(), count)
                continue
            if proc.poll() is not None:
                del procs[i]
            elif time.time() - last > stall_s:
                proc.kill()
                proc.wait()
                stuck = shards[i][count]  # first clip without a result
                with open(out, "a") as f:
                    f.write(json.dumps({"cache": stuck[0], "clip": stuck[1], "text": "", "timeout": True}) + "\n")
                print(f"asr: Whisper stalled on {stuck[1]}, recorded as empty", flush=True)
                count = flush(out, seen)
                start(i)
                procs[i] = (procs[i][0], out, time.time(), count)


def asr(jobs: list[tuple[str, Path, str]], cache: Path) -> dict[str, dict]:
    """Cached transcripts for `jobs`, re-scored with the current normaliser. Clips that
    were never transcribed (run `asr_many` first) are left out."""
    done = read_cache(cache)
    out = {}
    for clip, _, truth in jobs:
        if clip in done:
            row = dict(done[clip])
            row["errors"], row["words"] = word_errors(truth, row["text"])
            out[clip] = row
    return out
