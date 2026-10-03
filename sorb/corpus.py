"""Builds the two test sets.

* synthetic  LibriSpeech test-clean sentences sent through every simulated `Link`.
             Has a clean-channel reference, a transcript and oracle speech labels.
* real       Off-air recordings (audio and IQ) listed in sources/real.yaml.
             No reference; speech labels come from Silero VAD on the received audio.

Each clip is written as 48 kHz mono float WAV under data/corpus/, with `<clip>.vad.npy`
holding one bool per 10 ms frame (True = speech).
"""

from __future__ import annotations

import hashlib
import subprocess
import tarfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yaml
from scipy import ndimage
from tqdm import tqdm

from . import audio as A
from .channel import LINKS
from .demod import demodulate_iq

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE = DATA / "cache"
CORPUS = DATA / "corpus"
MANIFEST = CORPUS / "manifest.csv"
SOURCES = ROOT / "sources" / "real.yaml"
LIBRISPEECH_URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"
VAD_HOP = 480  # 10 ms at 48 kHz
LEAD_S, TAIL_S = 1.0, 1.0  # carrier keyed before and after the sentence
UTTERANCES_PER_SEX = 6


def stable_seed(*parts: str) -> int:
    return int.from_bytes(hashlib.sha256("/".join(parts).encode()).digest()[:4], "little")


# --------------------------------------------------------------------------- synthetic


def librispeech() -> Path:
    root = CACHE / "LibriSpeech"
    if not (root / "test-clean").is_dir():
        CACHE.mkdir(parents=True, exist_ok=True)
        archive = CACHE / "test-clean.tar.gz"
        if not archive.exists():
            download(LIBRISPEECH_URL, archive)
        with tarfile.open(archive) as tar:
            tar.extractall(CACHE, filter="data")
    return root


def pick_utterances(root: Path) -> list[tuple[str, str, Path, str]]:
    """One sentence of 8-15 s per speaker, first N female and N male speakers by id."""
    sex = {}
    for line in (root / "SPEAKERS.TXT").read_text().splitlines():
        if line.startswith(";"):
            continue
        cols = [c.strip() for c in line.split("|")]
        if len(cols) >= 3 and cols[2] == "test-clean":
            sex[cols[0]] = cols[1]
    chosen: dict[str, list] = {"F": [], "M": []}
    for speaker in sorted(sex, key=int):
        bucket = chosen[sex[speaker]]
        if len(bucket) >= UTTERANCES_PER_SEX:
            continue
        for trans in sorted((root / "test-clean" / speaker).glob("*/*.trans.txt")):
            for line in trans.read_text().splitlines():
                utt, text = line.split(" ", 1)
                flac = trans.parent / f"{utt}.flac"
                info = A.sf.info(str(flac))
                if 8.0 <= info.duration <= 15.0:
                    bucket.append((utt, sex[speaker], flac, text))
                    break
            if bucket and bucket[-1][0].startswith(speaker + "-"):
                break
    return chosen["F"] + chosen["M"]


def speech_labels_from_reference(reference: np.ndarray) -> np.ndarray:
    n = len(reference) // VAD_HOP
    power = np.mean(reference[: n * VAD_HOP].reshape(n, VAD_HOP) ** 2, axis=1)
    level = 10 * np.log10(power + 1e-12)
    active = level > np.percentile(level, 95) - 30
    active = ndimage.binary_closing(active, np.ones(15))  # bridge gaps < 150 ms
    return ndimage.binary_dilation(active, np.ones(5))  # 50 ms hangover each side


def build_synthetic(rows: list[dict]) -> None:
    root = librispeech()
    for utt, sex, flac, text in tqdm(pick_utterances(root), desc="synthetic"):
        speech = A.load(flac)
        speech = np.pad(speech, (int(LEAD_S * A.RATE), int(TAIL_S * A.RATE)))
        for link in LINKS:
            clip = f"{link.name}/{utt}"
            path = CORPUS / "synthetic" / f"{clip}.flac"
            ref = path.with_suffix(".ref.flac")
            vad = path.with_suffix(".vad.npy")
            if not (path.exists() and ref.exists() and vad.exists()):
                noisy, clean = link.run(speech, stable_seed(link.name, utt))
                A.save(path, noisy)
                A.save(ref, clean)
                np.save(vad, speech_labels_from_reference(clean))
            rows.append(
                dict(
                    clip=f"syn/{clip}", set="synthetic", path=str(path.relative_to(ROOT)),
                    reference=str(ref.relative_to(ROOT)), transcript=text, mode=link.mode,
                    condition=link.name, group=link.name, demod=link.demod, speaker_sex=sex,
                    source="LibriSpeech test-clean " + utt, license="CC BY 4.0",
                    duration_s=round(len(speech) / A.RATE, 2), description=link.description,
                )
            )


# --------------------------------------------------------------------------- real


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    headers = {"User-Agent": "sorb/0.1 (https://github.com/Newspicel/speech-over-radio-benchmark)"}
    for attempt in range(6):
        with requests.get(url, stream=True, timeout=60, headers=headers) as r:
            if r.status_code == 429 and attempt < 5:  # Wikimedia and archive.org rate-limit bursts
                time.sleep(float(r.headers.get("Retry-After", 0) or 0) or 20 * (attempt + 1))
                continue
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        break
    tmp.replace(dest)


def download_wav_slice(url: str, dest: Path, start_s: float, duration_s: float) -> int:
    """Fetches only `duration_s` of a big 16-bit IQ WAV via HTTP range requests (raw cs16)."""
    import struct

    head = requests.get(url, headers={"Range": "bytes=0-65535"}, timeout=60).content
    pos, rate = 12, None
    while pos < len(head) - 8:
        chunk, size = head[pos : pos + 4], struct.unpack("<I", head[pos + 4 : pos + 8])[0]
        if chunk == b"fmt ":
            rate = struct.unpack("<I", head[pos + 12 : pos + 16])[0]
        if chunk == b"data":
            first = pos + 8 + int(start_s * rate) * 4
            last = first + int(duration_s * rate) * 4 - 1
            dest.parent.mkdir(parents=True, exist_ok=True)
            with requests.get(url, headers={"Range": f"bytes={first}-{last}"}, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(dest.with_name(dest.name + ".part"), "wb") as f:
                    for block in r.iter_content(1 << 20):
                        f.write(block)
            dest.with_name(dest.name + ".part").replace(dest)
            dest.with_suffix(".rate").write_text(str(rate))
            return rate
        pos += 8 + size + (size & 1)
    raise ValueError("no data chunk in WAV header")


def decode_audio(src: Path, start: float, duration: float | None) -> np.ndarray:
    cmd = ["ffmpeg", "-v", "error", "-ss", str(start), "-i", str(src)]
    if duration:
        cmd += ["-t", str(duration)]
    cmd += ["-ac", "1", "-ar", str(A.RATE), "-f", "f32le", "-"]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.float32).copy()


_silero = None


def speech_labels_from_vad(x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    global _silero
    from silero_vad import get_speech_timestamps, load_silero_vad
    import torch

    if _silero is None:
        _silero = load_silero_vad(onnx=True)
    wav = torch.from_numpy(A.resample(x, A.RATE, 16_000))
    stamps = get_speech_timestamps(wav, _silero, threshold=threshold, sampling_rate=16_000,
                                   min_silence_duration_ms=150, speech_pad_ms=60)
    labels = np.zeros(len(x) // VAD_HOP, bool)
    for s in stamps:
        labels[s["start"] * 3 // VAD_HOP : s["end"] * 3 // VAD_HOP + 1] = True
    return labels


WINDOW_S = 45
SCAN_S = 20 * 60


def pick_window(src: Path) -> float:
    """Start (s) of the WINDOW_S excerpt with the most speech, capped at 80 % so that
    noise-only frames remain for the suppression metrics."""
    x = decode_audio(src, 0, SCAN_S)
    labels = speech_labels_from_vad(x).astype(float)
    per_s = labels[: len(labels) // 100 * 100].reshape(-1, 100).mean(axis=1)
    if len(per_s) <= WINDOW_S:
        return 0.0
    fraction = np.convolve(per_s, np.ones(WINDOW_S) / WINDOW_S, mode="valid")
    return float(np.argmax(np.minimum(fraction, 0.8) - 1e-3 * np.arange(len(fraction)) / len(fraction)))


def build_real(rows: list[dict]) -> None:
    if not SOURCES.exists():
        return
    entries = yaml.safe_load(SOURCES.read_text()) or []
    for entry in tqdm(entries, desc="real"):
        url = entry["url"]
        if "slice" in entry:  # [start_s, duration_s] of a large IQ WAV
            src = CACHE / "real" / f"{entry['source_file']}.cs16"
        else:
            src = CACHE / "real" / f"{entry['id']}{Path(url.split('?')[0]).suffix or '.bin'}"
        try:
            if not src.exists():
                if "slice" in entry:
                    download_wav_slice(url, src, *entry["slice"])
                else:
                    download(url, src)
            if "slice" in entry:
                entry = dict(entry, format="cs16", rate=float(src.with_suffix(".rate").read_text()))
        except Exception as error:  # a vanished upload should not stop the benchmark
            print(f"skip {entry['id']}: {error}")
            continue
        variants: list[tuple[str, str, np.ndarray | None]] = []
        if entry.get("kind", "audio") == "iq":
            for demod in entry.get("demods", [entry["modulation"]]):
                variants.append((f"{entry['id']}~{demod}", demod, None))
        else:
            variants.append((entry["id"], entry["modulation"], None))
        for clip_id, demod, _ in variants:
            path = CORPUS / "real" / f"{clip_id}.flac"
            vad = path.with_suffix(".vad.npy")
            if not path.exists():
                try:
                    if entry.get("kind", "audio") == "iq":
                        x = demodulate_iq(src, entry, demod)
                    else:
                        start = entry["start"] if "start" in entry else pick_window(src)
                        x = decode_audio(src, start, entry.get("duration", WINDOW_S))
                except Exception as error:
                    print(f"skip {clip_id}: {error}")
                    continue
                if len(x) < A.RATE * 2 or A.rms(x) < 1e-6:
                    print(f"skip {clip_id}: empty")
                    continue
                x = (x / (A.rms(x) + 1e-12) * 0.05).astype(np.float32)
                x = x * min(1.0, 0.99 / float(np.max(np.abs(x))))
                A.save(path, x)
            if not vad.exists():
                np.save(vad, speech_labels_from_vad(A.load(path)))
            info = A.sf.info(str(path))
            rows.append(
                dict(
                    clip=f"real/{clip_id}", set="real", path=str(path.relative_to(ROOT)),
                    reference="", transcript="", mode=entry["modulation"], condition=entry.get("service", ""),
                    group=entry.get("group", entry["modulation"]), demod=demod, speaker_sex="",
                    source=entry.get("page", url), license=entry.get("license", ""),
                    duration_s=round(info.duration, 2), description=entry.get("description", ""),
                )
            )


ATC_SETS = {
    # id: (url, page, license, utterances)
    "atco2": (
        "https://huggingface.co/datasets/Jzuluaga/atco2_corpus_1h/resolve/main/data/test-00000-of-00001-112f39d2f116a22b.parquet",
        "https://huggingface.co/datasets/Jzuluaga/atco2_corpus_1h", "ATCO2 1h test set, ATCO2 end-user data agreement", 60,
    ),
    "uwb-atcc": (
        "https://huggingface.co/datasets/Jzuluaga/uwb_atcc/resolve/main/data/test-00000-of-00001-3a021115ca23c2a5.parquet",
        "https://huggingface.co/datasets/Jzuluaga/uwb_atcc", "UWB-ATCC test split, CC BY-NC-SA 4.0", 60,
    ),
}


def build_atc(rows: list[dict]) -> None:
    """Real VHF AM air-traffic-control utterances with human transcripts (real-world WER)."""
    import io

    import soundfile as sf

    for name, (url, page, license_, count) in ATC_SETS.items():
        src = CACHE / "atc" / f"{name}.parquet"
        if not src.exists():
            download(url, src)
        table = pd.read_parquet(src)
        table["duration"] = table["duration"].astype(float)
        table = table[(table.duration >= 3) & (table.duration <= 12)].sort_values("id")
        table = table.sample(n=min(count, len(table)), random_state=7).sort_values("id")
        for row in table.itertuples():
            clip_id = f"{name}/{row.id}"
            path = CORPUS / "atc" / f"{clip_id}.flac"
            vad = path.with_suffix(".vad.npy")
            if not path.exists():
                x, rate = sf.read(io.BytesIO(row.audio["bytes"]), dtype="float32", always_2d=True)
                x = A.resample(x.mean(axis=1), rate, A.RATE)
                x = (x / (A.rms(x) + 1e-12) * 0.05).astype(np.float32)
                x = x * min(1.0, 0.99 / float(np.max(np.abs(x))))
                A.save(path, x)
            if not vad.exists():
                np.save(vad, speech_labels_from_vad(A.load(path)))
            rows.append(
                dict(
                    clip=f"real/{clip_id}", set="real", path=str(path.relative_to(ROOT)), reference="",
                    transcript=row.text, mode="am", condition="VHF airband ATC", group="atc", demod="am",
                    speaker_sex="", source=page, license=license_, duration_s=round(row.duration, 2),
                    description=f"{name} utterance with transcript",
                )
            )


HF_SHIFT_URL = "https://zenodo.org/api/records/5927810/files/shift_data.tar.gz/content"
HF_SHIFT_PAGE = "https://zenodo.org/records/5927810"
HF_PER_GROUP = 48


def build_hf_shift(rows: list[dict]) -> None:
    """Real HF SSB: LibriSpeech sent from Paderborn and received by KiwiSDRs across Europe
    (Heitkaemper et al., Zenodo 5927810, CC BY 4.0). OS0 = receiver tuned exactly (clean
    reference usable, delay re-measured per clip); OS300 = receiver 300 Hz off (WER only)."""
    import json

    root = CACHE / "zenodo-5927810" / "ShiftData_adjusted"
    if not root.is_dir():
        root.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(HF_SHIFT_URL, stream=True, timeout=120) as r, tarfile.open(fileobj=r.raw, mode="r|gz") as tar:
            tar.extractall(root.parent, filter="data")
    notes = json.loads((root / "annotations.json").read_text())
    rng = np.random.default_rng(5927810)
    for shift, group in (("OS0", "hf-real"), ("OS300", "hf-real-mistuned")):
        candidates = []
        for session in sorted((root / "noisy" / "eval_shift").glob(f"*_{shift}")):
            for wav in sorted(session.glob("*.wav")):
                receiver, number = wav.stem.split("__")
                note = notes[number]
                spans = [tuple(map(int, s.split(":"))) for s in note["activity"][0].split(", ")]
                for i, (a, b) in enumerate(spans):
                    if 3 * 8_000 <= b - a <= 15 * 8_000:
                        candidates.append((wav, receiver, number, i, a, b, note["short_transcriptions"][i]))
        picks = rng.choice(len(candidates), size=min(HF_PER_GROUP, len(candidates)), replace=False)
        for idx in sorted(picks):
            wav, receiver, number, i, a, b, text = candidates[idx]
            clip_id = f"{group}/{receiver}-{number}-{i}"
            path = CORPUS / "hf" / f"{clip_id}.flac"
            ref = path.with_suffix(".ref.flac")
            vad = path.with_suffix(".vad.npy")
            if not path.exists():
                noisy, rate = A.sf.read(str(wav), dtype="float32")
                clean, _ = A.sf.read(str(root / "clean" / "eval_shift" / f"clean_{number}.wav"), dtype="float32")
                lo, hi = max(0, a - rate), min(len(noisy), b + rate)  # 1 s of band noise each side
                seg = noisy[lo:hi]
                ref_seg = clean[lo:hi]
                if shift == "OS0":
                    lag = A.best_lag(ref_seg, seg, rate // 2)
                    seg = noisy[lo + lag : hi + lag] if lo + lag >= 0 else seg
                x = A.resample(seg, rate, A.RATE)
                gain = 0.05 / (A.rms(x) + 1e-12)
                x = x * gain
                peak = float(np.max(np.abs(x)))
                A.save(path, x * min(1.0, 0.99 / peak))
                if shift == "OS0":
                    r = A.resample(ref_seg, rate, A.RATE)
                    r = r * np.dot(x, r) / (np.dot(r, r) + 1e-12)  # match the speech level inside x
                    A.save(ref, r)
                labels = np.zeros(len(x) // VAD_HOP, bool)
                s0, s1 = (a - lo) * A.RATE // rate // VAD_HOP, (b - lo) * A.RATE // rate // VAD_HOP
                labels[s0 : s1 + 1] = True
                np.save(vad, labels)
            rows.append(
                dict(
                    clip=f"real/{clip_id}", set="real", path=str(path.relative_to(ROOT)),
                    reference=str(ref.relative_to(ROOT)) if shift == "OS0" else "", transcript=text,
                    mode="ssb", condition=f"HF SSB, KiwiSDR {receiver}, {shift.replace('OS', 'offset ')} Hz",
                    group=group, demod="ssb", speaker_sex="", source=HF_SHIFT_PAGE, license="CC BY 4.0",
                    duration_s=round(A.sf.info(str(path)).duration, 2),
                    description=f"LibriSpeech over a real HF link, received by {receiver}",
                )
            )


def build() -> pd.DataFrame:
    rows: list[dict] = []
    build_synthetic(rows)
    build_real(rows)
    build_atc(rows)
    build_hf_shift(rows)
    frame = pd.DataFrame(rows)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(MANIFEST, index=False)
    return frame


def manifest() -> pd.DataFrame:
    return pd.read_csv(MANIFEST, keep_default_na=False)
