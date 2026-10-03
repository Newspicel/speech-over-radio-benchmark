"""Runs method chains over the corpus with per-stage caching.

A chain is a tuple of stages. The output of every chain prefix is cached under
data/out/<hash of prefix>/, so `A`, `A -> B` and `A -> C` share A's output.

Backends
--------
dsp     in-process numpy function from `sorb.methods.DSP`
sdrmm   the Rust bridge in tools/sdrmm-denoise (real SDR-- code), spec is its stage chain
<env>   an isolated uv project in engines/<env>/ whose worker.py takes a job list
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import audio as A

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "out"
SDRMM_BIN = ROOT / "tools" / "sdrmm-denoise" / "target" / "release" / "sdrmm-denoise"
SDRMM_MODELS = ROOT / "models" / "sdrmm"
ENGINES = ROOT / "engines"
MAX_LAG = int(0.25 * A.RATE)
WORKER_THREADS = {"dpdfnet": 1, "onnx": 1, "rnnoise": 1, "deepfilternet": 2, "clearvoice": 2,
                  "torch": 6, "audiosronnx": 6}


@dataclass(frozen=True)
class Stage:
    backend: str
    spec: str

    def label(self) -> str:
        return f"{self.backend}:{self.spec}"


def prefix_dir(stages: tuple[Stage, ...]) -> Path:
    if not stages:
        return ROOT  # stage 0 reads straight from the corpus paths
    key = " -> ".join(s.label() for s in stages)
    return OUT / hashlib.sha1(key.encode()).hexdigest()[:12]


def output_path(stages: tuple[Stage, ...], clip: str, corpus_path: str) -> Path:
    if not stages:
        return ROOT / corpus_path
    return prefix_dir(stages) / f"{clip}.flac"


# --------------------------------------------------------------------------- backends


def _dsp_one(spec: str, src: Path, dst: Path) -> None:
    from .methods import DSP

    x = A.load(src)
    A.save(dst, A.fit(np.nan_to_num(DSP[spec](x, A.RATE)), len(x)))


def _run_dsp(stage: Stage, jobs: list[tuple[Path, Path]]) -> None:
    # Processes, not threads: the gain rules loop in Python and would hold the GIL.
    with ProcessPoolExecutor(max(1, os.cpu_count() // 2)) as pool:
        list(pool.map(_dsp_one, [stage.spec] * len(jobs), *zip(*jobs)))


def _run_sdrmm(stage: Stage, jobs: list[tuple[Path, Path]]) -> None:
    if not SDRMM_BIN.exists():
        subprocess.run(["cargo", "build", "--release"], cwd=SDRMM_BIN.parents[2], check=True)

    def one(job):
        src, dst = job
        x = A.load(src)
        proc = subprocess.run(
            [str(SDRMM_BIN), "--models", str(SDRMM_MODELS), stage.spec],
            input=x.astype("<f4").tobytes(), capture_output=True, check=False,
        )
        if proc.returncode:
            raise RuntimeError(proc.stderr.decode())
        A.save(dst, A.fit(np.frombuffer(proc.stdout, "<f4"), len(x)))

    with ThreadPoolExecutor(max(1, os.cpu_count() // 2)) as pool:
        list(pool.map(one, jobs))


def _run_worker(stage: Stage, jobs: list[tuple[Path, Path]]) -> None:
    env = ENGINES / stage.backend
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump([[str(s), str(d)] for s, d in jobs], f)
        job_file = f.name
    cmd = ["uv", "run", "--quiet", "--project", str(env), "python", str(env / "worker.py"),
           "--spec", stage.spec, "--jobs", job_file]
    models = ROOT / "models"
    environment = dict(
        os.environ, PYTHONPATH=str(ENGINES / "common"), VIRTUAL_ENV="",
        # Keep every engine's weights inside models/ so a run is self-contained.
        HF_HOME=str(models / "hf"), TORCH_HOME=str(models / "torch"),
        DPDFNET_CACHE_DIR=str(models / "dpdfnet"), XDG_DATA_HOME=str(models / "xdg"),
        PYTORCH_ENABLE_MPS_FALLBACK="1", TOKENIZERS_PARALLELISM="false",
        # Light frame-loop models get one thread each (many run side by side); big ones more.
        SORB_THREADS=str(WORKER_THREADS.get(stage.backend, 2)),
    )
    subprocess.run(cmd, check=True, env=environment, cwd=env)
    os.unlink(job_file)


def _measure_lag(jobs: list[tuple[Path, Path]]) -> int:
    """Constant processing delay of a stage: median cross-correlation lag over up to 24 clips."""
    lags = []
    for src, dst in jobs[:: max(1, len(jobs) // 24)][:24]:
        x, y = A.load(src), A.load(dst)
        if A.rms(y) > 1e-5:
            lags.append(A.best_lag(x, y, MAX_LAG))
    return int(np.median(lags)) if lags else 0


def _shift_all(jobs: list[tuple[Path, Path]], lag: int) -> None:
    if lag:
        for _, dst in jobs:
            A.save(dst, A.shift(A.load(dst), lag))


def run_stage(stages: tuple[Stage, ...], clips: list[tuple[str, str]]) -> dict:
    """Ensures the last stage of `stages` has run on every clip. Returns its metadata.

    Delay compensation (worker backends only) is measured once, over the whole stage, the
    first time the stage completes; clips added later get the same stored shift. Until
    stage.json exists no output has been shifted, so an interrupted run stays consistent."""
    stage = stages[-1]
    out_dir = prefix_dir(stages)
    meta_path = out_dir / "stage.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    all_jobs = [(output_path(stages[:-1], clip, path), output_path(stages, clip, path)) for clip, path in clips]
    jobs = [(src, dst) for src, dst in all_jobs if not dst.exists()]
    if not jobs and meta:
        return meta
    for _, dst in jobs:
        dst.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if jobs:
        if stage.backend == "dsp":
            _run_dsp(stage, jobs)
        elif stage.backend == "sdrmm":
            _run_sdrmm(stage, jobs)
        else:
            _run_worker(stage, jobs)
    elapsed = time.perf_counter() - started
    audio_s = sum(A.sf.info(str(d)).duration for _, d in jobs)
    # Only in-process DSP is zero-delay by construction. SDR-- neural stages report a
    # latency() ~20 ms shorter than their real delay, so they are measured like workers.
    aligned = stage.backend == "dsp"
    if meta:
        lag = meta.get("lag_samples", 0)
        if not aligned:
            _shift_all(jobs, lag)
    else:
        lag = 0 if aligned else _measure_lag(all_jobs)
        if not aligned:
            _shift_all(all_jobs, lag)
    meta = {
        "chain": [s.label() for s in stages],
        "lag_samples": lag,
        "seconds": meta.get("seconds", 0) + elapsed,
        "audio_seconds": meta.get("audio_seconds", 0) + audio_s,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta


def run_chain(stages: tuple[Stage, ...], clips: list[tuple[str, str]]) -> list[dict]:
    return [run_stage(stages[: i + 1], clips) for i in range(len(stages))]


def speed(stage: Stage, clips: list[tuple[str, str]], scratch: Path) -> float:
    """Real-time factor of one stage alone: inference seconds / audio seconds on `clips`."""
    jobs = [(ROOT / path, scratch / f"{i}.flac") for i, (_, path) in enumerate(clips)]
    scratch.mkdir(parents=True, exist_ok=True)
    if stage.backend in ("dsp", "sdrmm"):
        busy = audio_s = 0.0
        for src, dst in jobs:
            x = A.load(src)
            t0 = time.perf_counter()
            if stage.backend == "dsp":
                from .methods import DSP

                DSP[stage.spec](x, A.RATE)
            else:
                subprocess.run([str(SDRMM_BIN), "--models", str(SDRMM_MODELS), stage.spec],
                               input=x.astype("<f4").tobytes(), capture_output=True, check=True)
            busy += time.perf_counter() - t0
            audio_s += len(x) / A.RATE
        return busy / audio_s
    timing = scratch / "timing.json"
    os.environ["SORB_TIMING"] = str(timing)
    try:
        _run_worker(stage, jobs)
    finally:
        del os.environ["SORB_TIMING"]
    t = json.loads(timing.read_text())
    return t["seconds"] / t["audio_seconds"]


def realign(stage_chains: list[tuple[Stage, ...]], clips: list[tuple[str, str]], min_lag: int = 24) -> list[str]:
    """Re-measures every non-DSP stage's delay against its (already corrected) input, in
    chain order, and shifts outputs that are off by >= `min_lag` samples. Returns the
    prefix labels that changed (their scores must be recomputed)."""
    changed = []
    for stages in sorted(stage_chains, key=len):
        if stages[-1].backend == "dsp":
            continue
        meta_path = prefix_dir(stages) / "stage.json"
        if not meta_path.exists():
            continue
        jobs = [(output_path(stages[:-1], c, p), output_path(stages, c, p)) for c, p in clips]
        lag = _measure_lag(jobs)
        if abs(lag) < min_lag:
            continue
        _shift_all(jobs, lag)
        meta = json.loads(meta_path.read_text())
        meta["lag_samples"] = meta.get("lag_samples", 0) + lag
        meta_path.write_text(json.dumps(meta, indent=2))
        changed.append(" -> ".join(s.label() for s in stages))
        print(f"realigned {changed[-1]}: {lag} samples", flush=True)
    return changed
