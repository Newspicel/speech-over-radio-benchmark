"""sorb: speech-over-radio benchmark.

    sorb corpus                 build data/corpus (downloads LibriSpeech + real recordings)
    sorb process [-m NAME ...]  run methods (cached per chain prefix)
    sorb realign                re-measure stage delays and fix shifted outputs
    sorb score   [-m NAME ...]  intrusive + non-intrusive metrics -> results/clips.csv
    sorb asr     [-m NAME ...]  Whisper WER on every clip with a transcript
    sorb speed   [-m NAME ...]  real-time factor of each stage, run alone
    sorb level                  DPDFNet speech retention vs input level
    sorb report                 regenerate README.md from results/
    sorb all                    everything above, in order
"""

from __future__ import annotations

import argparse
import fnmatch
import json
from pathlib import Path

import pandas as pd

from . import corpus, engines, metrics
from .methods import METHODS, Method

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
METRICS = ROOT / "data" / "metrics"


def select(patterns: list[str] | None) -> list[Method]:
    if not patterns:
        return METHODS
    chosen = [m for m in METHODS if any(fnmatch.fnmatch(m.name, p) for p in patterns)]
    if not any(m.name == "input" for m in chosen):
        chosen.insert(0, METHODS[0])
    return chosen


def clips_of(frame: pd.DataFrame) -> list[tuple[str, str]]:
    return list(zip(frame["clip"], frame["path"]))


def metric_dir(method: Method) -> Path:
    key = engines.prefix_dir(method.stages).name if method.stages else "input"
    path = METRICS / key
    path.mkdir(parents=True, exist_ok=True)
    return path


def output_of(method: Method, clip: str, corpus_path: str) -> Path:
    return engines.output_path(method.stages, clip, corpus_path)


def cmd_process(methods: list[Method], frame: pd.DataFrame) -> None:
    """One thread per engine environment (methods inside one run in order), then chains."""
    from collections import defaultdict
    from concurrent.futures import ThreadPoolExecutor

    clips = clips_of(frame)
    lanes: dict[str, list[Method]] = defaultdict(list)
    chains = []
    for method in methods:
        if len(method.stages) == 1:
            backend = method.stages[0].backend
            if backend in ("dsp", "sdrmm"):
                lanes["local"].append(method)
            elif backend in ("dpdfnet", "onnx", "rnnoise"):  # light single-threaded loops: one lane each
                lanes[method.name].append(method)
            else:
                lanes[backend].append(method)
        elif method.stages:
            chains.append(method)

    def run(batch: list[Method]) -> None:
        for method in batch:
            print(f"process {method.name}", flush=True)
            try:
                engines.run_chain(method.stages, clips)
            except Exception as error:  # one broken engine must not stop the others
                print(f"FAILED {method.name}: {error}", flush=True)
            else:
                print(f"done {method.name}", flush=True)

    # At most SORB_LANES engine processes at once, to bound memory (~0.2-5 GB each).
    import os

    with ThreadPoolExecutor(min(len(lanes), int(os.environ.get("SORB_LANES", "6"))) or 1) as pool:
        list(pool.map(run, lanes.values()))
    run(chains)


def complete(method: Method, frame: pd.DataFrame) -> bool:
    """All outputs exist and every stage has written stage.json (i.e. delay-compensated)."""
    if any(not (engines.prefix_dir(method.stages[: i + 1]) / "stage.json").exists() for i in range(len(method.stages))):
        print(f"skip {method.name}: still processing", flush=True)
        return False
    missing = sum(not output_of(method, r.clip, r.path).exists() for r in frame.itertuples())
    if missing:
        print(f"skip {method.name}: {missing} outputs not processed yet", flush=True)
    return not missing


def cmd_score(methods: list[Method], frame: pd.DataFrame) -> None:
    rows = []
    for method in [m for m in methods if complete(m, frame)]:
        print(f"score {method.name}", flush=True)
        jobs = [
            (r.clip, r.path, str(output_of(method, r.clip, r.path)), r.reference)
            for r in frame.itertuples()
        ]
        for row in metrics.score(jobs, metric_dir(method) / "scores.jsonl"):
            rows.append({"method": method.name, **row})
    _merge(pd.DataFrame(rows), [m.name for m in methods])


def cmd_asr(methods: list[Method], frame: pd.DataFrame) -> None:
    transcribed = frame[frame["transcript"].astype(str).str.strip() != ""]
    ready = [m for m in methods if complete(m, transcribed)]
    groups = [(metric_dir(m) / "asr.jsonl",
               [(r.clip, output_of(m, r.clip, r.path), r.transcript) for r in transcribed.itertuples()])
              for m in ready]
    metrics.asr_many(groups, workers=int(__import__("os").environ.get("SORB_ASR_JOBS", "3")))
    rows = []
    for method, (cache, jobs) in zip(ready, groups):
        print(f"asr {method.name}", flush=True)
        for clip, row in metrics.asr(jobs, cache).items():
            rows.append({"method": method.name, "clip": clip, "asr_errors": row["errors"],
                         "asr_words": row["words"], "asr_text": row["text"]})
    _merge(pd.DataFrame(rows), [m.name for m in methods])


def _merge(new: pd.DataFrame, methods: list[str]) -> None:
    """Upserts per-clip rows into results/clips.csv, keyed by (method, clip)."""
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / "clips.csv"
    if path.exists():
        old = pd.read_csv(path, keep_default_na=False, na_values=[""])
        old = old.set_index(["method", "clip"])
        new = new.set_index(["method", "clip"])
        combined = old.combine_first(new)
        combined.update(new)
        frame = combined.reset_index()
    else:
        frame = new
    known = {m.name for m in METHODS}
    frame = frame[frame.method.isin(known)]
    frame.sort_values(["method", "clip"]).to_csv(path, index=False, float_format="%.4f")


def cmd_speed(methods: list[Method], frame: pd.DataFrame) -> None:
    """Times every distinct stage alone (machine otherwise idle) on 6 synthetic clips."""
    import shutil
    import tempfile

    syn = frame[frame["set"] == "synthetic"]
    picks = pd.concat([syn[syn["mode"] == mode].head(2) for mode in ("nfm", "am", "usb")])
    clips = clips_of(picks)
    path = RESULTS / "speed.csv"
    known = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["stage", "rtf"])
    rtf = dict(zip(known.stage, known.rtf))
    for stage in dict.fromkeys(s for m in methods for s in m.stages):
        if stage.label() in rtf:
            continue
        scratch = Path(tempfile.mkdtemp(prefix="sorb-speed-"))
        try:
            rtf[stage.label()] = engines.speed(stage, clips, scratch)
            print(f"speed {stage.label()}: RTF {rtf[stage.label()]:.4f}", flush=True)
        except Exception as error:
            print(f"speed {stage.label()} FAILED: {error}", flush=True)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
        RESULTS.mkdir(exist_ok=True)
        pd.DataFrame(sorted(rtf.items()), columns=["stage", "rtf"]).to_csv(path, index=False, float_format="%.5f")


def write_method_table(frame: pd.DataFrame) -> None:
    speed_path = RESULTS / "speed.csv"
    speed = dict(pd.read_csv(speed_path).values) if speed_path.exists() else {}
    rows = []
    for method in METHODS:
        stage_rtf = [speed.get(s.label()) for s in method.stages]
        rows.append(dict(
            method=method.name, family=method.family, description=method.description,
            chain=" → ".join(s.label() for s in method.stages),
            rtf=round(sum(stage_rtf), 5) if stage_rtf and all(r is not None for r in stage_rtf) else None,
        ))
    RESULTS.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS / "methods.csv", index=False)
    cols = ["clip", "set", "mode", "condition", "group", "demod", "speaker_sex", "source",
            "license", "duration_s", "description"]
    frame[cols].to_csv(RESULTS / "corpus.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(prog="sorb", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["corpus", "process", "realign", "score", "asr", "speed", "level", "report", "all"])
    parser.add_argument("-m", "--methods", nargs="*", help="method names or globs")
    parser.add_argument("--set", choices=["synthetic", "real"], help="limit to one test set")
    args = parser.parse_args()

    if args.command in ("corpus", "all"):
        corpus.build()
    if args.command == "report":
        from . import report

        write_method_table(corpus.manifest())
        report.write()
        return
    if args.command == "corpus":
        return
    if args.command == "level":
        args.methods = ["input"]
    frame = corpus.manifest()
    if args.set:
        frame = frame[frame["set"] == args.set]
    methods = select(args.methods)
    if args.command in ("process", "all"):
        cmd_process(methods, frame)
    if args.command == "realign":
        prefixes = {m.stages[: i + 1] for m in methods for i in range(len(m.stages))}
        for label in engines.realign(list(prefixes), clips_of(frame)):
            stale = METRICS / engines.prefix_dir(tuple(
                next(s for m in methods for s in m.stages if s.label() == part)
                for part in label.split(" -> "))).name / "scores.jsonl"
            stale.unlink(missing_ok=True)  # level/intrusive scores depend on timing; ASR does not
    if args.command in ("score", "all"):
        cmd_score(methods, frame)
    if args.command in ("asr", "all"):
        cmd_asr(methods, frame)
    if args.command in ("speed", "all"):
        cmd_speed(methods, frame)
    if args.command in ("level", "all"):
        import os
        import subprocess

        env = engines.ENGINES / "dpdfnet"
        subprocess.run(["uv", "run", "--quiet", "--project", str(env), "python", str(env / "level_sensitivity.py")],
                       check=True, cwd=env, env=dict(os.environ, VIRTUAL_ENV="",
                                                     DPDFNET_CACHE_DIR=str(ROOT / "models" / "dpdfnet")))
    if args.command == "all":
        from . import report

        write_method_table(corpus.manifest())
        report.write()


if __name__ == "__main__":
    main()
