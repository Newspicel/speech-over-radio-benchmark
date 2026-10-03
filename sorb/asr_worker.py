"""ASR worker process: `python -m sorb.asr_worker SHARD.json OUT.jsonl`.

Transcribes [cache, clip, audio] jobs with Whisper and appends {"cache", "clip", "text"}
lines to OUT as it goes, so the parent can watch progress and restart a stuck worker."""

import json
import sys
from pathlib import Path

from . import metrics


def main() -> None:
    shard, out = Path(sys.argv[1]), Path(sys.argv[2])
    done = set()
    if out.exists():
        done = {(r["cache"], r["clip"]) for r in map(json.loads, out.read_text().splitlines())}
    with open(out, "a") as f:
        for cache, clip, audio in json.loads(shard.read_text()):
            if (cache, clip) in done:
                continue
            text = metrics.transcribe([Path(audio)])[0]
            f.write(json.dumps({"cache": cache, "clip": clip, "text": text}) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
