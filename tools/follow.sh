#!/bin/sh
# Scores and transcribes methods as their outputs complete, while `sorb process` runs.
while true; do
  running=$(pgrep -f "sorb.cli process" >/dev/null && echo 1 || echo 0)
  uv run python -m sorb.cli score 2>&1 | grep -E "^score|Traceback|Error" | grep -v "^skip"
  uv run python -m sorb.cli asr 2>&1 | grep -E "^asr|Traceback|Error" | grep -v "^skip"
  [ "$running" = 0 ] && break
  sleep 300
done
echo FOLLOW-DONE
