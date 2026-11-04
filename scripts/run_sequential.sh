#!/usr/bin/env bash
# Run a job list (from scripts/make_jobs.py) one command after the other on this machine.
#   python scripts/make_jobs.py seattle -o jobs.txt && scripts/run_sequential.sh jobs.txt [first_line [last_line]]
# Commands whose log in logs/ already ends with "done" are skipped, so an interrupted run can be restarted.
set -uo pipefail
JOBS=${1:?usage: run_sequential.sh jobs.txt [first [last]]}
FIRST=${2:-1}; LAST=${3:-$(wc -l < "$JOBS")}
mkdir -p logs
export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd)${PYTHONPATH:+:$PYTHONPATH}"
for i in $(seq "$FIRST" "$LAST"); do
  cmd=$(sed -n "${i}p" "$JOBS")
  log=logs/job_$(printf %04d "$i").log
  if [ -f "$log" ] && [ "$(tail -n 1 "$log")" = "done" ]; then continue; fi
  echo "[$i/$LAST] $cmd"
  if eval "$cmd" > "$log" 2>&1; then echo done >> "$log"; else echo "FAILED (see $log)" >&2; fi
done
