#!/usr/bin/env bash
# Date-driven milestone drip. Runs daily; releases the next queued event only
# once its release_after date has arrived, one per day. The commit is made and
# dated on the real run day (no pre-stamped commit dates) — release_after only
# gates WHEN an already-built chunk becomes public, so the contribution graph
# fills organically across the year with zero manual involvement.
set -euo pipefail

rm -f .drip/_noop .drip/_commit_msg

next=$(sed -n 's/.*"next"[[:space:]]*:[[:space:]]*\([0-9][0-9]*\).*/\1/p' .drip/state.json)
[ -n "${next:-}" ] || { echo "ERROR: could not read .drip/state.json"; exit 1; }

dir=$(printf 'queue/%03d' "$next")
if [ ! -d "$dir" ]; then
  echo "Queue exhausted at #$next — nothing left to release."
  : > .drip/_noop
  exit 0
fi

today=$(date -u +%F)
due=$(tr -d '[:space:]' < "$dir/release_after")
if [[ "$today" < "$due" ]]; then
  echo "Next event #$next not due until $due (today is $today)."
  : > .drip/_noop
  exit 0
fi

echo "Releasing event #$next from $dir (due $due, today $today)."
git apply --whitespace=nowarn "$dir/diff.patch"
printf '{ "next": %d }\n' "$((next + 1))" > .drip/state.json
git add -A
cp "$dir/msg.txt" .drip/_commit_msg
echo "Staged event #$next; next is now $((next + 1))."
