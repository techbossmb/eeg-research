#!/bin/bash
# Download FACED raw EEG (BDF) files from the NEMAR public S3 bucket (CC-BY-4.0).
# Run from macOS Terminal inside ~/Code/eeg-research:
#     bash download_data.sh            # subjects 0-38  (~7 GB)
#     bash download_data.sh 0 122      # all 123 subjects (~34 GB)
# Safe to re-run: finished files are skipped, partial ones resume.
set -u
FIRST=${1:-0}
LAST=${2:-38}
BASE="https://nemar.s3.us-east-2.amazonaws.com/nm000112/objects"
cd "$(dirname "$0")"
for i in $(seq -f "%03g" "$FIRST" "$LAST"); do
  f="data/bids/sub-$i/eeg/sub-${i}_task-watchingVideoClips_eeg.bdf"
  mkdir -p "$(dirname "$f")"
  # the git checkout leaves a ~100-byte annex pointer; real files are 100+ MB
  if [ -f "$f" ] && [ "$(wc -c < "$f")" -gt 1000000 ] && [ ! -f "$f.part" ]; then
    echo "sub-$i done"; continue
  fi
    echo "sub-$i downloading..."
  key=$(awk -v s="$i" '$1==s{print $2}' annex_keys.tsv)
  if curl -fL --retry 5 -C - -o "$f.part" "$BASE/$key" \
     || curl -fL --retry 5 -C - -o "$f.part" "$BASE/sub-$i/eeg/sub-${i}_task-watchingVideoClips_eeg.bdf"; then
    mv "$f.part" "$f"
  else
    echo "  sub-$i FAILED (re-run the script to resume)"
  fi
done
echo "Finished. $(ls data/bids/sub-*/eeg/*.bdf 2>/dev/null | xargs -n1 wc -c 2>/dev/null | awk '$1>1000000' | wc -l) full BDF files present."
