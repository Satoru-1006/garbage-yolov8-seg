#!/usr/bin/env bash
# Fetch the OFFICIAL TACO release archive from the Zenodo mirror that the
# official README advertises (DOI 10.5281/zenodo.3587843, version v1.0).
#
# We use the DOI mirror rather than the per-image Flickr URLs in download.py
# because Flickr throttles the original-size (`_o`) URLs with HTTP 429 from this
# host, both concurrently and sequentially. The DOI mirror is the official
# fallback documented by the TACO authors and contains byte-identical images.
#
# Splits the 2.72 GB archive into N parallel byte ranges (curl -r), then joins
# and verifies the published MD5.
set -euo pipefail

URL="https://zenodo.org/api/records/3587843/files/TACO.zip/content"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="$ROOT/datasets/taco/raw"
ZIP="$OUT_DIR/TACO.zip"
EXPECTED_MD5="e9149407d883e21a8d224feef8210920"
SIZE=2719810615
N=16
PARTS="$OUT_DIR/.parts"

mkdir -p "$PARTS"
cd "$PARTS"

rm -f part_*
CHUNK=$(( (SIZE + N - 1) / N ))

for i in $(seq 0 $((N-1))); do
  start=$(( i * CHUNK ))
  end=$(( start + CHUNK - 1 ))
  if [ "$end" -ge "$SIZE" ]; then end=$(( SIZE - 1 )); fi
  (
    for attempt in 1 2 3 4 5; do
      if curl -s -f -L -r "${start}-${end}" -o "part_${i}" "$URL"; then
        sz=$(stat -c%s "part_${i}")
        if [ "$sz" -eq $(( end - start + 1 )) ]; then exit 0; fi
      fi
      echo "retry part_${i} attempt=${attempt}" >&2
      sleep 5
    done
    echo "FAILED part_${i} range=${start}-${end}" >&2
    exit 1
  ) &
done
wait

cat $(for i in $(seq 0 $((N-1))); do echo "part_${i}"; done) > "$ZIP"
rm -rf "$PARTS"

echo "downloaded: $(stat -c%s "$ZIP") bytes (expected $SIZE)"
echo "$EXPECTED_MD5  $ZIP" | md5sum -c -
