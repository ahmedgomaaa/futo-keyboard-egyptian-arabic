#!/usr/bin/env bash
# Reproducible end-to-end build of the Arabic (Egyptian) AOSP dictionary.
#
# Requires: bash, curl, python3 (>=3.8) with pyarrow, and a JRE (>=8).
# Everything is idempotent: already-downloaded inputs are reused.
#
#   ./build.sh                # full build into $ROOT/out
#   ROOT=/tmp/egdict ./build.sh
set -euo pipefail

ROOT="${ROOT:-$HOME/egdict}"
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts"
JAVA="${JAVA:-java}"
JAR="$ROOT/tools/dicttool_aosp.jar"

MIN_COUNT=5             # drop corpus tokens seen fewer than this many times
NEW_WORD_BAR=5          # extra bar for words absent from the MSA base
MAX_BIGRAMS=4           # bigrams emitted per word
N_FLOOR=200             # cap on the curated core-Egyptian floor list
FLOOR_F=250
BIGRAM_PARENTS=20000    # vocabulary rank cutoff for bigram parents
BIGRAM_NEXT=60000       # vocabulary rank cutoff for bigram continuations

# Set INCLUDE_OPENSUBS=1 to fold OpenSubtitles into the frequency list. Off by
# default: measured at 36.4 Egyptian markers per million tokens vs 15,611 for
# EFC, it is MSA fansub text and dilutes the Egyptian signal. See STATS.md 1.1.
INCLUDE_OPENSUBS="${INCLUDE_OPENSUBS:-0}"

mkdir -p "$ROOT"/{tools,base,corpora/{efc,fineweb,opensubs},work,out}

echo "== 1. tooling and base wordlist =="
[ -f "$JAR" ] || curl -sSL -o "$JAR" \
  "https://codeberg.org/Helium314/aosp-dictionaries/raw/branch/main/dicttool_aosp.jar"
# NOTE: we take the base from the published .combined, not by decompiling the
# .dict -- dicttool cannot decompile without a native JNI lib it does not ship.
[ -f "$ROOT/base/main_ar.combined" ] || curl -sSL -o "$ROOT/base/main_ar.combined" \
  "https://codeberg.org/Helium314/aosp-dictionaries/raw/branch/main/wordlists_experimental/main_ar.combined"

echo "== 2. corpora =="
# --- EFC-mini: Egyptian forum text, ~1.9GB, the best register match ---
if [ "$(ls "$ROOT/corpora/efc"/*.txt 2>/dev/null | wc -l)" -lt 70 ]; then
  python3 - "$ROOT/corpora/efc" <<'PY'
import json, sys, os, urllib.request
out = sys.argv[1]
r = urllib.request.Request(
    "https://huggingface.co/api/datasets/faisalq/EFC-mini/tree/main?recursive=1",
    headers={"User-Agent": "curl/8"})
files = [e["path"] for e in json.load(urllib.request.urlopen(r, timeout=60))
         if e["type"] == "file" and e["path"].endswith(".txt")]
with open(os.path.join(out, "urls.txt"), "w") as f:
    for p in files:
        f.write("https://huggingface.co/datasets/faisalq/EFC-mini/resolve/main/%s\n" % p)
print("EFC-mini: %d files to fetch" % len(files))
PY
  (cd "$ROOT/corpora/efc" && xargs -n1 -P8 curl -sSL --retry 5 --retry-all-errors -O < urls.txt)
fi

# --- FineWeb-2 arz: the whole train split is a single 1.1GB shard ---
[ -f "$ROOT/corpora/fineweb/arz_000.parquet" ] || curl -sSL --retry 5 --retry-all-errors \
  -o "$ROOT/corpora/fineweb/arz_000.parquet" \
  "https://huggingface.co/datasets/HuggingFaceFW/fineweb-2/resolve/main/data/arz_Arab/train/000_00000.parquet"

# --- OpenSubtitles (only when explicitly enabled) ---
if [ "$INCLUDE_OPENSUBS" = "1" ]; then
  # served without a reliable connection; loop with -C - to resume partial fetches
  for _ in 1 2 3 4 5 6 7 8; do
    sz=$(stat -c%s "$ROOT/corpora/opensubs/ar.txt.gz" 2>/dev/null || echo 0)
    [ "$sz" -ge 1012859699 ] && break
    curl -sSL --retry 5 --retry-all-errors -C - -o "$ROOT/corpora/opensubs/ar.txt.gz" \
      "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/mono/ar.txt.gz" || true
  done
  gzip -t "$ROOT/corpora/opensubs/ar.txt.gz"
fi

echo "== 3. frequency counting =="
python3 "$S/count_unigrams.py" efc     "$ROOT/work" "$ROOT"/corpora/efc/*.txt
python3 "$S/count_unigrams.py" fineweb "$ROOT/work" "$ROOT/corpora/fineweb/arz_000.parquet"
COUNTS=("$ROOT/work/counts_efc.tsv" "$ROOT/work/counts_fineweb.tsv")
if [ "$INCLUDE_OPENSUBS" = "1" ]; then
  python3 "$S/count_unigrams.py" opensubs "$ROOT/work" "$ROOT/corpora/opensubs/ar.txt.gz"
  COUNTS+=("$ROOT/work/counts_opensubs.tsv")
fi
python3 "$S/combine_counts.py" "$ROOT/work/counts_all.tsv" "${COUNTS[@]}"

echo "== 4. core Egyptian floor list =="
python3 "$S/derive_floor.py" "$ROOT/work/counts_all.tsv" "$ROOT/base/main_ar.combined" \
        "$N_FLOOR" "$ROOT/work/floor_words.tsv"

echo "== 5. bigrams (Egyptian-register sources only) =="
python3 "$S/count_bigrams.py" "$ROOT/work" "$ROOT/work/counts_all.tsv" \
        "$BIGRAM_PARENTS" "$BIGRAM_NEXT" \
        "$ROOT"/corpora/efc/*.txt "$ROOT/corpora/fineweb/arz_000.parquet"

echo "== 6. merge =="
python3 "$S/merge_build.py" \
  --base    "$ROOT/base/main_ar.combined" \
  --counts  "$ROOT/work/counts_all.tsv" \
  --bigrams "$ROOT/work/bigrams.tsv" \
  --out     "$ROOT/out/ar_eg.combined" \
  --min-count "$MIN_COUNT" --eg-min-count-new "$NEW_WORD_BAR" \
  --max-bigrams "$MAX_BIGRAMS" \
  --floor-file "$ROOT/work/floor_words.tsv" --floor-f "$FLOOR_F" \
  --offensive-file "$S/offensive.txt" \
  --report "$ROOT/work/merge_report.json"

echo "== 7. compile and validate =="
"$JAVA" -jar "$JAR" makedict -s "$ROOT/out/ar_eg.combined" -d "$ROOT/out/main_ar_eg.dict"
python3 "$S/validate.py" "$ROOT/out/ar_eg.combined" "$ROOT/out/main_ar_eg.dict" \
        --json "$ROOT/work/validate.json"

echo "== 8. stats =="
cp "$ROOT/work/floor_words.tsv" "$ROOT/out/floor_words.tsv"
python3 "$S/gen_stats.py"

ls -l "$ROOT/out/"
echo "done."
