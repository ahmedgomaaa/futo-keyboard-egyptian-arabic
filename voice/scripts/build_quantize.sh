set -e
ROOT=$HOME/egdict
W=$ROOT/tools/whisper.cpp
OUT=$ROOT/asr/ggml

echo "== build whisper.cpp (CPU is fine, we only need quantize + cli) =="
cd $W
cmake -B build -DCMAKE_BUILD_TYPE=Release -DWHISPER_BUILD_TESTS=OFF > /dev/null 2>&1
cmake --build build --config Release -j "$(nproc)" > /tmp/wcpp_build.log 2>&1 || {
  echo "BUILD FAILED"; tail -25 /tmp/wcpp_build.log; exit 1; }
echo "built:"; ls build/bin/ | head

echo
echo "== quantize for mobile =="
QB=$(find $W/build -name "quantize*" -type f -perm -u+x | head -1)
echo "quantize binary: $QB"
for Q in q5_1 q8_0; do
  "$QB" "$OUT/ggml-model.bin" "$OUT/ggml-egyptian-small-$Q.bin" $Q > /dev/null 2>&1 \
    && echo "  ok $Q" || echo "  FAILED $Q"
done
mv "$OUT/ggml-model.bin" "$OUT/ggml-egyptian-small-f16.bin"
ls -l "$OUT"
