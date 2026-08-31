set -e
cd $HOME/egdict
mkdir -p tools && cd tools
if [ ! -d whisper.cpp ]; then
  git clone --depth 1 https://github.com/ggml-org/whisper.cpp.git
fi
cd whisper.cpp
echo "=== conversion scripts available ==="
ls models/ | grep -iE "convert|quantize" || true
echo
echo "=== convert-h5-to-ggml.py usage header ==="
head -40 models/convert-h5-to-ggml.py 2>/dev/null || echo "MISSING convert-h5-to-ggml.py"
