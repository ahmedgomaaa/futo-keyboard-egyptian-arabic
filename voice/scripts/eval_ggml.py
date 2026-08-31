# -*- coding: utf-8 -*-
"""WER of the *quantized GGML* model via whisper.cpp -- the artifact we actually ship.

Confirms the HF->GGML->quantize chain preserved quality.
Usage: eval_ggml.py <ggml.bin> <n> [audio_ctx]
"""
import os, sys, glob, io, subprocess, tempfile, re, unicodedata
import pyarrow.parquet as pq
import soundfile as sf
import jiwer

ROOT = os.path.expanduser('~/egdict')
CLI = os.path.join(ROOT, 'tools/whisper.cpp/build/bin/whisper-cli')
DATA = os.path.join(ROOT, 'asr/casablanca_egypt')

DIAC = re.compile(u"[ـً-ٰٟۖ-ۭ]")
PUNCT = re.compile(u"[^ء-ي٠-٩a-zA-Z0-9\\s]")


def norm_ar(s):
    s = unicodedata.normalize("NFKC", s or "")
    s = DIAC.sub("", s)
    s = s.replace(u"أ", u"ا").replace(u"إ", u"ا").replace(u"آ", u"ا")
    s = s.replace(u"ى", u"ي").replace(u"ة", u"ه")
    s = PUNCT.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def main():
    model, n = sys.argv[1], int(sys.argv[2])
    actx = sys.argv[3] if len(sys.argv) > 3 else None

    rows = []
    for f in sorted(glob.glob(os.path.join(DATA, '*.parquet'))):
        pf = pq.ParquetFile(f)
        for b in pf.iter_batches(batch_size=64):
            d = b.to_pydict()
            for i in range(len(d['transcription'])):
                dur = d['duration'][i]
                if dur is None or not (0.8 <= dur <= 25.0):
                    continue
                ref = norm_ar(d['transcription'][i])
                if len(ref.split()) < 2:
                    continue
                rows.append((d['audio'][i]['bytes'], ref))
            if len(rows) >= n * 3:
                break
        if len(rows) >= n * 3:
            break
    rows.sort(key=lambda r: r[1])
    step = max(1, len(rows) // n)
    rows = rows[::step][:n]

    tmp = tempfile.mkdtemp()
    refs, hyps = [], []
    for i, (audio, ref) in enumerate(rows):
        arr, sr = sf.read(io.BytesIO(audio), dtype='float32')
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        if sr != 16000:
            import librosa
            arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
        wav = os.path.join(tmp, 'c.wav')
        sf.write(wav, arr, 16000, subtype='PCM_16')
        cmd = [CLI, '-m', model, '-f', wav, '-l', 'ar', '-nt', '-np', '-t', '8']
        if actx:
            cmd += ['-ac', actx]
        out = subprocess.run(cmd, capture_output=True, text=True).stdout
        hyps.append(norm_ar(out)); refs.append(ref)
        if (i + 1) % 25 == 0:
            sys.stderr.write("\r  %d/%d" % (i + 1, len(rows))); sys.stderr.flush()
    sys.stderr.write("\n")

    print("model      : %s" % os.path.basename(model))
    print("audio_ctx  : %s" % (actx or "full"))
    print("n          : %d" % len(refs))
    print("WER        : %.2f" % (100 * jiwer.wer(refs, hyps)))
    print("CER        : %.2f" % (100 * jiwer.cer(refs, hyps)))


if __name__ == '__main__':
    main()
