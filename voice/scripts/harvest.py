# -*- coding: utf-8 -*-
"""Harvest Egyptian podcast audio + word-timed captions into ASR training segments.

For each video:
  1. download the ar-orig auto-captions in json3 (word-level timings)
  2. download audio only, transcode to 16 kHz mono wav
  3. group words into 5-25 s utterances, splitting on pauses
  4. cut the audio at those boundaries and write (wav, text) pairs

Why json3 and not vtt: YouTube's VTT auto-captions are "rolling" (each cue repeats
the previous words), which makes utterance boundaries unreliable. json3 gives per-word
offsets, so segments line up with what was actually said.

Usage:
  harvest.py --ids-file ids.txt --out ~/egdict/asr/yt --max-hours 10
  harvest.py --search "الدحيح" --n 8 --out ~/egdict/asr/yt --max-hours 10
"""
import os, sys, json, re, glob, argparse, subprocess, unicodedata, shutil

YTDLP = os.path.expanduser("~/.local/bin/yt-dlp")

MIN_SEG, MAX_SEG = 3.0, 16.0     # seconds; keyboard dictation is short, not 25s
# Auto-caption word timings are continuous, so a "gap" must be measured between
# consecutive word STARTS, not between an estimated end and the next start --
# otherwise the gap is always ~0 and everything gets cut at MAX_SEG.
PAUSE_SPLIT = 0.90               # start-to-start interval that implies a pause
DIAC = re.compile(u"[ـً-ٰٟۖ-ۭ]")
BRACKET = re.compile(r"\[[^\]]{0,40}\]")
ENTITY = re.compile(r"&(?:[a-zA-Z]{1,10}|#\d{1,6});")
NONAR = re.compile(u"[^ء-ي٠-٩\\s]")


def clean(t):
    t = unicodedata.normalize("NFKC", t or "")
    t = ENTITY.sub(" ", t)
    t = BRACKET.sub(" ", t)
    t = DIAC.sub("", t)
    t = t.replace(">>", " ")
    t = NONAR.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


def run(cmd, timeout=1800):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def words_from_json3(path):
    """-> [(word, start_s, end_s)]"""
    try:
        j = json.load(open(path, encoding="utf-8"))
    except Exception:
        return []
    out = []
    for ev in j.get("events") or []:
        base = ev.get("tStartMs")
        if base is None or not ev.get("segs"):
            continue
        for s in ev["segs"]:
            w = (s.get("utf8") or "").strip()
            if not w or w == "\n":
                continue
            st = (base + (s.get("tOffsetMs") or 0)) / 1000.0
            out.append([w, st, st])
    out.sort(key=lambda x: x[1])
    for i in range(len(out) - 1):
        out[i][2] = min(out[i + 1][1], out[i][1] + 2.0)
    if out:
        out[-1][2] = out[-1][1] + 0.4
    return out


def group(words):
    """Group word timings into utterances on pauses / max length."""
    segs, cur = [], []
    for w, st, en in words:
        if cur:
            gap = st - cur[-1][1]          # start-to-start
            dur = en - cur[0][1]
            if gap > PAUSE_SPLIT or dur > MAX_SEG:
                segs.append(cur); cur = []
        cur.append((w, st, en))
    if cur:
        segs.append(cur)

    out = []
    for s in segs:
        st, en = s[0][1], s[-1][2]
        txt = clean(" ".join(w for w, _, _ in s))
        if en - st < MIN_SEG or en - st > MAX_SEG:
            continue
        if len(txt.split()) < 3:
            continue
        out.append((st, en, txt))
    return out


def video_ids(args):
    if args.ids_file:
        return [l.strip() for l in open(args.ids_file) if l.strip()
                and not l.startswith("#")]
    r = run([YTDLP, "--flat-playlist", "--playlist-end", str(args.n), "--dump-json",
             "--no-warnings", "--ignore-errors", "ytsearch%d:%s" % (args.n, args.search)])
    ids = []
    for line in r.stdout.splitlines():
        try:
            j = json.loads(line)
        except Exception:
            continue
        if (j.get("duration") or 0) >= 300:
            ids.append(j["id"])
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids-file")
    ap.add_argument("--search")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-hours", type=float, default=10.0)
    a = ap.parse_args()

    out = os.path.expanduser(a.out)
    wavdir = os.path.join(out, "wav")
    tmp = os.path.join(out, "tmp")
    for d in (wavdir, tmp):
        os.makedirs(d, exist_ok=True)
    manifest = os.path.join(out, "manifest.jsonl")
    done_ids = set()
    if os.path.exists(manifest):
        for l in open(manifest, encoding="utf-8"):
            try:
                done_ids.add(json.loads(l)["video"])
            except Exception:
                pass

    ids = video_ids(a)
    print("candidate videos: %d" % len(ids))
    total_s = 0.0
    mf = open(manifest, "a", encoding="utf-8")

    for vid in ids:
        if total_s / 3600.0 >= a.max_hours:
            break
        if vid in done_ids:
            print("  skip (already harvested) %s" % vid)
            continue
        for f in glob.glob(os.path.join(tmp, "*")):
            os.remove(f)

        # captions first -- no captions means the audio is useless to us
        run([YTDLP, "--skip-download", "--write-auto-subs", "--sub-langs", "ar-orig",
             "--sub-format", "json3", "--no-warnings",
             "-o", os.path.join(tmp, "c.%(ext)s"),
             "https://www.youtube.com/watch?v=" + vid], timeout=600)
        cap = glob.glob(os.path.join(tmp, "*.json3"))
        if not cap:
            print("  no captions, skipping %s" % vid)
            continue
        segs = group(words_from_json3(cap[0]))
        if not segs:
            print("  no usable segments, skipping %s" % vid)
            continue

        r = run([YTDLP, "-f", "bestaudio", "-x", "--audio-format", "wav",
                 "--postprocessor-args", "-ar 16000 -ac 1",
                 "--no-warnings", "-o", os.path.join(tmp, "a.%(ext)s"),
                 "https://www.youtube.com/watch?v=" + vid], timeout=2400)
        wav = glob.glob(os.path.join(tmp, "a.wav"))
        if not wav:
            print("  audio download failed %s (%s)" % (vid, r.stderr.strip()[:60]))
            continue
        src = wav[0]

        kept = 0
        vsecs = 0.0
        for i, (st, en, txt) in enumerate(segs):
            if total_s / 3600.0 >= a.max_hours:
                break
            name = "%s_%04d.wav" % (vid, i)
            dst = os.path.join(wavdir, name)
            rr = run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-ss", "%.3f" % st,
                      "-t", "%.3f" % (en - st), "-i", src, "-ar", "16000", "-ac", "1", dst],
                     timeout=120)
            if rr.returncode != 0 or not os.path.exists(dst):
                continue
            mf.write(json.dumps({"video": vid, "wav": name, "start": round(st, 2),
                                 "dur": round(en - st, 2), "text": txt},
                                ensure_ascii=False) + "\n")
            kept += 1
            vsecs += en - st
            total_s += en - st
        mf.flush()
        print("  %s -> %d segments, %.1f min (running total %.2f h)"
              % (vid, kept, vsecs / 60.0, total_s / 3600.0))

    mf.close()
    for f in glob.glob(os.path.join(tmp, "*")):
        try:
            os.remove(f)
        except Exception:
            pass
    print("\nHARVEST TOTAL: %.2f hours" % (total_s / 3600.0))


if __name__ == "__main__":
    main()
