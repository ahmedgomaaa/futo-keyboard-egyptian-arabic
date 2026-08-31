# -*- coding: utf-8 -*-
"""Score candidate channels: do they have Arabic auto-captions, and is the speech
actually Egyptian? Dialect density per 1k words separates Egyptian shows from
pan-Arab/Khaliji ones whose guests merely happen to be Egyptian.
"""
import json, subprocess, os, glob, re, collections, sys

YTDLP = os.path.expanduser("~/.local/bin/yt-dlp")
WORK = os.path.expanduser("~/egdict/work")
TMP = os.path.expanduser("~/egdict/asr/captest")
os.makedirs(TMP, exist_ok=True)

EGY = u"مش عايز عاوز كده كدا ده دي دى اللي اللى ايه ازاي عشان علشان دلوقتي احنا مفيش لسه كتير اوي اوى برضو بقى".split()


def vtt_text(path):
    lines = []
    for l in open(path, encoding="utf-8", errors="replace"):
        l = l.strip()
        if not l or l.startswith(("WEBVTT", "Kind:", "Language:")) or "-->" in l:
            continue
        l = re.sub(r"<[^>]+>", "", l)
        if l and (not lines or lines[-1] != l):
            lines.append(l)
    return " ".join(lines)


def score(vid):
    for f in glob.glob(os.path.join(TMP, "*.vtt")):
        os.remove(f)
    got = None
    for lang in ("ar-orig", "ar"):
        subprocess.run([YTDLP, "--skip-download", "--write-auto-subs", "--sub-langs", lang,
                        "--sub-format", "vtt", "--no-warnings",
                        "-o", os.path.join(TMP, "c.%(ext)s"),
                        "https://www.youtube.com/watch?v=" + vid],
                       capture_output=True, text=True, timeout=240)
        f = glob.glob(os.path.join(TMP, "*.vtt"))
        if f:
            got = (f[0], lang)
            break
    if not got:
        return None, None, 0, 0
    txt = vtt_text(got[0])
    w = txt.split()
    c = collections.Counter(w)
    hits = sum(c[x] for x in EGY)
    dens = 1000.0 * hits / max(len(w), 1)
    return got[1], len(w), hits, dens


samples = json.load(open(os.path.join(WORK, "channel_samples.json"), encoding="utf-8"))
print("%-42s %-8s %8s %7s  %s" % ("channel", "track", "words", "eg/1k", "verdict"))
print("-" * 92)
rows = []
for name, vid in samples.items():
    try:
        lang, nw, hits, dens = score(vid)
    except Exception as e:
        print("%-42s ERR %s" % (name[:42], str(e)[:30]))
        continue
    if not lang:
        print("%-42s %-8s %8s %7s  no captions" % (name[:42], "-", "-", "-"))
        continue
    verdict = "STRONG Egyptian" if dens >= 80 else ("Egyptian-ish" if dens >= 35 else "not Egyptian enough")
    print("%-42s %-8s %8d %7.1f  %s" % (name[:42], lang, nw, dens, verdict))
    rows.append((name, vid, lang, nw, dens))

rows.sort(key=lambda r: -r[4])
json.dump([{"channel": r[0], "sample_video": r[1], "track": r[2],
            "words": r[3], "egy_per_1k": round(r[4], 1)} for r in rows],
          open(os.path.join(WORK, "channel_scores.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print("\nsaved -> %s/channel_scores.json" % WORK)
