# -*- coding: utf-8 -*-
"""Collect video ids across the verified-Egyptian channels.

Spreads picks across channels rather than taking many episodes from one show:
speaker/acoustic diversity matters more than raw hours for ASR fine-tuning.
"""
import json, subprocess, os, sys, collections

YTDLP = os.path.expanduser("~/.local/bin/yt-dlp")

# queries chosen to hit the channels that scored as genuinely Egyptian
QUERIES = [
    u"بودكاست فايق ورايق ابراهيم فايق",
    u"الدحيح",
    u"بودكاست الدويتو",
    u"GLASSROOM بودكاست",
    u"بودكاست اثير Atheer",
    u"بودكاست بدون ورق",
    u"عبدالرحمن مجدي بودكاست",
    u"بودكاست مصري حلقة كاملة",
    u"بودكاست كورة مصري",
    u"بودكاست جرايم مصري",
    u"بودكاست تقني مصري",
    u"بودكاست تنمية بشرية مصري",
    u"GTalks بودكاست",
    u"TPP Network بودكاست",
    u"بودكاست روايتهم",
    u"سوالف بودكاست مصري",
    u"بودكاست فنانين مصريين",
    u"بودكاست أطباء مصريين",
]
PER_Q = int(sys.argv[1]) if len(sys.argv) > 1 else 6
PER_CHANNEL_CAP = int(sys.argv[2]) if len(sys.argv) > 2 else 3

seen, by_chan, rows = set(), collections.Counter(), []
for q in QUERIES:
    try:
        out = subprocess.run(
            [YTDLP, "--flat-playlist", "--playlist-end", str(PER_Q), "--dump-json",
             "--no-warnings", "--ignore-errors", "ytsearch%d:%s" % (PER_Q, q)],
            capture_output=True, text=True, timeout=300).stdout
    except subprocess.TimeoutExpired:
        continue
    for line in out.splitlines():
        try:
            j = json.loads(line)
        except Exception:
            continue
        vid, dur = j.get("id"), (j.get("duration") or 0)
        chan = j.get("channel") or j.get("uploader") or "?"
        if not vid or vid in seen or dur < 600:
            continue
        if by_chan[chan] >= PER_CHANNEL_CAP:
            continue
        seen.add(vid); by_chan[chan] += 1
        rows.append((vid, chan, dur, (j.get("title") or "")[:48]))

rows.sort(key=lambda r: r[1])
p = os.path.expanduser("~/egdict/work/harvest_ids.txt")
with open(p, "w", encoding="utf-8") as f:
    for vid, chan, dur, title in rows:
        f.write(vid + "\n")

print("collected %d videos, %.1f h of source across %d channels"
      % (len(rows), sum(r[2] for r in rows) / 3600.0, len(by_chan)))
for chan, n in by_chan.most_common():
    print("   %2d | %s" % (n, chan[:50]))
print("\n-> %s" % p)
