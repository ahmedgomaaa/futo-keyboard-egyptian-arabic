# -*- coding: utf-8 -*-
"""Validate the built .combined and the compiled .dict.

Usage: validate.py <ar_eg.combined> <main_ar_eg.dict> [--json out.json]
"""
import sys, os, json, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import egtext

MAGIC = 0x9BC13AFE

EG_TEST = u"""مش كده ده دي عشان علشان ايه ازاي دلوقتي عايز عاوز بقى خلاص يعني اهو
كمان برضو لسه شوية كتير ازيك ايوه اومال معلش حاجة اوي طب يلا مفيش احنا""".split()

MSA_TEST = u"""الذي التي هذا هذه ذلك كان يكون سوف قد لقد إلى على من عن في
مع بعد قبل حيث لكن حتى إذا كما أيضا جميع بعض كثير قليل الذين هؤلاء أولئك""".split()

# Egyptian words deliberately NOT in the floor list -- these exercise the
# corpus->f calibration rather than the hard-coded floor.
EG_NONFLOOR = u"""بتاعتنا مستني مستنية عاجبك بيحصل هيروح بنعمل بتقول
مبيعرفش هتيجي ماحدش عشرين حبتين اتفضل يابني بتحب""".split()


def parse_combined(path):
    words, bigrams = {}, {}
    dupes = []
    header = None
    with open(path, encoding="utf-8") as f:
        header = f.readline().rstrip("\n")
        cur = None
        for ln, line in enumerate(f, 2):
            s = line.rstrip("\n")
            if s.startswith(" word="):
                fl = dict(kv.split("=", 1) for kv in s.strip().split(",") if "=" in kv)
                w = fl["word"]
                if w in words:
                    dupes.append(w)
                words[w] = int(fl["f"])
                cur = w
            elif s.strip().startswith("bigram="):
                fl = dict(kv.split("=", 1) for kv in s.strip().split(",") if "=" in kv)
                bigrams.setdefault(cur, []).append((fl["bigram"], int(fl["f"])))
    return header, words, bigrams, dupes


def parse_dict_header(path):
    with open(path, "rb") as f:
        head = f.read(4096)
    magic = int.from_bytes(head[0:4], "big")
    fmt_version = int.from_bytes(head[4:6], "big")
    flags = int.from_bytes(head[6:8], "big")
    hsize = int.from_bytes(head[8:12], "big")
    body = head[12:hsize].decode("utf-8", "replace")
    parts = body.split("\x1f")
    attrs = {}
    for i in range(0, len(parts) - 1, 2):
        attrs[parts[i]] = parts[i + 1]
    return {"magic": magic, "format_version": fmt_version, "flags": flags,
            "header_size": hsize, "attrs": attrs}


def main():
    comb, dct = sys.argv[1], sys.argv[2]
    ok = True
    def check(cond, msg):
        nonlocal ok
        print(("  PASS  " if cond else "  FAIL  ") + msg)
        if not cond:
            ok = False

    header, words, bigrams, dupes = parse_combined(comb)
    nb = sum(len(v) for v in bigrams.values())
    print("== .combined ==")
    print("  header: %s" % header)
    print("  words: %d   bigrams: %d" % (len(words), nb))
    hattrs = dict(kv.split("=", 1) for kv in header.split(",") if "=" in kv)
    check(hattrs.get("locale") == "ar", "locale is plain 'ar' (got %r)" % hattrs.get("locale"))
    check(int(hattrs.get("version", 0)) > 18, "version > 18 (got %s)" % hattrs.get("version"))
    check(not dupes, "no duplicate word entries (%d dupes)" % len(dupes))
    bad_f = [w for w, f in words.items() if not (0 <= f <= 255)]
    check(not bad_f, "all f in 0..255 (%d bad)" % len(bad_f))
    nonarab = [w for w in list(words)[:200000] if not egtext.RE_TOKEN.fullmatch(w)]
    check(not nonarab, "sampled words are pure Arabic script (%d bad: %s)"
          % (len(nonarab), nonarab[:5]))
    orphan = [b for p, lst in bigrams.items() for b, _ in lst if b not in words]
    check(not orphan, "every bigram target exists as a word (%d orphans)" % len(orphan))

    print("== .dict binary ==")
    h = parse_dict_header(dct)
    print("  magic=0x%X format_version=%d header_size=%d" % (h["magic"], h["format_version"], h["header_size"]))
    print("  attrs: %s" % json.dumps(h["attrs"], ensure_ascii=False))
    check(h["magic"] == MAGIC, "magic number is 0x9BC13AFE")
    check(h["attrs"].get("locale") == "ar", "binary locale == 'ar'")
    check(int(h["attrs"].get("version", 0)) > 18, "binary version attr > 18")
    check(h["attrs"].get("dictionary", "").startswith("main:ar"), "binary dictionary == main:ar")
    size = os.path.getsize(dct)
    print("  size: %.2f MB" % (size / 1e6))
    check(size < 15e6, "size under 15 MB phone budget (%.2f MB)" % (size / 1e6))

    print("== test words ==")
    miss_eg = [w for w in EG_TEST if w not in words]
    miss_msa = [w for w in MSA_TEST if w not in words]
    print("  Egyptian (%d): %s" % (len(EG_TEST),
          " ".join("%s=%d" % (w, words[w]) for w in EG_TEST if w in words)))
    print("  MSA (%d): %s" % (len(MSA_TEST),
          " ".join("%s=%d" % (w, words[w]) for w in MSA_TEST if w in words)))
    check(not miss_eg, "all %d Egyptian test words present (missing: %s)" % (len(EG_TEST), miss_eg))
    check(not miss_msa, "all %d MSA test words present (missing: %s)" % (len(MSA_TEST), miss_msa))
    present_nf = [(w, words[w]) for w in EG_NONFLOOR if w in words]
    print("  Egyptian non-floored (%d/%d present): %s" % (
        len(present_nf), len(EG_NONFLOOR), " ".join("%s=%d" % t for t in present_nf)))
    check(len(present_nf) >= len(EG_NONFLOOR) * 0.8,
          "at least 80%% of non-floored Egyptian words present (%d/%d)"
          % (len(present_nf), len(EG_NONFLOOR)))
    check(all(f < 250 for _, f in present_nf),
          "non-floored words got calibrated f, not the floor value")
    lo_eg = [(w, words[w]) for w in EG_TEST if w in words and words[w] < 40]
    check(not lo_eg, "Egyptian test words have sane f >= 40 (low: %s)" % lo_eg[:8])

    print("\nRESULT: %s" % ("ALL CHECKS PASSED" if ok else "FAILURES PRESENT"))
    if "--json" in sys.argv:
        out = sys.argv[sys.argv.index("--json") + 1]
        hist = collections.Counter((f // 16) * 16 for f in words.values())
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"words": len(words), "bigrams": nb, "size": size,
                       "dict_header": h["attrs"], "f_hist": dict(sorted(hist.items()))},
                      f, ensure_ascii=False, indent=2)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
