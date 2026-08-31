# -*- coding: utf-8 -*-
"""Build the core-Egyptian floor list: words pinned to a high f so autocorrect
never fights them.

Selecting purely by "frequent in corpus but absent from the MSA base" does NOT
work -- on Egyptian forum text it surfaces football nouns (الزمالك، الدورى),
proper nouns, and porn spam (سكس، نيك), none of which belong at f=250. So the
list is curated by category and then *verified* against the corpus; anything
that fails the frequency check or hits the blocklist is reported and dropped.

Usage: derive_floor.py <counts.tsv> <base.combined> <n> <out.tsv> [--min-count N]
"""
import sys

# A. Egyptian core function / discourse words.
CAT_A = u"""مش كده كدا كدة ده دة دي دى ديه دول دا
عشان علشان ايه اية ازاي ازاى ليه فين منين امتى مين
دلوقتي دلوقتى دلوقت النهاردة امبارح بكرة بكره
عايز عاوز عايزة عاوزة عايزين عاوزين
بقى بقا يبقى هيبقى مبقاش خلاص يعني يعنى اهو اهي اهى اهه
كمان برضو برضه برضك لسه لسة لسا شوية شويه كتير خالص اوي اوى قوى
ازيك ازيكم ايوه ايوة ايوا اومال امال
احنا انتو انتوا هما انتي انتى
مكنش مفيش محدش ماشي ماشى طب طيب يلا يالا ياللا معلش
بتاع بتاعة بتوع بتاعي بتاعك بتاعها
عندي عندى عندك عنده عندنا معايا معاك معاه معاها ليا ليك ليها
جامد حلو وحش زي زى بلاش بجد
بيقول بيعمل هيكون عمال عارف عارفة فاهم فاهمة شايف عاجبني
ياريت ياعم ياض حاجة حاجات خلي خليك بص ينفع"""

# B. Egyptian orthographic variants (hamza-less alef, ya/alef-maqsura) of very
#    common words. These are the single biggest source of false typo flags,
#    and the MSA base almost entirely lacks them.
CAT_B = u"""فى ان انا الى التى الذى اللى اللي اى اي الا اذا ايضا
انى اني اننا انها انه لى هى وانا لكن الان اكيد اصلا دايما
اولا اخرى اخر تانى تاني تانية شىء شئ لأ اه اهلا للاسف
ولا حد كل يمكن لازم ممكن طبعا جدا"""

# Never floor these: spam / sexual terms and domain nouns that the frequency
# signal drags in. (Sexual terms are separately marked possibly_offensive.)
BLOCKLIST = u"""سكس نيك نيج زب كس طيز شرموط شرموطة عرص متناك لبوة مزة
افلام بوك اباحية جنس مساج
الزمالك الاهلى الأهلى الاهلي النادى الدورى الماتش ماتش شوبير ميدو
الاسماعيلى المصرى افريقيا ابراهيم عبدالله محمد احمد مصطفى
الغالىيارب ال ا هع اون العاب اغنية اغاني""".split()


def main():
    counts_path, base_path, n_out, out_path = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    min_count = 300
    if "--min-count" in sys.argv:
        min_count = int(sys.argv[sys.argv.index("--min-count") + 1])

    counts = {}
    with open(counts_path, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) == 2:
                counts[p[0]] = int(p[1])

    base = {}
    with open(base_path, encoding="utf-8") as f:
        f.readline()
        for line in f:
            if line.startswith(" word="):
                fl = dict(kv.split("=", 1) for kv in line.strip().split(",") if "=" in kv)
                base[fl["word"]] = int(fl.get("f", 0))

    block = set(BLOCKLIST)
    chosen, rejected, seen = [], [], set()
    for cat, words in (("A", CAT_A.split()), ("B", CAT_B.split())):
        for w in words:
            if w in seen:
                continue
            seen.add(w)
            c = counts.get(w, 0)
            if w in block:
                rejected.append((w, c, "blocklisted")); continue
            if c < min_count:
                rejected.append((w, c, "below min_count %d" % min_count)); continue
            chosen.append((w, c, base.get(w, 0), cat))

    chosen.sort(key=lambda x: -x[1])
    chosen = chosen[:n_out]

    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for w, c, bf, cat in chosen:
            f.write("%s\t%d\t%d\tcat%s\n" % (w, c, bf, cat))

    na = sum(1 for x in chosen if x[3] == "A")
    print("floor list: %d words (%d cat-A function/discourse, %d cat-B orthographic variants)"
          % (len(chosen), na, len(chosen) - na))
    print("  absent from MSA base: %d of %d" % (sum(1 for x in chosen if x[2] == 0), len(chosen)))
    if rejected:
        print("  rejected %d:" % len(rejected))
        for w, c, why in rejected:
            print("     %-12s count=%-8d %s" % (w, c, why))


if __name__ == "__main__":
    main()
