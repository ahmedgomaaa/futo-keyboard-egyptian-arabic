# -*- coding: utf-8 -*-
"""Sum per-corpus count TSVs into one. Usage: combine_counts.py OUT IN [IN ...]"""
import sys, collections
out = sys.argv[1]
total = collections.Counter()
for p in sys.argv[2:]:
    n = 0
    try:
        with open(p, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) == 2:
                    total[parts[0]] += int(parts[1]); n += 1
        print("  +%-28s %d types" % (p.split("/")[-1], n))
    except FileNotFoundError:
        print("  !! missing, skipped: %s" % p)
with open(out, "w", encoding="utf-8", newline="\n") as f:
    for t, c in total.most_common():
        f.write("%s\t%d\n" % (t, c))
print("combined -> %s : %d types, %d tokens" % (out, len(total), sum(total.values())))
