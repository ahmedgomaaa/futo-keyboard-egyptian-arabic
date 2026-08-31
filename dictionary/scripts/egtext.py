# -*- coding: utf-8 -*-
"""Shared text cleaning + tokenizing for the Egyptian Arabic dictionary build.

Filter rules (per spec):
  - strip HTML/markup, URLs, emails
  - keep Arabic-script tokens only (Latin-script tokens dropped)
  - normalize tatweel away, strip diacritics
  - do NOT normalize alef hamza forms, ta-marbuta, or ya/alef-maqsura
  - drop tokens longer than MAX_LEN chars
  - drop tokens with 3+ identical consecutive letters (elongation spam)
"""
import re, gzip, io, os

MAX_LEN = 20

# --- markup / noise removal (applied before tokenizing) ---
RE_TAG     = re.compile(r"<[^>]{0,400}>")
RE_ENTITY  = re.compile(r"&(?:[a-zA-Z]{1,10}|#\d{1,6}|#[xX][0-9a-fA-F]{1,6});")
RE_URL     = re.compile(r"(?:https?://|www\.)\S+", re.I)
RE_EMAIL   = re.compile(r"[^\s@]{1,64}@[^\s@]{1,255}\.[A-Za-z]{2,24}")
RE_WIKI    = re.compile(r"\[\[[^\]]{0,200}\]\]|\{\{[^\}]{0,200}\}\}")

# --- Arabic script handling ---
# Diacritics / marks to delete: harakat, superscript alef, Quranic marks, tatweel.
RE_DIACRITIC = re.compile(
    u"[ـ"              # tatweel
    u"ً-ٟ"        # fathatan..wavy hamza below
    u"ٰ"               # superscript alef
    u"ۖ-ۭ"        # Quranic annotation marks
    u"࣓-ࣿ"        # Arabic Extended-A combining marks
    u"​-‏‪-‮⁦-⁩﻿"  # zero-width / bidi controls
    u"]")

# Arabic *letters* only (no digits, no punctuation). Excludes U+0660-0669 (Arabic-Indic
# digits), U+066A-066D (punct), U+06F0-06F9 (extended digits).
ARABIC_LETTER = (
    u"ء-ؿ"        # hamza .. (letters incl. rare)
    u"ف-ي"        # feh .. yeh (0640 tatweel already stripped)
    u"ٮ-ە"        # dotless letters, extended letters, ta marbuta variants
    u"ۮ-ۯ"
    u"ۺ-ۿ"
    u"ݐ-ݿ"        # Arabic Supplement
    u"ࢠ-࣒"        # Arabic Extended-A letters
)
RE_TOKEN = re.compile(u"[" + ARABIC_LETTER + u"]+")

# 3+ identical consecutive letters
RE_ELONG = re.compile(r'(.)\1{2,}')


def clean_text(s):
    """Strip markup, URLs, emails, then diacritics/tatweel."""
    s = RE_URL.sub(" ", s)
    s = RE_EMAIL.sub(" ", s)
    s = RE_TAG.sub(" ", s)
    s = RE_WIKI.sub(" ", s)
    s = RE_ENTITY.sub(" ", s)
    s = RE_DIACRITIC.sub("", s)
    return s


def tokenize(s):
    """Clean then return list of Arabic-script tokens (unfiltered by length/elongation)."""
    return RE_TOKEN.findall(clean_text(s))


def token_ok(t):
    """Apply the drop rules. Returns None if OK, else the reason string."""
    if len(t) > MAX_LEN:
        return "too_long"
    if RE_ELONG.search(t):
        return "elongation"
    return None


# ---------------- corpus readers ----------------

def read_txt(path, encoding="utf-8"):
    with io.open(path, "r", encoding=encoding, errors="replace") as f:
        for line in f:
            yield line


def read_gz(path, encoding="utf-8"):
    with gzip.open(path, "rt", encoding=encoding, errors="replace") as f:
        for line in f:
            yield line


def read_parquet(path, column="text", batch_size=2000):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    cols = pf.schema_arrow.names
    col = column if column in cols else ("text" if "text" in cols else cols[0])
    for batch in pf.iter_batches(batch_size=batch_size, columns=[col]):
        for v in batch.column(0).to_pylist():
            if v:
                yield v


def read_any(path):
    low = path.lower()
    if low.endswith(".parquet"):
        return read_parquet(path)
    if low.endswith(".gz"):
        return read_gz(path)
    return read_txt(path)
