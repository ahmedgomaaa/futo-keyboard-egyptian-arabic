# Install guide — Egyptian Arabic for FUTO Keyboard

Five minutes, no technical knowledge needed. You'll download two files and
import them into FUTO Keyboard on your Android phone.

You need:

1. `main_ar_eg.dict` — the dictionary (fixes autocorrect on Egyptian words)
2. `ggml-egyptian-phase3-q5_1.bin` — the voice model (fixes dictation)

Both are on the [latest release page](https://github.com/ahmedgomaaa/futo-keyboard-egyptian-arabic/releases/latest).

---

## Part 1 — The dictionary

This stops FUTO from flagging everyday Egyptian words (مش، عايز، ازاي،
دلوقتي، ايوه) as typos.

1. On your phone, download `main_ar_eg.dict` from the release page
   (your Downloads folder is fine).
2. Open the FUTO Keyboard app.
3. Go to **Languages & Models**.
4. Under **العربية (Arabic)**, tap **Dictionary**.
5. Choose **Import** and select `main_ar_eg.dict` from Downloads.
6. Confirm. It replaces the current main Arabic dictionary.

**Check it worked:** type `عايز` or `ازاي` — neither should get the red
typo underline, and both should appear as suggestions.

> **Heads up:** FUTO keeps a single main dictionary per language. If you
> later import an Arabic **emoji** dictionary, it overwrites this one —
> just re-import `main_ar_eg.dict` to bring the Egyptian vocabulary back.

## Part 2 — The voice model

This replaces FUTO's stock voice model (which was trained on Modern
Standard Arabic and mishears Egyptian speech) with one fine-tuned on real
spoken Egyptian.

1. On your phone, download `ggml-egyptian-phase3-q5_1.bin` from the release
   page (about 190 MB).
2. Open the FUTO Keyboard app.
3. Go to **Settings → Voice Input**.
4. Choose **Add / import model** and select the `.bin` file.
5. Set it as the **active model** for Arabic.

**Check it worked:** open any app, switch to the Arabic keyboard, tap the
mic, and say something like «ازيك عامل ايه؟» — it should transcribe it
correctly instead of producing broken MSA.

---

## Fixing problems

| Problem | Fix |
|---|---|
| Egyptian words still underlined | Re-import `main_ar_eg.dict` (an emoji dict or an update may have replaced it) |
| Mic doesn't use the new model | Settings → Voice Input → make sure the imported model is **active** for Arabic |
| Autocorrect still fights you | Check **Settings → Text prediction** — FUTO's transformer prediction can override the dictionary for suggestions; the dictionary stops the *typo* flagging regardless |
| Import button missing | Update FUTO Keyboard — dictionary/voice import needs a recent version |
| Voice model won't load | The `.bin` must be the `q5_1` file (~190 MB). Re-download if it was interrupted; don't rename it |
