"""Simulate OCR / transcription errors without a real OCR engine."""

from __future__ import annotations

import re
from typing import Mapping

# Common OCR character confusions
_CONFUSIONS: Mapping[str, str] = {
    "0": "O",
    "O": "0",
    "1": "l",
    "l": "1",
    "I": "1",
    "5": "S",
    "S": "5",
    "8": "B",
    "B": "8",
    "rn": "m",
    "cl": "d",
    "vv": "w",
}


def apply_ocr_noise(text: str, rng, *, char_error_rate: float = 0.04) -> str:
    """Inject character substitutions, deletions, and spacing glitches.

    Parameters
    ----------
    text : str
        Clean source text.
    rng : numpy.random.Generator
        Seeded RNG for reproducibility.
    char_error_rate : float
        Approximate per-character corruption probability.
    """
    if char_error_rate <= 0 or not text:
        return text

    out: list[str] = []
    i = 0
    while i < len(text):
        # Digraph confusions first
        if i + 1 < len(text) and rng.random() < char_error_rate * 0.3:
            digraph = text[i : i + 2]
            if digraph in _CONFUSIONS:
                out.append(_CONFUSIONS[digraph])
                i += 2
                continue

        ch = text[i]
        if ch.isspace():
            # Random extra/missing space
            if rng.random() < char_error_rate * 0.5:
                if rng.random() < 0.5:
                    out.append(ch + " ")
                # else: drop space
            else:
                out.append(ch)
        elif rng.random() < char_error_rate:
            kind = rng.integers(0, 3)
            if kind == 0 and ch in _CONFUSIONS:
                out.append(_CONFUSIONS[ch])
            elif kind == 1:
                # deletion — skip
                pass
            else:
                # insert nearby char
                out.append(ch)
                out.append(chr(ord(ch) + int(rng.integers(-2, 3))) if ch.isalnum() else ch)
        else:
            out.append(ch)
        i += 1

    noisy = "".join(out)
    # Occasional line-break insertion mid-token (layout OCR artifact)
    if rng.random() < char_error_rate * 2:
        tokens = noisy.split()
        if len(tokens) > 4:
            idx = int(rng.integers(1, len(tokens) - 1))
            tokens[idx] = tokens[idx][: max(1, len(tokens[idx]) // 2)] + "\n" + tokens[idx][max(1, len(tokens[idx]) // 2) :]
            noisy = " ".join(tokens)
    return noisy


def normalize_for_match(s: str) -> str:
    """Light normalization used when aligning noisy text to gold fields."""
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s
