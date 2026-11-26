"""Decode MuPT's synchronized ABC without serializing parallel voices.

The <|> token delimits simultaneous bar blocks; barlines inside a block
separate voices by position. Empty duplicate delimiters are ignored.
"""
from __future__ import annotations
from fractions import Fraction
import re


def decode_mupt(text):
    text = text.replace("<n>", "\n").strip()
    if "<|>" not in text:
        return text, {"format": "standard_abc", "synchronized": False}
    header, *chunks = text.split("<|>")
    if not re.search(r"(?m)^K:", header):
        raise ValueError("MuPT synchronized ABC requires a key header before <|>")
    meter = re.search(r"(?m)^M:\s*(\S+)", header)
    meter = meter.group(1) if meter else "4/4"
    meter = {"C": "4/4", "C|": "2/2"}.get(meter, meter)
    unit = re.search(r"(?m)^L:\s*(\S+)", header)
    unit = Fraction(unit.group(1)) if unit else (Fraction(1, 16) if Fraction(meter) < Fraction(3, 4) else Fraction(1, 8))
    rest_length = Fraction(meter) / unit
    rest = "z" + (str(rest_length.numerator) if rest_length.denominator == 1 else str(rest_length))
    names = re.findall(r"(?m)^V:\s*(\S+)", header)
    voices = [[] for _ in names]
    block_count = missing = added = 0
    for chunk in chunks:
        if not chunk.strip():
            continue
        # Protect quoted chord/annotation strings containing literal barlines.
        quoted = []
        def protect(match):
            quoted.append(match.group(0))
            return f"@@ANNOTATION{len(quoted)-1}@@"
        protected = re.sub(r'"[^"\n]*"', protect, chunk.strip())
        # Opening repeat/bar markers belong to the first voice, not an empty
        # leading voice. A marker-only block contains no measure of music.
        protected = re.sub(r"^(?:(?:\|:|:\||\|\]|\|\||\|)\s*)+", "", protected)
        if not protected.strip():
            continue
        parts = re.split(r"\|\]|\|\||:\||\|:|\|", protected)
        if parts and not parts[-1].strip():
            parts.pop()
        while len(voices) < len(parts):
            voices.append([rest + " |"] * block_count)
            added += 1
        missing += max(0, len(voices)-len(parts))
        for i, target in enumerate(voices):
            value = parts[i].strip() if i < len(parts) else rest
            value = re.sub(r"@@ANNOTATION(\d+)@@", lambda m: quoted[int(m.group(1))], value)
            # Inline voice markers carry no new timing in synchronized format.
            value = re.sub(r"\[V:[^\]]+\]", "", value).strip()
            target.append((value or rest) + " |")
        block_count += 1
    if not block_count:
        raise ValueError("MuPT returned no nonempty bar blocks")
    # Use unique positional voice names; retain original declarations separately
    # in audit metadata, since missing/new voices must not be matched by pitch.
    header = re.sub(r"(?m)^V:.*(?:\n|$)", "", header).rstrip()
    standard = header + "\n" + "\n".join(
        f"V:{i+1}\n" + "\n".join(bars) for i, bars in enumerate(voices)) + "\n"
    return standard, {"format": "mupt_synchronized_abc", "synchronized": True,
        "bar_blocks": block_count, "voices": len(voices), "declared_voice_ids": names,
        "missing_voice_bars": missing, "added_voices": added,
        "trailing_block_closed": not bool(chunks[-1].strip()),
        "empty_delimiters": sum(not c.strip() for c in chunks),
        "missing_voice_policy": "full_meter_rest", "repeat_policy": "as_written"}
