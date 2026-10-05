#!/usr/bin/env python3
"""Bring SocaLabs Wavetable's factory presets and wavetables into bloop.

Reads the pinned `vendor/wavetable` checkout (fetch it with
`git -c submodule.vendor/wavetable.update=checkout submodule update --init
vendor/wavetable`) and writes, under this crate:

- `wavetables/<Bank>/<name>.<ext>`: the wavetables bloop may ship, copied
  byte for byte (FLAC, 2048 samples a frame), and the ones a shipped preset
  carries inside itself, unpacked to WAV;
- `wavetables/catalogue.tsv`: every one of them, in the order the Wavetable
  knob steps through them, with its frame size and whether the web build
  carries it;
- `src/presets_data.rs`: the presets bloop may ship, as Rust data.

Which wavetables bloop may ship is a licence question, answered in
`SHIPPED_BANKS`: the Adventure Kid tables are made from AKWF, which its
author put in the public domain (CC0), and Neu KatalYst says of theirs "feel
free to use them in all your other synths ... no credit necessary". The
rest of Wavetable's banks — Analog, Distorted, FM, Growl and Hyper — are
BVKER's free Serum wavetables, whose user agreement forbids using them "to
create competitive products, such as sample packs or preset banks"; they
stay out, and so does every preset that plays one. A preset's own embedded
wavetable ships only if it is the preset author's own work
(`SHIPPED_EMBEDDED`): the ones exported from Serum or Vital, which say so in
their `clm ` chunk, may be those programs' factory tables and stay out.

Run it again after moving the `vendor/wavetable` pin, or after changing
what is cleared. The output is deterministic.
"""

import argparse
import hashlib
import os
import shutil
import struct
import sys
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
CRATE = os.path.dirname(HERE)
DEFAULT_VENDOR = os.path.join(CRATE, "..", "..", "..", "vendor", "wavetable")

# Banks bloop may ship, in the Wavetable knob's order.
SHIPPED_BANKS = ["Adventure Kid", "Neu KatalYst"]
# Embedded wavetables bloop may ship: (preset author folder, table name).
SHIPPED_EMBEDDED = {
    # "Recorded on 16.05.2020 in Edison": the preset author's own audio.
    ("Flamedragonz", "Untitled project - 2020-May-16_7"),
    ("Flamedragonz", "Untitled project - 2020-May-16_6"),
}
EMBEDDED_BANK = "Presets"

# The two tables a fresh Wavetable plays, the original's own defaults
# being BVKER's.
INIT_TABLES = ("AKWP bw_saw", "AKWP bw_squ")

# Each shipped preset's TYPE in the Browser — Komplete's instrument tags,
# by hand from the preset's own tags and name.
TYPES = {
    "DT Siren 1": "Sound Effects",
    "DT Siren Complex 2": "Bass",
    "DT Siren complex": "Bass",
    "DT Sweeper 6": "Sound Effects",
    "VOX Dist Vocal I": "Vocal",
    "Guitar Pad": "Synth Pad",
    "Noise Riserish": "Sound Effects",
    "Whip": "Sound Effects",
}

# Parameters a preset may name that bloop does not have: the editor's own.
EDITOR_ONLY = {"uiLFO", "uiENV", "uiMOD"} | {
    f"lfo{n}{g}" for n in (1, 2, 3) for g in ("xgrid", "ygrid")
}

JUCE_B64 = ".ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+"
JUCE_B64_LOOKUP = {c: i for i, c in enumerate(JUCE_B64)}


def juce_base64(s):
    """juce::MemoryBlock::fromBase64Encoding: '<size>.<chars>', six bits a
    character from JUCE's own alphabet, least significant bit first."""
    size_str, _, chars = (s or "").partition(".")
    size = int(size_str) if size_str else 0
    out = bytearray(size)
    pos = 0
    for c in chars:
        v = JUCE_B64_LOOKUP.get(c)
        if v is None:
            continue
        for k in range(6):
            bit = pos + k
            if bit // 8 >= size:
                break
            if v >> k & 1:
                out[bit // 8] |= 1 << (bit % 8)
        pos += 6
    return bytes(out)


def wavetable_size(wav):
    """gin::getWavetableSize: the frame size a Serum-style `clm ` chunk
    names (`<!>2048 ...`), or -1."""
    at = wav.find(b"clm ")
    if at < 0:
        return -1
    text = wav[at + 8 : at + 8 + 16]
    if not text.startswith(b"<!>"):
        return -1
    digits = b""
    for b in text[3:]:
        if 48 <= b <= 57:
            digits += bytes([b])
        else:
            break
    return int(digits) if digits else -1


def natural_key(s):
    """juce::String::compareNatural, near enough for these names: digit
    runs compare as numbers, the rest case-insensitively."""
    key = []
    run = ""
    for ch in s:
        if ch.isdigit():
            run += ch
        else:
            if run:
                key.append((0, int(run), ""))
                run = ""
            key.append((1, 0, ch.lower()))
    if run:
        key.append((0, int(run), ""))
    return key


def f32(text):
    """A preset's value as the float the original stores, written so Rust
    reads back that float."""
    v = struct.unpack("<f", struct.pack("<f", float(text)))[0]
    r = repr(v)
    if "e" not in r and "." not in r:
        r += ".0"
    return r


def rust_str(s):
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def safe_file(name):
    keep = "".join(c if c.isalnum() or c in " -_." else "_" for c in name).strip()
    return keep or "table"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--vendor", default=DEFAULT_VENDOR)
    args = ap.parse_args()
    vendor = os.path.abspath(args.vendor)
    resources = os.path.join(vendor, "plugin", "Resources")
    flac_root = os.path.join(resources, "WavetablesFLAC")
    preset_root = os.path.join(resources, "Presets")
    if not os.path.isdir(flac_root):
        sys.exit(f"{flac_root} is missing: check out vendor/wavetable first")

    out_tables = os.path.join(CRATE, "wavetables")
    if os.path.isdir(out_tables):
        shutil.rmtree(out_tables)
    os.makedirs(out_tables)

    # Every table any bank has, so a preset's name can be told shipped or not.
    bank_of = {}
    for bank in sorted(os.listdir(flac_root), key=natural_key):
        for f in os.listdir(os.path.join(flac_root, bank)):
            if f.endswith(".wt2048"):
                bank_of[f[: -len(".wt2048")]] = bank

    presets = []
    embedded = []  # (name, file, size, bytes)
    used_tables = set()
    for author in sorted(os.listdir(preset_root), key=natural_key):
        folder = os.path.join(preset_root, author)
        for f in sorted(os.listdir(folder), key=natural_key):
            if not f.endswith(".xml"):
                continue
            root = ET.parse(os.path.join(folder, f)).getroot()
            state = root.find("state")
            params = {e.get("uid"): e.get("val") for e in root.iter("param")}
            enabled = {
                1: params.get("osc1enable", "1.0") != "0.0",
                2: params.get("osc2enable", "0.0") != "0.0",
            }
            tables = {}
            shippable = True
            for i in (1, 2):
                name = state.get(f"wt{i}")
                data = juce_base64(state.get(f"wt{i}Data"))
                if data:
                    if (author, name) in SHIPPED_EMBEDDED:
                        size = int(state.get(f"wt{i}Size", "-1") or -1)
                        if size <= 0:
                            size = wavetable_size(data)
                        tables[i] = ("embedded", name, data, size)
                    elif enabled[i]:
                        shippable = False
                elif bank_of.get(name) in SHIPPED_BANKS:
                    tables[i] = ("bank", name, None, 2048)
                elif enabled[i]:
                    shippable = False
            if not shippable:
                continue
            for i, (where, name, data, size) in tables.items():
                if where == "embedded":
                    digest = hashlib.sha256(data).hexdigest()
                    if not any(e[4] == digest for e in embedded):
                        embedded.append((name, safe_file(name) + ".wav", size, data, digest))
                else:
                    used_tables.add(name)
            mods = [
                (m.get("srcId"), m.get("dstId"), m.get("depth"))
                for m in root.iter("MODITEM")
                if m.get("enabled", "1") != "0"
            ]
            for m in root.iter("MODITEM"):
                for attr in m.attrib:
                    if attr not in ("srcId", "dstId", "depth", "enabled"):
                        sys.exit(f"{author}/{f}: a route sets {attr}, which bloop does not model")
            presets.append(
                {
                    "name": root.get("name"),
                    "author": author,
                    "tags": root.get("tags", ""),
                    "tables": {i: t[1] for i, t in tables.items()},
                    "params": [(k, v) for k, v in params.items() if k not in EDITOR_ONLY],
                    "mods": mods,
                }
            )

    # The catalogue: the shipped banks in Wavetable's order, then the
    # embedded tables in preset order.
    catalogue = []
    for bank in SHIPPED_BANKS:
        os.makedirs(os.path.join(out_tables, bank))
        names = sorted(
            (n for n, b in bank_of.items() if b == bank), key=natural_key
        )
        for n in names:
            src = os.path.join(flac_root, bank, n + ".wt2048")
            dst_rel = os.path.join(bank, n + ".wt2048")
            shutil.copyfile(src, os.path.join(out_tables, dst_rel))
            web = bank == "Adventure Kid" or n in used_tables
            catalogue.append((bank, n, dst_rel, "flac", 2048, web))
    if embedded:
        os.makedirs(os.path.join(out_tables, EMBEDDED_BANK))
    for name, file, size, data, _ in embedded:
        rel = os.path.join(EMBEDDED_BANK, file)
        with open(os.path.join(out_tables, rel), "wb") as fh:
            fh.write(data)
        catalogue.append((EMBEDDED_BANK, name, rel, "wav", size, True))

    index_of = {}
    for i, (bank, name, *_rest) in enumerate(catalogue):
        index_of.setdefault(name, i)

    with open(os.path.join(out_tables, "catalogue.tsv"), "w") as fh:
        fh.write("# bank\tname\tfile\tformat\tframe\tweb — written by tools/import.py\n")
        for bank, name, rel, fmt, size, web in catalogue:
            fh.write(f"{bank}\t{name}\t{rel}\t{fmt}\t{size}\t{'web' if web else 'native'}\n")

    lines = [
        "// Written by tools/import.py from vendor/wavetable's presets. Do not edit.",
        "",
        "use crate::presets::Preset;",
        "",
        f"/// The wavetables a fresh Wavetable plays: `{INIT_TABLES[0]}` and `{INIT_TABLES[1]}`.",
        f"pub const INIT_TABLES: [u16; 2] = [{index_of[INIT_TABLES[0]]}, {index_of[INIT_TABLES[1]]}];",
        "",
        "/// Wavetable's factory presets that bloop ships, by author then name.",
        "pub const PRESETS: &[Preset] = &[",
    ]
    for p in presets:
        tables = [
            str(index_of[p["tables"][i]]) if i in p["tables"] else "u16::MAX" for i in (1, 2)
        ]
        lines.append("    Preset {")
        lines.append(f"        name: {rust_str(p['name'])},")
        lines.append(f"        author: {rust_str(p['author'])},")
        lines.append(f"        tags: {rust_str(p['tags'])},")
        lines.append(f"        kind: {rust_str(TYPES.get(p['name'], ''))},")
        lines.append(f"        tables: [{tables[0]}, {tables[1]}],")
        lines.append("        params: &[")
        for k, v in p["params"]:
            lines.append(f"            ({rust_str(k)}, {f32(v)}),")
        lines.append("        ],")
        lines.append("        mods: &[")
        for s, d, depth in p["mods"]:
            lines.append(f"            ({rust_str(s)}, {rust_str(d)}, {f32(depth)}),")
        lines.append("        ],")
        lines.append("    },")
    lines.append("];")
    with open(os.path.join(CRATE, "src", "presets_data.rs"), "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"{len(presets)} presets, {len(catalogue)} wavetables "
          f"({sum(1 for c in catalogue if c[5])} on the web)")


if __name__ == "__main__":
    main()
