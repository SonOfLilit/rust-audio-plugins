#!/usr/bin/env python3
"""Bring SocaLabs Wavetable's factory presets and wavetables into bloop.

Reads the pinned `vendor/wavetable` checkout (fetch it with
`git -c submodule.vendor/wavetable.update=checkout submodule update --init
vendor/wavetable`, or point `--vendor` at another checkout of it) and
writes, under this crate:

- `wavetables/<Bank>/<name>.<ext>`: every wavetable, packed by
  `examples/pack_tables.rs` (which this runs through cargo): each in the
  smaller of Wavetable's own file (`.wt2048`, a 16-bit FLAC of 2048-sample
  frames) and the same samples with each frame stored as its difference
  from the one before (`.wtd`, `bank::Format::FlacDelta`), kept only when it
  decodes to the original bit for bit; and the tables presets carry inside
  themselves, under `Presets/`, packed the same way from the WAV inside;
- `wavetables/catalogue.tsv`: every one of them, in the order the Wavetable
  knob steps through them, with its frame size and whether the web build
  carries it;
- `src/presets_data.rs`: every factory preset, as Rust data.

What ships is what SocaLabs ships. Wavetable's installers put every bank
of `plugin/Resources/WavetablesFLAC` and every preset of
`plugin/Resources/Presets` beside the plug-in (`CMakeLists.txt`'s
`install`, `Installer/`), and bloop takes it that SocaLabs has a licence to
everything it distributes, so bloop ships all of it, credited as Wavetable
credits it (`LICENSE`): the Adventure Kid bank, made from Kristoffer
Ekstrand's Adventure Kid Waveforms (CC0); Neu KatalYst's bank, whose
author says to use it in any synth; BVKER's free Serum tables (Analog,
Distorted, FM, Growl and Hyper); and the presets by their authors, with
the tables some of them carry.

The web build carries less (`web_choice`): every table a preset plays
would put ~29 MB into the wasm, where the Adventure Kid bank and the eight
presets bloop shipped first took ~8 MB. It carries those, and then the
tables of as many more presets as fit in `WEB_BUDGET`. Every table and
every preset is still listed on the web, so the Browser and the Wavetable
knob show the same rows everywhere (the manual's gold tables are shared by
native and wasm); one the web lacks plays silent there.

Run it again after moving the `vendor/wavetable` pin. The output is
deterministic.
"""

import argparse
import hashlib
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
CRATE = os.path.dirname(HERE)
WORKSPACE = os.path.dirname(os.path.dirname(CRATE))
DEFAULT_VENDOR = os.path.join(WORKSPACE, "vendor", "wavetable")

# Where the tables presets carry are filed in the catalogue.
EMBEDDED_BANK = "Presets"

# The two tables a fresh Wavetable plays. The original's own defaults are
# Analog bank tables; bloop's Init has played these since before it
# shipped that bank, and changing them would change every Init.
INIT_TABLES = ("AKWP bw_saw", "AKWP bw_squ")

# What the web build carries (`web_choice`): this bank whole, the tables
# presets carry inside themselves, these tables (the manual's Wavetable
# position page sweeps NK - INERT), what these presets play (the presets
# bloop shipped first, which the tests and the manual play), and then as
# many more presets' tables as fit in the budget, cheapest first.
WEB_BANK = "Adventure Kid"
WEB_TABLES = ["NK - INERT"]
WEB_PRESETS = {
    "DT Siren 1",
    "DT Siren Complex 2",
    "DT Siren complex",
    "DT Sweeper 6",
    "VOX Dist Vocal I",
    "Guitar Pad",
    "Noise Riserish",
    "Whip",
}
WEB_BUDGET = 12_000_000

# Each preset's TYPE in the Browser comes from the first of its own tags
# that names one (`TAG_TYPES`), or from `TYPES` where the tags say nothing
# or say something else.
TAG_TYPES = {
    "kick": "Drums",
    "snare": "Drums",
    "drum": "Drums",
    "drums": "Drums",
    "hat": "Drums",
    "percussion": "Percussion",
    "bass": "Bass",
    "basses": "Bass",
    "fx": "Sound Effects",
    "sfx": "Sound Effects",
    "effect": "Sound Effects",
    "riser": "Sound Effects",
    "sweeper": "Sound Effects",
    "siren": "Sound Effects",
    "dive": "Sound Effects",
    "glitch": "Sound Effects",
    "glitchy": "Sound Effects",
    "lead": "Synth Lead",
    "leads": "Synth Lead",
    "pad": "Synth Pad",
    "key": "Piano / Keys",
    "keys": "Piano / Keys",
    "brass": "Brass",
    "string": "Bowed Strings",
    "pluck": "Synth Misc",
    "plucks": "Synth Misc",
    "plucked": "Synth Misc",
    "mallet": "Mallet Instruments",
    "bells": "Mallet Instruments",
    "vocalise": "Vocal",
    "vocals": "Vocal",
    "chord": "Synth Misc",
    "arp": "Synth Misc",
    "arpeggio": "Synth Misc",
    "sequence": "Synth Misc",
    "stab": "Synth Misc",
    "synth": "Synth Misc",
    "gate": "Synth Misc",
    "initial": "Synth Misc",
}
TYPES = {
    # A siren that is a bass line.
    "DT Siren Complex 2": "Bass",
    "DT Siren complex": "Bass",
    # Tagged only "wip".
    "DT Cymbal intro": "Drums",
    # No tags.
    "DT Noisy Pluck": "Synth Misc",
    # Tagged "Lead", and what they imitate.
    "LD Accordion I": "Reed Instruments",
    "LD EGuitar Dist I": "Guitar",
    # Tagged "Plucked": struck, not plucked.
    "PL Cow Bell II": "Percussion",
    "PL Marimba I": "Mallet Instruments",
    # Tagged "Pluck", and plucked strings.
    "Cimbalom": "Plucked Strings",
    "Shamisen": "Plucked Strings",
    "SteelS": "Plucked Strings",
    "NylonSteel": "Plucked Strings",
    # Tagged "Key", and organs.
    "CrazyOrgan": "Organ",
    "SomeChurch": "Organ",
    "IV's Organ Advanced": "Organ",
    "IV's Organ Basic": "Organ",
    "IV's Organ Superchorus": "Organ",
    # Tagged "Lead", and a flute.
    "Flutish": "Flute",
    # Tagged "Synth", and an organ.
    "Space Organ": "Organ",
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


def type_of(name, tags):
    if name in TYPES:
        return TYPES[name]
    for word in tags.lower().split():
        if word in TAG_TYPES:
            return TAG_TYPES[word]
    sys.exit(f"preset {name!r} (tags {tags!r}) has no TYPE: add it to TYPES")


def pack(jobs, out_tables):
    """Run examples/pack_tables.rs over (source, destination stem, frame)
    jobs; return (file, format, bytes) for each, in order."""
    lines = "".join(f"{src}\t{dst}\t{frame}\n" for src, dst, frame in jobs)
    run = subprocess.run(
        [
            "cargo",
            "run",
            "-q",
            "--release",
            "-p",
            "bloop-plugin-wavetable",
            "--example",
            "pack_tables",
            "--",
            out_tables,
        ],
        input=lines,
        capture_output=True,
        text=True,
        cwd=WORKSPACE,
        check=False,
    )
    if run.returncode != 0:
        sys.exit(f"pack_tables failed:\n{run.stderr}")
    rows = [line.split("\t") for line in run.stdout.splitlines()]
    if len(rows) != len(jobs):
        sys.exit(f"pack_tables packed {len(rows)} of {len(jobs)} tables:\n{run.stderr}")
    return [(f, fmt, int(n)) for f, fmt, n in rows]


def web_choice(presets, size_of, bank_tables, embedded):
    """The tables the web build carries: `WEB_BANK`, the tables presets
    carry inside themselves, `WEB_TABLES`, what `WEB_PRESETS` play, and then
    preset by preset, the one that costs least in tables not yet carried
    first, every preset's that fits in `WEB_BUDGET`."""
    carried = {("bank", n) for bank, n in bank_tables if bank == WEB_BANK}
    carried |= {("embedded", e[4]) for e in embedded}
    carried |= {("bank", n) for n in WEB_TABLES}
    for p in presets:
        if p["name"] in WEB_PRESETS:
            carried |= p["plays"]
    missing = WEB_PRESETS - {p["name"] for p in presets}
    if missing:
        sys.exit(f"WEB_PRESETS names presets there are not: {sorted(missing)}")
    total = sum(size_of[t] for t in carried)
    while True:
        costs = [
            (sum(size_of[t] for t in p["plays"] - carried), at)
            for at, p in enumerate(presets)
            if not p["plays"] <= carried
        ]
        if not costs:
            break
        cost, at = min(costs)
        if total + cost > WEB_BUDGET:
            break
        carried |= presets[at]["plays"]
        total += cost
    return carried


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

    # Every bank, in the order the original lists them: by folder, then by
    # name, naturally.
    banks = sorted(os.listdir(flac_root), key=natural_key)
    bank_tables = []  # (bank, name)
    bank_of = {}
    for bank in banks:
        names = sorted(
            (
                f[: -len(".wt2048")]
                for f in os.listdir(os.path.join(flac_root, bank))
                if f.endswith(".wt2048")
            ),
            key=natural_key,
        )
        for n in names:
            if n in bank_of:
                sys.exit(f"{n} is in both {bank_of[n]} and {bank}")
            bank_of[n] = bank
            bank_tables.append((bank, n))

    presets = []
    embedded = []  # (name, file stem, frame size, bytes, digest)
    for author in sorted(os.listdir(preset_root), key=natural_key):
        folder = os.path.join(preset_root, author)
        for f in sorted(os.listdir(folder), key=natural_key):
            if not f.endswith(".xml"):
                continue
            root = ET.parse(os.path.join(folder, f)).getroot()
            state = root.find("state")
            params = {e.get("uid"): e.get("val") for e in root.iter("param")}
            tables = {}  # oscillator -> ("bank", name) or ("embedded", digest)
            for i in (1, 2):
                name = state.get(f"wt{i}")
                data = juce_base64(state.get(f"wt{i}Data"))
                if data:
                    size = int(state.get(f"wt{i}Size", "-1") or -1)
                    if size <= 0:
                        size = wavetable_size(data)
                    if size <= 0:
                        sys.exit(f"{author}/{f}: wt{i} names no frame size")
                    digest = hashlib.sha256(data).hexdigest()
                    if not any(e[4] == digest for e in embedded):
                        stem = safe_file(name)
                        if any(e[1] == stem for e in embedded):
                            stem += " " + digest[:8]
                        embedded.append((name, stem, size, data, digest))
                    tables[i] = ("embedded", digest)
                elif name in bank_of:
                    tables[i] = ("bank", name)
                else:
                    sys.exit(f"{author}/{f}: wt{i} is {name!r}, which no bank has")
            for m in root.iter("MODITEM"):
                for attr in m.attrib:
                    if attr not in ("srcId", "dstId", "depth", "enabled"):
                        sys.exit(
                            f"{author}/{f}: a route sets {attr}, which bloop does not model"
                        )
            mods = [
                (m.get("srcId"), m.get("dstId"), m.get("depth"))
                for m in root.iter("MODITEM")
                if m.get("enabled", "1") != "0"
            ]
            name = root.get("name")
            tags = root.get("tags", "")
            enabled = {
                1: params.get("osc1enable", "1.0") != "0.0",
                2: params.get("osc2enable", "0.0") != "0.0",
            }
            presets.append(
                {
                    "name": name,
                    "author": author,
                    "tags": tags,
                    "kind": type_of(name, tags),
                    "tables": tables,
                    "plays": {t for i, t in tables.items() if enabled[i]},
                    "params": [
                        (k, v) for k, v in params.items() if k not in EDITOR_ONLY
                    ],
                    "mods": mods,
                }
            )

    # The catalogue: the banks in Wavetable's order, then the embedded
    # tables in preset order.
    with tempfile.TemporaryDirectory() as scratch:
        jobs = [
            (os.path.join(flac_root, bank, n + ".wt2048"), os.path.join(bank, n), 2048)
            for bank, n in bank_tables
        ]
        for name, stem, size, data, digest in embedded:
            src = os.path.join(scratch, digest + ".wav")
            with open(src, "wb") as fh:
                fh.write(data)
            jobs.append((src, os.path.join(EMBEDDED_BANK, stem), size))
        # Packed beside, not in place: the packer is built from this crate,
        # whose build script reads the tables there now.
        staged = os.path.join(scratch, "wavetables")
        packed = pack(jobs, staged)
        out_tables = os.path.join(CRATE, "wavetables")
        if os.path.isdir(out_tables):
            shutil.rmtree(out_tables)
        shutil.copytree(staged, out_tables)

    size_of = {
        ("bank", n): size for (_b, n), (_f, _fmt, size) in zip(bank_tables, packed)
    }
    for e, (_f, _fmt, size) in zip(embedded, packed[len(bank_tables) :]):
        size_of[("embedded", e[4])] = size
    web_tables = web_choice(presets, size_of, bank_tables, embedded)
    catalogue = []  # (bank, name, file, format, frame, web, bytes)
    for (bank, n), (file, fmt, size) in zip(bank_tables, packed):
        catalogue.append((bank, n, file, fmt, 2048, ("bank", n) in web_tables, size))
    index_of_digest = {}
    for (name, _stem, frame, _data, digest), (file, fmt, size) in zip(
        embedded, packed[len(bank_tables) :]
    ):
        index_of_digest[digest] = len(catalogue)
        catalogue.append(
            (
                EMBEDDED_BANK,
                name,
                file,
                fmt,
                frame,
                ("embedded", digest) in web_tables,
                size,
            )
        )
    index_of_name = {}
    for i, (bank, name, *_rest) in enumerate(catalogue):
        if bank != EMBEDDED_BANK:
            index_of_name[name] = i

    with open(os.path.join(out_tables, "catalogue.tsv"), "w") as fh:
        fh.write(
            "# bank\tname\tfile\tformat\tframe\tweb — written by tools/import.py\n"
        )
        for bank, name, rel, fmt, size, web, _bytes in catalogue:
            fh.write(
                f"{bank}\t{name}\t{rel}\t{fmt}\t{size}\t{'web' if web else 'native'}\n"
            )

    def index(table):
        where, key = table
        return index_of_digest[key] if where == "embedded" else index_of_name[key]

    lines = [
        "// Written by tools/import.py from vendor/wavetable's presets. Do not edit.",
        "",
        "use crate::presets::Preset;",
        "",
        f"/// The wavetables a fresh Wavetable plays: `{INIT_TABLES[0]}` and `{INIT_TABLES[1]}`.",
        f"pub const INIT_TABLES: [u16; 2] = [{index_of_name[INIT_TABLES[0]]}, "
        f"{index_of_name[INIT_TABLES[1]]}];",
        "",
        "/// Wavetable's factory presets, by author then name.",
        "pub const PRESETS: &[Preset] = &[",
    ]
    for p in presets:
        tables = [
            str(index(p["tables"][i])) if i in p["tables"] else "u16::MAX"
            for i in (1, 2)
        ]
        lines.append("    Preset {")
        lines.append(f"        name: {rust_str(p['name'])},")
        lines.append(f"        author: {rust_str(p['author'])},")
        lines.append(f"        tags: {rust_str(p['tags'])},")
        lines.append(f"        kind: {rust_str(p['kind'])},")
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

    on_web = sum(1 for p in presets if p["plays"] <= web_tables)
    print(f"{on_web} of {len(presets)} presets play on the web")
    total = sum(c[6] for c in catalogue)
    web = sum(c[6] for c in catalogue if c[5])
    delta = sum(1 for c in catalogue if c[3] == "flac-delta")
    print(
        f"{len(presets)} presets, {len(catalogue)} wavetables in {len(banks)} banks and "
        f"{len(embedded)} preset tables: {total} bytes ({delta} as frame deltas); "
        f"the web carries {sum(1 for c in catalogue if c[5])} of them, {web} bytes"
    )


if __name__ == "__main__":
    main()
