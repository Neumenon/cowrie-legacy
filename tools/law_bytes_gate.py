#!/usr/bin/env python3
"""Byte-space canonicality law (SPEC-v1 §0.2 / §5.3).

    for all byte strings B:
        if strict_decode(B) = V then encode(V) = B

Equivalently, expressed through the `recode` interface every implementation already exposes:

    recode --strict on B must either FAIL, or echo B back unchanged.

Echoing *different* bytes means strict decode accepted a non-canonical encoding and silently
normalised it — the exact failure this gate exists to catch.

The existing fixtures (testdata/v1_negative.json) state the law for 27 hand-picked inputs. This
gate states it for a generated population: every golden value's canonical encoding, mutated.

Mutation classes:
  * dictionary pollution — append a header-dictionary entry that sorts last and is referenced by
    nothing. A strict decoder must reject it (§2.4: "no unused entries").
  * byte mutation       — single-byte flips / increments / deletions at deterministic positions.
  * truncation          — the message cut short at deterministic offsets.

Usage:  python3 tools/law_bytes_gate.py [mutants_per_base]
"""
from __future__ import annotations

import json
import os
import random
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import cowrie_ref as ref  # noqa: E402

GOLDEN = json.load(open(os.path.join(ROOT, "testdata", "v1_golden.json")))
GO_CLI = os.environ.get("GO_CLI", "/tmp/cowrie-cli")
RUST = os.environ.get("RUST_RECODE", os.path.join(ROOT, "rust/target/release/examples/recode"))
TS_DIR = os.path.join(ROOT, "typescript")
TSX = os.path.join(TS_DIR, "node_modules/.bin/tsx")

IMPLS = {
    "go":   ([GO_CLI, "recode", "--strict"], None),
    "rust": ([RUST, "--strict"], None),
    "ts":   ([TSX, "recode.ts", "--strict"], TS_DIR),
}
_CODE = re.compile(rb"ERR_[A-Z0-9_]+")


def probe(cmd: list[str], raw: bytes, cwd: str | None) -> tuple[str, str]:
    p = subprocess.run(cmd, input=raw, capture_output=True, cwd=cwd)
    if p.returncode != 0 or not p.stdout:
        m = _CODE.search(p.stderr)
        return "REJECT", (m.group(0).decode() if m else "")
    return ("ACCEPT" if p.stdout == raw else "NORMALIZE"), ""


def canonical_bases() -> list[tuple[str, bytes]]:
    """Every golden value, re-encoded canonically by the reference."""
    out: list[tuple[str, bytes]] = []
    for name, vec in GOLDEN.items():
        if not isinstance(vec, dict):
            continue
        hexs = vec.get("canonical_hex") or vec.get("hex") or vec.get("encoded")
        if not hexs:
            continue
        try:
            raw = bytes.fromhex(hexs)
            out.append((name, ref.encode(ref.decode(raw, strict=True))))
        except Exception:
            continue
    return out


def dict_offset(b: bytes) -> int:
    """Offset of the dictionary-length uvarint (magic 4 + version 1 + compression 1)."""
    return 6


def pollute_dictionary(b: bytes) -> bytes | None:
    """Append a dictionary entry that sorts last and is referenced by nothing."""
    if len(b) < 8 or b[:4] != b"COWR":
        return None
    pos = dict_offset(b)
    n = b[pos]
    if n >= 0x7F:                       # multi-byte uvarint count: skip (not our population)
        return None
    p = pos + 1
    keys = []
    for _ in range(n):
        if p >= len(b):
            return None
        klen = b[p]
        if klen >= 0x7F:
            return None
        keys.append(b[p + 1 : p + 1 + klen])
        p += 1 + klen
    # a key that sorts strictly after every existing one, and is unique
    if keys:
        new_key = max(keys) + b"~"
    else:
        new_key = b"~"
    if len(new_key) >= 0x7F:
        return None
    entry = bytes([len(new_key)]) + new_key
    out = b[:pos] + bytes([n + 1]) + b[pos + 1 : p] + entry + b[p:]
    return out


def mutations(base: bytes, per_base: int, rng: random.Random) -> list[tuple[str, bytes]]:
    out: list[tuple[str, bytes]] = []
    polluted = pollute_dictionary(base)
    if polluted is not None:
        out.append(("dict-pollution", polluted))
    n = len(base)
    for i in range(per_base):
        op = i % 4
        if op == 0 and n > 0:
            pos = rng.randrange(n)
            m = bytearray(base)
            m[pos] ^= 1 << rng.randrange(8)
            out.append((f"flip@{pos}", bytes(m)))
        elif op == 1 and n > 0:
            pos = rng.randrange(n)
            m = bytearray(base)
            m[pos] = (m[pos] + rng.randrange(1, 5)) & 0xFF
            out.append((f"inc@{pos}", bytes(m)))
        elif op == 2 and n > 1:
            pos = rng.randrange(n)
            out.append((f"del@{pos}", base[:pos] + base[pos + 1 :]))
        elif op == 3 and n > 1:
            pos = rng.randrange(1, n)
            out.append((f"trunc@{pos}", base[:pos]))
    return out


def main() -> int:
    per_base = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    rng = random.Random(20261103)
    bases = canonical_bases()
    if not bases:
        print("!! no golden bases found")
        return 1

    print(f"Byte-space canonicality law — {len(bases)} canonical bases x ~{per_base} mutations")
    print("rule: strict recode must REJECT, or echo the input unchanged\n")

    total = 0
    violations: list[tuple[str, str, str]] = []
    controls = 0
    stats = {k: {"REJECT": 0, "ACCEPT": 0, "NORMALIZE": 0} for k in IMPLS}
    stats["python"] = {"REJECT": 0, "ACCEPT": 0, "NORMALIZE": 0}

    for name, base in bases:
        # CONTROL (§5.3 control precedence): the unmutated canonical encoding must be accepted,
        # and echoed unchanged, by every implementation. A decoder that rejects everything would
        # otherwise score perfectly on the mutation half of this gate.
        controls += 1
        try:
            v0 = ref.decode(base, strict=True)
            if ref.encode(v0) != base:
                violations.append((name, "control/canonical", "python"))
        except Exception:
            violations.append((name, "control/canonical", "python"))
        for lang, (cmd, cwd) in IMPLS.items():
            beh, code = probe(cmd, base, cwd)
            if beh != "ACCEPT":
                violations.append((name, "control/canonical", lang))
        for label, mutant in mutations(base, per_base, rng):
            total += 1
            # python reference, in process
            try:
                v = ref.decode(mutant, strict=True)
                stats["python"]["ACCEPT" if ref.encode(v) == mutant else "NORMALIZE"] += 1
                if ref.encode(v) != mutant:
                    violations.append((name, label, "python"))
            except Exception:
                stats["python"]["REJECT"] += 1
            # the three out-of-process implementations
            for lang, (cmd, cwd) in IMPLS.items():
                beh, _code = probe(cmd, mutant, cwd)
                stats[lang][beh] += 1
                if beh == "NORMALIZE":
                    violations.append((name, label, lang))

    print(f"{'impl':10}{'REJECT':>9}{'ACCEPT':>9}{'NORMALIZE':>11}")
    for lang, s in stats.items():
        print(f"{lang:10}{s['REJECT']:>9}{s['ACCEPT']:>9}{s['NORMALIZE']:>11}")

    print(f"\n{total} mutants tested, {controls} canonical controls")
    if violations:
        print(f"\n❌ {len(violations)} canonicality failures (a control that rejects canonical input, or\n   a mutant accepted with different bytes — both are defects):")
        seen = set()
        for name, label, lang in violations[:20]:
            key = (name, label)
            if key in seen:
                continue
            seen.add(key)
            print(f"   {name:24} {label:16} {lang}")
        return 1
    print("\n✅ law holds: every strict-accepted mutant was already canonical")
    return 0


if __name__ == "__main__":
    sys.exit(main())
