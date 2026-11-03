"""The byte-space canonicality law, stated in the reference's own test suite (SPEC §0.2 / §5.3).

    for all B:  if strict_decode(B) = V  then  encode(V) = B

The curated fixtures in testdata/v1_negative.json are examples of this law. This test states it
for a generated population: every golden value's canonical encoding, mutated. The mutation that
matters most is dictionary pollution — an entry appended to the header dictionary that no object
field references (§2.4 forbids it), which strict decode used to accept and silently normalise.
"""
from __future__ import annotations

import json
import os
import random

import pytest

import cowrie_ref as ref

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while not os.path.exists(os.path.join(ROOT, "testdata", "v1_golden.json")):
    parent = os.path.dirname(ROOT)
    if parent == ROOT:
        raise RuntimeError("repo root (testdata/v1_golden.json) not found")
    ROOT = parent
GOLDEN = json.load(open(os.path.join(ROOT, "testdata", "v1_golden.json")))


def canonical_bases() -> list[bytes]:
    out = []
    for _name, vec in GOLDEN.items():
        if not isinstance(vec, dict):
            continue
        hexs = vec.get("canonical_hex")
        if not hexs:
            continue
        try:
            out.append(ref.encode(ref.decode(bytes.fromhex(hexs), strict=True)))
        except Exception:
            continue
    return out


def pollute(b: bytes) -> bytes | None:
    """Append a header-dictionary entry that sorts last and is referenced by nothing."""
    if len(b) < 8 or b[:4] != b"COWR":
        return None
    pos = 6
    n = b[pos]
    if n >= 0x7F:
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
    new_key = (max(keys) if keys else b"") + b"~"
    if len(new_key) >= 0x7F:
        return None
    return b[:pos] + bytes([n + 1]) + b[pos + 1 : p] + bytes([len(new_key)]) + new_key + b[p:]


def mutations(base: bytes, per_base: int, rng: random.Random) -> list[bytes]:
    out = []
    p = pollute(base)
    if p is not None:
        out.append(p)
    n = len(base)
    for i in range(per_base):
        op = i % 3
        if op == 0 and n:
            pos = rng.randrange(n)
            m = bytearray(base)
            m[pos] ^= 1 << rng.randrange(8)
            out.append(bytes(m))
        elif op == 1 and n > 1:
            pos = rng.randrange(n)
            out.append(base[:pos] + base[pos + 1 :])
        elif op == 2 and n > 1:
            out.append(base[: rng.randrange(1, n)])
    return out


@pytest.mark.parametrize("per_base", [6])
def test_strict_accept_implies_canonical(per_base: int) -> None:
    rng = random.Random(20261103)
    bases = canonical_bases()
    assert bases, "no canonical bases found"
    checked = 0
    for base in bases:
        for mutant in mutations(base, per_base, rng):
            checked += 1
            try:
                v = ref.decode(mutant, strict=True)
            except Exception:
                continue  # rejection is always lawful
            assert ref.encode(v) == mutant, (
                f"strict decode accepted non-canonical bytes: {mutant.hex()} "
                f"(canonical is {ref.encode(v).hex()})"
            )
    assert checked > 100


def test_unused_dictionary_entry_is_rejected() -> None:
    """The case that motivated the law: a redundant dictionary entry (§2.4)."""
    scalar = bytes.fromhex("434f5752010001016145")
    obj = bytes.fromhex("434f575201000201610162d10045")

    for raw in (scalar, obj):
        with pytest.raises(Exception) as exc:
            ref.decode(raw, strict=True)
        assert "NON_CANONICAL" in str(exc.value)

    # lenient decode still accepts and normalises — that is the documented behaviour
    assert ref.decode(scalar, strict=False) == 5
    assert ref.decode(obj, strict=False) == {"a": 5}
