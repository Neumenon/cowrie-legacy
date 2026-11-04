#!/usr/bin/env python3
"""Constructor gate: build the same value in every implementation and compare the bytes.

Every other cross-language gate feeds an implementation bytes it already had (decode → re-encode).
That proves the decoders agree. It says nothing about the *encoders*, because in Go, Rust and
TypeScript a value built in code never otherwise reaches the wire.

This gate asks each implementation to construct a fixed case list itself and print `name<TAB>hex`.
The Python reference is the oracle. Any difference is either a wrong encoding or a self-reject.

Run: python3 tools/constructor_gate.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import cowrie_ref as ref  # noqa: E402
from cowrie_ref import Decimal128  # noqa: E402

GO_CLI = os.environ.get("GO_CLI", "/tmp/cowrie-cli")
RUST = os.environ.get("RUST_RECODE", os.path.join(ROOT, "rust/target/release/examples/recode"))
TS_DIR = os.path.join(ROOT, "typescript")
TSX = os.path.join(TS_DIR, "node_modules/.bin/tsx")


def _dec(scale: int, coef0: int) -> Decimal128:
    b = bytearray(16)
    b[0] = coef0
    return Decimal128(coefficient=int.from_bytes(bytes(b), "little"), scale=scale)


def oracle() -> dict[str, bytes]:
    """The reference builds the same cases; its bytes are the answer key."""
    return {
        "null": ref.encode(None),
        "bool_true": ref.encode(True),
        "int_5": ref.encode(5),
        "int_neg17": ref.encode(-17),
        "int_127": ref.encode(127),
        "uint_42": ref.encode(42),                 # a uint that fits Int is an Int
        "uint_2p63": ref.encode(2**63),
        "bigint_1": ref.encode(1),                 # a BigInt that fits Int is an Int
        "bigint_2p64": ref.encode(2**64),
        "dec_2_150": ref.encode(_dec(2, 150)),     # unreduced -> scale 1, coefficient 15
        "dec_2_42": ref.encode(_dec(2, 42)),
        "float_nan": ref.encode(float("nan")),
        "float_negzero": ref.encode(-0.0),
        "str_unicode": ref.encode("héllo"),
        "bytes": ref.encode(b"\x00\x01\x02"),
        "dt_neg": ref.encode(ref.Datetime(-1)),
        "arr": ref.encode([1, 2, 3]),
        "obj": ref.encode({"a": 1, "b": 2}),
    }


IMPLS = {
    "go": ([GO_CLI, "cases"], None),
    "rust": ([RUST, "--cases"], None),
    "ts": ([TSX, "recode.ts", "--cases"], TS_DIR),
}


def run_cases(cmd, cwd) -> dict[str, bytes]:
    p = subprocess.run(cmd, capture_output=True, cwd=cwd)
    if p.returncode != 0:
        msg = p.stderr.decode("utf-8", "replace").strip().splitlines()
        raise RuntimeError(f"exit {p.returncode}: {msg[0] if msg else 'no stderr'}")
    out = {}
    for line in p.stdout.decode().splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            try:
                out[parts[0]] = bytes.fromhex(parts[1])
            except ValueError:
                continue
    return out


def main() -> int:
    want = oracle()
    names = list(want)
    failures: list[str] = []

    print(f"Constructor gate — {len(names)} values built in code, compared to the reference\n")
    print(f"{'case':14}" + "".join(f"{k:>8}" for k in IMPLS))
    for name in names:
        row = f"{name:14}"
        for lang, (cmd, cwd) in IMPLS.items():
            try:
                got = run_cases(cmd, cwd).get(name)
            except Exception as exc:  # noqa: BLE001
                row += f"{'ERR':>8}"
                failures.append(f"{lang}/{name}: {exc}")
                continue
            if got is None:
                row += f"{'MISSING':>8}"
                failures.append(f"{lang}/{name}: no case emitted")
            elif got == want[name]:
                row += f"{'ok':>8}"
            else:
                row += f"{'DIFF':>8}"
                failures.append(f"{lang}/{name}: {got.hex()} != {want[name].hex()} (oracle)")
        print(row)

    print()
    if failures:
        print(f"❌ {len(failures)} constructor divergences:")
        for f in failures[:24]:
            print("   ", f)
        return 1
    print("✅ every implementation constructs the same bytes as the reference")
    return 0


if __name__ == "__main__":
    sys.exit(main())
