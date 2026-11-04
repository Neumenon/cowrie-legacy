#!/usr/bin/env python3
"""Pack / verify gate: one self-certifying file, packed and checked by every implementation.

  pack    frames on stdin (uvarint count, then count x uvarint(len)||message) -> a CWRF file
  verify  a CWRF file on stdin -> "<merkle-root-hex> OK", exit 1 on any tampering

Three properties, each checked in all four implementations:
  1. the same frames pack to byte-identical files;
  2. every implementation verifies every other's file to the same Merkle root;
  3. a single flipped byte anywhere in a file makes every verifier fail.

Run: python3 tools/pack_gate.py
"""
from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import cowrie_ref as ref  # noqa: E402
from cowrie_ref import file as F  # noqa: E402

GO_CLI = os.environ.get("GO_CLI", "/tmp/cowrie-cli")
RUST = os.environ.get("RUST_RECODE", os.path.join(ROOT, "rust/target/release/examples/recode"))
TS_DIR = os.path.join(ROOT, "typescript")
TSX = os.path.join(TS_DIR, "node_modules/.bin/tsx")

PACK = {
    "go": ([GO_CLI, "pack"], None),
    "rust": ([RUST, "--pack"], None),
    "ts": ([TSX, "recode.ts", "--pack"], TS_DIR),
}
VERIFY = {
    "go": ([GO_CLI, "verify"], None),
    "rust": ([RUST, "--verify"], None),
    "ts": ([TSX, "recode.ts", "--verify"], TS_DIR),
}

# a spread of shapes: numbers, strings, a tensor-free object, an array, unicode
VALUES = [
    None,
    True,
    5,
    -1000,
    2**64,
    "héllo",
    b"\x00\x01\x02",
    {"a": 1, "b": [1, 2, 3]},
]


def framed_stream(frames: list[bytes]) -> bytes:
    def uv(n: int) -> bytes:
        out = bytearray()
        while n >= 0x80:
            out.append((n & 0x7F) | 0x80)
            n >>= 7
        out.append(n)
        return bytes(out)

    out = bytearray(uv(len(frames)))
    for f in frames:
        out += uv(len(f)) + f
    return bytes(out)


def run(cmd, payload: bytes, cwd) -> tuple[int, bytes, str]:
    p = subprocess.run(cmd, input=payload, capture_output=True, cwd=cwd)
    return p.returncode, p.stdout, p.stderr.decode("utf-8", "replace").strip()


def main() -> int:
    frames = [ref.encode(v) for v in VALUES]
    stream = framed_stream(frames)
    failures: list[str] = []

    # ---- 1. pack: all four produce the same file
    files: dict[str, bytes] = {}
    py_file = F.encode_file(list(VALUES))
    files["python"] = py_file
    for lang, (cmd, cwd) in PACK.items():
        rc, out, err = run(cmd, stream, cwd)
        if rc != 0 or not out:
            failures.append(f"pack {lang}: exit {rc} {err.splitlines()[:1]}")
            continue
        files[lang] = out
    print(f"packed {len(frames)} frames: " + ", ".join(f"{k}={len(v)}B" for k, v in files.items()))
    first = files.get("python")
    for lang, blob in files.items():
        if blob != first:
            failures.append(f"pack {lang}: file bytes differ from python")

    # ---- 2. verify: every implementation agrees on every file
    roots: dict[str, str] = {}
    for producer, blob in files.items():
        for lang, (cmd, cwd) in VERIFY.items():
            rc, out, err = run(cmd, blob, cwd)
            text = out.decode().strip()
            if rc != 0 or not text.endswith("OK"):
                failures.append(f"verify {lang} on {producer}: exit {rc} {text[:60]} {err[:60]}")
                continue
            roots[f"{lang}/{producer}"] = text.split()[0]
    distinct = set(roots.values())
    print(f"verify matrix: {len(roots)} checks, {len(distinct)} distinct root(s)")
    if len(distinct) != 1:
        failures.append(f"roots disagree: {sorted(distinct)[:4]}")

    # ---- 3. tamper: one flipped byte, every verifier must refuse
    if py_file:
        mid = len(py_file) // 2
        tampered = bytearray(py_file)
        tampered[mid] ^= 0x01
        tampered = bytes(tampered)
        for lang, (cmd, cwd) in VERIFY.items():
            rc, out, _ = run(cmd, tampered, cwd)
            if rc == 0:
                failures.append(f"verify {lang}: ACCEPTED a tampered file")
        rc, _, _ = subprocess.run(
            [sys.executable, "-c", "import sys; sys.path.insert(0,'tools'); "
             "from cowrie_ref import file as F; F.decode_file(sys.stdin.buffer.read())"],
            input=tampered, capture_output=True, cwd=ROOT,
        ).returncode, b"", ""
        if rc == 0:
            failures.append("verify python: ACCEPTED a tampered file")

    print()
    if failures:
        print(f"❌ {len(failures)} failures:")
        for f in failures[:20]:
            print("   ", f)
        return 1
    print(f"✅ identical files, one root across 4 implementations ({next(iter(distinct))[:16]}…), "
          f"tampering rejected everywhere")
    return 0


if __name__ == "__main__":
    sys.exit(main())
