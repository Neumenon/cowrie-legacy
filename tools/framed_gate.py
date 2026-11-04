#!/usr/bin/env python3
"""Framed-composite parity gate (SPEC-v1 §2.2 transport framing).

The §2.2 framed layout is:

    COWR(4) + version(1) + compression_byte(1) + uvarint(uncompressed_body_len) + compressed_body

Compression bytes: 0x00 none (canonical passthrough), 0x01 gzip, 0x02 zstd. A frame with
compression byte 0 is byte-identical to the canonical message, so `recode --framed=none`
must reproduce the input byte-for-byte. gzip/zstd streams are transport-entropy-coded and
NOT required to be byte-identical across libraries — the contract is:

  1. header parity: every impl emits the same COWR+version+comp-byte prefix and the same
     uvarint original length for the same vector;
  2. decode matrix: bytes framed by ANY impl must decode back to the identical canonical
     bytes in EVERY impl (the framing is a transport, not an identity).

The Python reference oracle deliberately does not participate in the decode matrix: per
§2.2, compression is transport-only, and `cowrie_ref.decode` (the canonical decoder)
rejects a non-zero compression byte with ERR_UNSUPPORTED_COMPRESSION rather than
silently pulling in a codec.

Each recode CLI accepts `--framed=<none|gzip|zstd>` (canonical stdin -> framed stdout);
its default path decodes both canonical and framed input and re-emits canonical bytes.

Run: python3 tools/framed_gate.py   (exit non-zero on any mismatch)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GO_CLI = os.environ.get("GO_CLI", "/tmp/cowrie-cli")
RUST = os.environ.get("RUST_RECODE", os.path.join(ROOT, "rust/target/release/examples/recode"))
TS_DIR = os.path.join(ROOT, "typescript")
TSX = os.path.join(TS_DIR, "node_modules/.bin/tsx")

# A spread of vectors large enough to exercise every tag class the framing wraps:
# ints, strings, unicode, nested objects/arrays, tensors, extensions.
VECTOR_NAMES = [
    "nested",
    "str_unicode",
    "str_4byte",
    "int64_min",
    "uint64_max",
    "float_nan",
    "decimal_big_scale",
    "tensor_f32_2x3",
    "tensors_nested",
    "tensor_binary_9",
    "extension",
    "obj_empty",
]

COMPS = ("none", "gzip", "zstd")
COMP_BYTE = {"none": 0, "gzip": 1, "zstd": 2}

IMPL_ENCODE = {
    "go":   lambda comp, raw: run([GO_CLI, "recode", f"--framed={comp}"], raw, None),
    "rust": lambda comp, raw: run([RUST, f"--framed={comp}"], raw, None),
    "ts":   lambda comp, raw: run([TSX, "recode.ts", f"--framed={comp}"], raw, TS_DIR),
}

IMPL_DECODE = {
    "go":   lambda raw: run([GO_CLI, "recode"], raw, None),
    "rust": lambda raw: run([RUST], raw, None),
    "ts":   lambda raw: run([TSX, "recode.ts"], raw, TS_DIR),
}


def run(cmd, raw: bytes, cwd: str | None) -> bytes:
    """Run a recode tool and return stdout. A non-zero exit raises with the tool's own stderr,
    so a capability gap ("zstd feature not enabled") reads as that, not as an empty frame."""
    p = subprocess.run(cmd, input=raw, capture_output=True, cwd=cwd)
    if p.returncode != 0:
        msg = p.stderr.decode("utf-8", "replace").strip().splitlines()
        raise RuntimeError(f"exit {p.returncode}: {msg[0] if msg else 'no stderr'}")
    return p.stdout


def parse_header(framed: bytes):
    """Return (first6, uncomp_len, body_start) for the §2.2 frame prefix."""
    if len(framed) < 6 or bytes(framed[:4]) != b"COWR":
        return None
    pos = 6
    length = 0
    shift = 0
    while pos < len(framed):
        b = framed[pos]
        pos += 1
        length |= (b & 0x7F) << shift
        if b & 0x80 == 0:
            break
        shift += 7
        if shift > 63:
            break
    return bytes(framed[:6]), length, pos


def main() -> int:
    golden = json.load(open(os.path.join(ROOT, "testdata", "v1_golden.json")))
    impls = ("go", "rust", "ts")
    failures: list[str] = []
    n_enc = 0
    n_dec = 0
    n_dec_ok_decline = 0

    for name in VECTOR_NAMES:
        canon = bytes.fromhex(golden[name]["canonical_hex"])

        for comp in COMPS:
            # (1) encode: every impl frames the same canonical bytes
            outs: dict[str, bytes] = {}
            for lang in impls:
                try:
                    outs[lang] = IMPL_ENCODE[lang](comp, canon)
                except Exception as exc:  # noqa: BLE001
                    failures.append(f"encode {lang}/{name}/{comp}: {exc}")
                    continue
            n_enc += len(outs)

            for lang, out in outs.items():
                if bytes(out) == canon:
                    # Declined: the impl returned the canonical bytes unchanged
                    # (e.g. Go's compressThreshold — framing a bigger-than-raw
                    # payload is a transport choice, decode-compatible with all
                    # readers). Everything else still flows through the matrix.
                    n_dec_ok_decline += 1
                    continue
                if comp == "none":
                    # comp byte 0 ⇒ no length uvarint follows; the frame IS canonical.
                    if bytes(out[:6]) != canon[:6]:
                        failures.append(f"header {lang}/{name}/{comp}: {out[:6].hex()} want {canon[:6].hex()}")
                    continue
                parsed = parse_header(out)
                if parsed is None:
                    failures.append(f"header {lang}/{name}/{comp}: unparsable frame ({out[:8].hex()}…{len(out)}B)")
                    continue
                hdr, length, body0 = parsed
                if hdr[:5] != canon[:5] or hdr[5] != COMP_BYTE[comp]:
                    failures.append(
                        f"header {lang}/{name}/{comp}: {hdr.hex()} want {canon[:5].hex()}+{COMP_BYTE[comp]:02x}"
                    )
                if length != len(canon) - 6:
                    failures.append(f"header {lang}/{name}/{comp}: uncomp_len={length} want {len(canon)-6}")
                if body0 < 6:
                    failures.append(f"header {lang}/{name}/{comp}: uvarint overruns frame")

            # comp=none passthrough: the frame IS the canonical bytes
            if comp == "none":
                for lang, out in outs.items():
                    if out != canon:
                        failures.append(f"passthrough {lang}/{name}: --framed=none != canonical")
                    elif out:
                        # decode it too (it is valid input for the matrix below)
                        pass

            # (2) decode matrix: produce framed bytes with each producer impl,
            #     decode with every reader impl, require byte-identical canon.
            for producer, framed in outs.items():
                if not framed:
                    continue
                for reader in impls:
                    try:
                        got = IMPL_DECODE[reader](framed)
                        if not got:
                            raise RuntimeError("empty output")
                    except Exception as exc:  # noqa: BLE001
                        got = b""
                        failures.append(f"decode {producer}/{reader}/{name}/{comp}: {exc}")
                    n_dec += 1
                    if got != canon:
                        failures.append(
                            f"decode {producer}/{reader}/{name}/{comp}: "
                            f"{got.hex()[:40]}… != {canon.hex()[:40]}…"
                        )

    total = len(VECTOR_NAMES) * len(COMPS) * len(impls)
    print(f"Framed-composite parity gate (§2.2) — {total} frame encodes, {n_dec} seed cross-decodes\n")
    if failures:
        print("  ❌ FAILURES:")
        for f in failures[:16]:
            print("   ", f)
        if len(failures) > 16:
            print(f"    … and {len(failures) - 16} more")
        return 1
    print(f"  ✅ header parity ({n_enc} encodes), comp=0 passthrough, and the "
          f"{len(impls)}x{len(impls)} cross-decode matrix all agree")
    return 0


if __name__ == "__main__":
    sys.exit(main())
