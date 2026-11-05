#!/usr/bin/env bash
# Pre-publish verification: prove the artifacts that would be pushed to a registry are not broken.
#
# Publishes nothing. Run this immediately before `cargo publish` / `npm publish` / `twine upload`,
# on the exact commit you intend to ship.
#
#   bash tools/release_check.sh            # full check, including the gate suite
#   bash tools/release_check.sh --no-gates # skip the gate suite (only if you just ran it)
#
# Exit code 0 = every artifact verified. Non-zero = do not publish.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${COWRIE_PY:-python3}"
WITH_GATES=1
[ "${1:-}" = "--no-gates" ] && WITH_GATES=0

fail=0
note() { printf '  %s\n' "$*"; }
bad()  { printf '  ❌ %s\n' "$*"; fail=$((fail+1)); }
ok()   { printf '  ✅ %s\n' "$*"; }

echo "== 1. version consistency"
RUST_V=$(grep -m1 '^version' rust/Cargo.toml | cut -d'"' -f2)
TS_V=$(node -p "require('$ROOT/typescript/package.json').version" 2>/dev/null || echo "?")
PY_V=$(grep -m1 '^version' tools/pyproject.toml | cut -d'"' -f2)
note "crates.io cowrie-rs   $RUST_V"
note "npm       cowrie-codec $TS_V"
note "PyPI      cowrie-ref   $PY_V"
if [ "$RUST_V" = "$TS_V" ] && [ "$TS_V" = "$PY_V" ]; then ok "all three agree"; else bad "versions disagree"; fi

echo "== 2. tree state"
if [ -z "$(git status --porcelain)" ]; then ok "clean working tree"; else bad "uncommitted changes — publish from a clean tree"; fi
note "HEAD $(git rev-parse --short HEAD) · $(git log -1 --pretty=%s)"
# Compare against the remote itself: a stale remote-tracking ref reports a pushed tree as ahead.
REMOTE_SHA=$(git ls-remote origin refs/heads/main 2>/dev/null | cut -f1)
if [ -z "$REMOTE_SHA" ]; then
  note "could not reach origin — push state unverified"
elif [ "$REMOTE_SHA" = "$(git rev-parse HEAD)" ]; then
  ok "HEAD is exactly what origin/main has"
else
  bad "origin/main is ${REMOTE_SHA:0:8} but HEAD is $(git rev-parse --short HEAD) — publish what the repo shows, not what your laptop has"
fi

echo "== 3. Rust: cargo package (packages, then compiles standalone)"
if ( cd rust && cargo package --allow-dirty --quiet >/tmp/rc-cargo.log 2>&1 ); then
  CRATE=$(ls rust/target/package/cowrie-rs-*.crate 2>/dev/null | head -1)
  SZ=""; [ -n "$CRATE" ] && SZ=" ($(du -h "$CRATE" | cut -f1))"
  ok "cowrie-rs packages and compiles standalone${SZ}"
else
  bad "cargo package failed — see /tmp/rc-cargo.log"; tail -5 /tmp/rc-cargo.log | sed 's/^/     /'
fi

echo "== 4. npm: tarball must contain the built module"
( cd typescript && npm run build >/tmp/rc-npm-build.log 2>&1 ) || bad "npm run build failed — see /tmp/rc-npm-build.log"
LIST=$(cd typescript && npm pack --dry-run 2>&1)
if echo "$LIST" | grep -q 'dist/index.js'; then ok "dist/index.js in tarball"; else bad "tarball has no dist/index.js — check 'prepare' and tsconfig outDir"; fi
if echo "$LIST" | grep -q 'dist/gen2/index.js'; then ok "dist/gen2/index.js in tarball"; else bad "tarball has no dist/gen2/index.js"; fi
FILES=$(echo "$LIST" | grep -oE 'total files: [0-9]+' | grep -oE '[0-9]+')
SIZE=$(echo "$LIST" | grep -oE 'package size: [0-9.]+ kB' | grep -oE '[0-9.]+ kB')
note "tarball: ${FILES:-?} files, ${SIZE:-?}"
if [ "${FILES:-0}" -lt 5 ]; then bad "tarball suspiciously small ($FILES files) — it probably ships no code"; fi

echo "== 5. npm: the built entry point actually encodes"
( cd typescript && node -e '
const c = require("./dist/index.js");
const v = c.SJ.object({ a: c.SJ.int64(1) });
const hex = Buffer.from(c.encode(v)).toString("hex");
if (hex !== "434f57520100010161d10041") { console.error("unexpected bytes: " + hex); process.exit(1); }
if (!c.decode(Buffer.from(hex, "hex"))) { console.error("decode failed"); process.exit(1); }
' >/tmp/rc-npm-smoke.log 2>&1 ) && ok "require('cowrie-codec') encodes {\"a\":1} to the canonical bytes" \
  || { bad "built module smoke test failed — see /tmp/rc-npm-smoke.log"; tail -3 /tmp/rc-npm-smoke.log | sed 's/^/     /'; }

echo "== 6. Python: wheel builds and imports"
if ( cd tools && "$PY" -m build --wheel --outdir /tmp/rc-wheel . >/tmp/rc-py.log 2>&1 ); then
  W=$(ls /tmp/rc-wheel/cowrie_ref-*.whl 2>/dev/null | head -1)
  ok "wheel: $(basename "${W:-none}")"
else
  note "python -m build unavailable — falling back to an editable install check"
  ( "$PY" -m pip install -e "tools[test]" -q >/tmp/rc-py.log 2>&1 ) && ok "editable install resolves" \
    || { bad "python package does not install — see /tmp/rc-py.log"; tail -3 /tmp/rc-py.log | sed 's/^/     /'; }
fi
PYTHONPATH=tools "$PY" -c "
import cowrie_ref as r
b = r.encode({'a': 1})
assert b.hex() == '434f57520100010161d10041', b.hex()
" 2>/dev/null && ok "cowrie_ref encodes {\"a\":1} to the canonical bytes" || bad "cowrie_ref oracle disagrees with the canonical bytes"

if [ "$WITH_GATES" -eq 1 ]; then
  echo "== 7. gate suite (against freshly built binaries)"
  rm -f /tmp/cowrie-cli
  ( cd go && go build -o /tmp/cowrie-cli ./cmd/cowrie ) || bad "go build failed"
  ( cd rust && cargo build --release --example recode --features zstd >/dev/null 2>&1 ) || bad "rust recode build failed"
  if COWRIE_PY="$PY" bash tools/run_all_gates.sh >/tmp/rc-gates.log 2>&1; then
    ok "$(grep -cE '^>> \[' /tmp/rc-gates.log) gates pass"
  else
    bad "a gate failed — see /tmp/rc-gates.log"; grep -E 'GATE FAILED' /tmp/rc-gates.log | tail -3 | sed 's/^/     /'
  fi
fi

echo
if [ "$fail" -eq 0 ]; then
  echo "✅ every artifact verified — the publish commands in RELEASE.md are safe to run"
  exit 0
fi
echo "❌ $fail check(s) failed — do not publish"
exit 1
