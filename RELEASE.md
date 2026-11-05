# Releasing Cowrie

Three packages, one format. They are versioned together and must ship from the same commit:
`cowrie-rs` (crates.io), `cowrie-codec` (npm), `cowrie-ref` (PyPI, the executable spec).

**Current status: not published anywhere.** `tools/release_check.sh` verifies the artifacts; no
registry credentials exist in this repository or on the maintainer's machine, so publishing is a
deliberate act that needs a token, not a script that runs itself.

## Before anything

```bash
bash tools/release_check.sh        # ~4 min; exits non-zero if any artifact would ship broken
git push                            # publish what the repo shows, not what your laptop has
```

The check verifies version agreement across the three manifests, a clean pushed tree,
`cargo package` (which compiles the crate standalone), an npm tarball that actually contains
`dist/`, the built module encoding `{"a":1}` to `434f57520100010161d10041`, the wheel, and all
fifteen gates against freshly built binaries.

## One-time credentials

```bash
cargo login                     # crates.io token: https://crates.io/settings/tokens
npm login                       # or: npm config set //registry.npmjs.org/:_authToken <token>
python -m twine --version       # PyPI: pip install twine; token in ~/.pypirc or TWINE_PASSWORD
```

None of these are stored in the repository. Do not paste tokens into a shell that records history
on a shared machine.

## Publish

Order matters only in that the Rust crate is the one with a build step; all three are independent.

```bash
# 1. Rust
cd rust && cargo publish

# 2. TypeScript — `prepare` builds dist/ before packing; verify the file list in the output
cd typescript && npm publish --access public

# 3. Python (the reference implementation, so the spec has an installable form)
cd tools && python -m build && python -m twine upload dist/cowrie_ref-*.whl
```

## Verify the release, from the registry, not from the repo

In an empty directory on a machine that has never seen this checkout:

```bash
cargo new /tmp/rc && cd /tmp/rc && cargo add cowrie-rs && cargo run -- \
  'cowrie_rs::gen2::encode(&cowrie_rs::gen2::Value::Object(...))'   # encode one value

npm i cowrie-codec && node -e 'const c=require("cowrie-codec");
  console.log(Buffer.from(c.encode(c.SJ.object({a:c.SJ.int64(1)}))).toString("hex"))'

pip install cowrie-ref && python -c "import cowrie_ref as r; print(r.encode({'a':1}).hex())"
```

All three must print `434f57520100010161d10041`. If they do not, the release is broken and the
packages must be yanked, not patched in place.

## Rules

- **crates.io is permanent.** A version can be yanked, never deleted or reused. A bad `0.9.0` means
  `0.9.1` forever; it does not mean a corrected `0.9.0`.
- **Never publish from a dirty or unpushed tree.** The tarball must be reproducible from the tag.
- **Tag the commit** after all three succeed: `git tag -a v0.9.0 -m "Cowrie v1 0.9.0" && git push --tags`.
- **If one registry fails and two succeed**, publish the fix as the next patch version on all three.
  Never leave the three at different versions with the same wire format.
