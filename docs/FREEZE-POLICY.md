# Cowrie freeze policy

**Status:** pre-1.0 (0.9.0). Nothing in this document is in force yet. It says what will be frozen,
when, and what a change after the freeze costs — so that the decision can be argued now, while it
is still free.

---

## Why this document exists

A content address is a promise about the future. When someone stores `cowrie hash` next to an
artifact, they are underwriting that the same bytes still decode to the same value in five years,
in a language that does not exist yet. A format that might change its mind about ordering, number
form, or padding is not an identity system; it is an encoding that happens to be stable this week.

The freeze is not a marketing milestone. It is the moment the promise becomes one we can keep.

---

## What freezes at 1.0

**Core** — the byte-level contract:

1. **The value model** (§1): the type set, their equality rules, the number domain
   (Int / Uint / BigInt, the smallest legal form), Float64 semantics (−0.0 ≡ +0.0, one canonical
   NaN), String as byte-preserving UTF-8, Datetime as int64 nanoseconds.
2. **The canonical encoding** (§2): tag assignments, minimal integer widths, the header dictionary
   rule (byte-sorted, unique, every entry referenced), 64-byte tensor alignment with zero padding,
   Decimal128 in lowest terms, sub-byte dtype packing.
3. **The hash** (§3): SHA-256 over the canonical bytes, multihash framing, the 34-byte identity.
   This freezes first and hardest — it is the part an adopter cannot re-derive.
4. **The file format** (§7): CWRF framing, footer layout, the Merkle construction and its domain
   separation, sealed-file semantics.

## What does not freeze

- **Profiles** (§8) — Embedding, Media, Trace, Eval: schemas over Core. They may evolve, because a
  profile change is a new object shape, not a new encoding rule.
- **Tag-space growth** — reserved tags may be assigned in a future version. A v1 decoder rejects
  them today (ERR_RESERVED_TAG), so growth is additive and never reinterprets existing bytes.
- **Tooling, CLIs, bindings, APIs** — free to change at any time. Identity is the wire, not the API.
- **Compression** — transport only (§2.2). Canonical form is uncompressed; framing may add codecs.

---

## The rule after the freeze

> **No change may alter the bytes any frozen value produces, or the identity any frozen bytes carry.**

A change that would is not a change to Core. It is a new tag, a new profile, or a new format
version — and it ships alongside, never underneath.

Three things this forbids outright, because each has a tempting justification:

- **Reinterpreting an existing tag** — even to fix a bug. If v1 encodes X as bytes B, a decoder
  that reads B as Y is a fork wearing the same name.
- **Relaxing a canonical rule** — "accept both spellings for compatibility" destroys the property
  the format exists to provide. Lenient decode is allowed as an explicit opt-out for reading legacy
  data; it is never the default and never what produces an identity.
- **Changing the hash input** — not the domain separation, not the multihash wrapper, not the
  canonicalisation that precedes it.

## If a defect is found after the freeze

1. **Errata, published.** The defect, the affected byte patterns, and the workaround go into the
   specification as a dated erratum with its own hash.
2. **Never a silent change.** If a fix requires different bytes, it is a new format version with a
   new magic/version byte. Old files keep their identities forever.
3. **The gates grow, they never shrink.** `count_guard` enforces that the conformance corpus only
   ever grows — a fix adds fixtures, it does not delete the case it broke.

---

## Preconditions for declaring the freeze

The freeze is a claim that the contract is right, not that the code is finished. It may only be
declared when all of the following hold. Each is checkable from the repository.

| # | precondition | how it is checked |
|---|---|---|
| 1 | Every gate green on every push | `tools/run_all_gates.sh` (15 gates) is the CI job, not a suggestion |
| 2 | Four implementations agree on the bytes they **construct**, not only the bytes they receive | `constructor` gate: same value built in Go, Rust, TS, Python → identical bytes |
| 3 | Strict decode never accepts bytes and returns a different value | `law_bytes` gate: 73 canonical controls + 511 mutated encodings |
| 4 | The normative specification is in the repository and versioned | `_md-archive/docs/SPEC-v1.md`; the freeze publishes its hash in `FREEZE.sha256` |
| 5 | The negative corpus covers every rejection the spec names, with exact error codes | `negative --strict` gate (29 fixtures today; grows, never shrinks) |
| 6 | Hostile input is fuzzed, not merely tested | libFuzzer targets in CI, restored from `attic/` |
| 7 | A defect found post-freeze has a written errata process | this document, §"If a defect is found" |
| 8 | At least one deployment the maintainers do not control runs it in production | evidence, not intent |

**As of 0.9.0:** 1–3 hold and are enforced by CI. 4 holds (the spec is versioned in the
repository); publishing its hash is part of the freeze itself. 5 is close — 29 fixtures with
exact-code checking, but nobody has audited it against every rejection the spec names. 6 is not
wired: the fuzz workflow is in `attic/`. 7 is this document, unratified. 8 does not hold and
cannot be manufactured.

---

## What freezing costs, and when to pay it

After the freeze, every Core mistake is permanent — it must be lived with or versioned around.
Before it, a mistake costs a search-and-replace.

That asymmetry argues for freezing **late** (fix everything first) and, once frozen, for treating
the specification as the artifact and the implementations as ports of it.

The specific gates on the way there, in order:

1. Restore fuzzing to CI (precondition 6).
2. Get one external deployment using it for real artifact identity (precondition 8).
3. Only then publish 1.0 and this policy as ratified.

Freezing on a schedule rather than on evidence is how formats acquire the bugs they can never fix.
