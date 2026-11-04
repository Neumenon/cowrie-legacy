# Cowrie

**One value. One byte string. One hash. The same in every language.**

---

Most data formats answer "how do I turn a value into bytes?"
Cowrie answers a harder question: **"how do I make the bytes *be* the value's identity?"**

That difference is the whole project.

---

## 0. The problem this exists for

You have three services. A Go ingester, a Python trainer, a TypeScript dashboard. Each one
serializes the same record. Each one produces different bytes, because:

- JSON has no key order. `{"a":1,"b":2}` and `{"b":2,"a":1}` are the same value and different bytes.
- JSON has no canonical number form. `1`, `1.0`, `1e0` are the same number and three byte strings.
- MessagePack has the same freedom plus a choice of int widths.
- CBOR *has* a canonical mode — but it is optional, so in practice nothing enforces it.

So when you try to use the hash as a cache key, a dedup key, or a provenance ID, you get
**three hashes for one record**. Caching silently misses. Dedup silently fails. Nobody notices,
because nothing errors — the data is all "correct" and none of it is *identical*.

> **The rule Cowrie is built on:** there is exactly **one** byte string for a given value, and the
> decoder refuses to accept any other. Not "usually one". Not "one if you remember to sort". One.

That is not a feature you can bolt on. It has to be the design from byte zero.

---

## 1. The invariant

```
canonical_encode : Value → Bytes      is a total function
decode           : Bytes → Value      is its inverse
content_address  : Value → 34-byte multihash
```

Three consequences, and they are all the same consequence:

| property | statement |
|---|---|
| **Determinism** | `encode(v)` is the same bytes on every machine, every run, every language |
| **Bijectivity** | `decode(encode(v)) == v` **and** `encode(decode(b)) == b` |
| **Identity** | `sha256(encode(v))` **is** `v`'s name — the hash needs no separate canonicalization pass |

The third one is the product. Everything else is machinery to make it true.

> **Doctrine:** Core is physics, profiles are chemistry, applications are biology.
> A type may enter Core **only if** its value domain, equality, canonical encoding, error behavior,
> and conformance fixtures are fully specified *without external state*. Anything that needs to look
> something up is not Core.

That rule is why `TensorRef`, graph types, and media codecs are **not** wire variants — they are
profiles expressed over the primitives. It is the rule that keeps the format from growing a second
source of entropy.

---

## 2. What the invariant buys you

Four properties, each one a direct consequence, none of them available from JSON or MessagePack:

1. **Identity across languages.** Go writes the record, Python hashes it, TypeScript looks it up.
   Same 34 bytes. No canonicalization step, no "please sort your keys" note in the docs.

2. **Dedup across untrusted producers.** Two services emit the same logical record with different
   key order. You get one content address, not two. You don't have to trust either producer.

3. **Localization, not just detection.** A file has a Merkle root over its frames, so you can find
   *which* frame broke without re-parsing the file.

4. **Zero-copy tensors.** Tensor data sits on a 64-byte boundary relative to byte 0 of the message,
   and frames sit on 64-byte boundaries in files. So `mmap(file)`, add an offset, and hand a `&[f32]`
   straight to your GPU. No decode, no copy.

---

## 3. The bytes, by hand

This is `{"name": "Alice", "scores": [1, 2, 3]}`. Thirty-three bytes. Every one of them accounted for.

```
offset   bytes                    meaning
──────   ──────────────────────   ─────────────────────────────────────────────
[ 0.. 3] 43 4f 57 52             "COWR" — magic. Four bytes, not two, so it can
                                 never collide with a small integer.
[ 4]     01                       version 1
[ 5]     00                       compression: none. Canonical is always none;
                                 compression is transport, never identity.
[ 6]     02                       dictionary length = 2 entries
[ 7]     04                       key length 4
[ 8..11] 6e 61 6d 65             "name"
[12]     06                       key length 6
[13..18] 73 63 6f 72 65 73       "scores"     ← keys are byte-sorted, always
[19]     d2                       FIXMAP, 2 fields
[20]     00                       dict index 0 → "name"
[21]     05                       tag: String
[22]     05                       length 5
[23..27] 41 6c 69 63 65          "Alice"
[28]     01                       dict index 1 → "scores"
[29]     c3                       FIXARRAY, 3 items
[30]     41                       FIXINT 1     ← 0x40 + value
[31]     42                       FIXINT 2
[32]     43                       FIXINT 3
```

Three things worth noticing, because they are the design in miniature:

- **The dictionary is in the header, and it is sorted.** Keys are written once, then referenced by
  index. Sorting them is what makes key order non-semantic — there is no other order to produce.
- **Small integers are one byte.** `0x40..0xBF` is `0..127` with the value in the tag itself. A list of
  small ints is a list of bytes.
- **Nothing is redundant.** No type annotation for the map, no length on the array beyond the fixarray
  tag, no whitespace to normalize.

Its content address:

```
12 20 91f7d42a00c157c37f0929b15e90d8c785dbe50581cc651aafc338f6e5e8aad1
└┬┘ └┬┘ └──────────────────────────────┬───────────────────────────────┘
 │   │                                  └─ 32 bytes of SHA-256
 │   └─ 0x20 = length 32
 └─ 0x12 = multihash sha2-256
```

That prefix is the point: the algorithm is **named in the address**, so it can rotate after 1.0
without invalidating a decade of stored hashes. All four implementations produce those exact bytes.

Same value, four languages, one hash. That is the claim, and `tools/run_all_gates.sh` is what
proves it rather than asserting it.

---

## 4. Why each rule exists

Canonical formats fail when a rule exists "because the spec says so". Every rule here has a
failure it prevents:

| rule | what breaks without it |
|---|---|
| Varints must be **minimal** (no trailing zero groups) | Two encodings of `5` → two hashes → dedup misses |
| Integers use the **smallest form**: FIXINT → Int → Uint → BigInt | `5` as Int and `5` as BigInt are different bytes for the same number |
| `Uint` exists **only** above `i64::MAX` | `Uint(5)` and `Int(5)` both legal → canonical set has a hole |
| Floats: `−0.0` → `+0.0`, one canonical quiet NaN | `-0.0 == 0.0` is true, so they must encode identically or equality ≠ identity |
| **No scalar float32** — f32 lives only in tensors | A value could be written two ways; the format's own docs called this bug "K10" |
| Dictionary is **byte-sorted and unique** | Key order becomes semantic; two records, two hashes |
| Decimal128 in **lowest terms** | `1.50` and `1.5` are the same number and two byte strings |
| BigInt **minimal two's-complement** | Leading zero bytes give an unbounded family of encodings |
| Tensor data on a **64-byte boundary**, zero padding | Unaligned data kills zero-copy, and non-zero padding is a covert channel |
| Sub-byte dtypes packed **LSB-first**, trailing bits zero | Quantized tensors would have many valid spellings |
| **Reserved tags reject**, not skip | Silent skipping means two parsers disagree about what a document says |
| Compression is **not** part of identity | The same value compressed two ways would have two addresses |

And the inversion that makes it real: **the strict decoder refuses well-formed-but-non-canonical
input.** It does not silently re-canonicalize. A decoder that "helpfully" fixes your bytes destroys
the property, because now it accepts documents that other implementations would hash differently.

Proof, executed:

```
canonical      43 4f 57 52 01 00 00 45          (FIXINT 5)
non-canonical  43 4f 57 52 01 00 00 09 05       (Uint 5 — legal-looking, wrong)
strict decode  ERR_NON_CANONICAL: uint fits Int
```

---

## 5. How it is enforced

A canonical format with four implementations has one failure mode that kills it: **drift**. Four
codebases, four interpretations, four hash functions pretending to be one.

The answer here is an **oracle**:

```
        ┌──────────────┐
        │ Python ref   │  deliberately the dumbest, most readable implementation
        │  (oracle)    │  ~1.2k lines, no cleverness, generates the fixtures
        └──────┬───────┘
               │ golden vectors (73)
     ┌─────────┼─────────┬─────────┐
     ▼         ▼         ▼         ▼
    Go       Rust       TS      Python
     └─────────┴─────────┴─────────┘
               │
        byte-identical, or the build is red
```

Eleven gates run on every push, and one red gate blocks merge:

| gate | what it proves |
|---|---|
| `count_guard` | the conformance corpus only ever grows |
| `pytest cowrie_ref` | the oracle is internally consistent (236 tests) |
| `conformance` | 73 golden vectors × 4 languages, byte-identical |
| `content_address` | all four agree on the SHA-256 multihash |
| `file_identity` | all four agree on the CWRF Merkle root, and re-encode files byte-identically |
| `tensor_view` | all four agree on where tensor data starts (the zero-copy contract) |
| `fingerprint` | all four agree on `fingerprint64` |
| `dataset_identity` | all four agree on `dataset_root` |
| `negative` (lenient) | 27 malformed fixtures rejected by every decoder |
| `negative --strict` | the same 27 rejected with the *exact* expected `ERR_*` code |
| `fuzz_differential` | 150 random values: Go/Rust/TS byte-identical to the oracle |

```bash
bash tools/run_all_gates.sh     # the only command that matters
```

The error *codes* are part of the ABI. `ERR_NON_CANONICAL`, `ERR_RESERVED_TAG`,
`ERR_DUPLICATE_KEY` — the message is free text, the code is law, and all four implementations must
agree on which one fires.

---

## 6. The file layer

A single value is a message. A dataset is many values. Cowrie files package them:

```
"CWRF" · version · reserved · frame_count
  frame_len · align-pad · frame_bytes        ← each frame is one complete COWR message
  ...                                         each frame starts on a 64-byte file offset
FOOTER · frame_count · (offset, len) pairs · merkle_root(34)
u64 footer_offset · "CWRF"                   ← seek-from-end trailer: O(1) random access
```

A real 125-byte file containing one frame, annotated:

```
[ 0.. 3] 43 57 52 46              "CWRF"
[ 4]     01                       version
[ 5]     00                       reserved (must be 0)
[ 6]     01                       frame count = 1
[ 7]     0c                       frame length = 12
[ 8..63] 00 × 56                  pad — so the frame starts at exactly byte 64
[64..75] 43 4f 57 52 01 00 01 01 61 d1 00 41     the frame: {"a": 1}
[76]     01                       footer: frame count = 1
[77]     40                       frame offset = 64
[78]     0c                       frame length = 12
[79..112] 12 20 745e853d…         merkle_root, 34 bytes
[113..120] 4c 00 00 00 00 00 00 00  footer_offset = 76, u64 LE
[121..124] 43 57 52 46            "CWRF" trailer
```

The Merkle construction is RFC 6962 with domain separation, **promote-odd** (a lone node is carried
up, never duplicated — that is the CVE-2012-2459 fix), and **count-bound**:

```
leaf(f)    = SHA-256(0x00 ‖ f)
node(a,b)  = SHA-256(0x01 ‖ a ‖ b)
root       = SHA-256(0x02 ‖ uvarint(count) ‖ tree_digest) → 0x12 0x20 ‖ root
```

A lone leaf and a node can never be confused, and two different frame lists can never share a root.
That is what makes "find the broken frame" cheap: walk the hashes, skip the parse.

Note the deliberate distinction: a **frame's** content address is a bare SHA-256 of its canonical
bytes. A **file's** identity is a Merkle root over `0x00`-prefixed leaves. They are different
functions on purpose; conflating them would let a subtree masquerade as a value.

---

## 7. The dataset layer

Above files, the same idea once more. A dataset is an **ordered list of sealed shard files** plus a
manifest. Identity is `dataset_root` — a Merkle root over the shards' 34-byte file roots — and
crucially it is **location-independent**: the same data at a different URI is the same dataset.

That split is the useful part:

- `dataset_root` = **what the data is** (order-sensitive, counted, no URIs)
- `content_address(manifest)` = **where this copy lives** (binds URIs too)

Which gives you lazy verification: download the manifest, check that `dataset_root(listed roots)`
equals the root you trust, then fetch **only the shards you need** and check each one's Merkle root
against its listed entry. Two checks, no full download, and tampering with any shard or its order
fails one of them.

---

## 8. Profiles: chemistry over physics

Anything that needs external state is a **schema over Core**, not a new tag.

| profile | shape |
|---|---|
| **TensorRef** | `Object{hash: Bytes, dtype, shape, byteLen, store?}` — identity is `hash`, never the ref itself; `store` is a hint, not semantics |
| **Embedding** | `Object{id, vector: Tensor<f32>[D], model, modelVersion, metric, normalized, createdAt}` — the vector is a real tensor, not a float array |
| **Trace / Eval / TrainingBatch** | schemas over Object/Array/Tensor |

The payoff: **a profile needs no decoder work.** Every Cowrie implementation already reads it,
because it is just an object.

> **Release gate, not advice:** before Core freezes at 1.0, the Embedding, Media, and Trace profiles
> must be fully expressible over the locked Core with no Core change. Thawing Core after 1.0 breaks
> determinism forever. So the profiles get built *first* and the freeze happens *after*.

---

## 9. What Cowrie is not

Being precise about this saves everyone time:

- **Not a compression format.** Against zstd'd MessagePack the size difference is ~1%. If you only
  want smaller bytes, use zstd on JSON and stop reading.
- **Not a schema language.** No IDL, no codegen, no validation layer. It gives you bytes and identity.
- **Not a streaming format.** The header dictionary forces a buffering pre-pass on encode. That is a
  deliberate trade: identity beats single-pass write. Growth is sealed files, not an appendable stream.
- **Not human-readable**, and not trying to be. `cowrie recode` will render JSON for debugging.
- **Not frozen yet.** Pre-1.0, 0.9.0. Tag space is reserved and rejected conservatively so it can be
  extended without breaking anyone.

---

## 10. Measured

Compiled Go, Apple M3 Ultra, one 54 KB payload (50 KB binary blob + a prompt + a 768-element f32
embedding), same process. Three runs, medians; the host was also running other work (load ≈ 5), so
the absolute times are pessimistic — the ratios are the signal.

The MessagePack column is the comparison a MessagePack user would actually write. The *packed*
column carries the same bytes Cowrie carries, so it isolates the format rather than the encoding
choice:

| | Cowrie | MessagePack (float array) | MessagePack (packed bytes) | JSON |
|---|---|---|---|---|
| encode | 7,124 ns | 15,531 ns | **5,692 ns** | 64,319 ns |
| decode (strict) | 6,579 ns | 21,387 ns | 6,951 ns | 157,045 ns |
| size | 54,439 B | 55,171 B | **54,448 B** | 75,167 B |
| allocs (encode) | 12 | 4 | 3 | 12 |
| allocs (decode) | 15 | 780 | 23 | 1,574 |

Read it honestly.

- **Against an element-wise float array, the tensor type wins decisively.** 2.2× faster to encode,
  3.3× faster to decode, and 780 allocations collapse to 15. That is the mistake MessagePack makes
  on numeric payloads — every float gets a type tag and a boxed value — and Cowrie's type system
  makes it unrepresentable.
- **Against a hand-packed blob, the speed advantage is gone.** MessagePack encodes 25% faster
  (5,692 vs 7,124 ns), Cowrie decodes 5% faster, and the sizes differ by nine bytes. If you are
  willing to pack your own bytes and carry your own dtype and shape, MessagePack serves you at the
  same cost. Cowrie's claim is that you should not have to.
- **Identity is not in this table.** None of these timings include hashing, and none of the three
  alternatives gives you a byte string two services are guaranteed to agree on.

### What the dictionary buys

The header codes every object key once and references it by index. Measured with the reference
encoder on four real shapes:

| shape | encoded | dictionary | inlining keys would add | net |
|---|---:|---:|---:|---:|
| one small object, 3 keys | 1,062 B | 23 B | 19 B | −4 B (−0.4%) |
| agent trace, 50 events × 5 keys | 1,328 B | 20 B | 700 B | **+680 B (+51%)** |
| dataset shard, 500 rows × 6 keys | 11,022 B | 49 B | 21,000 B | **+20,951 B (+190%)** |
| wide unique keys, 200 objects × 600 keys | 7,553 B | 5,672 B | 5,070 B | −602 B (−8%) |

The dictionary costs four bytes on a lone small object, returns half the message on an agent trace,
and nearly triples a dataset shard. It is a loss only when every key is distinct — and then it is a
loss of 8%, not a doubling.

Rust, borrowed (zero-copy) decoder vs owned decoder on a 3.2 KB agent-trace frame:

| | µs/frame | allocations/frame |
|---|---|---|
| `decode` (owned) | 1.918 | 57.2 |
| `decode_ref` (borrowed) | **0.528** | **4.4** |

**3.6× faster, 52.8 fewer allocations per frame** — the 64-byte tensor alignment cashing in.

Honest ledger — where it loses:

- **MessagePack allocates less on encode** (4 vs 12; 3 packed). Cowrie is faster against the
  float-array form, not leaner.
- **The Python reference is 12–24× slower than MessagePack.** It is the oracle, deliberately dumb.
  Throughput comes from Go and Rust.
- **Against zstd'd competitors, size is a wash.** The archive profile (whole dataset as one
  canonical array, dictionary coded once) reaches 1.25% of JSON against MessagePack's 1.42% —
  inches, not orders.
- **Benchmarks are one payload.** These numbers say nothing about small messages, deep nesting,
  mobile CPUs, or the cost of hashing, which none of them include.

---

## 11. Use it

```bash
git clone https://github.com/Neumenon/cowrie.git && cd cowrie
bash tools/run_all_gates.sh      # prove it to yourself first
```

**Go**

```go
v := cowrie.Object(
    cowrie.Member{Key: "name", Value: cowrie.String("Alice")},
    cowrie.Member{Key: "scores", Value: cowrie.Array(cowrie.Int64(1), cowrie.Int64(2))},
)
data, _ := cowrie.Encode(v)
addr := cowrie.AddressOfBytes(data)   // 34-byte multihash
```

**Rust**

```rust
let v = gen2::Value::String("Alice".into());
let wire = gen2::encode(&v)?;
let addr = gen2::address_of_bytes(&wire);
```

**Python** (the reference)

```python
from cowrie_ref import encode, decode, content_address
blob = encode({"name": "Alice", "scores": [1, 2, 3]})
addr = content_address({"name": "Alice", "scores": [1, 2, 3]})
```

**TypeScript**

```javascript
const { fromAny, encode, contentAddress } = require("cowrie-codec");
const v = fromAny({ name: "Alice", scores: [1, 2, 3] });   // JS can't tell int from float
const addr = contentAddress(v);
```

The APIs differ on purpose: Python takes plain objects, JS needs `fromAny` because its numbers are
one type, Go and Rust make you say what you mean.

---

## 12. Status

**0.9.0 — pre-1.0 release candidate.** Four implementations, eleven gates, all green.

The wire format is fixture-pinned but **not frozen**. Reserved tags are rejected, so growth happens
through the Extension envelope and through profiles — never by re-using a tag. At 1.0 the Core byte
layout freezes and the profiles must already exist (see §8).

**License: Apache-2.0.**

---

## Appendix: the tag space

```
0x00 Null      0x01 false      0x02 true       0x03 Int (svarint)
0x04 Float64   0x05 String     0x06 Array      0x07 Object (dict-indexed)
0x08 Bytes     0x09 Uint       0x0A Decimal128 0x0B Datetime (ns, LE)
0x0C UUID      0x0D BigInt     0x0E Extension  0x20 Tensor
0x24 Bitmask
0x40-0xBF FIXINT    (value = tag − 0x40, 0..127, one byte)
0xC0-0xCF FIXARRAY  (count = tag − 0xC0, 0..15)
0xD0-0xDF FIXMAP    (count = tag − 0xD0, 0..15)
0xE0-0xEF FIXNEG    (value = −1 − (tag − 0xE0), −1..−16, one byte)
everything else → ERR_RESERVED_TAG
```

All multi-byte numbers are **little-endian**. UUID is the single exception (RFC 4122 field order).
That exception is written down, tested, and the only one in the format.

---

*The interesting property of a canonical format is not that it is small. It is that two strangers
can compute the same 32 bytes from the same idea, on different machines, in different languages,
years apart, without talking to each other — and both be right.*
