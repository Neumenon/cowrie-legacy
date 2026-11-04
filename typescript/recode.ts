// Cross-language conformance harness: decode a Cowrie v1 stream from stdin and re-encode
// canonical bytes to stdout. Used by the golden recode gate. Run: tsx recode.ts < in.cow
//
// Strict mode (SPEC-v1 §5.3): pass --strict or set STRICT=1 to reject well-formed-but-
// non-canonical input (decode raises ERR_NON_CANONICAL -> non-zero exit). Without it,
// behavior is unchanged (lenient decode).
//
// Address mode (SPEC-v1 §3): pass --addr to print the content address (multihash SHA-256
// of the canonical wire bytes) as lowercase hex instead of the canonical bytes. Combines
// with --strict.
//
// File modes (SPEC-v1 §7): read a Cowrie FILE (CWRF container) on stdin.
//   --file-id     : decode+verify, print the merkle_root as one line of lowercase hex (68 chars).
//   --file-recode : decode, RE-ENCODE the file, write raw bytes to stdout (reproduces input
//                   byte-for-byte for canonical files).
import { encode, decode, encodeFramed, decodeFramed, addressOfBytes, decodeFile, encodeFile, fileIdentity, tensorSpans, schemaFingerprint64, merkleRoot, Compression, SJ } from "./src/gen2/index";

const strict = process.argv.includes("--strict") || process.env.STRICT === "1";
// Dataset/stream layer Merkle root (SPEC-v1 §7 dataset_root): read uvarint(n) then n
// repetitions of (uvarint(len) || blob_bytes) from stdin, compute the §7 merkleRoot over
// that ORDERED list of blobs (shard file-roots), print the 34-byte multihash as 68 hex chars.
const datasetRoot = process.argv.includes("--dataset-root");
const addr = process.argv.includes("--addr");
const fileId = process.argv.includes("--file-id");
const fileRecode = process.argv.includes("--file-recode");
const tensorSpansMode = process.argv.includes("--tensor-spans");
// Fingerprint mode (SPEC-v1 §4): decode and print the §4 FPC-grammar fingerprint64 as
// 16 lowercase hex chars. Combines with --strict.
const fingerprint = process.argv.includes("--fingerprint");
// §2.2 framed composite mode: canonical stdin -> framed stdout (gzip|zstd|none).
const framedArg = process.argv.find((a) => a.startsWith("--framed="));
const framedMode = framedArg ? framedArg.split("=", 2)[1] : null;

// --- constructor-gate cases: build each value in code and print name<TAB>hex ----------------
if (process.argv.includes("--cases")) {
  const dec = (scale: number, c0: number) => {
    const c = new Uint8Array(16);
    c[0] = c0;
    return SJ.decimal128(scale, c);
  };
  const cases: [string, any][] = [
    ["null", SJ.null()],
    ["bool_true", SJ.bool(true)],
    ["int_5", SJ.int64(5)],
    ["int_neg17", SJ.int64(-17)],
    ["int_127", SJ.int64(127)],
    ["uint_42", SJ.uint64(42)],
    ["uint_2p63", SJ.uint64(9223372036854775808n)],
    ["bigint_1", SJ.bigint(1n)],
    ["bigint_2p64", SJ.bigint(18446744073709551616n)],
    ["dec_2_150", dec(2, 150)],
    ["dec_2_42", dec(2, 42)],
    ["float_nan", SJ.float64(NaN)],
    ["float_negzero", SJ.float64(-0)],
    ["str_unicode", SJ.string("h\u00e9llo")],
    ["bytes", SJ.bytes(new Uint8Array([0, 1, 2]))],
    ["dt_neg", SJ.datetime64(-1n)],
    ["arr", SJ.array([SJ.int64(1), SJ.int64(2), SJ.int64(3)])],
    ["obj", SJ.object({ a: SJ.int64(1), b: SJ.int64(2) })],
  ];
  for (const [name, v] of cases) {
    process.stdout.write(`${name}\t${Buffer.from(encode(v)).toString("hex")}\n`);
  }
  process.exit(0);
}

const chunks: Buffer[] = [];
process.stdin.on("data", (c: Buffer) => chunks.push(c));
process.stdin.on("end", () => {
  const data = new Uint8Array(Buffer.concat(chunks));
  try {
    if (datasetRoot) {
      let pos = 0;
      const readUvarint = (): number => {
        let shift = 0n;
        let val = 0n;
        for (;;) {
          const b = data[pos];
          pos++;
          val |= BigInt(b & 0x7f) << shift;
          if ((b & 0x80) === 0) break;
          shift += 7n;
        }
        return Number(val);
      };
      const n = readUvarint();
      const blobs: Uint8Array[] = [];
      for (let i = 0; i < n; i++) {
        const len = readUvarint();
        blobs.push(data.subarray(pos, pos + len));
        pos += len;
      }
      process.stdout.write(Buffer.from(merkleRoot(blobs)).toString("hex") + "\n");
    } else if (fingerprint) {
      const fp = schemaFingerprint64(decode(data, { strict }));
      process.stdout.write(fp.toString(16).padStart(16, "0") + "\n");
    } else if (tensorSpansMode) {
      // Phase 2 zero-copy tensor locator: one line per tensor in document order,
      // "<data_offset> <data_len> <dtype> <shape>" (shape comma-joined or "-" for rank-0).
      const spans = tensorSpans(data, { strict });
      let out = "";
      for (const s of spans) {
        const shape = s.shape.length === 0 ? "-" : s.shape.join(",");
        out += `${s.dataOffset} ${s.dataLen} ${s.dtype} ${shape}\n`;
      }
      process.stdout.write(out);
    } else if (fileId) {
      process.stdout.write(Buffer.from(fileIdentity(data)).toString("hex") + "\n");
    } else if (fileRecode) {
      const frames = decodeFile(data, true);
      process.stdout.write(Buffer.from(encodeFile(frames)));
    } else if (framedMode !== null) {
      let compression: Compression;
      if (framedMode === "none") compression = Compression.NONE;
      else if (framedMode === "gzip") compression = Compression.GZIP;
      else if (framedMode === "zstd") compression = Compression.ZSTD;
      else throw new Error(`unknown --framed compression ${framedMode} (use none|gzip|zstd)`);
      const val = decodeFramed(data, { strict });
      process.stdout.write(Buffer.from(encodeFramed(val, compression)));
    } else {
      const val = decodeFramed(data, { strict });
      const canonical = encode(val);
      if (addr) {
        process.stdout.write(Buffer.from(addressOfBytes(canonical)).toString("hex") + "\n");
      } else {
        process.stdout.write(Buffer.from(canonical));
      }
    }
  } catch (e) {
    process.stderr.write(String(e));
    process.exit(1);
  }
});
