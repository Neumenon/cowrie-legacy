#![no_main]
use std::collections::BTreeMap;

use cowrie_rs::gen2::{decode, decode_framed, encode, Value};
use libfuzzer_sys::fuzz_target;

/// Build a Value from fuzzer bytes, consuming input linearly.
///
/// The previous builder recursed on the *remaining* slice for each child, so a byte like 0xF6 (array
/// of three, each recursing on the same tail) grew 3^depth: a 73-byte input asked for billions of
/// nodes and the harness OOM'd before the code under test ran at all. A cursor plus a depth cap makes
/// the cost proportional to the input.
struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn byte(&mut self) -> Option<u8> {
        let b = *self.data.get(self.pos)?;
        self.pos += 1;
        Some(b)
    }

    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let n = n.min(self.data.len() - self.pos);
        let out = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Some(out)
    }

    fn u64(&mut self) -> Option<u64> {
        let mut b = [0u8; 8];
        let n = self.data.len().saturating_sub(self.pos).min(8);
        b[..n].copy_from_slice(self.take(n)?);
        self.pos += 0;
        Some(u64::from_le_bytes(b))
    }
}

fn value_from_bytes(c: &mut Cursor, depth: usize) -> Option<Value> {
    if depth > 8 {
        return None;
    }
    let tag = c.byte()?;
    match tag % 10 {
        0 => Some(Value::Null),
        1 => Some(Value::Bool(tag & 0x10 != 0)),
        2 => Some(Value::Int(c.u64()? as i64)),
        3 => Some(Value::Float(f64::from_bits(c.u64()?))),
        4 => {
            let len = ((tag >> 4) & 0x0F) as usize;
            let s = String::from_utf8_lossy(c.take(len)?).into_owned();
            Some(Value::String(s))
        }
        5 => {
            let len = ((tag >> 4) & 0x0F) as usize;
            Some(Value::Bytes(c.take(len)?.to_vec()))
        }
        6 => {
            let count = (tag & 0x03) as usize;
            let mut items = Vec::with_capacity(count);
            for _ in 0..count {
                items.push(value_from_bytes(c, depth + 1)?);
            }
            Some(Value::Array(items))
        }
        7 => {
            let count = (tag & 0x03) as usize;
            let mut map = BTreeMap::new();
            for i in 0..count {
                let key = format!("k{}{}", i, c.byte()?);
                map.insert(key, value_from_bytes(c, depth + 1)?);
            }
            Some(Value::Object(map))
        }
        // BigInt from arbitrary bytes: exercises the §5.3 coercion into Int/Uint.
        8 => {
            let len = ((tag >> 4) & 0x0F) as usize;
            Some(Value::BigInt(c.take(len)?.to_vec()))
        }
        // Decimal128 from a scale and a 16-byte coefficient: exercises lowest-terms reduction.
        _ => {
            let scale = c.u64()? as i64;
            let mut coef = [0u8; 16];
            let avail = c.take(16).unwrap_or(&[]);
            coef[..avail.len()].copy_from_slice(avail);
            let mut wire = Vec::with_capacity(17);
            // varint(zigzag(scale)) as the decoder would have produced it
            let mut z = ((scale << 1) ^ (scale >> 63)) as u64;
            loop {
                let b = (z & 0x7F) as u8;
                z >>= 7;
                if z == 0 {
                    wire.push(b);
                    break;
                }
                wire.push(b | 0x80);
            }
            wire.extend_from_slice(&coef);
            Some(Value::Decimal(wire))
        }
    }
}

fuzz_target!(|data: &[u8]| {
    let mut c = Cursor { data, pos: 0 };
    while c.pos < data.len() {
        let Some(val) = value_from_bytes(&mut c, 0) else {
            break;
        };
        let Ok(encoded) = encode(&val) else {
            continue;
        };
        let decoded = match decode(&encoded) {
            Ok(d) => d,
            Err(e) => panic!("encode succeeded but decode failed for {:?}: {:?}", val, e),
        };
        let re_encoded = encode(&decoded).expect("re-encode of a decoded value must succeed");
        assert_eq!(
            encoded, re_encoded,
            "non-canonical round-trip for {:?}",
            val
        );
        // The framed form must agree with the bare form's payload.
        if let Ok(framed) = cowrie_rs::gen2::encode_framed(&decoded, cowrie_rs::gen2::Compression::None) {
            if let Ok(unframed) = decode_framed(&framed) {
                assert_eq!(
                    encode(&unframed).unwrap(),
                    re_encoded,
                    "framed round-trip disagrees"
                );
            }
        }
    }
});
