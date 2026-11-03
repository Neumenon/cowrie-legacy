//! Borrowed, zero-copy Cowrie decoder (gen2).
//!
//! Same wire format, same canonical checks, and the same error surface as the owning
//! [`super::decode`] decoder — but String/Bytes/Decimal/Uuid/BigInt/Ext/Tensor payloads come
//! back as borrows of the input instead of fresh allocations. The cost moves to the caller:
//! the input must outlive the decoded value (`ValueRef<'a>` borrows `'a`).
//!
//! Borrowed decoding is safe on canonical messages because tensor data runs are 64-byte
//! aligned contiguous little-endian runs (SPEC-v1 §2.5), dictionaries are once-per-message,
//! and every other byte sequence is length-prefixed in place.
//!
//! Conformance: `decode_ref` must accept exactly the byte strings [`super::decode`] accepts
//! and produce equal values (via [`ValueRef::to_owned`]).

use super::decode::{DecodeOptions, tensor_expected_bytes, tensor_total_bits};
use super::tags;
use super::types::{CowrieError, DType};
use crate::{MAGIC, VERSION};
use std::collections::BTreeMap;

/// Borrowed tensor: shape is decoded (dims are uvarints, they must be), data is a view.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorView<'a> {
    pub dtype: DType,
    pub shape: Vec<u64>,
    pub data: &'a [u8],
}

/// Borrowed extension payload.
#[derive(Debug, Clone, PartialEq)]
pub struct ExtView<'a> {
    pub type_id: u64,
    pub payload: &'a [u8],
}

/// Borrowed Cowrie value.
#[derive(Debug, Clone, PartialEq)]
pub enum ValueRef<'a> {
    Null,
    Bool(bool),
    Int(i64),
    Uint(u64),
    Float(f64),
    /// svarint(scale) bytes followed by the 16-byte LE coefficient, as on the wire.
    Decimal(&'a [u8]),
    String(&'a str),
    Bytes(&'a [u8]),
    DateTime(i64),
    Uuid(&'a [u8]),
    BigInt(&'a [u8]),
    Array(Vec<ValueRef<'a>>),
    Object(BTreeMap<&'a str, ValueRef<'a>>),
    Tensor(TensorView<'a>),
    Ext(ExtView<'a>),
    Bitmask {
        count: u64,
        bits: &'a [u8],
    },
}

impl ValueRef<'_> {
    /// Convert to the owning [`super::types::Value`]. Allocates only for the data that is
    /// genuinely owned in the owning model (strings, byte runs, containers).
    pub fn to_owned(&self) -> super::types::Value {
        use super::types::{ExtData, TensorData, Value};
        match self {
            ValueRef::Null => Value::Null,
            ValueRef::Bool(b) => Value::Bool(*b),
            ValueRef::Int(i) => Value::Int(*i),
            ValueRef::Uint(u) => Value::Uint(*u),
            ValueRef::Float(f) => Value::Float(*f),
            ValueRef::Decimal(d) => Value::Decimal(d.to_vec()),
            ValueRef::String(s) => Value::String((*s).to_string()),
            ValueRef::Bytes(b) => Value::Bytes(b.to_vec()),
            ValueRef::DateTime(dt) => Value::DateTime(*dt),
            ValueRef::Uuid(u) => Value::Uuid(
                <[u8; 16]>::try_from(*u).expect("decoder guarantees 16-byte uuid"),
            ),
            ValueRef::BigInt(b) => Value::BigInt(b.to_vec()),
            ValueRef::Array(a) => Value::Array(a.iter().map(ValueRef::to_owned).collect()),
            ValueRef::Object(o) => Value::Object(
                o.iter()
                    .map(|(k, v)| ((*k).to_string(), v.to_owned()))
                    .collect(),
            ),
            ValueRef::Tensor(t) => {
                Value::Tensor(TensorData::new(t.dtype, t.shape.clone(), t.data.to_vec()))
            }
            ValueRef::Ext(e) => Value::Ext(ExtData {
                type_id: e.type_id,
                payload: e.payload.to_vec(),
            }),
            ValueRef::Bitmask { count, bits } => Value::Bitmask {
                count: *count,
                bits: bits.to_vec(),
            },
        }
    }
}

/// Decode Cowrie bytes with zero copies, using default options.
pub fn decode_ref(data: &[u8]) -> Result<ValueRef<'_>, CowrieError> {
    decode_ref_with_options(data, &DecodeOptions::default())
}

/// STRICT zero-copy decode (SPEC-v1 §5.3): identical rejections to `decode_strict`.
pub fn decode_ref_strict(data: &[u8]) -> Result<ValueRef<'_>, CowrieError> {
    decode_ref_with_options(
        data,
        &DecodeOptions {
            strict: true,
            ..DecodeOptions::default()
        },
    )
}

/// Zero-copy decode with configurable limits.
pub fn decode_ref_with_options<'a>(
    data: &'a [u8],
    opts: &DecodeOptions,
) -> Result<ValueRef<'a>, CowrieError> {
    let mut reader = ReaderRef::new(data, opts);
    reader.decode()
}

struct ReaderRef<'a> {
    data: &'a [u8],
    pos: usize,
    dict: Vec<&'a str>,
    depth: usize,
    opts: DecodeOptions,
}

impl<'a> ReaderRef<'a> {
    fn new(data: &'a [u8], opts: &DecodeOptions) -> Self {
        ReaderRef {
            data,
            pos: 0,
            dict: Vec::new(),
            depth: 0,
            opts: opts.clone(),
        }
    }

    fn decode(&mut self) -> Result<ValueRef<'a>, CowrieError> {
        if self.remaining() < 4 {
            return Err(CowrieError::Truncated);
        }
        if &self.data[0..4] != MAGIC {
            return Err(CowrieError::InvalidMagic);
        }
        self.pos = 4;

        let version = self.read_byte()?;
        if version != VERSION {
            return Err(CowrieError::InvalidVersion(version));
        }
        let _flags = self.read_byte()?;

        let dict_len = self.read_uvarint()? as usize;
        if dict_len > self.opts.max_dict_len {
            return Err(CowrieError::TooLarge);
        }
        self.dict = Vec::with_capacity(dict_len);
        let mut prev: Option<&'a [u8]> = None;
        for _ in 0..dict_len {
            let key_len = self.read_uvarint()? as usize;
            if key_len > self.opts.max_string_len {
                return Err(CowrieError::TooLarge);
            }
            let raw = self.slice(key_len)?;
            if self.opts.strict {
                if let Some(prev) = prev {
                    if raw == prev {
                        return Err(CowrieError::DuplicateKey);
                    }
                    if raw < prev {
                        return Err(CowrieError::NonCanonical(
                            "dictionary not byte-sorted".into(),
                        ));
                    }
                }
            }
            let key = std::str::from_utf8(raw).map_err(|_| CowrieError::InvalidUtf8)?;
            prev = Some(raw);
            self.dict.push(key);
        }

        let value = self.decode_value()?;

        if self.pos < self.data.len() {
            return Err(CowrieError::TrailingData {
                pos: self.pos,
                remaining: self.data.len() - self.pos,
            });
        }
        Ok(value)
    }

    fn decode_value(&mut self) -> Result<ValueRef<'a>, CowrieError> {
        if self.depth > self.opts.max_depth {
            return Err(CowrieError::TooDeep);
        }
        self.depth += 1;

        let tag = self.read_byte()?;
        let value = match tag {
            tags::NULL => ValueRef::Null,
            tags::TRUE => ValueRef::Bool(true),
            tags::FALSE => ValueRef::Bool(false),
            tags::INT64 => {
                let z = self.read_uvarint()?;
                let v = zigzag_decode(z);
                if self.opts.strict && (-16..=127).contains(&v) {
                    return Err(CowrieError::NonCanonical(
                        "int in FIXINT/FIXNEG range".into(),
                    ));
                }
                ValueRef::Int(v)
            }
            tags::UINT64 => {
                let u = self.read_uvarint()?;
                if self.opts.strict && u <= i64::MAX as u64 {
                    return Err(CowrieError::NonCanonical("uint fits Int".into()));
                }
                ValueRef::Uint(u)
            }
            tags::FLOAT64 => {
                let bytes = self.slice_fixed::<8>()?;
                if self.opts.strict {
                    if *bytes == [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80] {
                        return Err(CowrieError::NonCanonical("negative zero".into()));
                    }
                    let v = f64::from_le_bytes(*bytes);
                    if v.is_nan() && *bytes != [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf8, 0x7f] {
                        return Err(CowrieError::NonCanonical("non-canonical NaN".into()));
                    }
                }
                ValueRef::Float(f64::from_le_bytes(*bytes))
            }
            tags::DECIMAL128 => {
                let scale_start = self.pos;
                let _scale = self.read_svarint()?;
                let scale_bytes = &self.data[scale_start..self.pos];
                let coeff_bytes = self.slice_fixed::<16>()?;
                if self.opts.strict {
                    let coeff = i128::from_le_bytes(*coeff_bytes);
                    let non_canonical = if coeff == 0 {
                        _scale != 0
                    } else {
                        coeff % 10 == 0
                    };
                    if non_canonical {
                        return Err(CowrieError::NonCanonical(
                            "decimal not lowest terms".into(),
                        ));
                    }
                }
                let start = scale_start;
                let end = scale_bytes.len() + start + 16;
                ValueRef::Decimal(&self.data[start..end])
            }
            tags::STRING => {
                let s = self.read_str()?;
                ValueRef::String(s)
            }
            tags::BYTES => {
                let len = self.read_uvarint()? as usize;
                if len > self.opts.max_bytes_len {
                    return Err(CowrieError::TooLarge);
                }
                ValueRef::Bytes(self.slice(len)?)
            }
            tags::DATETIME64 => {
                let b = self.slice_fixed::<8>()?;
                ValueRef::DateTime(i64::from_le_bytes(*b))
            }
            tags::UUID128 => {
                ValueRef::Uuid(self.slice_fixed::<16>()?)
            }
            tags::BIGINT => {
                let len = self.read_uvarint()? as usize;
                if len > self.opts.max_bytes_len {
                    return Err(CowrieError::TooLarge);
                }
                let raw = self.slice(len)?;
                if self.opts.strict {
                    check_bigint_canonical(raw)?;
                }
                ValueRef::BigInt(raw)
            }
            tags::ARRAY => {
                let len = self.read_uvarint()? as usize;
                if self.opts.strict && len <= 15 {
                    return Err(CowrieError::NonCanonical(
                        "array should use FIXARRAY".into(),
                    ));
                }
                if len > self.opts.max_array_len {
                    return Err(CowrieError::TooLarge);
                }
                let mut arr = Vec::with_capacity(len);
                for _ in 0..len {
                    arr.push(self.decode_value()?);
                }
                ValueRef::Array(arr)
            }
            tags::OBJECT => {
                let len = self.read_uvarint()? as usize;
                if self.opts.strict && len <= 15 {
                    return Err(CowrieError::NonCanonical("object should use FIXMAP".into()));
                }
                if len > self.opts.max_object_len {
                    return Err(CowrieError::TooLarge);
                }
                let mut obj = BTreeMap::new();
                let mut last: i64 = -1;
                for _ in 0..len {
                    let key_idx = self.read_uvarint()? as usize;
                    if key_idx >= self.dict.len() {
                        return Err(CowrieError::InvalidDictIndex {
                            index: key_idx,
                            dict_len: self.dict.len(),
                        });
                    }
                    if self.opts.strict && (key_idx as i64) <= last {
                        return Err(CowrieError::NonCanonical(
                            "object fields not in ascending dictIdx".into(),
                        ));
                    }
                    last = key_idx as i64;
                    let key = self.dict[key_idx];
                    let val = self.decode_value()?;
                    obj.insert(key, val);
                }
                ValueRef::Object(obj)
            }
            tags::EXT => {
                let type_id = self.read_uvarint()?;
                let len = self.read_uvarint()? as usize;
                if len > self.opts.max_ext_len {
                    return Err(CowrieError::TooLarge);
                }
                let payload = self.slice(len)?;
                ValueRef::Ext(ExtView { type_id, payload })
            }
            tags::TENSOR => {
                let dtype = DType::try_from(self.read_byte()?)?;
                let rank = self.read_byte()? as usize;
                if rank > self.opts.max_rank {
                    return Err(CowrieError::RankExceeded {
                        rank,
                        max: self.opts.max_rank,
                    });
                }
                let mut shape = Vec::with_capacity(rank);
                for _ in 0..rank {
                    shape.push(self.read_uvarint()?);
                }
                let data_len = self.read_uvarint()? as usize;
                if data_len > self.opts.max_bytes_len {
                    return Err(CowrieError::TooLarge);
                }
                if let Some(expected) = tensor_expected_bytes(dtype, &shape) {
                    if data_len != expected {
                        return Err(CowrieError::InvalidData(format!(
                            "tensor dataLen {} does not match shape/dtype (expected {} bytes)",
                            data_len, expected
                        )));
                    }
                }
                let pad = (super::decode::TENSOR_ALIGN - (self.pos % super::decode::TENSOR_ALIGN))
                    % super::decode::TENSOR_ALIGN;
                let pad_bytes = self.slice(pad)?;
                if self.opts.strict && pad_bytes.iter().any(|&b| b != 0) {
                    return Err(CowrieError::NonCanonical(
                        "tensor alignment padding not zero".into(),
                    ));
                }
                let data = self.slice(data_len)?;
                if self.opts.strict {
                    if let Some(total_bits) = tensor_total_bits(dtype, &shape) {
                        let rem = (total_bits % 8) as u32;
                        if rem != 0 {
                            if let Some(&last_byte) = data.last() {
                                if last_byte >> rem != 0 {
                                    return Err(CowrieError::NonCanonical(
                                        "tensor sub-byte padding not zero".into(),
                                    ));
                                }
                            }
                        }
                    }
                }
                ValueRef::Tensor(TensorView {
                    dtype,
                    shape,
                    data,
                })
            }
            tags::BITMASK => {
                let count = self.read_uvarint()?;
                let byte_len = count.div_ceil(8) as usize;
                if byte_len > self.opts.max_bytes_len {
                    return Err(CowrieError::TooLarge);
                }
                let bits = self.slice(byte_len)?;
                if self.opts.strict {
                    let rem = (count % 8) as u32;
                    if rem != 0 {
                        if let Some(&last_byte) = bits.last() {
                            if last_byte >> rem != 0 {
                                return Err(CowrieError::NonCanonical(
                                    "bitmask trailing bits not zero".into(),
                                ));
                            }
                        }
                    }
                }
                ValueRef::Bitmask { count, bits }
            }
            t if (tags::FIXINT_BASE..=tags::FIXINT_MAX).contains(&t) => {
                ValueRef::Int((t - tags::FIXINT_BASE) as i64)
            }
            t if (tags::FIXARRAY_BASE..=tags::FIXARRAY_MAX).contains(&t) => {
                let len = (t - tags::FIXARRAY_BASE) as usize;
                if len > self.opts.max_array_len {
                    return Err(CowrieError::TooLarge);
                }
                let mut arr = Vec::with_capacity(len);
                for _ in 0..len {
                    arr.push(self.decode_value()?);
                }
                ValueRef::Array(arr)
            }
            t if (tags::FIXMAP_BASE..=tags::FIXMAP_MAX).contains(&t) => {
                let len = (t - tags::FIXMAP_BASE) as usize;
                if len > self.opts.max_object_len {
                    return Err(CowrieError::TooLarge);
                }
                let mut obj = BTreeMap::new();
                let mut last: i64 = -1;
                for _ in 0..len {
                    let key_idx = self.read_uvarint()? as usize;
                    if key_idx >= self.dict.len() {
                        return Err(CowrieError::InvalidDictIndex {
                            index: key_idx,
                            dict_len: self.dict.len(),
                        });
                    }
                    if self.opts.strict && (key_idx as i64) <= last {
                        return Err(CowrieError::NonCanonical(
                            "object fields not in ascending dictIdx".into(),
                        ));
                    }
                    last = key_idx as i64;
                    let key = self.dict[key_idx];
                    let val = self.decode_value()?;
                    obj.insert(key, val);
                }
                ValueRef::Object(obj)
            }
            t if (tags::FIXNEG_BASE..=tags::FIXNEG_MAX).contains(&t) => {
                ValueRef::Int(-1 - (t - tags::FIXNEG_BASE) as i64)
            }
            // §2.3 reserved: every non-core tag rejects, exactly like the owning decoder.
            _ => return Err(CowrieError::InvalidTag(tag)),
        };

        self.depth -= 1;
        Ok(value)
    }

    fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    fn read_byte(&mut self) -> Result<u8, CowrieError> {
        if self.pos >= self.data.len() {
            return Err(CowrieError::Truncated);
        }
        let b = self.data[self.pos];
        self.pos += 1;
        Ok(b)
    }

    fn slice(&mut self, len: usize) -> Result<&'a [u8], CowrieError> {
        let end = self
            .pos
            .checked_add(len)
            .ok_or(CowrieError::Truncated)?;
        if end > self.data.len() {
            return Err(CowrieError::Truncated);
        }
        let out = &self.data[self.pos..end];
        self.pos = end;
        Ok(out)
    }

    fn slice_fixed<const N: usize>(&mut self) -> Result<&'a [u8; N], CowrieError> {
        let end = self
            .pos
            .checked_add(N)
            .ok_or(CowrieError::Truncated)?;
        if end > self.data.len() {
            return Err(CowrieError::Truncated);
        }
        let out = &self.data[self.pos..end];
        let arr: &'a [u8; N] = out
            .try_into()
            .expect("slice length checked to be exactly N");
        self.pos = end;
        Ok(arr)
    }

    fn read_str(&mut self) -> Result<&'a str, CowrieError> {
        let len = self.read_uvarint()? as usize;
        if len > self.opts.max_string_len {
            return Err(CowrieError::TooLarge);
        }
        let raw = self.slice(len)?;
        std::str::from_utf8(raw).map_err(|_| CowrieError::InvalidUtf8)
    }

    fn read_uvarint(&mut self) -> Result<u64, CowrieError> {
        let mut result: u64 = 0;
        let mut shift: u32 = 0;
        let mut nbytes: usize = 0;
        loop {
            let b = self.read_byte()?;
            nbytes += 1;
            if shift >= 64 || (shift == 63 && (b & 0x7f) > 1) {
                return Err(CowrieError::InvalidData("uvarint overflows 64 bits".into()));
            }
            result |= ((b & 0x7f) as u64) << shift;
            if b & 0x80 == 0 {
                if b == 0 && nbytes > 1 {
                    return Err(CowrieError::InvalidData("overlong varint".into()));
                }
                break;
            }
            shift += 7;
        }
        Ok(result)
    }

    fn read_svarint(&mut self) -> Result<i64, CowrieError> {
        let z = self.read_uvarint()?;
        Ok(zigzag_decode(z))
    }
}

/// Strict-mode canonical-form check for BigInt raw little-endian two's-complement bytes.
/// Mirrors the owning decoder exactly (§5.3).
fn check_bigint_canonical(raw: &[u8]) -> Result<(), CowrieError> {
    if raw.is_empty() {
        return Err(CowrieError::NonCanonical("bigint fits Int/Uint".into()));
    }
    if raw.len() >= 2 {
        let last = raw[raw.len() - 1];
        let prev = raw[raw.len() - 2];
        let prev_sign = prev & 0x80 != 0;
        if (last == 0x00 && !prev_sign) || (last == 0xFF && prev_sign) {
            return Err(CowrieError::NonCanonical("non-minimal bigint".into()));
        }
    }
    let negative = raw[raw.len() - 1] & 0x80 != 0;
    if negative {
        if raw.len() <= 8 {
            return Err(CowrieError::NonCanonical("bigint fits Int/Uint".into()));
        }
    } else {
        let mut sig = raw.len();
        if raw[sig - 1] == 0x00 {
            sig -= 1;
        }
        if sig <= 8 {
            return Err(CowrieError::NonCanonical("bigint fits Int/Uint".into()));
        }
    }
    Ok(())
}

fn zigzag_decode(z: u64) -> i64 {
    ((z >> 1) as i64) ^ (-((z & 1) as i64))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gen2::encode::encode;
    use crate::gen2::decode::decode;
    use crate::gen2::types::{TensorData, Value};

    fn blob200() -> Vec<u8> {
        vec![9; 200]
    }

    /// Canonical-byte equality is the right parity predicate here: `PartialEq` on
    /// `Value` reports `Float(NaN) != Float(NaN)` even when both decoders are correct.
    /// `encode` normalises every zero (including -0.0) and every NaN to its single
    /// canonical form (SPEC-v1 §3), so byte equality is bit-exact and strictly
    /// stronger than `==`.
    fn canonical(v: &Value) -> Vec<u8> {
        encode(v).expect("canonical encode")
    }

    fn assert_parity(v: &Value) {
        let bytes = encode(v).expect("encode");
        let owned = decode(&bytes).expect("owned decode");
        let borrowed = decode_ref(&bytes).expect("borrowed decode");
        assert_eq!(
            canonical(&borrowed.to_owned()),
            canonical(&owned),
            "parity failed (canonical bytes)"
        );
        let strict_owned = crate::gen2::decode::decode_strict(&bytes);
        let strict_borrowed = decode_ref_strict(&bytes);
        match (strict_owned, strict_borrowed) {
            (Ok(a), Ok(b)) => assert_eq!(
                canonical(&b.to_owned()),
                canonical(&a),
                "strict parity failed (canonical bytes)"
            ),
            (Err(e1), Err(e2)) => assert_eq!(e1.err_code(), e2.err_code()),
            other => panic!("strict divergence: {:?}", other),
        }
    }

    #[test]
    fn test_ref_parity_primitives() {
        let vals: Vec<Value> = vec![
            Value::Null,
            Value::Bool(true),
            Value::Bool(false),
            Value::Int(42),
            Value::Int(-42),
            Value::Int(i64::MAX),
            Value::Int(i64::MIN),
            Value::Uint(u64::MAX),
            Value::Float(0.0),
            Value::Float(std::f64::consts::PI),
            Value::Float(f64::NAN),
            Value::Float(-0.0),
            Value::String("hello".into()),
            Value::String("".into()),
            Value::String("ünïcode ✓ 测试".into()),
            Value::Bytes(vec![1, 2, 3]),
            Value::Bytes(vec![]),
            Value::DateTime(1_762_000_000_000_000_000),
            Value::Uuid([7u8; 16]),
            Value::BigInt(vec![0x80, 0x00]),
            Value::Bitmask {
                count: 13,
                bits: vec![0b1010_0101, 0],
            },
        ];
        for v in &vals {
            assert_parity(v);
        }
    }

    #[test]
    fn test_ref_parity_containers_and_tensor() {
        let v = Value::object(vec![
            ("name", Value::String("Alice".into())),
            ("seq", Value::Int(7)),
            (
                "tags",
                Value::Array(vec![Value::String("x".into()), Value::Int(1)]),
            ),
            (
                "embedding",
                Value::Tensor(TensorData::new(
                    DType::Float32,
                    vec![2, 3],
                    (0..24).collect(),
                )),
            ),
            (
                "nested",
                Value::Object(BTreeMap::from([
                    ("deep".to_string(), Value::Bool(true)),
                    ("blob".to_string(), Value::Bytes(blob200())),
                ])),
            ),
        ]);
        assert_parity(&v);
    }

    #[test]
    fn test_ref_parity_borrowed_data_is_really_borrowed() {
        let v = Value::String("borscht".into());
        let bytes = encode(&v).unwrap();
        let r = decode_ref(&bytes).unwrap();
        match r {
            ValueRef::String(s) => {
                let ptr = s.as_ptr();
                // gen2 layout for a bare string: "COWR"(4) + version(1) + flags(1)
                // + dict-count uvarint(1) + STRING tag(1) + length uvarint(1) => payload at 9.
                let payload = b"borscht";
                let expected_off = bytes
                    .windows(payload.len())
                    .position(|w| w == payload)
                    .expect("string payload must appear in the encoded bytes");
                assert_eq!(expected_off, 9, "gen2 bare-string payload offset");
                assert_eq!(
                    unsafe { ptr.offset_from(bytes.as_ptr()) } as usize,
                    expected_off,
                    "borrowed slice must point at the payload, not a copy"
                );
                assert_eq!(s.as_bytes(), &bytes[expected_off..expected_off + payload.len()]);
            }
            other => panic!("expected string, got {:?}", other),
        }
    }

    #[test]
    fn test_ref_rejects_reserved_and_corrupt() {
        let raw = b"XX\x02\x00\x00";
        assert!(matches!(decode_ref(raw), Err(CowrieError::InvalidMagic)));
        assert!(matches!(decode_ref(b"SJ"), Err(CowrieError::Truncated)));
        use crate::gen2::types::EdgeData;
        let edge = Value::Edge(EdgeData::new("a", "b", "T", Default::default()));
        // 0x36 is reserved in v1: the encoder refuses it by default...
        assert!(matches!(
            encode(&edge),
            Err(CowrieError::InvalidTag(0x36))
        ));
        // ...and when explicitly forced, the decoder still refuses the bytes.
        let bytes = crate::gen2::encode::encode_with_options(
            &edge,
            &crate::gen2::encode::EncodeOptions {
                allow_reserved: true,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(matches!(decode_ref(&bytes), Err(CowrieError::InvalidTag(0x36))));
    }
}
