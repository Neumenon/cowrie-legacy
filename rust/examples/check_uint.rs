//! Verify the reported wart: does `encode(Uint(n))` emit bytes that our own
//! strict decoder rejects? Run: cargo run --release --example check_uint
use cowrie_rs::gen2::decode::{decode, decode_strict};
use cowrie_rs::gen2::encode::encode;
use cowrie_rs::gen2::types::Value;

fn main() {
    let cases = [
        ("Uint(5)", Value::Uint(5)),
        ("Uint(i64::MAX as u64)", Value::Uint(i64::MAX as u64)),
        ("Uint(u64::MAX)", Value::Uint(u64::MAX)),
        ("Int(5)", Value::Int(5)),
    ];
    for (label, v) in cases {
        let bytes = encode(&v).expect("encode");
        let lenient = decode(&bytes);
        let strict = decode_strict(&bytes);
        println!(
            "{:<24} tag=0x{:02x} len={:<3} lenient={:<4} strict={}",
            label,
            bytes[6],
            bytes.len(),
            if lenient.is_ok() { "ok" } else { "ERR" },
            match strict {
                Ok(_) => "ok".to_string(),
                Err(e) => format!("{} ({})", e.err_code(), e),
            }
        );
    }
}
