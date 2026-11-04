//! Bench: owning decode vs borrowed zero-copy decode on a record corpus.
//!
//! Corpus: an array of 100 records {id, session, agent, tags[..], flags, embedding 384xf64}.
//! Run: cargo run --release --example borrow_bench

use std::collections::BTreeMap;
use std::time::Instant;

use cowrie_rs::gen2::decode::decode;
use cowrie_rs::gen2::decode_ref::{decode_ref, ValueRef};
use cowrie_rs::gen2::types::{DType, TensorData, Value};

fn corpus(n: usize, emb_len: usize) -> Value {
    let mut recs = Vec::with_capacity(n);
    for i in 0..n {
        let mut obj = BTreeMap::new();
        obj.insert("id".to_string(), Value::Int(i as i64));
        obj.insert("session".to_string(), Value::String(format!("sess_{}", i / 50)));
        obj.insert("agent".to_string(), Value::String("planner".to_string()));
        obj.insert(
            "tags".to_string(),
            Value::Array(vec![
                Value::String("x".into()),
                Value::String("y".into()),
                Value::String("z".into()),
            ]),
        );
        obj.insert("tokens_in".to_string(), Value::Int((1200 + i) as i64));
        obj.insert("tokens_out".to_string(), Value::Int(88));
        obj.insert("ok".to_string(), Value::Bool(true));
        obj.insert(
            "embedding".to_string(),
            Value::Tensor(TensorData {
                dtype: DType::Float64,
                shape: vec![emb_len as u64],
                data: vec![0u8; emb_len * 8],
            }),
        );
        recs.push(Value::Object(obj));
    }
    Value::Array(recs)
}

fn main() {
    let v = corpus(100, 384);
    let bytes = cowrie_rs::gen2::encode::encode(&v).expect("encode");
    println!("corpus: 100 records x 384 f64, {} bytes", bytes.len());

    let iters = 2000;
    for _ in 0..50 {
        let _ = decode(&bytes).expect("warm decode");
        let _ = decode_ref(&bytes).expect("warm borrow");
    }

    let t0 = Instant::now();
    let mut sink_owned = 0usize;
    for _ in 0..iters {
        let v = decode(&bytes).expect("decode");
        sink_owned = sink_owned.wrapping_add(
            match &v {
                Value::Array(a) => a.len(),
                _ => 0,
            },
        );
    }
    let t_own = t0.elapsed().as_secs_f64();

    let t0 = Instant::now();
    let mut sink_ref = 0usize;
    for _ in 0..iters {
        let v = decode_ref(&bytes).expect("borrow");
        if let ValueRef::Array(arr) = &v {
            if let Some(ValueRef::Object(o)) = arr.first() {
                for k in o.keys() {
                    std::hint::black_box(k);
                }
            }
            sink_ref = sink_ref.wrapping_add(arr.len());
        }
    }
    let t_ref = t0.elapsed().as_secs_f64();

    let mb = (bytes.len() as f64 * iters as f64) / 1e6;
    println!(
        "decode    (owning) : {:>7.0} ns/op  {:>8.0} MB/s   sink={}",
        t_own * 1e9 / iters as f64,
        mb / t_own,
        sink_owned
    );
    println!(
        "decode_ref (borrow): {:>7.0} ns/op  {:>8.0} MB/s   sink={}",
        t_ref * 1e9 / iters as f64,
        mb / t_ref,
        sink_ref
    );
    println!("borrowed speedup  : {:.2}x", t_own / t_ref);
}
