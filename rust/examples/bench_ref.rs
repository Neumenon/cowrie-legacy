//! Borrowed-vs-owned decode benchmark for gen2.
//!
//! Measures, on the same encoded payload:
//!   * wall time per decode
//!   * number of heap allocations per decode (counting global allocator)
//! for `decode` (owned: every string/bytes/tensor is cloned into a new Vec/String)
//! versus `decode_ref` (borrowed: slices point straight into the input buffer).
//!
//! Run: cargo run --release --example bench_ref

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use cowrie_rs::gen2::decode::decode;
use cowrie_rs::gen2::decode_ref::decode_ref;
use cowrie_rs::gen2::encode::encode;
use cowrie_rs::gen2::types::{DType, TensorData, Value};

struct Counting;

static ALLOCS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

fn reset() {
    ALLOCS.store(0, Ordering::Relaxed);
    BYTES.store(0, Ordering::Relaxed);
}

fn snapshot() -> (usize, usize) {
    (ALLOCS.load(Ordering::Relaxed), BYTES.load(Ordering::Relaxed))
}

/// A frame shaped like a real agent-trace record: strings, nested object, blob, f32 embedding.
fn make_frame() -> Value {
    let mut emb = Vec::with_capacity(384 * 4);
    for i in 0..384u32 {
        emb.extend_from_slice(&(i as f32 * 0.001).to_le_bytes());
    }
    Value::object(vec![
        ("trace_id", Value::String("01J8Z3K4QW7X9N2V5B6M8C1T4R".into())),
        ("span", Value::String("planner.think.step".into())),
        ("model", Value::String("eureka-community-27b".into())),
        ("prompt_tokens", Value::Int(1_842)),
        ("completion_tokens", Value::Int(576)),
        ("cost_usd_micros", Value::Int(1_437)),
        ("utterance", Value::String(
            "Summarise the three highest-signal findings from the fabric reclaim run \
             and list the rebuild recipe for each deleted weight bundle.".into(),
        )),
        ("payload", Value::Bytes((0..1200u32).map(|i| (i % 251) as u8).collect())),
        (
            "meta",
            Value::object(vec![
                ("host", Value::String("omen".into())),
                ("lane", Value::String("trunk-program".into())),
                ("attempt", Value::Int(3)),
            ]),
        ),
        (
            "embedding",
            Value::Tensor(TensorData::new(DType::Float32, vec![384], emb)),
        ),
    ])
}

fn time_it<F: FnMut()>(iters: usize, mut f: F) -> f64 {
    // warm up
    for _ in 0..(iters / 10).max(1) {
        f();
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        f();
    }
    t0.elapsed().as_secs_f64() * 1e6 / iters as f64 // microseconds per op
}

fn main() {
    let value = make_frame();
    let bytes = encode(&value).expect("encode");
    let iters = 20_000;

    println!("payload: {} bytes ({} fields + 384xf32 tensor)", bytes.len(), 10);

    // ---- owned decode -------------------------------------------------------
    reset();
    let owned_us = time_it(iters, || {
        let v = decode(black_box(&bytes)).expect("owned decode");
        black_box(&v);
    });
    let (owned_allocs, owned_bytes) = snapshot();

    // ---- borrowed decode ----------------------------------------------------
    reset();
    let ref_us = time_it(iters, || {
        let v = decode_ref(black_box(&bytes)).expect("borrowed decode");
        // touch a borrowed field so the work cannot be optimised away
        if let cowrie_rs::gen2::decode_ref::ValueRef::Object(map) = &v {
            black_box(map.len());
        }
    });
    let (ref_allocs, ref_bytes) = snapshot();

    println!();
    println!("{:<24}{:>14}{:>16}{:>14}", "decoder", "us/frame", "allocs/frame", "bytes/frame");
    println!("{:<24}{:>14.3}{:>16.1}{:>14.0}", "decode (owned)", owned_us,
             owned_allocs as f64 / iters as f64, owned_bytes as f64 / iters as f64);
    println!("{:<24}{:>14.3}{:>16.1}{:>14.0}", "decode_ref (borrowed)", ref_us,
             ref_allocs as f64 / iters as f64, ref_bytes as f64 / iters as f64);
    println!();
    println!("speed delta   : {:.2}x faster ({:+.3} us/frame)", owned_us / ref_us, ref_us - owned_us);
    println!("alloc delta   : {:+.1} allocs/frame", (ref_allocs as f64 - owned_allocs as f64) / iters as f64);
}
