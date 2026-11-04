package bench

import (
	. "github.com/Neumenon/cowrie/go"
	"encoding/binary"
	"math"
	"math/rand"
	"testing"

	msgpack "github.com/vmihailenco/msgpack/v5"
)

// Compiled-vs-compiled fight: cowrie (Any API with tensorized embeddings)
// vs github.com/vmihailenco/msgpack/v5 on the SAME record shape as the bull
// case: nested record + 384xf32 embedding. msgpack gets the embedding as raw
// []byte (the size-fair encoding); cowrie gets a tensor.

type benchRecord struct {
	Seq       int64  `msgpack:"seq"`
	Session   string `msgpack:"session"`
	Agent     string `msgpack:"agent"`
	TsNs      int64  `msgpack:"ts_ns"`
	Level     string `msgpack:"level"`
	Action    string `msgpack:"action"`
	LatencyMs int64  `msgpack:"latency_ms"`
	TokensIn  int64  `msgpack:"tokens_in"`
	Ok        bool   `msgpack:"ok"`
	Embedding []byte `msgpack:"embedding"`
}

func newBenchRecord(rng *rand.Rand) *benchRecord {
	emb := make([]float32, 384)
	for i := range emb {
		emb[i] = rng.Float32() - 0.5
	}
	raw := make([]byte, len(emb)*4)
	for i, f := range emb {
		binary.LittleEndian.PutUint32(raw[i*4:], math.Float32bits(f))
	}
	return &benchRecord{
		Seq:       int64(rng.Intn(2000)),
		Session:   "sess_0",
		Agent:     "planner",
		TsNs:      1762000000000000000 + int64(rng.Intn(1000000)),
		Level:     "info",
		Action:    "tool_call",
		LatencyMs: int64(3 + rng.Intn(1800)),
		TokensIn:  int64(50 + rng.Intn(9000)),
		Ok:        true,
		Embedding: raw,
	}
}

func cowrieRecord(rec *benchRecord) *Value {
	m := map[string]*Value{
		"seq":        Int64(rec.Seq),
		"session":    String(rec.Session),
		"agent":      String(rec.Agent),
		"ts_ns":      Int64(rec.TsNs),
		"level":      String(rec.Level),
		"action":     String(rec.Action),
		"latency_ms": Int64(rec.LatencyMs),
		"tokens_in":  Int64(rec.TokensIn),
		"ok":         Bool(rec.Ok),
		"embedding":  Tensor(DTypeFloat32, []uint64{384}, rec.Embedding),
	}
	return ObjectFromMap(m)
}

func BenchmarkCowrieEncodeRecord(b *testing.B) {
	rng := rand.New(rand.NewSource(7))
	rec := newBenchRecord(rng)
	v := cowrieRecord(rec)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := Encode(v)
		mustOk(b, err)
	}
}

func BenchmarkMsgpackEncodeRecord(b *testing.B) {
	rng := rand.New(rand.NewSource(7))
	rec := newBenchRecord(rng)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := msgpack.Marshal(rec)
		mustOk(b, err)
	}
}

func BenchmarkCowrieDecodeRecord(b *testing.B) {
	rng := rand.New(rand.NewSource(7))
	v := cowrieRecord(newBenchRecord(rng))
	data, err := Encode(v)
	mustOk(b, err)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := Decode(data)
		mustOk(b, err)
	}
}

func BenchmarkMsgpackDecodeRecord(b *testing.B) {
	rng := rand.New(rand.NewSource(7))
	rec := newBenchRecord(rng)
	data, err := msgpack.Marshal(rec)
	mustOk(b, err)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var out benchRecord
		err := msgpack.Unmarshal(data, &out)
		mustOk(b, err)
	}
}

func mustOk(b *testing.B, err error) {
	b.Helper()
	if err != nil {
		b.Fatal(err)
	}
}
