// Package bench is a standalone benchmark module: Cowrie vs encoding/json vs the
// real msgpack library (github.com/vmihailenco/msgpack/v5), on one v1-legal
// multimodal payload.
//
// It lives in its own module on purpose. Adding msgpack to the library's go.mod
// would push that requirement into every consumer's module graph for the sake of
// one benchmark number; a nested module keeps the library's dependency surface
// exactly as it is.
//
// Run:
//
//	cd bench && go test -bench . -benchtime 300ms
package bench

import (
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"math"
	"testing"

	cowrie "github.com/Neumenon/cowrie/go"
	"github.com/vmihailenco/msgpack/v5"
)

const blobSize = 50 * 1024

func blob() []byte {
	b := make([]byte, blobSize)
	for i := range b {
		b[i] = byte(i % 256)
	}
	return b
}

func floats() []float32 {
	f := make([]float32, 768)
	for i := range f {
		f[i] = float32(i) * 0.001
	}
	return f
}

func floatBytes(f []float32) []byte {
	out := make([]byte, len(f)*4)
	for i, v := range f {
		binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(v))
	}
	return out
}

func makeCowrie() *cowrie.Value {
	return cowrie.Object(
		cowrie.Member{Key: "blob", Value: cowrie.Bytes(blob())},
		cowrie.Member{Key: "prompt", Value: cowrie.String(promptText)},
		cowrie.Member{Key: "embedding", Value: cowrie.Tensor(cowrie.DTypeFloat32, []uint64{1, 768}, floatBytes(floats()))},
	)
}

const promptText = "Describe the contents of this image in detail, including objects, colors, and spatial relationships."

// makeMsgpack uses the same logical data: binary blob as []byte (msgpack bin),
// prompt as str, embedding as an array of float32.
func makeMsgpack() map[string]any {
	return map[string]any{
		"blob":      blob(),
		"prompt":    promptText,
		"embedding": floats(),
	}
}

func makeJSON() map[string]any {
	return map[string]any{
		"blob":      base64.StdEncoding.EncodeToString(blob()),
		"prompt":    promptText,
		"embedding": map[string]any{"shape": []int{1, 768}, "dtype": "float32", "data": floats()},
	}
}

// ------------------------------------------------------------------ encode

func BenchmarkEncode(b *testing.B) {
	cv := makeCowrie()
	mv := makeMsgpack()
	jv := makeJSON()

	b.Run("cowrie", func(b *testing.B) {
		b.ReportAllocs()
		var n int
		for i := 0; i < b.N; i++ {
			out, err := cowrie.Encode(cv)
			if err != nil {
				b.Fatal(err)
			}
			n = len(out)
		}
		b.SetBytes(int64(n))
	})

	b.Run("msgpack", func(b *testing.B) {
		b.ReportAllocs()
		var n int
		for i := 0; i < b.N; i++ {
			out, err := msgpack.Marshal(mv)
			if err != nil {
				b.Fatal(err)
			}
			n = len(out)
		}
		b.SetBytes(int64(n))
	})

	b.Run("json", func(b *testing.B) {
		b.ReportAllocs()
		var n int
		for i := 0; i < b.N; i++ {
			out, err := json.Marshal(jv)
			if err != nil {
				b.Fatal(err)
			}
			n = len(out)
		}
		b.SetBytes(int64(n))
	})
}

// ------------------------------------------------------------------ decode

func BenchmarkDecode(b *testing.B) {
	cv := makeCowrie()
	mv := makeMsgpack()
	jv := makeJSON()

	cBytes, err := cowrie.Encode(cv)
	if err != nil {
		b.Fatal(err)
	}
	mBytes, err := msgpack.Marshal(mv)
	if err != nil {
		b.Fatal(err)
	}
	jBytes, err := json.Marshal(jv)
	if err != nil {
		b.Fatal(err)
	}

	b.Run("cowrie_strict", func(b *testing.B) {
		b.ReportAllocs()
		opts := cowrie.DecodeOptions{Strict: true}
		for i := 0; i < b.N; i++ {
			if _, err := cowrie.DecodeWithOptions(cBytes, opts); err != nil {
				b.Fatal(err)
			}
		}
	})

	b.Run("msgpack", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			var out map[string]any
			if err := msgpack.Unmarshal(mBytes, &out); err != nil {
				b.Fatal(err)
			}
		}
	})

	b.Run("json", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			var out map[string]any
			if err := json.Unmarshal(jBytes, &out); err != nil {
				b.Fatal(err)
			}
		}
	})
}

// ------------------------------------------------------------------- report

// TestSizeReport prints the byte counts and proves the Cowrie payload is a
// strictly valid v1 message (something the old in-repo benchmark could not say).
func TestSizeReport(t *testing.T) {
	cBytes, err := cowrie.Encode(makeCowrie())
	if err != nil {
		t.Fatalf("cowrie encode: %v", err)
	}
	if _, err := cowrie.DecodeWithOptions(cBytes, cowrie.DecodeOptions{Strict: true}); err != nil {
		t.Fatalf("cowrie strict decode of our own bytes: %v", err)
	}
	mBytes, err := msgpack.Marshal(makeMsgpack())
	if err != nil {
		t.Fatalf("msgpack marshal: %v", err)
	}
	jBytes, err := json.Marshal(makeJSON())
	if err != nil {
		t.Fatalf("json marshal: %v", err)
	}
	t.Logf("payload sizes: cowrie=%d msgpack=%d json=%d  (cowrie/json=%.3f, msgpack/json=%.3f)",
		len(cBytes), len(mBytes), len(jBytes),
		float64(len(cBytes))/float64(len(jBytes)), float64(len(mBytes))/float64(len(jBytes)))
}
