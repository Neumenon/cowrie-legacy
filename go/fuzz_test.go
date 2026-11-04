// Fuzz targets. Run with: go test -fuzz=FuzzCanonicalLaw -fuzztime=60s .
//
// These target the invariant, not just panic-freedom: if strict decode accepts a byte string, that
// byte string IS canonical, so re-encoding the value must reproduce it exactly. tools/law_bytes_gate.py
// checks this over a generated corpus; fuzzing checks it over inputs nobody thought to generate.
package cowrie

import (
	"bytes"
	"testing"
)

var fuzzSeeds = [][]byte{
	{0x43, 0x4f, 0x57, 0x52, 0x01, 0x00, 0x00, 0x45},                   // Int(5)
	{0x43, 0x4f, 0x57, 0x52, 0x01, 0x00, 0x02, 0x61, 0x62, 0xd2, // {"a":1,"b":2}
		0x00, 0x41, 0x01, 0x42},
	{0x43, 0x4f, 0x57, 0x52, 0x01, 0x00, 0x00, 0xc0},                   // null
	{0x43, 0x4f, 0x57, 0x52, 0x01, 0x00, 0x00, 0xf7},                   // canonical NaN
	{0x43, 0x4f, 0x57, 0x52, 0x01, 0x00, 0x00, 0x0d, 0x09, 0x00, 0x00, // 2^64 as BigInt
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01},
}

func FuzzDecode(f *testing.F) {
	for _, s := range fuzzSeeds {
		f.Add(s)
	}
	f.Fuzz(func(t *testing.T, data []byte) {
		// Arbitrary input must never panic, whatever it is.
		_, _ = Decode(data)
	})
}

// FuzzCanonicalLaw is the property the format exists to provide.
func FuzzCanonicalLaw(f *testing.F) {
	for _, s := range fuzzSeeds {
		f.Add(s)
	}
	f.Fuzz(func(t *testing.T, data []byte) {
		opts := DecodeOptions{Strict: true}
		v, err := DecodeWithOptions(data, opts)
		if err != nil {
			return // rejection is always allowed; only acceptance makes a promise
		}
		out, encErr := Encode(v)
		if encErr != nil {
			t.Fatalf("strict decode accepted %x, but re-encode failed: %v", data, encErr)
		}
		if !bytes.Equal(out, data) {
			t.Fatalf("strict decode accepted non-canonical bytes:\n  in  = %x\n  out = %x", data, out)
		}
	})
}

func FuzzFile(f *testing.F) {
	f.Add([]byte{0x43, 0x57, 0x52, 0x46, 0x01, 0x00, 0x00})
	f.Add([]byte("CWRF\x01\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x0e\x01\x00\x45"))
	f.Fuzz(func(t *testing.T, data []byte) {
		if _, err := DecodeFile(data); err != nil {
			return
		}
		// A file that validates must also re-encode to identical bytes (§7: one canonical
		// file per ordered value list) and carry a stable identity.
		again, err := RecodeFile(data)
		if err != nil {
			t.Fatalf("valid file failed to recode: %v", err)
		}
		if !bytes.Equal(again, data) {
			t.Fatalf("file validated but re-encoded differently:\n  in  = %x\n  out = %x", data, again)
		}
		if _, err := FileIdentity(data); err != nil {
			t.Fatalf("valid file has no identity: %v", err)
		}
	})
}
