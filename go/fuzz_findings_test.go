package cowrie

import (
	"bytes"
	"encoding/hex"
	"errors"
	"testing"
)

// Both cases below were found by `go test -fuzz=FuzzCanonicalLaw` within seconds. They are the
// regression pin: a strict decoder must either reject these bytes or reproduce them exactly.

// A bare message is uncompressed (§2.2). Accepting a non-zero compression byte silently downgrades
// the byte string, so the same value gets two spellings and two identities.
func TestCompressionByteInBareMessageIsRejected(t *testing.T) {
	data, err := hex.DecodeString("434f575201300041")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := Decode(data); !errors.Is(err, ErrUnsupportedCompression) {
		t.Fatalf("want ErrUnsupportedCompression, got %v", err)
	}
	if _, err := DecodeWithOptions(data, DecodeOptions{Strict: true}); !errors.Is(err, ErrUnsupportedCompression) {
		t.Fatalf("strict: want ErrUnsupportedCompression, got %v", err)
	}
}

// SPEC §2.3 says the decimal scale is an int64 svarint. A decoder that stores it in int32 truncates
// 6469928589 to -2120006003 and re-encodes different bytes — a silently different number.
func TestDecimalScaleBeyondInt32RoundTrips(t *testing.T) {
	data, err := hex.DecodeString("434f57520100000a9a9a9a9a3030303030303030303030303030313030")
	if err != nil {
		t.Fatal(err)
	}
	v, err := DecodeWithOptions(data, DecodeOptions{Strict: true})
	if err != nil {
		t.Fatalf("strict decode: %v", err)
	}
	if v.Type() != TypeDecimal128 {
		t.Fatalf("want Decimal128, got %v", v.Type())
	}
	out, err := Encode(v)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	if !bytes.Equal(out, data) {
		t.Fatalf("round-trip changed the bytes:\n  in  = %x\n  out = %x", data, out)
	}
}
