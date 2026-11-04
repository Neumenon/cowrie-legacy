package bench

import (
	. "github.com/Neumenon/cowrie/go"
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"testing"

	msgpack "github.com/vmihailenco/msgpack/v5"
)

// The task people actually pay for: "give me record #137's embedding out of a
// 2,500-frame store." CWRF's footer makes it a footer walk + one frame decode;
// structured formats with no per-record byte identity must parse the whole
// stream to reach one record. Everything runs from memory; no disk tricks.

const taskN = 2500

var (
	taskCowrieFile []byte
	taskMsgpackAll []byte
	taskRecs       []benchRecord
)

func buildTaskData(b *testing.B) {
	b.Helper()
	if taskCowrieFile == nil {
		var err error
		vals := make([]*Value, 0, taskN)
		recs := make([]benchRecord, 0, taskN)
		rng := rand.New(rand.NewSource(9))
		for i := 0; i < taskN; i++ {
			rec := newBenchRecord(rng)
			rec.Session = fmt.Sprintf("sess_%d", i/50)
			recs = append(recs, *rec)
			vals = append(vals, cowrieRecord(rec))
		}
		taskCowrieFile, err = EncodeFile(vals)
		if err != nil {
			b.Fatal(err)
		}
		taskMsgpackAll = msgpackMust(b, recs)
		taskRecs = recs
	}
}

func mustEncode(b *testing.B, v *Value) []byte {
	b.Helper()
	data, err := Encode(v)
	mustOk(b, err)
	return data
}

func msgpackMust(b *testing.B, recs []benchRecord) []byte {
	b.Helper()
	data, err := msgpack.Marshal(recs)
	mustOk(b, err)
	return data
}

// readUvarint decodes a LEB128 unsigned varint at pos and returns the value and the position
// after it. Local copy: this module cannot reach the library's unexported reader.
func readUvarint(data []byte, pos int) (uint64, int, error) {
	var value uint64
	var shift uint
	for {
		if pos >= len(data) {
			return 0, pos, fmt.Errorf("uvarint: unexpected end of data")
		}
		b := data[pos]
		pos++
		if shift >= 64 {
			return 0, pos, fmt.Errorf("uvarint: overflow")
		}
		value |= uint64(b&0x7f) << shift
		if b&0x80 == 0 {
			return value, pos, nil
		}
		shift += 7
	}
}

// frameAt reads frame k via the CWRF footer: no body parsing.
func frameAt(data []byte, k int) ([]byte, error) {
	if len(data) < 12 {
		return nil, fmt.Errorf("cowrie file: too short")
	}
	// The u64 footer offset (little-endian) sits 12 bytes from the end; it POINTS
	// at the footer region (count + (off,len) pairs + merkle root).
	fo := int(binary.LittleEndian.Uint64(data[len(data)-12:]))
	pos := fo
	n, adv, err := readUvarint(data, pos)
	if err != nil {
		return nil, err
	}
	pos = adv
	if k < 0 || uint64(k) >= n {
		return nil, fmt.Errorf("frame %d out of range (count %d)", k, n)
	}
	for i := 0; i < k; i++ {
		_, advA, err := readUvarint(data, pos)
		if err != nil {
			return nil, err
		}
		pos = advA
		_, advB, err := readUvarint(data, pos)
		if err != nil {
			return nil, err
		}
		pos = advB
	}
	foff, advA, err := readUvarint(data, pos)
	if err != nil {
		return nil, err
	}
	pos = advA
	flen, advB, err := readUvarint(data, pos)
	if err != nil {
		return nil, err
	}
	if advB < 0 {
		return nil, fmt.Errorf("impossible")
	}
	return data[foff : foff+flen], nil
}

func u32le(b []byte) uint32 {
	return uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24
}

func f32SumLe(raw []byte) float64 {
	s := 0.0
	for i := 0; i+4 <= len(raw); i += 4 {
		s += float64(math.Float32frombits(u32le(raw[i:])))
	}
	return s
}

func BenchmarkTaskReadOneEmbeddingFromCWRF(b *testing.B) {
	buildTaskData(b)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		frame, err := frameAt(taskCowrieFile, 137)
		mustOk(b, err)
		v, err := Decode(frame)
		mustOk(b, err)
		emb := v.Get("embedding")
		if emb == nil || emb.Type() != TypeTensor {
			b.Fatal("missing embedding")
		}
		_ = f32SumLe(emb.Tensor().Data)
	}
}

func BenchmarkTaskMsgpackReadOneEmbedding(b *testing.B) {
	buildTaskData(b)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		recs, err := msgpackUnpackRecords(taskMsgpackAll)
		mustOk(b, err)
		_ = f32SumLe(recs[137].Embedding)
	}
}

func msgpackUnpackRecords(data []byte) ([]benchRecord, error) {
	var out []benchRecord
	if err := msgpack.Unmarshal(data, &out); err != nil {
		return nil, err
	}
	return out, nil
}
