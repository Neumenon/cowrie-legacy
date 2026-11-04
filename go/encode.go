package cowrie

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"math/big"
	"sort"
	"sync"
)

// Buffer pool for encoding - reduces allocations in hot paths
var bufferPool = sync.Pool{
	New: func() any {
		return &buffer{data: make([]byte, 0, 4096)}
	},
}

// maxPoolBufSize is the cap above which buffers are not returned to the pool.
const maxPoolBufSize = 1 << 20 // 1MB

// ErrReservedTag is returned by the encoder for value kinds whose wire tags v1
// reserves (SPEC-v1 §2.3): TensorRef(0x21), Image(0x22), Audio(0x23), Node(0x35),
// Edge(0x36), NodeBatch(0x37), EdgeBatch(0x38). Decoders reject those tags, so
// encoding them by default would produce a message no conforming v1 reader —
// including this package's own Decode — can read. Set EncodeOptions.AllowReserved
// to emit pre-v1 (attic) bytes anyway.
var ErrReservedTag = errors.New("cowrie: value uses a tag reserved in v1")

// putBuf returns a buffer to the pool if it's not oversized.
func putBuf(buf *buffer) {
	if cap(buf.data) <= maxPoolBufSize {
		bufferPool.Put(buf)
	}
}

// Encode encodes a value to Cowrie v2 binary format.
func Encode(v *Value) ([]byte, error) {
	return encodeTop(v, false)
}

// encodeTop is the shared entry point; allowReserved selects the tag policy.
func encodeTop(v *Value, allowReserved bool) ([]byte, error) {
	buf := bufferPool.Get().(*buffer)
	buf.data = buf.data[:0] // reset length, keep capacity

	if err := encode(buf, v, allowReserved); err != nil {
		putBuf(buf)
		return nil, err
	}

	// Copy result (can't return pooled buffer)
	out := make([]byte, len(buf.data))
	copy(out, buf.data)

	putBuf(buf)
	return out, nil
}

// EncodeAppend encodes v and appends the result to dst.
// Useful for avoiding allocations when the caller manages buffers.
func EncodeAppend(dst []byte, v *Value) ([]byte, error) {
	buf := buffer{data: dst}
	if err := encode(&buf, v, false); err != nil {
		return nil, err
	}
	return buf.data, nil
}

// EncodeTo encodes a value to an io.Writer.
func EncodeTo(w io.Writer, v *Value) error {
	data, err := Encode(v)
	if err != nil {
		return err
	}
	_, err = w.Write(data)
	return err
}

// buffer is a simple byte buffer for encoding.
type buffer struct {
	data []byte
}

func (b *buffer) bytes() []byte {
	return b.data
}

func (b *buffer) writeByte(c byte) {
	b.data = append(b.data, c)
}

func (b *buffer) write(p []byte) {
	b.data = append(b.data, p...)
}

func (b *buffer) writeUvarint(v uint64) {
	var buf [10]byte
	n := binary.PutUvarint(buf[:], v)
	b.write(buf[:n])
}

func (b *buffer) writeString(s string) {
	b.writeUvarint(uint64(len(s)))
	b.write([]byte(s))
}

func (b *buffer) writeUint16LE(v uint16) {
	var buf [2]byte
	binary.LittleEndian.PutUint16(buf[:], v)
	b.write(buf[:])
}

func (b *buffer) writeUint32LE(v uint32) {
	var buf [4]byte
	binary.LittleEndian.PutUint32(buf[:], v)
	b.write(buf[:])
}

func (b *buffer) writeInt32LE(v int32) {
	var buf [4]byte
	binary.LittleEndian.PutUint32(buf[:], uint32(v))
	b.write(buf[:])
}

// zigzagEncode encodes a signed integer using zigzag encoding.
func zigzagEncode(n int64) uint64 {
	return uint64((n << 1) ^ (n >> 63))
}

// dictionary for field name encoding
type dict struct {
	keys   []string
	lookup map[string]int

	// allowReserved carries the encoder's tag policy down the recursive walk. The
	// dictionary is the one structure already threaded through every encode path,
	// so the flag rides here rather than widening encodeValue/encodeAny signatures.
	allowReserved bool
}

func newDict() *dict {
	return &dict{
		lookup: make(map[string]int),
	}
}

func (d *dict) add(key string) int {
	if idx, ok := d.lookup[key]; ok {
		return idx
	}
	idx := len(d.keys)
	d.keys = append(d.keys, key)
	d.lookup[key] = idx
	return idx
}

func (d *dict) get(key string) int {
	return d.lookup[key]
}

// sortGlobal byte-sorts the whole dictionary and re-indexes it, so dict order is the
// SPEC-v1 §2.4 canonical (global byte-sorted) order rather than DFS-discovery order.
func (d *dict) sortGlobal() {
	sort.Slice(d.keys, func(i, j int) bool { return d.keys[i] < d.keys[j] })
	for i, k := range d.keys {
		d.lookup[k] = i
	}
}

// collectKeys recursively collects all object keys into the dictionary.
func collectKeys(v *Value, d *dict) {
	if v == nil {
		return
	}
	switch v.typ {
	case TypeArray:
		for _, item := range v.arrayVal {
			collectKeys(item, d)
		}
	case TypeObject:
		for _, m := range v.objectVal {
			d.add(m.Key)
			collectKeys(m.Value, d)
		}
	case TypeNode:
		// Collect node property keys (including nested maps)
		collectKeysFromAnyMap(v.nodeVal.Props, d)
	case TypeEdge:
		// Collect edge property keys (including nested maps)
		collectKeysFromAnyMap(v.edgeVal.Props, d)
	case TypeNodeBatch:
		// Collect property keys from all nodes
		for _, node := range v.nodeBatchVal.Nodes {
			collectKeysFromAnyMap(node.Props, d)
		}
	case TypeEdgeBatch:
		// Collect property keys from all edges
		for _, edge := range v.edgeBatchVal.Edges {
			collectKeysFromAnyMap(edge.Props, d)
		}
	}
}

// collectKeysFromAnyMap recursively collects keys from a map[string]any.
// Keys are visited in sorted order so that dictionary indices are deterministic.
func collectKeysFromAnyMap(m map[string]any, d *dict) {
	for _, k := range sortedMapKeys(m) {
		d.add(k)
		// Recursively collect keys from nested maps and arrays
		collectKeysFromAnyValue(m[k], d)
	}
}

// collectKeysFromAnyValue recursively collects keys from nested structures.
func collectKeysFromAnyValue(v any, d *dict) {
	if v == nil {
		return
	}
	switch val := v.(type) {
	case map[string]any:
		collectKeysFromAnyMap(val, d)
	case []any:
		for _, item := range val {
			collectKeysFromAnyValue(item, d)
		}
	case *Value:
		collectKeys(val, d)
	}
}

// encode writes the complete Cowrie v2 format to the buffer.
func encode(buf *buffer, v *Value, allowReserved bool) error {
	// Build dictionary (global byte-sorted, §2.4)
	d := newDict()
	d.allowReserved = allowReserved
	collectKeys(v, d)
	d.sortGlobal()

	// Write header (SPEC-v1 §2.2: COWR + version + compression byte)
	buf.writeByte(Magic0)
	buf.writeByte(Magic1)
	buf.writeByte(Magic2)
	buf.writeByte(Magic3)
	buf.writeByte(Version)
	buf.writeByte(0) // compression = none (0x00)

	// Write dictionary
	buf.writeUvarint(uint64(len(d.keys)))
	for _, key := range d.keys {
		buf.writeString(key)
	}

	// Write root value
	return encodeValue(buf, v, d)
}

// encodeValue writes a single value to the buffer.
func encodeValue(buf *buffer, v *Value, d *dict) error {
	if v == nil {
		buf.writeByte(TagNull)
		return nil
	}

	switch v.typ {
	case TypeNull:
		buf.writeByte(TagNull)

	case TypeBool:
		if v.boolVal {
			buf.writeByte(TagTrue)
		} else {
			buf.writeByte(TagFalse)
		}

	case TypeInt64:
		writeIntInto(buf, v.int64Val)

	case TypeUint64:
		writeUintInto(buf, v.uint64Val)

	case TypeFloat64:
		buf.writeByte(TagFloat64)
		var bits [8]byte
		binary.LittleEndian.PutUint64(bits[:], uint64FromFloat64(v.float64Val))
		buf.write(bits[:])

	case TypeDecimal128:
		writeDecimalInto(buf, v.decimal128.Scale, v.decimal128.Coef)

	case TypeString:
		buf.writeByte(TagString)
		buf.writeString(v.stringVal)

	case TypeBytes:
		buf.writeByte(TagBytes)
		buf.writeUvarint(uint64(len(v.bytesVal)))
		buf.write(v.bytesVal)

	case TypeDatetime64:
		buf.writeByte(TagDatetime64)
		var bits [8]byte
		binary.LittleEndian.PutUint64(bits[:], uint64(v.datetime64))
		buf.write(bits[:])

	case TypeUUID128:
		buf.writeByte(TagUUID128)
		buf.write(v.uuid128[:])

	case TypeBigInt:
		writeBigIntInto(buf, v.bigintVal)

	case TypeArray:
		// v3 inline encoding: FIXARRAY for length 0-15
		n := len(v.arrayVal)
		if n <= 15 {
			buf.writeByte(byte(TagFixarrayBase + n))
		} else {
			buf.writeByte(TagArray)
			buf.writeUvarint(uint64(n))
		}
		for _, item := range v.arrayVal {
			if err := encodeValue(buf, item, d); err != nil {
				return err
			}
		}

	case TypeObject:
		// v3 inline encoding: FIXMAP for count 0-15
		n := len(v.objectVal)
		if n <= 15 {
			buf.writeByte(byte(TagFixmapBase + n))
		} else {
			buf.writeByte(TagObject)
			buf.writeUvarint(uint64(n))
		}
		// §2.4 canonical: object fields MUST be emitted in strictly ascending
		// dictIdx order. The dictionary is globally byte-sorted, so we order the
		// members by their resolved dict index (equivalently, by key) here rather
		// than relying on source/insertion order.
		for _, m := range sortedMembersByDictIdx(v.objectVal, d) {
			idx := d.get(m.Key)
			buf.writeUvarint(uint64(idx))
			if err := encodeValue(buf, m.Value, d); err != nil {
				return err
			}
		}

	// v2.1 Extension Types
	case TypeTensor:
		// Validate dataLen against shape/dtype on encode too, so we never emit
		// bytes the decoder would reject (symmetric with decode.go's check).
		if expected, ok := tensorExpectedBytes(v.tensorVal.DType, v.tensorVal.Dims); ok {
			if uint64(len(v.tensorVal.Data)) != expected {
				return fmt.Errorf("cowrie: tensor dataLen %d does not match shape/dtype (expected %d bytes)", len(v.tensorVal.Data), expected)
			}
		}
		buf.writeByte(TagTensor)
		buf.writeByte(byte(v.tensorVal.DType))
		buf.writeByte(byte(len(v.tensorVal.Dims))) // rank
		for _, dim := range v.tensorVal.Dims {
			buf.writeUvarint(dim)
		}
		buf.writeUvarint(uint64(len(v.tensorVal.Data)))
		// §2.5: tensor data MUST begin at a 64-byte boundary relative to byte 0
		// of the message. len(buf.data) is the absolute message offset (encoding
		// starts at byte 0), so pad = (-offset) mod 64 zero bytes before the data.
		// Position-determined: exactly one canonical byte-string.
		pad := (-len(buf.data)) & (TensorAlign - 1)
		for i := 0; i < pad; i++ {
			buf.writeByte(0)
		}
		buf.write(v.tensorVal.Data)

	case TypeTensorRef:
		if !d.allowReserved {
			return ErrReservedTag
		}
		buf.writeByte(TagTensorRef)
		buf.writeByte(v.tensorRefVal.StoreID)
		buf.writeUvarint(uint64(len(v.tensorRefVal.Key)))
		buf.write(v.tensorRefVal.Key)

	case TypeImage:
		if !d.allowReserved {
			return ErrReservedTag
		}
		buf.writeByte(TagImage)
		buf.writeByte(byte(v.imageVal.Format))
		buf.writeUint16LE(v.imageVal.Width)
		buf.writeUint16LE(v.imageVal.Height)
		buf.writeUvarint(uint64(len(v.imageVal.Data)))
		buf.write(v.imageVal.Data)

	case TypeAudio:
		if !d.allowReserved {
			return ErrReservedTag
		}
		buf.writeByte(TagAudio)
		buf.writeByte(byte(v.audioVal.Encoding))
		buf.writeUint32LE(v.audioVal.SampleRate)
		buf.writeByte(v.audioVal.Channels)
		buf.writeUvarint(uint64(len(v.audioVal.Data)))
		buf.write(v.audioVal.Data)

	case TypeNode:
		if !d.allowReserved {
			return ErrReservedTag
		}
		buf.writeByte(TagNode)
		// ID
		buf.writeString(v.nodeVal.ID)
		// Labels
		buf.writeUvarint(uint64(len(v.nodeVal.Labels)))
		for _, label := range v.nodeVal.Labels {
			buf.writeString(label)
		}
		// Properties (dictionary-coded keys, sorted for determinism)
		buf.writeUvarint(uint64(len(v.nodeVal.Props)))
		for _, k := range sortedMapKeys(v.nodeVal.Props) {
			buf.writeUvarint(uint64(d.get(k)))
			if err := encodeAny(buf, v.nodeVal.Props[k], d); err != nil {
				return err
			}
		}

	case TypeEdge:
		if !d.allowReserved {
			return ErrReservedTag
		}
		buf.writeByte(TagEdge)
		// Source and destination IDs
		buf.writeString(v.edgeVal.From)
		buf.writeString(v.edgeVal.To)
		// Edge type
		buf.writeString(v.edgeVal.Type)
		// Properties (dictionary-coded keys, sorted for determinism)
		buf.writeUvarint(uint64(len(v.edgeVal.Props)))
		for _, k := range sortedMapKeys(v.edgeVal.Props) {
			buf.writeUvarint(uint64(d.get(k)))
			if err := encodeAny(buf, v.edgeVal.Props[k], d); err != nil {
				return err
			}
		}

	case TypeNodeBatch:
		if !d.allowReserved {
			return ErrReservedTag
		}
		buf.writeByte(TagNodeBatch)
		buf.writeUvarint(uint64(len(v.nodeBatchVal.Nodes)))
		for _, node := range v.nodeBatchVal.Nodes {
			if err := encodeNodeData(buf, &node, d); err != nil {
				return err
			}
		}

	case TypeEdgeBatch:
		if !d.allowReserved {
			return ErrReservedTag
		}
		buf.writeByte(TagEdgeBatch)
		buf.writeUvarint(uint64(len(v.edgeBatchVal.Edges)))
		for _, edge := range v.edgeBatchVal.Edges {
			if err := encodeEdgeData(buf, &edge, d); err != nil {
				return err
			}
		}

	case TypeBitmask:
		buf.writeByte(TagBitmask)
		buf.writeUvarint(v.bitmaskVal.Count)
		buf.write(v.bitmaskVal.Bits)

	case TypeUnknownExt:
		// Re-encode unknown extensions to enable round-trip preservation
		buf.writeByte(TagExt)
		buf.writeUvarint(v.unknownExtVal.ExtType)
		buf.writeUvarint(uint64(len(v.unknownExtVal.Payload)))
		buf.write(v.unknownExtVal.Payload)

	default:
		return fmt.Errorf("cowrie: unsupported type %v (%d)", v.typ, v.typ)
	}

	return nil
}

// uint64FromFloat64 converts a float64 to its canonical bit representation (§3).
// Normalizes on encode so the encoder never emits bytes its own strict decoder
// would reject: -0.0 is written as +0.0 (all-zero 8 bytes) and any NaN
// (including signaling / non-canonical bit patterns) is written as the one
// canonical quiet NaN 0x7FF8000000000000 (LE bytes 000000000000f87f).
// Mirrors tools/cowrie_ref/encode.py encode_value float branch.
func uint64FromFloat64(f float64) uint64 {
	if f == 0.0 {
		// covers +0.0 and -0.0 -> canonical +0.0
		return 0
	}
	if f != f {
		// any NaN -> canonical quiet NaN
		return 0x7FF8000000000000
	}
	return math.Float64bits(f)
}

// ============================================================
// Graph type encoding helpers
// ============================================================

// encodeNodeData encodes a NodeData struct (without the tag byte).
func encodeNodeData(buf *buffer, node *NodeData, d *dict) error {
	// ID
	buf.writeString(node.ID)
	// Labels
	buf.writeUvarint(uint64(len(node.Labels)))
	for _, label := range node.Labels {
		buf.writeString(label)
	}
	// Properties (dictionary-coded keys, sorted for determinism)
	buf.writeUvarint(uint64(len(node.Props)))
	for _, k := range sortedMapKeys(node.Props) {
		buf.writeUvarint(uint64(d.get(k)))
		if err := encodeAny(buf, node.Props[k], d); err != nil {
			return err
		}
	}
	return nil
}

// encodeEdgeData encodes an EdgeData struct (without the tag byte).
func encodeEdgeData(buf *buffer, edge *EdgeData, d *dict) error {
	// Source and destination IDs
	buf.writeString(edge.From)
	buf.writeString(edge.To)
	// Edge type
	buf.writeString(edge.Type)
	// Properties (dictionary-coded keys, sorted for determinism)
	buf.writeUvarint(uint64(len(edge.Props)))
	for _, k := range sortedMapKeys(edge.Props) {
		buf.writeUvarint(uint64(d.get(k)))
		if err := encodeAny(buf, edge.Props[k], d); err != nil {
			return err
		}
	}
	return nil
}

// encodeAny encodes a Go any value to Cowrie.
// Supports basic types: nil, bool, int/int64, uint/uint64, float64, string, []byte,
// []any, map[string]any, and *Value.
func encodeAny(buf *buffer, v any, d *dict) error {
	if v == nil {
		buf.writeByte(TagNull)
		return nil
	}
	switch val := v.(type) {
	case bool:
		if val {
			buf.writeByte(TagTrue)
		} else {
			buf.writeByte(TagFalse)
		}
	case int:
		if val >= 0 && val <= 127 {
			buf.writeByte(byte(TagFixintBase + val))
		} else if val >= -16 && val <= -1 {
			buf.writeByte(byte(TagFixnegBase + (-1 - val)))
		} else {
			buf.writeByte(TagInt64)
			buf.writeUvarint(zigzagEncode(int64(val)))
		}
	case int64:
		if val >= 0 && val <= 127 {
			buf.writeByte(byte(TagFixintBase + int(val)))
		} else if val >= -16 && val <= -1 {
			buf.writeByte(byte(TagFixnegBase + int(-1-val)))
		} else {
			buf.writeByte(TagInt64)
			buf.writeUvarint(zigzagEncode(val))
		}
	case uint:
		buf.writeByte(TagUint64)
		buf.writeUvarint(uint64(val))
	case uint64:
		buf.writeByte(TagUint64)
		buf.writeUvarint(val)
	case float64:
		buf.writeByte(TagFloat64)
		var fbuf [8]byte
		binary.LittleEndian.PutUint64(fbuf[:], uint64FromFloat64(val))
		buf.write(fbuf[:])
	case string:
		buf.writeByte(TagString)
		buf.writeString(val)
	case []byte:
		buf.writeByte(TagBytes)
		buf.writeUvarint(uint64(len(val)))
		buf.write(val)
	case []any:
		if len(val) <= 15 {
			buf.writeByte(byte(TagFixarrayBase + len(val)))
		} else {
			buf.writeByte(TagArray)
			buf.writeUvarint(uint64(len(val)))
		}
		for _, item := range val {
			if err := encodeAny(buf, item, d); err != nil {
				return err
			}
		}
	case map[string]any:
		if len(val) <= 15 {
			buf.writeByte(byte(TagFixmapBase + len(val)))
		} else {
			buf.writeByte(TagObject)
			buf.writeUvarint(uint64(len(val)))
		}
		for _, k := range sortedMapKeys(val) {
			v := val[k]
			buf.writeUvarint(uint64(d.get(k)))
			if err := encodeAny(buf, v, d); err != nil {
				return err
			}
		}
	case *Value:
		if val == nil {
			buf.writeByte(TagNull)
		} else {
			return encodeValue(buf, val, d)
		}
	default:
		return fmt.Errorf("cowrie: unsupported property type %T", v)
	}
	return nil
}

// sortedMembersByDictIdx returns the object members ordered by their resolved
// dictionary index (§2.4 canonical: fields ascending by dictIdx). The dictionary
// is globally byte-sorted, so dictIdx order matches key byte order. Returns a new
// slice; the input is not mutated. Duplicate keys retain their relative order.
func sortedMembersByDictIdx(members []Member, d *dict) []Member {
	out := make([]Member, len(members))
	copy(out, members)
	sort.SliceStable(out, func(i, j int) bool {
		return d.get(out[i].Key) < d.get(out[j].Key)
	})
	return out
}

// EncodeOptions controls encoding behavior.
type EncodeOptions struct {
	// Deterministic sorts object keys lexicographically for reproducible output.
	// This is essential for hashing, caching, and content addressing.
	Deterministic bool

	// OmitNull excludes null values from encoded objects.
	// Useful for sparse data and reducing payload size.
	OmitNull bool

	// AllowReserved permits encoding value kinds whose wire tags v1 reserves
	// (TensorRef, Image, Audio, Node, Edge, NodeBatch, EdgeBatch). Off by default
	// because every v1 decoder rejects those tags; enable only to reproduce
	// pre-v1 (attic) bytes.
	AllowReserved bool
}

// EncodeWithOptions encodes a value with explicit options.
func EncodeWithOptions(v *Value, opts EncodeOptions) ([]byte, error) {
	if opts.OmitNull {
		v = omitNulls(v)
	}
	if opts.Deterministic {
		v = canonicalize(v)
	}
	return encodeTop(v, opts.AllowReserved)
}

// omitNulls returns a copy of v with all null values removed from objects.
// For non-object types, returns the same value. Recursively processes nested values.
func omitNulls(v *Value) *Value {
	if v == nil {
		return nil
	}

	switch v.Type() {
	case TypeObject:
		members := v.Members()
		if len(members) == 0 {
			return v
		}

		// Filter out null values and recursively process nested values
		filtered := make([]Member, 0, len(members))
		for _, m := range members {
			if m.Value != nil && m.Value.Type() != TypeNull {
				filtered = append(filtered, Member{
					Key:   m.Key,
					Value: omitNulls(m.Value),
				})
			}
		}

		return Object(filtered...)

	case TypeArray:
		arr := v.Array()
		if len(arr) == 0 {
			return v
		}

		// Recursively process array elements (don't filter nulls from arrays)
		processed := make([]*Value, len(arr))
		for i, elem := range arr {
			processed[i] = omitNulls(elem)
		}

		return Array(processed...)

	default:
		return v
	}
}

// canonicalize returns a copy of v with all object keys sorted lexicographically.
// For non-object types, returns the same value. For objects and arrays, recursively
// canonicalizes nested values.
func canonicalize(v *Value) *Value {
	if v == nil {
		return nil
	}

	switch v.Type() {
	case TypeObject:
		members := v.Members()
		if len(members) == 0 {
			return v
		}

		// Create sorted copy of members
		sorted := make([]Member, len(members))
		copy(sorted, members)
		sortMembers(sorted)

		// Recursively canonicalize nested values
		for i := range sorted {
			sorted[i].Value = canonicalize(sorted[i].Value)
		}

		return Object(sorted...)

	case TypeArray:
		arr := v.Array()
		if len(arr) == 0 {
			return v
		}

		// Recursively canonicalize array elements
		canonical := make([]*Value, len(arr))
		for i, elem := range arr {
			canonical[i] = canonicalize(elem)
		}

		return Array(canonical...)

	default:
		// Scalars and other types don't need canonicalization
		return v
	}
}

// sortedMapKeys returns the keys of a map[string]any sorted by UTF-8 byte order.
// Used to make graph property encoding deterministic.
func sortedMapKeys(m map[string]any) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// sortMembers sorts members by key in lexicographic order.
// The decoder accepts objects up to DefaultMaxObjectLen (1M) fields, so this
// must not be quadratic — deterministic encoding is pitched for hashing/caching
// where large objects are plausible.
func sortMembers(members []Member) {
	sort.SliceStable(members, func(i, j int) bool {
		return members[i].Key < members[j].Key
	})
}

// decimalCanonical reduces a Decimal128 to lowest terms (§5.3): factors of ten move from the
// coefficient into the scale, and a zero coefficient collapses to scale 0.
func decimalCanonical(scale int64, coef [16]byte) (int64, [16]byte) {
	c := bigIntFromLE(coef[:])
	if c.Sign() == 0 {
		return 0, [16]byte{}
	}
	ten := big.NewInt(10)
	rem := new(big.Int)
	for scale != math.MinInt32 {
		q := new(big.Int)
		q.QuoRem(c, ten, rem)
		if rem.Sign() != 0 {
			break
		}
		c = q
		scale-- // c/10 x 10^-(s-1) == c x 10^-s
	}
	return scale, bigIntToLE16(c)
}

// bigIntToLE16 renders an integer as the 16-byte little-endian two's-complement Decimal128
// coefficient (§2.3).
func bigIntToLE16(c *big.Int) [16]byte {
	var out [16]byte
	v := new(big.Int).Set(c)
	if v.Sign() < 0 {
		v.Add(v, new(big.Int).Lsh(big.NewInt(1), 128))
	}
	b := v.Bytes() // big-endian magnitude
	for i, j := 0, len(b)-1; j >= 0 && i < 16; i, j = i+1, j-1 {
		out[i] = b[j]
	}
	return out
}

// ---- canonical number writers (§5.3) ----------------------------------------
// Shared by the contiguous encoder (encodeValue) and the scatter-gather encoder
// (collectSegments): one implementation, so the two paths cannot drift.

// writeIntInto writes an int64 in its canonical form: FIXINT for 0..127, FIXNEG for -1..-16,
// otherwise the INT64 tag with a zigzag varint.
func writeIntInto(buf *buffer, v int64) {
	if v >= 0 && v <= 127 {
		buf.writeByte(byte(TagFixintBase + v))
		return
	}
	if v >= -16 && v <= -1 {
		buf.writeByte(byte(TagFixnegBase + (-1 - v)))
		return
	}
	buf.writeByte(TagInt64)
	buf.writeUvarint(zigzagEncode(v))
}

// writeUintInto writes a uint64 in its canonical form: a uint that fits i64 is an Int
// (tools/cowrie_ref number_domain), otherwise the UINT64 tag.
func writeUintInto(buf *buffer, u uint64) {
	if u <= math.MaxInt64 {
		writeIntInto(buf, int64(u))
		return
	}
	buf.writeByte(TagUint64)
	buf.writeUvarint(u)
}

// writeDecimalInto writes a Decimal128 in lowest terms (§5.3), scale as an SVARINT (§2.3).
func writeDecimalInto(buf *buffer, scale int64, coef [16]byte) {
	scale, coef = decimalCanonical(scale, coef)
	buf.writeByte(TagDecimal128)
	buf.writeUvarint(zigzagEncode(int64(scale)))
	buf.write(coef[:])
}

// writeBigIntInto writes a BigInt, coercing to Int or Uint first when the value fits one of them
// (a BigInt is only canonical for values that fit neither).
func writeBigIntInto(buf *buffer, raw []byte) {
	if c := bigIntFromLE(raw); c != nil {
		if c.IsInt64() {
			writeIntInto(buf, c.Int64())
			return
		}
		if c.Sign() > 0 && c.BitLen() <= 64 {
			buf.writeByte(TagUint64)
			buf.writeUvarint(c.Uint64())
			return
		}
	}
	buf.writeByte(TagBigInt)
	buf.writeUvarint(uint64(len(raw)))
	buf.write(raw)
}
