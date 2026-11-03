package cowrie

// encodeReserved encodes a value whose wire tag v1 reserves (TensorRef, Image,
// Audio, Node, Edge, NodeBatch, EdgeBatch). Plain Encode refuses these by design,
// so the tests that prove the retired bytes still round-trip — and that the
// decoder still rejects them — force the opt-in here.
func encodeReserved(v *Value) ([]byte, error) {
	return EncodeWithOptions(v, EncodeOptions{AllowReserved: true})
}
