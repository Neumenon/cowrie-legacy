module github.com/Neumenon/cowrie/bench

go 1.21

require (
	github.com/Neumenon/cowrie/go v0.0.0
	github.com/vmihailenco/msgpack/v5 v5.4.1
)

require (
	github.com/klauspost/compress v1.17.0 // indirect
	github.com/vmihailenco/tagparser/v2 v2.0.0 // indirect
)

replace github.com/Neumenon/cowrie/go => ../go
