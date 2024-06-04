dev-size:
	du ./target/debug/rust_openes -h

prod-size:
	du ./target/release/rust_openes -h

check:
	cargo check

fmt:
	cargo fmt

lint:
	# flag for not make check redundant
	cargo clippy --no-default-features 

build:
	cargo build

test:
	clear && cargo test --tests

prep:
	clear && make fmt lint

run:
	clear && make build && cargo run

rel:
	clear && make build && cargo run --release
