WASM_BINDGEN_FORK := https://github.com/audreyt/wasm-bindgen
WASM_BINDGEN_BIN  := $(HOME)/.cargo/bin/wasm-bindgen

# Install the forked wasm-bindgen CLI if not already present.
$(WASM_BINDGEN_BIN):
	cargo install --git $(WASM_BINDGEN_FORK) --branch main wasm-bindgen-cli --locked

.PHONY: wasm-bindgen
wasm-bindgen: $(WASM_BINDGEN_BIN)

# Build the WASM package targeting wasm64 (WebAssembly memory64 proposal).
# Requires nightly Rust and the wasm64-unknown-unknown target via -Z build-std.
.PHONY: build
build: wasm-bindgen
	cargo +nightly build --release \
		--target wasm64-unknown-unknown \
		--no-default-features --features web \
		-Z build-std=std,panic_abort
	wasm-bindgen \
		--target web \
		--out-dir pkg \
		target/wasm64-unknown-unknown/release/web_rwkv.wasm

# Force-reinstall after the fork is updated.
.PHONY: install-tools
install-tools:
	cargo install --git $(WASM_BINDGEN_FORK) --branch main wasm-bindgen-cli --locked --force
