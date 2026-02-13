.PHONY: test run clean

STEP ?= step01

test:
	cargo test --bin $(STEP)

run:
	cargo run --bin $(STEP)

clean:
	cargo clean
