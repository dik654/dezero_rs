.PHONY: test clean

STEP ?= step01

test:
	cargo test --test $(STEP)

clean:
	cargo clean
