.PHONY: test clean

test:
ifdef STEP
	cargo test --test $(STEP)
else
	cargo test
endif

clean:
	cargo clean
