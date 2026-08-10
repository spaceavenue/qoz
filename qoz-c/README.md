# qoz-capi

C bindings for `qoz`, built as `libqoz` (`.so` + `.a`). 
Header is `include/qoz.h`, generated from `src/lib.rs` by [cbindgen](https://github.com/mozilla/cbindgen).

## Building

```bash
cargo build --release -p qoz-capi
# -> ../target/release/libqoz.so   (cdylib)
# -> ../target/release/libqoz.a    (staticlib)
```

## Regenerating the header

Only needed if you change `src/lib.rs`'s public API:

```bash
cargo install cbindgen   # once
cd qoz-capi
cbindgen --config cbindgen.toml --output include/qoz.h
```

## Using it from C

```c
#include <qoz.h>

QozDesc desc = { .width = w, .height = h, .channels = 4, .colorspace = 0 };
size_t out_size;
uint8_t *encoded = qoz_encode(pixels, &desc, /*filter=*/0, /*level=*/9, /*tile_rows=*/0, &out_size);
if (!encoded) { fprintf(stderr, "%s\n", qoz_last_error()); }
/* ... */
qoz_free(encoded, out_size);
```

Link with `-lqoz` dynamically, or link `libqoz.a` directly plus `-lpthread -ldl -lm` (transitive deps of the Rust standard library on Linux). 
See `examples/roundtrip.c` for a full working example; it's built and run (dynamically *and* statically linked, plus under Valgrind with zero leaks/errors) as part of verifying this library.

## API shape

- `qoz_get_desc(data, size, *desc) -> bool` - peek header only, no decode.
- `qoz_decode(data, size, *desc) -> uint8_t*` - decode to a fresh buffer; free with `qoz_free(ptr, desc.width*desc.height*desc.channels)`.
- `qoz_encode(pixels, *desc, level, tile_rows, *out_size) -> uint8_t*` - free with `qoz_free(ptr, out_size)`.
- `qoz_max_encoded_len(width, height, channels, tile_rows) -> size_t` - worst-case size, for pre-allocating.
- `qoz_last_error() -> const char*` - set on the last failed call on the current thread; NULL if the last call succeeded.
- `qoz_free(ptr, len)` - the *only* correct way to free anything returned by `qoz_decode`/`qoz_encode`. Never call libc `free()` on them - the allocation comes from Rust's global allocator, not `malloc`. Conversely, never pass a `malloc`'d pointer to `qoz_free`.
- `qoz_version() -> const char*` - static, no need to free.

## Safety notes

- Every exported function catches Rust panics internally (`catch_unwind`) and converts them into a `qoz_last_error()` message instead of unwinding into C, which is undefined behavior.
- Null pointers are checked explicitly and reported as an error rather than dereferenced.
- All buffer/length pairs are the caller's responsibility to track correctly - the library does not store lengths alongside returned pointers.
