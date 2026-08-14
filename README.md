# QOZ

A lossless image format using zstd (de)compression to store raw image data.
Raw pixel bytes are split into horizontal tiles, with each tile compressed
into its own zstd frame.
Decoding images simply decompresses the tiles into an output buffer. Tiles 
are decompressed in parallel across CPU cores.

This format was inspired by QOZ for it's simplicity.
## File layout

```
[magic: "QOZ1"]
[width: u32 LE]
[height: u32 LE]
[channels: u8]
[colorspace: u8]
[reserved: u8]
[reserved: u8]
[tile_rows: u32 LE]
[tile_count: u32 LE]
[tile_len: u64 LE] * tile_count <- compressed tile length table
[tile bytes] * tile_count       <- compressed, concatenated zstd frames
```

It's just a 20-byte fixed header + an 8-byte-per-tile length table.

## Project structure:

`qoz/`    : the library. find header format and encode+decode
`qoz-cli/`: the `qoz` binary. convert between qoz and anything image-rs recognizes by extension
`qoz-c/`  : the `C` bindings.

## Parallel tiling

Each tile is compressed and decompressed independently, so `qoz` splits
work across `std::thread::available_parallelism()` threads on both encode and
decode. 
Tile row-height defaults to `ceil(height / num_cpus)`.
On a single-core machine this is exactly one tile.

## CLI usage

```bash
# Encode any image-rs-supported format into .qoz
qoz encode photo.png photo.qoz
qoz encode photo.png photo.qoz --level 15

# Decode back to any image-rs-supported format (by output extension)
qoz decode photo.qoz photo_out.png

# Inspect a .qoz file's header
qoz info photo.qoz

# Head-to-head decode benchmark against the `qoi` crate, on your own image
qoz bench photo.png --iterations 50
qoz bench photo.png --level 9 --iterations 50
```

