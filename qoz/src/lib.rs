//! QOZ: An image format using zstd compression for fast decode. Inspired by the QOI format.
//!
//! We encode interleaved pixel bytes that are split into horizontal row tiles, with each tile being
//! compressed indepdently as its own zstd frame.
//! To decode, we look up tile offsets and decompress each tile in parallel straight into the output
//! buffer.
//!
//! File layout:
//! ```text
//! [magic: 4 bytes "QOZ1"]
//! [channels: u8]
//! [colorspace: u8]
//! [reserved: u8]
//! [reserved: u8]
//! [width: u32 LE]
//! [height: u32 LE]
//! [tile_rows: u32 LE]
//! [tile_count: u32 LE]
//! [tile_len: u64 LE] * tile_count       <- compressed length table
//! [tile compressed bytes] * tile_count  <- concatenated zstd frames
//! ```
//!
//! Tiles are horizontal row bands. Because pixel data is row-major, a tile's raw bytes are always a
//! contiguous slice of the full buffer.

use std::{io, thread};

use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use thiserror::Error;

pub const MAGIC: [u8; 4] = *b"QOZ1";
pub const HEADER_LEN: usize = 4 + 4 + 16; // magic + 4 u8 fields + 4 u32 fields

#[derive(Debug, Error)]
pub enum QozError {
    #[error("input buffer length {actual} does not match width*height*channels ({expected})")]
    SizeMismatch { expected: usize, actual: usize },
    #[error("invalid channel count: {0} (must be 1-4)")]
    InvalidChannels(u8),
    #[error("data too short to contain a valid QOZ header")]
    Truncated,
    #[error("bad magic bytes: expected {MAGIC:?}")]
    BadMagic,
    #[error("tile length table or tile data is truncated/corrupt")]
    CorruptTileTable,
    #[error("zstd error: {0}")]
    Zstd(#[from] io::Error),
}

#[derive(Debug, Clone, Copy)]
pub struct Header {
    pub channels: u8,
    pub colorspace: u8,
    pub width: u32,
    pub height: u32,
    pub tile_rows: u32,
    pub tile_count: u32,
}

impl Header {
    pub fn bytes_per_pixel(&self) -> usize {
        self.channels as usize
    }

    pub fn total_bytes(&self) -> usize {
        self.width as usize * self.height as usize * self.bytes_per_pixel()
    }
}

/// Pick a default tile height so tile_count ~= available parallelism.
/// With 1 core this is a single tile/zstd frame.
pub fn default_tile_rows(height: u32) -> u32 {
    let threads = thread::available_parallelism()
        .map(|n| n.get() as u32)
        .unwrap_or(1);
    let rows = height.div_ceil(threads);
    rows.max(1)
}

fn tile_row_ranges(height: u32, tile_rows: u32) -> Vec<(u32, u32)> {
    let mut ranges = Vec::new();
    let mut row = 0u32;
    while row < height {
        let rows = tile_rows.min(height - row);
        ranges.push((row, rows));
        row += rows;
    }
    if ranges.is_empty() {
        // zero-height image edge case
        ranges.push((0, 0));
    }
    ranges
}

// Encode

pub struct EncodeOptions {
    pub channels: u8,
    pub colorspace: u8,
    /// zstd compression level. Decode speed is nearly independent of this;
    /// higher levels mostly cost more encode time in exchange for better
    /// ratio (and sometimes *faster* decode, since longer matches shift
    /// work from entropy-coded literals to memcpy).
    pub level: i32,
    /// Rows per tile. 0 = pick automatically from available CPU cores.
    pub tile_rows: u32,
}

impl Default for EncodeOptions {
    fn default() -> Self {
        Self {
            channels: 4,
            colorspace: 0,
            level: 9,
            tile_rows: 0,
        }
    }
}

pub fn encode(
    pixels: &[u8],
    width: u32,
    height: u32,
    opts: &EncodeOptions,
) -> Result<Vec<u8>, QozError> {
    let channels = opts.channels;
    if !(1..=4).contains(&channels) {
        return Err(QozError::InvalidChannels(channels));
    }
    let expected = width as usize * height as usize * channels as usize;
    if pixels.len() != expected {
        return Err(QozError::SizeMismatch {
            expected,
            actual: pixels.len(),
        });
    }

    let tile_rows = if opts.tile_rows == 0 {
        default_tile_rows(height)
    } else {
        opts.tile_rows
    };
    let ranges = tile_row_ranges(height, tile_rows);
    let row_stride = width as usize * channels as usize;

    // Slice the input into per-tile byte ranges.
    let tile_slices: Vec<&[u8]> = ranges
        .iter()
        .map(|(start, rows)| {
            let a = *start as usize * row_stride;
            let b = a + *rows as usize * row_stride;
            &pixels[a..b]
        })
        .collect();

    let compressed: Result<Vec<Vec<u8>>, QozError> = tile_slices
        .into_par_iter()
        .map_init(
            || zstd::bulk::Compressor::new(opts.level),
            |compressor_res, tile| match compressor_res {
                Ok(compressor) => compressor.compress(tile).map_err(QozError::Zstd),
                Err(e) => Err(QozError::Zstd(io::Error::new(e.kind(), "zstd init error"))),
            },
        )
        .collect();
    let compressed = compressed?;

    // Assemble the file
    // header size + length table + concatenated compressed tiles
    let tile_count = compressed.len() as u32;
    let mut out = Vec::with_capacity(
        HEADER_LEN + 8 * compressed.len() + compressed.iter().map(|c| c.len()).sum::<usize>(),
    );
    out.extend_from_slice(&MAGIC);
    out.push(channels);
    out.push(opts.colorspace);
    out.push(0); // reserved
    out.push(0); // reserved
    out.extend_from_slice(&width.to_le_bytes());
    out.extend_from_slice(&height.to_le_bytes());
    out.extend_from_slice(&tile_rows.to_le_bytes());
    out.extend_from_slice(&tile_count.to_le_bytes());
    compressed
        .iter()
        .for_each(|c| out.extend_from_slice(&(c.len() as u64).to_le_bytes()));
    compressed.iter().for_each(|c| out.extend_from_slice(c));
    Ok(out)
}

// Decode

fn parse_header(data: &[u8]) -> Result<Header, QozError> {
    if data.len() < HEADER_LEN {
        return Err(QozError::Truncated);
    }
    if data[0..4] != MAGIC {
        return Err(QozError::BadMagic);
    }
    let channels = data[4];
    let colorspace = data[5];
    // data[6] reserved
    // data[7] reserved
    let width = u32::from_le_bytes(data[8..12].try_into().unwrap());
    let height = u32::from_le_bytes(data[12..16].try_into().unwrap());
    let tile_rows = u32::from_le_bytes(data[16..20].try_into().unwrap());
    let tile_count = u32::from_le_bytes(data[20..24].try_into().unwrap());
    if !(1..=4).contains(&channels) {
        return Err(QozError::InvalidChannels(channels));
    }
    Ok(Header {
        width,
        height,
        channels,
        colorspace,
        tile_rows,
        tile_count,
    })
}

/// Parse just the header, without decompressing image data.
pub fn read_header(data: &[u8]) -> Result<Header, QozError> {
    parse_header(data)
}

// Decode and return the header + decompressed data.
pub fn decode(data: &[u8]) -> Result<(Header, Vec<u8>), QozError> {
    let header = parse_header(data)?;
    let mut out = vec![0u8; header.total_bytes()];
    decode_into(data, &mut out)?;
    Ok((header, out))
}

/// Decode into a caller-provided buffer, which must be exactly width*height*channels
/// bytes. Returns the header as well.
pub fn decode_into(data: &[u8], out: &mut [u8]) -> Result<Header, QozError> {
    let header = parse_header(data)?;
    if out.len() != header.total_bytes() {
        return Err(QozError::SizeMismatch {
            expected: header.total_bytes(),
            actual: out.len(),
        });
    }

    // Get the length of tiles from the tile length table.
    let table_start = HEADER_LEN;
    let table_len = 8 * header.tile_count as usize;
    if data.len() < table_start + table_len {
        return Err(QozError::CorruptTileTable);
    }
    let mut tile_lens = Vec::with_capacity(header.tile_count as usize);
    for i in 0..header.tile_count as usize {
        let offset = table_start + i * 8;
        let len = u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap()) as usize;
        tile_lens.push(len);
    }

    let ranges = tile_row_ranges(header.height, header.tile_rows.max(1));
    if ranges.len() != header.tile_count as usize {
        return Err(QozError::CorruptTileTable);
    }

    let row_stride = header.width as usize * header.channels as usize;
    let mut blob_offset = table_start + table_len;
    let mut tile_blobs: Vec<&[u8]> = Vec::with_capacity(tile_lens.len());
    for len in &tile_lens {
        if blob_offset + len > data.len() {
            return Err(QozError::CorruptTileTable);
        }
        tile_blobs.push(&data[blob_offset..blob_offset + len]);
        blob_offset += len;
    }

    // Split the output buffer into disjoint per-tile mutable slices up front, then hand each
    // (compressed tile, output slice) pair to a worker thread.
    let mut out_slices: Vec<&mut [u8]> = Vec::with_capacity(ranges.len());
    {
        let mut rest = out;
        for (_, rows) in &ranges {
            let n = *rows as usize * row_stride;
            let (a, b) = rest.split_at_mut(n);
            out_slices.push(a);
            rest = b;
        }
    }
    let results: Result<Vec<()>, QozError> = out_slices
        .into_par_iter()
        .zip(tile_blobs.into_par_iter())
        .map_init(
            || zstd::bulk::Decompressor::new(),
            |decompressor_res, (out_slice, blob)| match decompressor_res {
                Ok(decompressor) => {
                    let n = decompressor
                        .decompress_to_buffer(blob, out_slice)
                        .map_err(QozError::Zstd)?;
                    if n != out_slice.len() {
                        return Err(QozError::CorruptTileTable);
                    }
                    Ok(())
                }
                Err(e) => Err(QozError::Zstd(std::io::Error::new(
                    e.kind(),
                    "zstd init error",
                ))),
            },
        )
        .collect();
    results?;

    Ok(header)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_image(w: u32, h: u32, channels: u8) -> Vec<u8> {
        let mut v = Vec::with_capacity(w as usize * h as usize * channels as usize);
        for y in 0..h {
            for x in 0..w {
                for c in 0..channels {
                    // Some structure (gradient + a flat block).
                    let val = ((x + y * 2 + c as u32 * 7) % 251) as u8;
                    v.push(val);
                }
            }
        }
        v
    }

    fn roundtrip(w: u32, h: u32, channels: u8, tile_rows: u32) {
        let pixels = make_test_image(w, h, channels);
        let opts = EncodeOptions {
            channels,
            colorspace: 0,
            level: 3,
            tile_rows,
        };
        let encoded = encode(&pixels, w, h, &opts).unwrap();
        let (header, decoded) = decode(&encoded).unwrap();
        assert_eq!(header.width, w);
        assert_eq!(header.height, h);
        assert_eq!(header.channels, channels);
        assert_eq!(
            decoded, pixels,
            "roundtrip mismatch w={w} h={h} channels={channels} tile_rows={tile_rows}"
        );
    }

    #[test]
    fn roundtrip_various_sizes() {
        for &channels in &[1u8, 2, 3, 4] {
            roundtrip(1, 1, channels, 0);
            roundtrip(1, 1, channels, 1);
            roundtrip(37, 1, channels, 0);
            roundtrip(17, 5, channels, 2); // tile smaller than image, ragged last tile
            roundtrip(64, 64, channels, 0);
            roundtrip(257, 129, channels, 16); // odd sizes vs tile boundary
            roundtrip(300, 200, channels, 0); // auto tiling
        }
    }

    #[test]
    fn zero_height_image() {
        let opts = EncodeOptions {
            channels: 4,
            colorspace: 0,
            level: 3,
            tile_rows: 0,
        };
        let encoded = encode(&[], 10, 0, &opts).unwrap();
        let (header, decoded) = decode(&encoded).unwrap();
        assert_eq!(header.width, 10);
        assert_eq!(header.height, 0);
        assert!(decoded.is_empty());
    }

    #[test]
    fn size_mismatch_is_rejected() {
        let opts = EncodeOptions::default();
        let err = encode(&[0u8; 10], 4, 4, &opts).unwrap_err();
        assert!(matches!(err, QozError::SizeMismatch { .. }));
    }

    #[test]
    fn bad_magic_is_rejected() {
        let data = vec![0u8; HEADER_LEN + 8];
        let err = decode(&data).unwrap_err();
        assert!(matches!(err, QozError::BadMagic));
    }
}
