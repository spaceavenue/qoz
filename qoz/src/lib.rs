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
//! [width: u32 LE]
//! [height: u32 LE]
//! [channels: u8]
//! [colorspace: u8]
//! [reserved: u8]
//! [reserved: u8]
//! [tile_rows: u32 LE]
//! [tile_count: u32 LE]
//! [tile_len: u64 LE] * tile_count       <- compressed length table
//! [tile compressed bytes] * tile_count  <- concatenated zstd frames
//! ```
//!
//! Tiles are horizontal row bands and are contiguous slices of the full buffer.

mod header;

use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use thiserror::Error;
use zstd::zstd_safe::CParameter::WindowLog;
use zstd::zstd_safe::DParameter;

use crate::QozError::{InvalidChannels, InvalidColorspace};

pub const MAGIC: [u8; 4] = *b"QOZ1";
// magic + 4 (1-byte) u8 fields + 4 (4-byte) u32 fields
pub const HEADER_LEN: usize = 4 + 4 + 16;
pub const MAX_PIXELS: u32 = u32::MAX;

#[derive(Debug, Error)]
pub enum QozError {
    #[error("input buffer length {actual} does not match width * height * channels ({expected})")]
    SizeMismatch { expected: usize, actual: usize },
    #[error("invalid dimensions: {w}x{h}. both must be non-zero and width * height must be < 2^32")]
    InvalidDimensions { w: u32, h: u32 },
    #[error("invalid channel count: {0} (must be 1-4)")]
    InvalidChannels(u8),
    #[error("unsupported colorspace: {0} (must be 0 (Srgb) or 1 (Linear))")]
    InvalidColorspace(u8),
    #[error("data too short to contain a valid QOZ header: expected {HEADER_LEN}, got {actual}")]
    Truncated { actual: usize },
    #[error("bad magic bytes: expected {MAGIC:?}")]
    BadMagic,
    #[error("tile length table requires {required} bytes, but only {available} remain")]
    TruncatedTileTable { required: usize, available: usize },
    #[error("tile {index} offset extends beyond file bounds")]
    TileOutOfBounds { index: usize },
    #[error("decompressed tile size ({actual}) does not match expected size ({expected})")]
    TileDecompressionSizeMismatch { expected: usize, actual: usize },
    #[error("zstd error: {0}")]
    Zstd(#[from] std::io::Error),
}

/// The color channels of a pixel in the image.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum Channels {
    /// Grayscale.
    Gray = 1,
    /// Grayscale with alpha channel.
    GrayA = 2,
    /// Red, Green, Blue.
    Rgb = 3,
    /// Red, Green, Blue, Alpha.
    #[default]
    Rgba = 4,
}
impl std::fmt::Display for Channels {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Channels::Gray => f.write_str("Grayscale (1)"),
            Channels::GrayA => f.write_str("Grayscale w/ Alpha (2)"),
            Channels::Rgb => f.write_str("RGB (3)"),
            Channels::Rgba => f.write_str("RGBA (4)"),
        }
    }
}
impl TryFrom<u8> for Channels {
    type Error = QozError;

    fn try_from(channels: u8) -> Result<Self, Self::Error> {
        match channels {
            1 => Ok(Channels::Gray),
            2 => Ok(Channels::GrayA),
            3 => Ok(Channels::Rgb),
            4 => Ok(Channels::Rgba),
            _ => Err(InvalidChannels(channels)),
        }
    }
}
impl From<Channels> for u8 {
    fn from(channel: Channels) -> Self {
        channel as Self
    }
}

/// The colorspace of an image.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum ColorSpace {
    #[default]
    Srgb = 0,
    Linear = 1,
}
impl std::fmt::Display for ColorSpace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ColorSpace::Srgb => f.write_str("srgb"),
            ColorSpace::Linear => f.write_str("linear"),
        }
    }
}
impl TryFrom<u8> for ColorSpace {
    type Error = QozError;

    fn try_from(colorspace: u8) -> Result<Self, Self::Error> {
        match colorspace {
            0 => Ok(ColorSpace::Srgb),
            1 => Ok(ColorSpace::Linear),
            _ => Err(InvalidColorspace(colorspace)),
        }
    }
}
impl From<ColorSpace> for u8 {
    fn from(channel: ColorSpace) -> Self {
        channel as Self
    }
}

/// Image header. Consists of channels, color space, width, height, number of rows per zstd
/// compressed tile and total number of tiles.
/// Notes:
/// * Both width and height must be non-zero.
/// * Maximum number of pixels is 2^32 (~4 GP or ~400MP).
/// * tile rows are calculated from the image height; see `[default_tile_rows()]`.
/// * tile count depends on tile rows, and will by default try to match the number of CPU cores the
///   image was encoded on.
#[derive(Debug, Clone, Copy)]
pub struct Header {
    /// Width of the image.
    pub width: u32,
    /// Height of the image.
    pub height: u32,
    /// Number of color channels per pixel.
    pub channels: Channels,
    /// Color space of the image. Purely informative and does not affect encode/decode.
    pub colorspace: ColorSpace,
    /// Number of rows per zstd compressed tiles.
    pub tile_rows: u32,
    /// Number of zstd compressed tiles.
    pub tile_count: u32,
}
impl Header {
    /// Creates a new header and validates image dimensions.
    #[inline]
    pub const fn try_new(
        width: u32,
        height: u32,
        channels: Channels,
        colorspace: ColorSpace,
        tile_rows: u32,
        tile_count: u32,
    ) -> Result<Self, QozError> {
        if width == 0 || height == 0 {
            return Err(QozError::InvalidDimensions {
                w: width,
                h: height,
            });
        }
        let n_pixels = width.checked_mul(height);
        if n_pixels.is_none() {
            return Err(QozError::InvalidChannels(u8::MAX));
        }
        Ok(Self {
            width,
            height,
            channels,
            colorspace,
            tile_rows,
            tile_count,
        })
    }

    /// Returns the bytes per pixel.
    #[inline]
    pub fn bytes_per_pixel(&self) -> usize {
        self.channels as usize
    }
    /// Returns the total pixels.
    #[inline]
    pub fn num_pixels(&self) -> usize {
        self.width.saturating_mul(self.height) as usize
    }
    /// Returns the total bytes.
    #[inline]
    pub fn num_bytes(&self) -> usize {
        self.num_pixels() * self.bytes_per_pixel()
    }
    /// Decodes the header from byte array.
    #[inline]
    pub fn decode_header(data: impl AsRef<[u8]>) -> Result<Header, QozError> {
        let data = data.as_ref();
        if data.len() < HEADER_LEN {
            return Err(QozError::Truncated { actual: data.len() });
        }
        if data[0..4] != MAGIC {
            return Err(QozError::BadMagic);
        }
        let width = u32::from_le_bytes(data[4..8].try_into().unwrap());
        let height = u32::from_le_bytes(data[8..12].try_into().unwrap());
        let channels = data[12].try_into()?;
        let colorspace = data[13].try_into()?;
        // data[14..16] reserved
        let tile_rows = u32::from_le_bytes(data[16..20].try_into().unwrap());
        let tile_count = u32::from_le_bytes(data[20..24].try_into().unwrap());
        Self::try_new(width, height, channels, colorspace, tile_rows, tile_count)
    }

    /// Encodes the header as a byte array.
    #[inline]
    pub fn encode_header(&self) -> [u8; HEADER_LEN] {
        let mut out = [0; HEADER_LEN];
        out[..4].copy_from_slice(&MAGIC);
        out[4..8].copy_from_slice(&self.width.to_le_bytes());
        out[8..12].copy_from_slice(&self.height.to_le_bytes());
        out[12] = self.channels.into();
        out[13] = self.colorspace.into();
        out[14..16].copy_from_slice(&[0u8; 2]);
        out[16..20].copy_from_slice(&self.tile_rows.to_le_bytes());
        out[20..24].copy_from_slice(&self.tile_count.to_le_bytes());
        out
    }

    /// Creates a new header with modified channels.
    #[inline]
    pub const fn with_channels(mut self, channels: Channels) -> Self {
        self.channels = channels;
        self
    }

    /// Creates a new header with modified color space.
    #[inline]
    pub const fn with_colorspace(mut self, colorspace: ColorSpace) -> Self {
        self.colorspace = colorspace;
        self
    }
}

/// Pick a default tile height so tile_count ~= available parallelism. With 1 core this is a single
/// tile/zstd frame.
#[inline]
pub fn default_tile_rows(height: u32) -> u32 {
    let threads = std::thread::available_parallelism()
        .map(|n| n.get() as u32)
        .unwrap_or(1);
    let rows = height.div_ceil(threads);
    rows.max(1)
}

/// The upper bound of the the encoded image's size. This is the worst case and the compressed size
/// will generally be smaller.
///
/// Can be used to preallocate space for a buffer to encode the image into.
pub fn encode_max_len(width: u32, height: u32, channels: Channels, tile_rows: u32) -> usize {
    if width == 0 || height == 0 {
        return 0;
    }
    let tile_rows = if tile_rows == 0 {
        default_tile_rows(height)
    } else {
        tile_rows
    };
    let mut ranges = Vec::new();
    let mut row = 0u32;
    while row < height {
        let rows = tile_rows.min(height - row);
        ranges.push(rows);
        row += rows;
    }
    let row_stride = width as usize * u8::from(channels) as usize;
    let tile_upper_bound = ranges
        .iter()
        .map(|rows| zstd::zstd_safe::compress_bound(*rows as usize * row_stride))
        .sum::<usize>();
    HEADER_LEN + 8 * ranges.len() + tile_upper_bound
}

/// Encode options.
pub struct EncodeOptions {
    pub channels: Channels,
    pub colorspace: ColorSpace,
    /// zstd compression level. Decode speed is nearly independent of this, so higher levels mostly
    /// cost more encode time in exchange for better ratio, and sometimes faster decode, since
    /// longer matches shift work from entropy-coded literals to memcpy.
    pub level: i32,
    /// Rows per tile. 0 = pick automatically from available CPU cores.
    pub tile_rows: u32,
}

impl Default for EncodeOptions {
    #[inline]
    fn default() -> Self {
        Self {
            channels: Channels::default(),
            colorspace: ColorSpace::default(),
            level: 9,
            tile_rows: 0,
        }
    }
}
impl EncodeOptions {
    #[inline]
    pub fn new(channels: Channels, colorspace: ColorSpace, level: i32, tile_rows: u32) -> Self {
        Self {
            channels,
            colorspace,
            level,
            tile_rows,
        }
    }
}

/// Encode raw pixel bytes into a new `Vec<u8>`.
pub fn encode(
    pixels: impl AsRef<[u8]>,
    width: u32,
    height: u32,
    opts: &EncodeOptions,
) -> Result<Vec<u8>, QozError> {
    let mut buf = vec![0u8; encode_max_len(width, height, opts.channels, opts.tile_rows)];
    let bytes_written = encode_into_buf(pixels.as_ref(), width, height, opts, &mut buf)?;
    buf.truncate(bytes_written);
    Ok(buf)
}

/// Encode raw pixel bytes into a provided buffer.
pub fn encode_into_buf(
    pixels: impl AsRef<[u8]>,
    width: u32,
    height: u32,
    opts: &EncodeOptions,
    mut buf: impl AsMut<[u8]>,
) -> Result<usize, QozError> {
    if width == 0 || height == 0 {
        return Err(QozError::InvalidDimensions {
            w: width,
            h: height,
        });
    }
    let pixels = pixels.as_ref();
    let buf = buf.as_mut();
    let channels = opts.channels;
    let colorspace = opts.colorspace;
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
    let chunk_size = width as usize * channels as usize * tile_rows as usize;
    let tile_slices = pixels.chunks(chunk_size).collect::<Vec<&[u8]>>();
    let tile_count = tile_slices.len();
    let tile_len_table_size = tile_count * 8;
    let data_offset = HEADER_LEN + tile_len_table_size;

    // Each tile's length is initially the zstd compressions's upper bound and they'll be
    // compacted to their true size later.
    let tile_upper_bounds = tile_slices
        .iter()
        .map(|tile| zstd::zstd_safe::compress_bound(tile.len()))
        .collect::<Vec<usize>>();

    // Slice the output into per-tile byte slices, which we can then hand over to a thread. This
    // allows us to decode into the buffer directly, preventing a temporary copy of the compressed
    // data being created in memory.
    let mut out_slices: Vec<&mut [u8]> = Vec::with_capacity(tile_count);
    {
        let mut rest = &mut buf[data_offset..];
        for &bound in &tile_upper_bounds {
            let (a, b) = rest.split_at_mut(bound);
            out_slices.push(a);
            rest = b;
        }
    }

    // Compress data. Returns a Vec of bytes written for each tile.
    let true_lens: Result<Vec<usize>, QozError> = tile_slices
        .into_par_iter()
        .zip(out_slices)
        .map_init(
            || zstd::bulk::Compressor::new(opts.level),
            |compressor_res, (tile, out_slice)| match compressor_res {
                Ok(compressor) => {
                    compressor.set_parameter(WindowLog(tile.len().ilog2()))?;
                    let n = compressor
                        .compress_to_buffer(tile, out_slice)
                        .map_err(QozError::Zstd)?;
                    Ok(n)
                }
                Err(e) => Err(QozError::Zstd(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("zstd initialization failed: {e}"),
                ))),
            },
        )
        .collect();
    let true_lens = true_lens?;

    // Compact the tiles.
    let mut read_offset = data_offset;
    let mut write_offset = data_offset;
    tile_upper_bounds
        .iter()
        .zip(true_lens.iter())
        .for_each(|(&max_bound, &true_len)| {
            // Only move data if there's a gap (Tile 0 is always already in place)
            if read_offset != write_offset {
                let read_end = read_offset + true_len;
                buf.copy_within(read_offset..read_end, write_offset);
            }
            read_offset += max_bound;
            write_offset += true_len;
        });

    // Assemble the file
    // header + length table + concatenated compressed tiles
    let header = Header::try_new(
        width,
        height,
        channels,
        colorspace,
        tile_rows,
        tile_count as u32,
    )?
    .encode_header();

    // header
    buf[0..HEADER_LEN].copy_from_slice(&header);

    // tile length table
    true_lens.iter().enumerate().for_each(|(i, &len)| {
        let start = HEADER_LEN + (i * 8);
        let end = start + 8;
        buf[start..end].copy_from_slice(&(len as u64).to_le_bytes());
    });
    // compressed tiles
    // let mut current_offset = HEADER_LEN + tile_len_table_size;
    // for c in out_slices.iter() {
    //     let end_offset = current_offset + c.len();
    //     buf[current_offset..end_offset].copy_from_slice(c);
    //     current_offset = end_offset;
    // }
    Ok(write_offset)
}

/// Parse the header, without decompressing image data.
#[inline]
pub fn read_header(data: &[u8]) -> Result<Header, QozError> {
    Header::decode_header(data)
}

/// Decode and return header + decompressed data.
#[inline]
pub fn decode(data: &[u8]) -> Result<(Header, Vec<u8>), QozError> {
    let header = Header::decode_header(data)?;
    // let mut out = vec![0u8; header.num_bytes()];
    let mut out = Vec::with_capacity(header.num_bytes());
    unsafe {
        // SAFETY: Immediately passed to decode_into_buf, which guarantees every single byte is
        // overwritten by the zstd decompressor or returns an error; see `n != out_slice.len()
        // below.
        out.set_len(header.num_bytes());
    }
    decode_into_buf(data, &mut out)?;
    Ok((header, out))
}

/// Decode into a provided buffer, which must be exactly width*height*channels bytes. Returns the
/// header as well.
pub fn decode_into_buf(data: &[u8], out: &mut [u8]) -> Result<Header, QozError> {
    let header = Header::decode_header(data)?;
    if header.width == 0 || header.height == 0 {
        return Err(QozError::InvalidDimensions {
            w: header.width,
            h: header.height,
        });
    }
    if out.len() != header.num_bytes() {
        return Err(QozError::SizeMismatch {
            expected: header.num_bytes(),
            actual: out.len(),
        });
    }

    // Get the length of tiles from the tile length table.
    let table_start = HEADER_LEN;
    let table_len = 8 * header.tile_count as usize;
    if data.len() < table_start + table_len {
        return Err(QozError::TruncatedTileTable {
            required: data.len(),
            available: table_start + table_len,
        });
    }
    let mut tile_lens = Vec::with_capacity(header.tile_count as usize);
    for i in 0..header.tile_count as usize {
        let offset = table_start + i * 8;
        let len = u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap()) as usize;
        tile_lens.push(len);
    }

    // let row_stride = header.width as usize * header.channels as usize;
    let mut blob_offset = table_start + table_len;
    let mut tile_blobs: Vec<&[u8]> = Vec::with_capacity(tile_lens.len());
    for len in &tile_lens {
        if blob_offset + len > data.len() {
            return Err(QozError::TileOutOfBounds {
                index: blob_offset + len,
            });
        }
        tile_blobs.push(&data[blob_offset..blob_offset + len]);
        blob_offset += len;
    }

    let chunk_size =
        header.width as usize * header.channels as usize * header.tile_rows.max(1) as usize;
    let out_slices = out.chunks_mut(chunk_size).collect::<Vec<&mut [u8]>>();

    let results: Result<Vec<()>, QozError> = out_slices
        .into_par_iter()
        .zip(tile_blobs.into_par_iter())
        .map_init(
            || zstd::bulk::Decompressor::new(),
            |decompressor_res, (out_slice, blob)| match decompressor_res {
                Ok(decompressor) => {
                    decompressor.set_parameter(DParameter::WindowLogMax(blob.len().ilog2()))?;
                    let n = decompressor
                        .decompress_to_buffer(blob, out_slice)
                        .map_err(QozError::Zstd)?;
                    if n != out_slice.len() {
                        return Err(QozError::TileDecompressionSizeMismatch {
                            expected: out_slice.len(),
                            actual: n,
                        });
                    }
                    Ok(())
                }
                Err(e) => Err(QozError::Zstd(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("zstd initialization failed: {e}"),
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

    fn make_test_image(w: u32, h: u32, channels: Channels) -> Vec<u8> {
        let mut v = Vec::with_capacity(w as usize * h as usize * channels as usize);
        for y in 0..h {
            for x in 0..w {
                for c in 0..channels.into() {
                    // Some structure (gradient + a flat block).
                    let val = ((x + y * 2 + c as u32 * 7) % 251) as u8;
                    v.push(val);
                }
            }
        }
        v
    }

    fn roundtrip(w: u32, h: u32, channels: Channels, tile_rows: u32) {
        let pixels = make_test_image(w, h, channels);
        let opts = EncodeOptions {
            channels: channels,
            colorspace: ColorSpace::default(),
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
        for &channels in &[
            Channels::Gray,
            Channels::GrayA,
            Channels::Rgb,
            Channels::Rgba,
        ] {
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
            channels: Channels::default(),
            colorspace: ColorSpace::default(),
            level: 3,
            tile_rows: 0,
        };
        let err_h = encode(&[], 10, 0, &opts).unwrap_err();
        assert!(matches!(err_h, QozError::InvalidDimensions { w: 10, h: 0 }));

        let err_w = encode(&[], 0, 10, &opts).unwrap_err();
        assert!(matches!(err_w, QozError::InvalidDimensions { w: 0, h: 10 }));
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
