//! C bindings for qoz (`libqoz`).
//!
//! We pass a buffer in, get a malloc'd buffer out, and free it explicitly. See
//! `qoz.h` (and the README for how to (re)generate it) for the C-facing declarations and doc
//! comments.
//!
//! # Panic safety
//! Every exported function's body runs inside `catch_unwind`. A Rust panic must never unwind across
//! an `extern "C"` boundary (that's undefined behavior once it hits foreign frames on the stack) -
//! so instead of risking that, a panic here is caught, turned into a `qoz_last_error()` message,
//! and the function returns its documented error value (NULL or false) instead of taking the host
//! process down with it. This only works because the workspace does *not* build with `panic =
//! "abort"`; see the workspace `Cargo.toml` for why.
//!
//! # Memory ownership
//! Buffers returned by [`qoz_decode`] and [`qoz_encode`] are allocated by Rust's global allocator
//! and not libc `malloc`. So, they must be freed with [`qoz_free`] (which frees through that same
//! allocator) and never with libc's `free()`.

use std::cell::RefCell;
use std::ffi::CString;
use std::os::raw::c_char;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::{ptr, slice};

use qoz::{Channels, ColorSpace, EncodeOptions};

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = RefCell::new(None);
}

fn set_last_error(msg: impl std::fmt::Display) {
    let msg = msg.to_string();
    let c = CString::new(msg).unwrap_or_else(|_| {
        CString::new("qoz: error message contained an interior NUL byte").unwrap()
    });
    LAST_ERROR.with(|e| *e.borrow_mut() = Some(c));
}

fn clear_last_error() {
    LAST_ERROR.with(|e| *e.borrow_mut() = None);
}

/// Description of a raw pixel buffer / decoded image.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct QozDesc {
    pub width: u32,
    pub height: u32,
    /// 1 = grey, 2 = grey+alpha, 3 = rgb, 4 = rgba
    pub channels: u8,
    /// 0 = srgb, 1 = linear (purely informative, like QOI's colorspace byte)
    pub colorspace: u8,
}

fn channels_from_desc(desc: &QozDesc) -> Result<Channels, String> {
    Channels::try_from(desc.channels).map_err(|e| e.to_string())
}

fn colorspace_from_desc(desc: &QozDesc) -> Result<ColorSpace, String> {
    ColorSpace::try_from(desc.colorspace).map_err(|e| e.to_string())
}

/// Move a `Vec<u8>` onto the heap as an exactly-sized allocation and hand
/// back a thin pointer to it. The length is *not* stored alongside it -
/// callers must track it themselves (qoz_decode's is width*height*channels;
/// qoz_encode's is written to *out_size) and pass it back to [`qoz_free`].
fn vec_to_raw(v: Vec<u8>) -> *mut u8 {
    let boxed = v.into_boxed_slice(); // guarantees capacity == len, no excess
    Box::into_raw(boxed) as *mut u8
}

/// Run `f`, catching any panic instead of letting it unwind across the FFI
/// boundary. Returns `err_value` and sets `qoz_last_error()` on either an
/// `Err` or a panic; clears the last error and returns the value on `Ok`.
fn ffi_guard<T>(err_value: T, f: impl FnOnce() -> Result<T, String>) -> T {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(v)) => {
            clear_last_error();
            v
        }
        Ok(Err(msg)) => {
            set_last_error(msg);
            err_value
        }
        Err(panic) => {
            let msg = panic
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| panic.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "unknown panic payload".to_string());
            set_last_error(format!("internal panic in qoz: {msg}"));
            err_value
        }
    }
}

/// Description of the last error on the calling thread, or NULL if the
/// most recent `qoz_*` call on this thread succeeded. The pointer is only
/// valid until the next `qoz_*` call on the same thread.
#[unsafe(no_mangle)]
pub extern "C" fn qoz_last_error() -> *const c_char {
    LAST_ERROR.with(|e| match &*e.borrow() {
        Some(c) => c.as_ptr(),
        None => ptr::null(),
    })
}

/// Library version string (e.g. "0.1.0"), NUL-terminated, static storage.
#[unsafe(no_mangle)]
pub extern "C" fn qoz_version() -> *const c_char {
    static VERSION: &str = concat!(env!("CARGO_PKG_VERSION"), "\0");
    VERSION.as_ptr() as *const c_char
}

/// Parse just the header of an in-memory `.qoz` buffer into `*desc`,
/// without decoding pixel data. Returns `true` on success. `data` must
/// point to at least `size` readable bytes.
///
/// # Safety
/// `data` must be valid for reads of `size` bytes, and `desc` must be a
/// valid, aligned, writable `QozDesc*`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qoz_get_desc(data: *const u8, size: usize, desc: *mut QozDesc) -> bool {
    if data.is_null() || desc.is_null() {
        set_last_error("qoz_get_desc: null pointer argument");
        return false;
    }
    ffi_guard(false, || {
        let bytes = unsafe { slice::from_raw_parts(data, size) };
        let header = qoz::read_header(bytes).map_err(|e| e.to_string())?;
        unsafe {
            *desc = QozDesc {
                width: header.width,
                height: header.height,
                channels: header.channels.into(),
                colorspace: header.colorspace.into(),
            }
        };
        Ok(true)
    })
}

/// Decode an in-memory `.qoz` buffer. On success, fills `*desc` and
/// returns a freshly allocated pixel buffer of exactly
/// `desc->width * desc->height * channels` bytes (row-major, interleaved).
/// Returns NULL on error - call [`qoz_last_error`] for details.
///
/// Free the returned buffer with `qoz_free(ptr, width*height*channels)`.
///
/// # Safety
/// `data` must be valid for reads of `size` bytes, and `desc` must be a
/// valid, aligned, writable `QozDesc*`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qoz_decode(data: *const u8, size: usize, desc: *mut QozDesc) -> *mut u8 {
    if data.is_null() || desc.is_null() {
        set_last_error("qoz_decode: null pointer argument");
        return ptr::null_mut();
    }
    ffi_guard(ptr::null_mut(), || {
        let bytes = unsafe { slice::from_raw_parts(data, size) };
        let (header, pixels) = qoz::decode(bytes).map_err(|e| e.to_string())?;
        unsafe {
            *desc = QozDesc {
                width: header.width,
                height: header.height,
                channels: header.channels.into(),
                colorspace: header.colorspace.into(),
            }
        };
        Ok(vec_to_raw(pixels))
    })
}

/// Encode raw pixels (`desc->width * desc->height * channels` bytes,
/// row-major interleaved) into a new `.qoz` buffer.
///
/// - `level`: zstd compression level (try 9); mostly affects ratio and encode time, not decode
///   time.
/// - `tile_rows`: rows per independently-decodable tile; 0 = pick automatically from available CPU
///   cores.
///
/// On success, writes the encoded length to `*out_size` and returns a
/// malloc'd buffer; free it with `qoz_free(ptr, *out_size)`. Returns NULL
/// on error - call [`qoz_last_error`] for details.
///
/// # Safety
/// `pixels` must be valid for reads of `desc->width * desc->height *
/// channels` bytes. `desc` and `out_size` must be valid, aligned, and (for
/// `out_size`) writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qoz_encode(
    pixels: *const u8,
    desc: *const QozDesc,
    level: i32,
    tile_rows: u32,
    out_size: *mut usize,
) -> *mut u8 {
    if pixels.is_null() || desc.is_null() || out_size.is_null() {
        set_last_error("qoz_encode: null pointer argument");
        return ptr::null_mut();
    }
    ffi_guard(ptr::null_mut(), || {
        let desc = unsafe { *desc };
        let channels = channels_from_desc(&desc)?;
        let colorspace = colorspace_from_desc(&desc)?;
        let n_bytes = desc.width as usize * desc.height as usize * u8::from(channels) as usize;
        let pixel_slice = unsafe { slice::from_raw_parts(pixels, n_bytes) };
        let opts = EncodeOptions {
            channels,
            colorspace,
            level,
            tile_rows,
        };
        let encoded =
            qoz::encode(pixel_slice, desc.width, desc.height, &opts).map_err(|e| e.to_string())?;
        unsafe { *out_size = encoded.len() };
        Ok(vec_to_raw(encoded))
    })
}

/// Upper bound on the encoded size for the given dimensions - useful for
/// pre-allocating a buffer on the caller's side, if desired. `channels`
/// must be 1-4; returns 0 for an invalid channel count.
// #[unsafe(no_mangle)]
// pub extern "C" fn qoz_max_encoded_len(
//     width: u32,
//     height: u32,
//     channels: u8,
//     tile_rows: u32,
// ) -> usize {
//     match Channels::try_from(channels) {
//         Ok(c) => qoz::max_encoded_len(width, height, c, tile_rows),
//         Err(_) => 0,
//     }
// }

/// Free a buffer returned by [`qoz_decode`] or [`qoz_encode`]. `len` must
/// be exactly the byte length documented for that call. Safe to call with
/// `ptr == NULL` (no-op).
///
/// # Safety
/// `ptr` must either be NULL, or a pointer previously returned by
/// `qoz_decode`/`qoz_encode` that has not already been freed, and `len`
/// must exactly match the length documented for that call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qoz_free(ptr: *mut u8, len: usize) {
    if ptr.is_null() {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| {
        drop(unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr, len)) });
    }));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_through_c_api() {
        let w = 32u32;
        let h = 16u32;
        let channels = 4u8;
        let mut pixels = vec![0u8; (w * h * channels as u32) as usize];
        for (i, b) in pixels.iter_mut().enumerate() {
            *b = (i % 256) as u8;
        }

        let desc = QozDesc {
            width: w,
            height: h,
            channels,
            colorspace: 0,
        };
        let mut out_size: usize = 0;
        let encoded_ptr = unsafe { qoz_encode(pixels.as_ptr(), &desc, 9, 0, &mut out_size) };
        assert!(!encoded_ptr.is_null(), "encode failed: {:?}", unsafe {
            std::ffi::CStr::from_ptr(qoz_last_error())
        });
        let encoded = unsafe { slice::from_raw_parts(encoded_ptr, out_size) }.to_vec();

        let mut decoded_desc = QozDesc {
            width: 0,
            height: 0,
            channels: 0,
            colorspace: 0,
        };
        let decoded_ptr = unsafe { qoz_decode(encoded.as_ptr(), encoded.len(), &mut decoded_desc) };
        assert!(!decoded_ptr.is_null());
        assert_eq!(decoded_desc.width, w);
        assert_eq!(decoded_desc.height, h);
        assert_eq!(decoded_desc.channels, channels);
        let decoded_len =
            (decoded_desc.width * decoded_desc.height * decoded_desc.channels as u32) as usize;
        let decoded = unsafe { slice::from_raw_parts(decoded_ptr, decoded_len) };
        assert_eq!(decoded, pixels.as_slice());

        unsafe {
            qoz_free(encoded_ptr, out_size);
            qoz_free(decoded_ptr, decoded_len);
        }
    }

    #[test]
    fn bad_magic_reports_error_not_panic() {
        let data = vec![0u8; 64];
        let mut desc = QozDesc {
            width: 0,
            height: 0,
            channels: 0,
            colorspace: 0,
        };
        let ptr = unsafe { qoz_decode(data.as_ptr(), data.len(), &mut desc) };
        assert!(ptr.is_null());
        let err = unsafe { std::ffi::CStr::from_ptr(qoz_last_error()) };
        assert!(err.to_str().unwrap().contains("magic"));
    }

    #[test]
    fn null_pointers_are_rejected_not_ub() {
        let mut desc = QozDesc {
            width: 0,
            height: 0,
            channels: 0,
            colorspace: 0,
        };
        assert!(unsafe { qoz_decode(ptr::null(), 10, &mut desc) }.is_null());
        assert!(!unsafe { qoz_get_desc(ptr::null(), 10, &mut desc) });
    }
}
