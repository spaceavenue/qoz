use std::error::Error;
use std::fs::{File, OpenOptions};
use std::io::Cursor;
use std::path::PathBuf;
use std::time::Instant;

use clap::{Parser, Subcommand};
use image::{ImageError, ImageReader};
use memmap2::{Mmap, MmapMut};
use qoz::EncodeOptions;

type BoxErr = Box<dyn Error>;

#[derive(Parser)]
#[command(
    name = "qoz",
    version,
    about = "Convert images to/from QOZ: a minimal format storing zstd-compressed raw pixel tiles, tuned for fast decoding"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Encode any image-rs supported format into .qoz
    Encode {
        input: PathBuf,
        output: PathBuf,
        /// zstd compression level, from 1 to 22 (default = 9).
        /// Mostly affects ratio & encode time, not decode time.
        #[arg(long, default_value_t = 9)]
        level: i32,
        /// Rows per tile. Each tile is a zstd frame decoded on its own thread).
        /// 0 = pick automatically from available CPU cores (default).
        #[arg(long, default_value_t = 0)]
        tile_rows: u32,
        /// Force RGBA output even if the source has no alpha.
        #[arg(long)]
        force_alpha: bool,
    },
    /// Decode a .qoz file to any image-rs-supported format (by extension).
    Decode { input: PathBuf, output: PathBuf },
    /// Print header info about a .qoz file.
    Info { input: PathBuf },
    /// Benchmark qoz decode and encode speed against the qoi crate on a given source image.
    Bench {
        input: PathBuf,
        #[arg(long, default_value_t = 50)]
        iterations: u32,
        #[arg(long, default_value_t = 9)]
        level: i32,
        #[arg(long, default_value_t = 0)]
        tile_rows: u32,
    },
}

fn load_image_raw(path: &PathBuf, force_alpha: bool) -> Result<(Vec<u8>, u32, u32, u8), BoxErr> {
    let in_mmap = unsafe { Mmap::map(&File::open(path)?)? };
    let img = ImageReader::new(Cursor::new(in_mmap))
        .with_guessed_format()
        .map_err(ImageError::from)
        .and_then(|i| i.decode())?;

    if img.color().has_alpha() || force_alpha {
        let buf = img.into_rgba8();
        let (w, h) = buf.dimensions();
        Ok((buf.into_raw(), w, h, 4))
    } else {
        let buf = img.into_rgb8();
        let (w, h) = buf.dimensions();
        Ok((buf.into_raw(), w, h, 3))
    }
}

fn save_pixels(
    path: &PathBuf,
    w: u32,
    h: u32,
    channels: u8,
    pixels: Vec<u8>,
) -> Result<(), BoxErr> {
    match channels {
        1 => image::GrayImage::from_raw(w, h, pixels)
            .ok_or("pixel buffer size mismatch")?
            .save(path)?,
        3 => image::RgbImage::from_raw(w, h, pixels)
            .ok_or("pixel buffer size mismatch")?
            .save(path)?,
        4 => image::RgbaImage::from_raw(w, h, pixels)
            .ok_or("pixel buffer size mismatch")?
            .save(path)?,
        other => {
            return Err(format!(
                "cannot save a {other}-channel image, the format only supports 1-4 channel(s)"
            )
            .into());
        }
    }
    Ok(())
}

fn cmd_encode(
    input: PathBuf,
    output: PathBuf,
    level: i32,
    tile_rows: u32,
    force_alpha: bool,
) -> Result<(), BoxErr> {
    let (pixels, w, h, channels) = load_image_raw(&input, force_alpha)?;

    let opts = EncodeOptions {
        channels,
        colorspace: 0,
        level,
        tile_rows,
    };

    let t0 = Instant::now();
    let encoded = qoz::encode(&pixels, w, h, &opts)?;
    let dt = t0.elapsed();

    // Create/truncate the output file and create an Mmap with Read+Write permissions. This works
    // here because we know the length of the encoded output.
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(&output)?;

    file.set_len(encoded.len() as u64)?;
    let mut out_mmap = unsafe { MmapMut::map_mut(&file)? };
    out_mmap.copy_from_slice(&encoded);
    out_mmap.flush()?;

    let raw_len = pixels.len();
    println!(
        "{}x{} ({} ch) -> {}\n  raw: {} bytes -> qoz: {} bytes ({:.1}% of raw)\n  encode time: {:.2?} ms",
        w,
        h,
        channels,
        output.display(),
        raw_len,
        encoded.len(),
        100.0 * encoded.len() as f64 / raw_len as f64,
        dt
    );

    Ok(())
}

fn cmd_decode(input: PathBuf, output: PathBuf) -> Result<(), BoxErr> {
    let in_mmap = unsafe { Mmap::map(&File::open(input)?)? };

    let t0 = Instant::now();
    let (header, pixels) = qoz::decode(&in_mmap)?;
    let dt = t0.elapsed();

    let mb_s = (pixels.len() as f64 / 1e6) / dt.as_secs_f64();
    println!(
        "decoded {}x{} ({} ch) in {:.2?} ms ({:.0} MB/s)",
        header.width, header.height, header.channels, dt, mb_s
    );

    save_pixels(
        &output,
        header.width,
        header.height,
        header.channels,
        pixels,
    )?;
    println!("saved -> {}", output.display());

    Ok(())
}

fn cmd_info(input: PathBuf) -> Result<(), BoxErr> {
    let data = std::fs::read(&input)?;
    let header = qoz::read_header(&data)?;
    println!("file:       {}", input.display());
    println!("dimensions: {}x{}", header.width, header.height);
    println!("channels:   {}", header.channels);
    println!("colorspace: {}", header.colorspace);
    println!("tile_rows:  {}", header.tile_rows);
    println!("tile_count: {}", header.tile_count);
    println!("raw size:   {} bytes", header.total_bytes());
    println!(
        "file size:  {} bytes ({:.1}% of raw)",
        data.len(),
        100.0 * data.len() as f64 / header.total_bytes().max(1) as f64
    );
    Ok(())
}

fn cmd_bench(input: PathBuf, iterations: u32, level: i32, tile_rows: u32) -> Result<(), BoxErr> {
    let (pixels, w, h, channels) = load_image_raw(&input, false)?;
    if channels != 3 && channels != 4 {
        return Err(
            "bench only supports 3 or 4 channel source images (qoi itself requires 3 or 4)".into(),
        );
    }

    let opts = EncodeOptions {
        channels,
        colorspace: 0,
        level,
        tile_rows,
    };
    let qoz_data = qoz::encode(&pixels, w, h, &opts)?;
    let qoi_data = qoi::encode_to_vec(&pixels, w, h)?;

    let (_, qoz_roundtrip) = qoz::decode(&qoz_data)?;
    assert_eq!(
        qoz_roundtrip, pixels,
        "qoz roundtrip did not match source pixels!"
    );
    let (_, qoi_roundtrip) = qoi::decode_to_vec(&qoi_data)?;
    assert_eq!(
        qoi_roundtrip, pixels,
        "qoi roundtrip did not match source pixels!"
    );

    let mut qoz_buf = vec![0u8; pixels.len()];
    let mut qoi_buf = vec![0u8; pixels.len()];

    // Warm-up passes (excluded from timing).
    qoz::decode_into(&qoz_data, &mut qoz_buf)?;
    qoi::decode_to_buf(&mut qoi_buf, &qoi_data)?;

    let t0 = Instant::now();
    for _ in 0..iterations {
        qoz::decode_into(&qoz_data, &mut qoz_buf)?;
    }
    let qoz_dt = t0.elapsed() / iterations;

    let t0 = Instant::now();
    for _ in 0..iterations {
        qoi::decode_to_buf(&mut qoi_buf, &qoi_data)?;
    }
    let qoi_dt = t0.elapsed() / iterations;

    let mb = pixels.len() as f64 / 1e6;
    let qoz_mbs = mb / qoz_dt.as_secs_f64();
    let qoi_mbs = mb / qoi_dt.as_secs_f64();

    println!(
        "image: {} ({}x{}, {} channels, {} raw bytes)",
        input.display(),
        w,
        h,
        channels,
        pixels.len()
    );
    println!(
        "threads available: {}",
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    );
    println!();
    println!(
        "{:<8} {:>12} {:>8} {:>12} {:>10}",
        "format", "size(B)", "ratio", "decode/iter", "MB/s"
    );
    println!(
        "{:<8} {:>12} {:>7.1}% {:>9.3} ms {:>10.0}",
        "qoz",
        qoz_data.len(),
        100.0 * qoz_data.len() as f64 / pixels.len() as f64,
        qoz_dt.as_secs_f64() * 1000.0,
        qoz_mbs
    );
    println!(
        "{:<8} {:>12} {:>7.1}% {:>9.3} ms {:>10.0}",
        "qoi",
        qoi_data.len(),
        100.0 * qoi_data.len() as f64 / pixels.len() as f64,
        qoi_dt.as_secs_f64() * 1000.0,
        qoi_mbs
    );
    println!();
    println!(
        "qoz decode is {:.2}x qoi decode throughput (level={}, {} tiles)",
        qoz_mbs / qoi_mbs,
        level,
        qoz::read_header(&qoz_data)?.tile_count
    );
    Ok(())
}

fn main() -> Result<(), BoxErr> {
    let cli = Cli::parse();
    match cli.command {
        Command::Encode {
            input,
            output,
            level,
            tile_rows,
            force_alpha,
        } => cmd_encode(input, output, level, tile_rows, force_alpha),
        Command::Decode { input, output } => cmd_decode(input, output),
        Command::Info { input } => cmd_info(input),
        Command::Bench {
            input,
            iterations,
            level,
            tile_rows,
        } => cmd_bench(input, iterations, level, tile_rows),
    }
}
