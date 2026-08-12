use std::fs::{File, OpenOptions};
use std::io::Cursor;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, bail};
use clap::{Parser, Subcommand};
use image::ExtendedColorType;
use memmap2::{Mmap, MmapMut};
use qoz::{Channels, ColorSpace, EncodeOptions};

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
    /// Rows per tile. Each tile is a zstd frame decoded on its own thread.
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

fn load_image_raw(
  path: &PathBuf,
  force_alpha: bool,
) -> Result<(Vec<u8>, u32, u32, Channels), anyhow::Error> {
  let in_mmap = unsafe { Mmap::map(&File::open(path)?)? };
  let img = image::ImageReader::new(Cursor::new(in_mmap))
    .with_guessed_format()
    .map_err(image::ImageError::from)
    .and_then(|i| i.decode())
    .context("failed to decode image-rs image.")?;

  let (w, h) = image::GenericImageView::dimensions(&img);
  let has_color = img.color().has_color();
  let has_alpha = img.color().has_alpha();

  let (buf, channels) = if !has_color && !has_alpha {
    (img.into_luma8().into_raw(), Channels::Gray)
  } else if !has_color && has_alpha {
    (img.into_luma_alpha8().into_raw(), Channels::GrayA)
  } else if has_alpha || force_alpha {
    (img.into_rgba8().into_raw(), Channels::Rgba)
  } else {
    (img.into_rgb8().into_raw(), Channels::Rgb)
  };

  Ok((buf, w, h, channels))
}

fn save_pixels(
  path: &PathBuf,
  w: u32,
  h: u32,
  channels: Channels,
  pixels: &[u8],
) -> Result<(), anyhow::Error> {
  let channels = match channels {
    Channels::Gray => ExtendedColorType::L8,
    Channels::GrayA => ExtendedColorType::La8,
    Channels::Rgb => ExtendedColorType::Rgb8,
    Channels::Rgba => ExtendedColorType::Rgba8,
  };
  image::save_buffer(path, pixels, w, h, channels).context("Failed to save image to disk.")?;
  Ok(())
}

fn cmd_encode(
  input: PathBuf,
  output: PathBuf,
  level: i32,
  tile_rows: u32,
  force_alpha: bool,
) -> Result<(), anyhow::Error> {
  let (pixels, w, h, channels) =
    load_image_raw(&input, force_alpha).context("Failed to load raw image.")?;

  let opts = EncodeOptions {
    channels,
    colorspace: ColorSpace::default(),
    level,
    tile_rows,
  };
  let raw_len = pixels.len();
  let max_len = qoz::encode_max_len(w, h, opts.channels, opts.tile_rows);

  // Create/truncate the output file and create an Mmap with Read+Write permissions. The file is
  // created with max possible compressed len, and then truncated later. This way the can be
  // written directly to the disk instead of allocating a Vec.
  let file = OpenOptions::new()
    .read(true)
    .write(true)
    .create(true)
    .truncate(true)
    .open(&output)?;
  file.set_len(max_len as u64)?;

  let mut out_mmap = unsafe { MmapMut::map_mut(&file).context("Failed to mmap input file.")? };

  // out_mmap must be dropped before truncating file len, so it is scoped here.
  let (bytes_written, dt) = {
    let t0 = Instant::now();
    let written = qoz::encode_into_buf(&pixels, w, h, &opts, &mut out_mmap)
      .context("failed to encode image.")? as u64;
    let dt = t0.elapsed();
    out_mmap.flush().context("Failed to flush mmap to disk.")?;
    (written, dt)
  };
  file.set_len(bytes_written)?;

  println!(
    "{w}x{h} ({channels}) -> {}\n\traw: {raw_len} bytes -> qoz: {bytes_written} bytes ({:.1}% of raw)\n\tencode time: {dt:.2?} ms",
    output.display(),
    // encoded.len(),
    100.0 * bytes_written as f64 / raw_len as f64,
  );

  Ok(())
}

fn cmd_decode(input: PathBuf, output: PathBuf) -> Result<(), anyhow::Error> {
  let in_mmap = unsafe { Mmap::map(&File::open(input)?)? };

  let t0 = Instant::now();
  let (header, pixels) = qoz::decode(&in_mmap).context("failed to decode image.")?;
  let dt = t0.elapsed();

  let mb_s = (pixels.len() as f64 / 1e6) / dt.as_secs_f64();
  println!(
    "decoded {}x{} ({}) in {dt:.2?} ms ({mb_s:.0} MB/s)",
    header.width, header.height, header.channels
  );

  save_pixels(
    &output,
    header.width,
    header.height,
    header.channels,
    pixels.as_slice(),
  )?;
  println!("saved -> {}", output.display());

  Ok(())
}

fn cmd_info(input: PathBuf) -> Result<(), anyhow::Error> {
  let data = unsafe { Mmap::map(&File::open(&input)?)? };
  let header = qoz::read_header(&data).context("Failed to decode qoz header.")?;
  println!("file:       {}", input.display());
  println!("dimensions: {}x{}", header.width, header.height);
  println!("channels:   {}", header.channels);
  println!("colorspace: {}", header.colorspace);
  println!("tile_rows:  {}", header.tile_rows);
  println!("tile_count: {}", header.tile_count);
  println!("raw size:   {} bytes", header.num_bytes());
  println!(
    "file size:  {} bytes ({:.1}% of raw)",
    data.len(),
    100.0 * data.len() as f64 / header.num_bytes().max(1) as f64
  );
  Ok(())
}

fn cmd_bench(
  input: PathBuf,
  iterations: u32,
  level: i32,
  tile_rows: u32,
) -> Result<(), anyhow::Error> {
  let (pixels, w, h, channels) = load_image_raw(&input, false)?;
  if channels != Channels::Rgb && channels != Channels::Rgba {
    bail!("bench only supports 3 or 4 channel source images (qoi requires 3 or 4)");
  }

  let opts = EncodeOptions {
    channels,
    colorspace: ColorSpace::default(),
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
  qoz::decode_into_buf(&qoz_data, &mut qoz_buf)?;
  qoi::decode_to_buf(&mut qoi_buf, &qoi_data)?;

  let t0 = Instant::now();
  for _ in 0..iterations {
    qoz::decode_into_buf(&qoz_data, &mut qoz_buf)?;
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
    "image: {} ({w}x{h}, {channels}, {} raw bytes)",
    input.display(),
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
    "{:<8} {:>12} {:>7.1}% {:>9.3} ms {qoz_mbs:>10.0}",
    "qoz",
    qoz_data.len(),
    100.0 * qoz_data.len() as f64 / pixels.len() as f64,
    qoz_dt.as_secs_f64() * 1000.0,
  );
  println!(
    "{:<8} {:>12} {:>7.1}% {:>9.3} ms {qoi_mbs:>10.0}",
    "qoi",
    qoi_data.len(),
    100.0 * qoi_data.len() as f64 / pixels.len() as f64,
    qoi_dt.as_secs_f64() * 1000.0,
  );
  println!();
  println!(
    "qoz decode is {:.2}x qoi decode throughput (level={level}, {} tiles)",
    qoz_mbs / qoi_mbs,
    qoz::read_header(&qoz_data)?.tile_count
  );
  Ok(())
}

fn main() -> Result<(), anyhow::Error> {
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
