mod decoder;
mod encoder;
mod engine;
mod frame;

use std::path::{Path, PathBuf};

use anyhow::{bail, Context};
use clap::Parser;

use decoder::{open_video_decoder, DecodeOptions};
use encoder::create_video_encoder;
use engine::{create_engine, EngineOptions};

#[derive(Debug, Parser)]
#[command(name = "segmenter")]
#[command(about = "Run Robust Video Matting over a video and write a mask video")]
struct Args {
    #[arg(long)]
    input: PathBuf,

    #[arg(long)]
    output: PathBuf,

    #[arg(long)]
    model_path: PathBuf,

    #[arg(long)]
    max_dimension: Option<u32>,

    #[arg(long, default_value_t = 0.25)]
    downsample_ratio: f32,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    validate_video_path(&args.input, "input")?;
    validate_video_path(&args.output, "output")?;

    if !args.input.exists() {
        bail!("input file does not exist: {}", args.input.display());
    }
    if args.max_dimension == Some(0) {
        bail!("--max-dimension must be greater than zero when provided");
    }

    let mut decoder = open_video_decoder(
        &args.input,
        DecodeOptions {
            max_dimension: args.max_dimension,
        },
    )
    .with_context(|| format!("failed to open video decoder for {}", args.input.display()))?;
    let mut encoder = create_video_encoder(&args.output).with_context(|| {
        format!(
            "failed to create video encoder for {}",
            args.output.display()
        )
    })?;
    let mut engine = create_engine(EngineOptions {
        model_path: args.model_path.clone(),
        downsample_ratio: args.downsample_ratio,
    })
    .with_context(|| {
        format!(
            "failed to initialize segmentation engine for {}",
            args.model_path.display()
        )
    })?;

    let mut frames = 0u64;
    while let Some(frame) = decoder.read_frame()? {
        let mask = engine.segment(&frame)?;
        encoder.send_frame(&mask)?;
        frames += 1;
    }

    if frames == 0 {
        bail!("input video did not produce any frames");
    }

    encoder.finalize()?;
    Ok(())
}

fn validate_video_path(path: &Path, label: &str) -> anyhow::Result<()> {
    let extension = path
        .extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| extension.to_ascii_lowercase())
        .with_context(|| format!("{label} path must have a video extension"))?;

    match extension.as_str() {
        "mp4" | "mov" => Ok(()),
        _ => bail!(
            "{label} path extension .{} is not supported; this version accepts video files only (.mp4, .mov)",
            extension
        ),
    }
}
