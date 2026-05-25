mod decoder;
mod encoder;
mod engine;
mod frame;

use std::{
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use anyhow::{bail, Context};
use clap::Parser;

use decoder::{open_video_decoder, DecodeOptions};
use encoder::create_video_encoder;
use engine::{create_engine, engine_label_for_model, EngineOptions};
use frame::MediaTime;

const PROGRESS_INTERVAL: Duration = Duration::from_secs(1);

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

    let engine_label = engine_label_for_model(&args.model_path)?;
    eprintln!(
        "INFO: Version {} Arch {}",
        env!("CARGO_PKG_VERSION"),
        std::env::consts::ARCH
    );
    eprintln!(
        "INFO: Using {engine_label} engine from model \"{}\"",
        args.model_path.display()
    );
    eprintln!(
        "INFO: Generating segmentation mask for file \"{}\"",
        args.input.display()
    );
    eprintln!("INFO: Writing output file \"{}\"", args.output.display());

    let mut decoder = open_video_decoder(
        &args.input,
        DecodeOptions {
            max_dimension: args.max_dimension,
        },
    )
    .with_context(|| format!("failed to open video decoder for {}", args.input.display()))?;
    let duration = decoder.duration();
    let mut encoder = create_video_encoder(&args.output).with_context(|| {
        format!(
            "failed to create video encoder for {}",
            args.output.display()
        )
    })?;
    encoder.set_expected_duration(duration)?;
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

    let mut progress = ProgressReporter::new(duration);
    progress.emit_initial(0);

    let mut frames = 0u64;
    while let Some(frame) = decoder.read_frame()? {
        let frame_time = frame.time;
        let mask = engine.segment(&frame)?;
        encoder.send_frame(&mask)?;
        frames += 1;
        progress.maybe_emit(frame_time, frames);
    }

    if frames == 0 {
        bail!("input video did not produce any frames");
    }

    progress.emit_final(frames);
    encoder.finalize()?;
    eprintln!("DONE");
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

struct ProgressReporter {
    duration_seconds: Option<f64>,
    last_emit: Option<Instant>,
    last_reported_seconds: Option<f64>,
    latest_current_seconds: Option<f64>,
}

impl ProgressReporter {
    fn new(duration: Option<MediaTime>) -> Self {
        Self {
            duration_seconds: duration.map(MediaTime::as_seconds),
            last_emit: None,
            last_reported_seconds: None,
            latest_current_seconds: None,
        }
    }

    fn emit_initial(&mut self, frames: u64) {
        self.emit(0.0, frames, Instant::now());
    }

    fn maybe_emit(&mut self, current: MediaTime, frames: u64) {
        let now = Instant::now();
        self.latest_current_seconds = Some(current.as_seconds());
        if self
            .last_emit
            .is_some_and(|last_emit| now.duration_since(last_emit) < PROGRESS_INTERVAL)
        {
            return;
        }

        self.emit(current.as_seconds(), frames, now);
    }

    fn emit_final(&mut self, frames: u64) {
        match self.duration_seconds {
            Some(duration) if !same_displayed_time(self.last_reported_seconds, duration) => {
                self.emit(duration, frames, Instant::now());
            }
            None => {
                if let Some(current) = self.latest_current_seconds {
                    if same_displayed_time(self.last_reported_seconds, current) {
                        return;
                    }
                    self.emit(current, frames, Instant::now());
                }
            }
            _ => {}
        }
    }

    fn emit(&mut self, current_seconds: f64, frames: u64, now: Instant) {
        let current = self.clamp_current(current_seconds);
        match self.duration_seconds {
            Some(duration) => eprintln!("PROGRESS: {:.2}/{:.2} {frames}", current, duration),
            None => eprintln!("PROGRESS: {:.2}/NaN {frames}", current),
        }
        self.last_reported_seconds = Some(current);
        self.last_emit = Some(now);
    }

    fn clamp_current(&self, current_seconds: f64) -> f64 {
        let current = current_seconds.max(0.0);
        match self.duration_seconds {
            Some(duration) => current.min(duration),
            None => current,
        }
    }
}

fn same_displayed_time(current: Option<f64>, target: f64) -> bool {
    current.is_some_and(|current| (current - target).abs() < 0.005)
}
