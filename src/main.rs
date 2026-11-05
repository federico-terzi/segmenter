mod decoder;
mod encoder;
mod engine;
mod frame;
mod processing;

use std::{
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use anyhow::{bail, Context};
use clap::Parser;

use decoder::{open_video_decoder, DecodeOptions};
use encoder::{create_video_encoder, EncodeOptions};
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

    /// Overlap decoding, inference and encoding (default: on for Metal and Windows).
    #[arg(
        long,
        action = clap::ArgAction::Set,
        num_args = 0..=1,
        default_missing_value = "true"
    )]
    pipeline: Option<bool>,

    /// Faster Windows-compatible H.264 (default: on for macOS Metal).
    /// Use false to retain the original compression settings.
    #[cfg(target_os = "macos")]
    #[arg(long, action = clap::ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    h264_fast: Option<bool>,
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
    #[cfg(target_os = "macos")]
    if args.h264_fast == Some(true) && engine_label != "Metal" {
        bail!("--h264-fast is supported only by the native macOS Metal path");
    }
    let encode_options = EncodeOptions {
        #[cfg(target_os = "macos")]
        h264_fast: args.h264_fast.unwrap_or(engine_label == "Metal"),
    };
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

    let mode = if args
        .pipeline
        .unwrap_or(engine_label == "Metal" || cfg!(target_os = "windows"))
    {
        processing::Mode::Pipelined
    } else {
        processing::Mode::Sequential
    };
    eprintln!("INFO: Processing mode: {mode:?}");
    let mut progress = ProgressReporter::new(None);
    let frames = processing::run(
        mode,
        || {
            open_video_decoder(
                &args.input,
                DecodeOptions {
                    max_dimension: args.max_dimension,
                    #[cfg(target_os = "macos")]
                    native_buffers: engine_label == "Metal",
                },
            )
            .with_context(|| format!("failed to open video decoder for {}", args.input.display()))
        },
        || {
            create_video_encoder(&args.output, encode_options).with_context(|| {
                format!(
                    "failed to create video encoder for {}",
                    args.output.display()
                )
            })
        },
        engine.as_mut(),
        |event| match event {
            processing::Progress::Started(duration) => {
                progress = ProgressReporter::new(duration);
                progress.emit_initial(0);
            }
            processing::Progress::Frame(time, frames) => {
                progress.maybe_emit(time, frames);
            }
        },
    )?;
    progress.emit_final(frames);
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

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(extra: &[&str]) -> Result<Args, clap::Error> {
        Args::try_parse_from(
            [
                "segmenter",
                "--input",
                "input.mp4",
                "--output",
                "output.mp4",
                "--model-path",
                "model.onnx",
            ]
            .into_iter()
            .chain(extra.iter().copied()),
        )
    }

    #[test]
    fn pipeline_default_is_selected_after_detecting_engine() {
        assert_eq!(parse(&[]).unwrap().pipeline, None);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn fast_h264_can_be_disabled_without_affecting_pipeline_selection() {
        assert_eq!(parse(&[]).unwrap().h264_fast, None);
        for (flags, expected) in [
            (vec!["--h264-fast"], true),
            (vec!["--h264-fast", "true"], true),
            (vec!["--h264-fast=false"], false),
        ] {
            let args = parse(&flags).unwrap();
            assert_eq!(args.h264_fast, Some(expected));
            assert_eq!(args.pipeline, None);
        }
        assert!(parse(&["--h264-fast", "maybe"]).is_err());
    }

    #[test]
    fn pipeline_flag_accepts_bare_and_explicit_boolean_values() {
        for (flags, expected) in [
            (vec!["--pipeline"], true),
            (vec!["--pipeline", "true"], true),
            (vec!["--pipeline", "false"], false),
            (vec!["--pipeline=true"], true),
            (vec!["--pipeline=false"], false),
        ] {
            assert_eq!(parse(&flags).unwrap().pipeline, Some(expected));
        }
        assert!(parse(&["--pipeline=invalid"]).is_err());
    }
}
