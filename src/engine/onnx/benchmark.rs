//! Opt-in real-model benchmark. No downloads or additional dependencies.
use super::*;
use crate::{
    decoder::{open_video_decoder, DecodeOptions},
    frame::MediaTime,
};
use std::{
    io::{Read, Write},
    time::Instant,
};

/// A completion-timed probe for extracted, fixed-shape ONNX blocks. Inputs are
/// uploaded once; outputs stay on DirectML. This is not end-to-end video timing.
#[test]
#[cfg(target_os = "windows")]
#[ignore = "requires SEGMENTER_BENCH_GRAPH; research only"]
fn graph_benchmark() -> anyhow::Result<()> {
    use ort::{
        execution_providers::DirectMLExecutionProvider,
        memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType},
        session::{builder::GraphOptimizationLevel, Session},
        tensor::TensorElementType,
        value::ValueType,
    };
    let path = std::env::var("SEGMENTER_BENCH_GRAPH")?;
    if let Ok(output) = std::env::var("SEGMENTER_BENCH_INFER_SHAPES") {
        Session::builder()?
            .with_intra_threads(1)?
            .with_optimization_level(GraphOptimizationLevel::Level1)?
            .with_optimized_model_path(output)?
            .commit_from_file(&path)?;
        return Ok(());
    }
    let paths = if std::path::Path::new(&path).is_dir() {
        let mut paths = fs::read_dir(&path)?
            .map(|v| v.map(|v| v.path()))
            .collect::<Result<Vec<_>, _>>()?;
        paths.retain(|p| p.extension().is_some_and(|e| e == "onnx"));
        paths.sort();
        paths
    } else {
        vec![PathBuf::from(path)]
    };
    let device: i32 = std::env::var("SEGMENTER_BENCH_DEVICE")
        .unwrap_or("0".into())
        .parse()?;
    let mut failed = 0;
    for path in paths {
        let capture = std::env::var_os("SEGMENTER_BENCH_CAPTURE").is_some();
        let builder = Session::builder()?
            .with_memory_pattern(false)?
            .with_parallel_execution(false)?
            .with_intra_op_spinning(false)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?;
        let builder = if capture {
            builder.with_config_entry("ep.dml.enable_graph_capture", "true")?
        } else {
            builder
        };
        let builder = if std::env::var_os("SEGMENTER_BENCH_NO_METACOMMANDS").is_some() {
            use ort::AsPointer;
            let mut builder = builder;
            let device_value = std::ffi::CString::new(device.to_string())?;
            let keys = [c"device_id".as_ptr(), c"disable_metacommands".as_ptr()];
            let values = [device_value.as_ptr(), c"true".as_ptr()];
            // A provider option, not a session config entry. ORT copies the
            // strings during this call; their storage stays alive until return.
            unsafe {
                ort::error::status_to_result((ort::api().SessionOptionsAppendExecutionProvider)(
                    builder.ptr_mut(),
                    c"DML".as_ptr(),
                    keys.as_ptr(),
                    values.as_ptr(),
                    keys.len(),
                ))?;
            }
            builder
        } else {
            builder.with_execution_providers([DirectMLExecutionProvider::default()
                .with_device_id(device)
                .build()
                .error_on_failure()])?
        };
        let session = builder.commit_from_file(&path);
        let mut session = match session {
            Ok(session) => session,
            Err(error) => {
                eprintln!("GRAPH {} COMPILE_FAILED {error}", path.display());
                failed += 1;
                continue;
            }
        };
        let mut binding = session.create_binding()?;
        let mut inputs = Vec::new();
        for input in &session.inputs {
            let ValueType::Tensor { ty, shape, .. } = &input.input_type else {
                anyhow::bail!("not a tensor");
            };
            anyhow::ensure!(
                shape.iter().all(|d| *d > 0),
                "dynamic input {} {shape:?}",
                input.name
            );
            let len = shape.iter().product::<i64>() as usize;
            let tensor = match ty {
                TensorElementType::Float16 => {
                    Tensor::from_array((shape.to_vec(), vec![f16::from_f32(0.1); len]))?.upcast()
                }
                TensorElementType::Float32 => {
                    Tensor::from_array((shape.to_vec(), vec![0.1_f32; len]))?.upcast()
                }
                _ => anyhow::bail!("unsupported input type {ty:?}"),
            };
            let tensor = if capture {
                tensor.to(AllocationDevice::DIRECTML, device)?
            } else {
                tensor
            };
            binding.bind_input(&input.name, &tensor)?;
            inputs.push(tensor);
        }
        let memory = MemoryInfo::new(
            AllocationDevice::DIRECTML,
            0,
            AllocatorType::Device,
            MemoryType::Default,
        )?;
        let allocator = ort::memory::Allocator::new(&session, memory.clone())?;
        for output in &session.outputs {
            if capture {
                let ValueType::Tensor { ty, shape, .. } = &output.output_type else {
                    anyhow::bail!("not tensor");
                };
                anyhow::ensure!(
                    shape.iter().all(|d| *d > 0),
                    "dynamic output {}",
                    output.name
                );
                let value = ort::value::DynTensor::new(&allocator, *ty, shape.to_vec())?;
                binding.bind_output(&output.name, value)?;
            } else {
                binding.bind_output_to_device(&output.name, &memory)?;
            }
        }
        let mut times = Vec::new();
        for i in 0..600 {
            let start = Instant::now();
            let _outputs = session.run_binding(&binding)?;
            binding.synchronize_outputs()?;
            if i >= 100 {
                times.push(start.elapsed().as_secs_f64() * 1000.);
            }
        }
        let mean = times.iter().sum::<f64>() / times.len() as f64;
        times.sort_by(f64::total_cmp);
        eprintln!(
            "GRAPH {} completed_runs={} mean_ms={mean:.4} median_ms={:.4} p95_ms={:.4}",
            path.display(),
            times.len(),
            times[times.len() / 2],
            times[times.len() * 95 / 100]
        );
    }
    anyhow::ensure!(failed == 0, "{failed} graphs failed to compile");
    Ok(())
}

#[test]
#[cfg(target_os = "windows")]
#[ignore = "GPU-resident RVM experiment; requires SEGMENTER_BENCH_MODEL"]
fn resident_network_benchmark() -> anyhow::Result<()> {
    let model = PathBuf::from(std::env::var("SEGMENTER_BENCH_MODEL")?);
    let ratio = std::env::var("SEGMENTER_BENCH_RATIO")
        .unwrap_or("1".into())
        .parse()?;
    let width = std::env::var("SEGMENTER_BENCH_WIDTH")
        .unwrap_or("480".into())
        .parse()?;
    let height = std::env::var("SEGMENTER_BENCH_HEIGHT")
        .unwrap_or("360".into())
        .parse()?;
    let mut engine = OnnxEngine::new(model, ratio)?;
    engine.ensure_initialized(width, height)?;
    let state = engine.state.as_mut().unwrap();
    anyhow::ensure!(
        state.recurrent_memory.allocation_device() == ort::memory::AllocationDevice::DIRECTML,
        "benchmark requires DirectML"
    );
    if let Ok(path) = std::env::var("SEGMENTER_BENCH_EXPORT_STATIC") {
        let original = fs::read(&engine.model_path)?;
        let prepared = prepare_primary_model(&original, width, height, ratio)?;
        let model = match &prepared.source {
            PreparedModelSource::Bytes(b) => b,
            _ => &original,
        };
        let mut shapes = vec![(SRC_INPUT, vec![1, 3, height as i64, width as i64])];
        shapes.extend(
            [R1_INPUT, R2_INPUT, R3_INPUT, R4_INPUT]
                .into_iter()
                .zip(&state.recurrent)
                .map(|(n, v)| (n, v.shape().to_vec())),
        );
        fs::write(path, specialize::specialize(model, &shapes, ratio)?)?;
    }
    state.binding.bind_input(SRC_INPUT, &state.src)?;
    if state.uses_downsample_ratio {
        state
            .binding
            .bind_input(DOWNSAMPLE_RATIO_INPUT, &engine.downsample_ratio)?;
    }
    state
        .binding
        .bind_output_to_device(PHA_OUTPUT, &state.recurrent_memory)?;
    let mut times = Vec::new();
    for i in 0..600 {
        let start = Instant::now();
        for (name, value) in [R1_INPUT, R2_INPUT, R3_INPUT, R4_INPUT]
            .into_iter()
            .zip(&state.recurrent)
        {
            state.binding.bind_input(name, value)?;
        }
        for name in [R1_OUTPUT, R2_OUTPUT, R3_OUTPUT, R4_OUTPUT] {
            state
                .binding
                .bind_output_to_device(name, &state.recurrent_memory)?;
        }
        let mut outputs = state.session.run_binding(&state.binding)?;
        state.binding.synchronize_outputs()?;
        state.recurrent = take_recurrent_outputs(&mut outputs)?;
        if i >= 100 {
            times.push(start.elapsed().as_secs_f64() * 1000.);
        }
    }
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    times.sort_by(f64::total_cmp);
    eprintln!(
        "RESIDENT completed_runs={} mean_ms={mean:.4} median_ms={:.4} p95_ms={:.4}",
        times.len(),
        times[times.len() / 2],
        times[times.len() * 95 / 100]
    );
    Ok(())
}
#[test]
#[ignore = "requires SEGMENTER_BENCH_MODEL; see docs/rvm-performance.md"]
fn inference_benchmark() -> anyhow::Result<()> {
    let model = PathBuf::from(std::env::var("SEGMENTER_BENCH_MODEL")?);
    let ratio: f32 = std::env::var("SEGMENTER_BENCH_RATIO")
        .unwrap_or("1".into())
        .parse()?;
    let dimension: u32 = std::env::var("SEGMENTER_BENCH_DIMENSION")
        .unwrap_or("480".into())
        .parse()?;
    let count: usize = std::env::var("SEGMENTER_BENCH_FRAMES")
        .unwrap_or("150".into())
        .parse()?;
    anyhow::ensure!(count > 10, "need more than 10 frames");
    let mut frames = Vec::new();
    if let Ok(path) = std::env::var("SEGMENTER_BENCH_VIDEO") {
        let mut decoder = open_video_decoder(
            std::path::Path::new(&path),
            DecodeOptions {
                max_dimension: Some(dimension),
            },
        )?;
        while frames.len() < count {
            let Some(frame) = decoder.read_frame()? else {
                break;
            };
            frames.push(frame);
        }
    } else {
        for i in 0..count {
            let mut data = vec![0; dimension as usize * 270 * 4];
            for (p, pixel) in data.chunks_exact_mut(4).enumerate() {
                pixel.copy_from_slice(&[
                    (p.wrapping_mul(13) + i * 7) as u8,
                    (p / 9 + i) as u8,
                    (p / 29) as u8,
                    255,
                ]);
            }
            frames.push(VideoFrame::new_bgra(
                dimension,
                270,
                dimension * 4,
                MediaTime::new(i as i64, 30)?,
                data,
            )?);
        }
    }
    anyhow::ensure!(frames.len() > 10, "video needs more than 10 frames");
    let mut engine = OnnxEngine::new(model, ratio)?;
    let mut reference = std::env::var("SEGMENTER_BENCH_REFERENCE")
        .ok()
        .map(fs::File::open)
        .transpose()?;
    let mut output = std::env::var("SEGMENTER_BENCH_SAVE")
        .ok()
        .map(fs::File::create)
        .transpose()?;
    let mut times = Vec::new();
    let mut total_ms = 0.0;
    let repeats: usize = std::env::var("SEGMENTER_BENCH_REPEATS")
        .unwrap_or("1".into())
        .parse()?;
    let reset_sequence = std::env::var_os("SEGMENTER_BENCH_RESET_SEQUENCE").is_some();
    let warmup: usize = std::env::var("SEGMENTER_BENCH_WARMUP")
        .unwrap_or("10".into())
        .parse()?;
    anyhow::ensure!(
        repeats > 0 && warmup < frames.len() * repeats,
        "invalid repeat/warmup count"
    );
    let (mut total, mut abs_error, mut large, mut max_error) = (0u64, 0u64, 0u64, 0u8);
    for (i, frame) in frames
        .iter()
        .cycle()
        .take(frames.len() * repeats)
        .enumerate()
    {
        if reset_sequence && i > 0 && i % frames.len() == 0 {
            let state = engine
                .state
                .as_mut()
                .context("missing initialized session")?;
            state.recurrent = session::initial_recurrent_states(state.precision, &state.session)?;
        }
        let start = Instant::now();
        let mask = engine.segment(frame)?;
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        total_ms += elapsed;
        if i == 0 {
            eprintln!("BENCH first_frame_ms={elapsed:.3}");
        }
        if i >= warmup {
            times.push(elapsed);
        }
        let alpha: Vec<u8> = mask.data.chunks_exact(4).map(|p| p[0]).collect();
        if let Some(output) = &mut output {
            output.write_all(&alpha)?;
        }
        if let Some(reference) = &mut reference {
            let mut expected = vec![0; alpha.len()];
            reference.read_exact(&mut expected)?;
            if let Some(directory) = std::env::var_os("SEGMENTER_BENCH_PREVIEWS") {
                if [0, 30, 75, 149].contains(&i) {
                    preview(
                        std::path::Path::new(&directory),
                        i,
                        frame,
                        &expected,
                        &alpha,
                    )?;
                }
            }
            for (a, b) in alpha.iter().zip(expected) {
                let delta = a.abs_diff(b);
                abs_error += u64::from(delta);
                max_error = max_error.max(delta);
                large += u64::from(delta > 4);
                total += 1;
            }
        }
    }
    if let Some(reference) = &mut reference {
        anyhow::ensure!(reference.read(&mut [0])? == 0, "reference has extra frames");
    }
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    times.sort_by(f64::total_cmp);
    eprintln!("BENCH frames={} warmup={warmup} reset_sequence={reset_sequence} width={} height={} ratio={ratio} total_inference_ms={total_ms:.3} mean_ms={mean:.3} median_ms={:.3} p95_ms={:.3} fps={:.2}", frames.len()*repeats, frames[0].width, frames[0].height, times[times.len()/2], times[(times.len()*95/100).min(times.len()-1)], 1000.0/mean);
    if total > 0 {
        eprintln!(
            "QUALITY pixels={total} mae_u8={:.6} max_u8={max_error} fraction_over_4={:.8}",
            abs_error as f64 / total as f64,
            large as f64 / total as f64
        );
        anyhow::ensure!(
            abs_error as f64 / total as f64 <= 0.05
                && max_error <= 8
                && large as f64 / total as f64 <= 0.0001,
            "mask equivalence threshold exceeded"
        );
    }
    if std::env::var_os("SEGMENTER_BENCH_PROFILE").is_some() {
        eprintln!(
            "PROFILE {}",
            engine.state.as_mut().unwrap().session.end_profiling()?
        );
    }
    Ok(())
}

// Source, reference composite, candidate composite, and 16x absolute mask error.
pub(super) fn preview(
    directory: &std::path::Path,
    index: usize,
    frame: &VideoFrame,
    expected: &[u8],
    actual: &[u8],
) -> anyhow::Result<()> {
    fs::create_dir_all(directory)?;
    let mut image = image::RgbImage::new(frame.width * 4, frame.height);
    for y in 0..frame.height {
        for x in 0..frame.width {
            let p = (y * frame.bytes_per_row + x * 4) as usize;
            let rgb = [frame.data[p + 2], frame.data[p + 1], frame.data[p]];
            let pixel = (y * frame.width + x) as usize;
            image.put_pixel(x, y, image::Rgb(rgb));
            for (column, alpha) in [expected[pixel], actual[pixel]].into_iter().enumerate() {
                let bg = [40u32, 200, 80];
                let composite = std::array::from_fn(|c| {
                    ((u32::from(rgb[c]) * u32::from(alpha)
                        + bg[c] * (255 - u32::from(alpha))
                        + 127)
                        / 255) as u8
                });
                image.put_pixel(
                    x + frame.width * (column as u32 + 1),
                    y,
                    image::Rgb(composite),
                );
            }
            let error = expected[pixel].abs_diff(actual[pixel]).saturating_mul(16);
            image.put_pixel(x + frame.width * 3, y, image::Rgb([error; 3]));
        }
    }
    image.save(directory.join(format!("comparison-{index:04}.png")))?;
    Ok(())
}
