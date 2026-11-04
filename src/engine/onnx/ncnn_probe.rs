//! Opt-in Vulkan comparison. Uses the supplied RVM weights, not another model.
use super::*;
use crate::decoder::{open_video_decoder, DecodeOptions};
use std::{
    ffi::{c_char, c_void, CStr, CString},
    time::Instant,
};
extern "C" {
    fn rvm_ncnn_error() -> *const c_char;
    fn rvm_ncnn_create(
        root: *const c_char,
        w: i32,
        h: i32,
        adapter: i32,
        flags: u32,
    ) -> *mut c_void;
    fn rvm_ncnn_destroy(p: *mut c_void);
    fn rvm_ncnn_reset(p: *mut c_void) -> i32;
    fn rvm_ncnn_run(p: *mut c_void, rgb: *const f32, alpha: *mut f32) -> i32;
}
struct Probe(*mut c_void);
impl Drop for Probe {
    fn drop(&mut self) {
        unsafe { rvm_ncnn_destroy(self.0) }
    }
}
fn checked(status: i32) -> anyhow::Result<()> {
    anyhow::ensure!(
        status == 0,
        "{}",
        unsafe { CStr::from_ptr(rvm_ncnn_error()) }.to_string_lossy()
    );
    Ok(())
}
#[test]
#[ignore = "one-frame ncnn/DirectML intermediate diagnostic"]
fn ncnn_diagnostic() -> anyhow::Result<()> {
    use ort::{
        execution_providers::DirectMLExecutionProvider,
        memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType},
        session::Session,
        value::ValueType,
    };
    let root = std::env::var("SEGMENTER_NCNN_MODEL").unwrap_or("target/rvm-ncnn-cam".into());
    let croot = CString::new(root.clone())?;
    let mut decoder = open_video_decoder(
        &PathBuf::from(std::env::var("SEGMENTER_BENCH_VIDEO")?),
        DecodeOptions {
            max_dimension: Some(480),
        },
    )?;
    let frame = decoder.read_frame()?.context("empty video")?;
    let n = (frame.width * frame.height) as usize;
    let mut pixels = vec![f16::ZERO; n * 3];
    super::tensor::preprocess_f16(&frame, &mut pixels)?;
    let rgb: Vec<f32> = pixels.iter().map(|v| v.to_f32()).collect();
    let mut alpha = vec![0.; n];
    let raw = unsafe {
        rvm_ncnn_create(
            croot.as_ptr(),
            frame.width as i32,
            frame.height as i32,
            0,
            16,
        )
    };
    if raw.is_null() {
        checked(-1)?;
    }
    let probe = Probe(raw);
    checked(unsafe { rvm_ncnn_run(probe.0, rgb.as_ptr(), alpha.as_mut_ptr()) })?;
    let mut session = Session::builder()?
        .with_memory_pattern(false)?
        .with_parallel_execution(false)?
        .with_intra_threads(1)?
        .with_execution_providers([DirectMLExecutionProvider::default()
            .build()
            .error_on_failure()])?
        .commit_from_file(format!("{root}/diagnostic.onnx"))?;
    let mut binding = session.create_binding()?;
    let src = Tensor::from_array(([1, 3, frame.height as usize, frame.width as usize], pixels))?;
    binding.bind_input("src", &src)?;
    let mut states = Vec::new();
    for name in [R1_INPUT, R2_INPUT, R3_INPUT, R4_INPUT] {
        let v = session
            .inputs
            .iter()
            .find(|v| v.name == name)
            .context("missing recurrence")?;
        let ValueType::Tensor { shape, .. } = &v.input_type else {
            anyhow::bail!("not tensor")
        };
        let t = Tensor::from_array((
            shape.to_vec(),
            vec![f16::ZERO; shape.iter().product::<i64>() as usize],
        ))?;
        binding.bind_input(name, &t)?;
        states.push(t);
    }
    let cpu = MemoryInfo::new(
        AllocationDevice::CPU,
        0,
        AllocatorType::Device,
        MemoryType::Default,
    )?;
    for v in &session.outputs {
        binding.bind_output_to_device(&v.name, &cpu)?;
    }
    let outputs = session.run_binding(&binding)?;
    for (name, v) in outputs.iter() {
        let data = v.try_extract_tensor::<f16>()?.1;
        fs::write(
            format!("{root}/dml-{name}.f32"),
            data.iter()
                .flat_map(|v| v.to_f32().to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
    }
    Ok(())
}
#[test]
#[ignore = "Vulkan RVM real-video quality and completed-call timing; pinned ncnn SDK required"]
fn ncnn_video_benchmark() -> anyhow::Result<()> {
    let root = CString::new(
        std::env::var("SEGMENTER_NCNN_MODEL").unwrap_or("target/rvm-ncnn-cam".into()),
    )?;
    let adapter = std::env::var("SEGMENTER_NCNN_DEVICE")
        .unwrap_or("0".into())
        .parse()?;
    let flags = std::env::var("SEGMENTER_NCNN_FLAGS")
        .unwrap_or("0".into())
        .parse()?;
    let repeats: usize = std::env::var("SEGMENTER_BENCH_REPEATS")
        .unwrap_or("1".into())
        .parse()?;
    anyhow::ensure!(repeats > 0, "empty repeat count");
    let path = PathBuf::from(std::env::var("SEGMENTER_BENCH_VIDEO")?);
    let mut decoder = open_video_decoder(
        &path,
        DecodeOptions {
            max_dimension: Some(480),
        },
    )?;
    let mut frames = Vec::new();
    while frames.len() < 150 {
        let Some(f) = decoder.read_frame()? else {
            break;
        };
        frames.push(f);
    }
    anyhow::ensure!(frames.len() == 150, "expected 150 frames");
    let w = frames[0].width;
    let h = frames[0].height;
    let n = (w * h) as usize;
    let reference = fs::read(std::env::var("SEGMENTER_BENCH_REFERENCE")?)?;
    anyhow::ensure!(
        reference.len() == n * frames.len(),
        "reference size mismatch"
    );
    let raw = unsafe { rvm_ncnn_create(root.as_ptr(), w as i32, h as i32, adapter, flags) };
    if raw.is_null() {
        checked(-1)?;
    }
    let probe = Probe(raw);
    let mut times = Vec::new();
    let mut masks = Vec::with_capacity(reference.len());
    let mut half = vec![f16::ZERO; 3 * n];
    let mut rgb = vec![0f32; 3 * n];
    let mut alpha = vec![0f32; n];
    for (i, frame) in frames
        .iter()
        .cycle()
        .take(frames.len() * repeats)
        .enumerate()
    {
        if i % frames.len() == 0 {
            checked(unsafe { rvm_ncnn_reset(probe.0) })?;
            masks.clear();
        }
        let start = Instant::now();
        if flags & 1 != 0 {
            super::tensor::preprocess_f16(frame, &mut half)?;
            for (a, b) in rgb.iter_mut().zip(&half) {
                *a = b.to_f32();
            }
        } else {
            // Vulkan upload casts to FP16, matching the original source tensor.
            super::tensor::preprocess_f32(frame, &mut rgb)?;
        }
        checked(unsafe { rvm_ncnn_run(probe.0, rgb.as_ptr(), alpha.as_mut_ptr()) })?;
        anyhow::ensure!(
            alpha.iter().all(|v| v.is_finite()),
            "nonfinite Vulkan alpha"
        );
        masks.extend(alpha.iter().map(|a| (a.clamp(0., 1.) * 255.).round() as u8));
        if i >= 30 {
            times.push(start.elapsed().as_secs_f64() * 1000.);
        }
    }
    let mut sum = 0u64;
    let mut max = 0u8;
    let mut over4 = 0usize;
    for (a, b) in masks.iter().zip(&reference) {
        let d = a.abs_diff(*b);
        sum += u64::from(d);
        max = max.max(d);
        over4 += usize::from(d > 4);
    }
    let mae = sum as f64 / masks.len() as f64;
    times.sort_by(f64::total_cmp);
    eprintln!("NCNN_VIDEO flags={flags} device={adapter} measured={} mean_ms={:.4} median_ms={:.4} p95_ms={:.4} mae_u8={mae:.6} max_u8={max} fraction_over4={:.8}",times.len(),times.iter().sum::<f64>()/times.len() as f64,times[times.len()/2],times[times.len()*95/100],over4 as f64/masks.len() as f64);
    if let Ok(path) = std::env::var("SEGMENTER_BENCH_SAVE") {
        fs::write(path, &masks)?;
    }
    if let Ok(path) = std::env::var("SEGMENTER_BENCH_PREVIEWS") {
        for i in [0, 30, 75, 149] {
            super::benchmark::preview(
                std::path::Path::new(&path),
                i,
                &frames[i],
                &reference[i * n..(i + 1) * n],
                &masks[i * n..(i + 1) * n],
            )?;
        }
    }
    anyhow::ensure!(
        mae <= 0.05 && max <= 8 && over4 as f64 / masks.len() as f64 <= 0.0001,
        "Vulkan quality threshold exceeded"
    );
    Ok(())
}
