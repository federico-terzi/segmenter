//! GPU preprocessing, fixed recurrent buffers, packed mask and optional DML capture.
//! Kept separate from the application; the actual sequence quality gate is mandatory.
use super::*;

#[test]
#[ignore = "GPU IEEE binary16 conversion audit; compile fp16_round_probe.hlsl first"]
fn dx12_half_rounding() -> anyhow::Result<()> {
    let mut values: Vec<f32> = (0..=u16::MAX).map(|v| f16::from_bits(v).to_f32()).collect();
    for bits in 0..0x7bffu16 {
        let a = f16::from_bits(bits).to_f32();
        let b = f16::from_bits(bits + 1).to_f32();
        let middle = ((a + b) * 0.5).to_bits();
        for offset in [-1i64, 0, 1] {
            let value = f32::from_bits((i64::from(middle) + offset) as u32);
            values.push(value);
            values.push(-value);
        }
    }
    for value in [65519.996f32, 65520., 65520.004, f32::MAX, f32::INFINITY] {
        values.push(value);
        values.push(-value);
    }
    let tested = values.len();
    let w = 1024;
    let h = tested.div_ceil(w);
    values.resize(w * h, 0.);
    let raw: Vec<f16> = values
        .iter()
        .flat_map(|v| {
            [
                f16::from_bits(v.to_bits() as u16),
                f16::from_bits((v.to_bits() >> 16) as u16),
            ]
        })
        .collect();
    let probe = Probe::new()?;
    probe.setup(
        &std::env::var("SEGMENTER_HALF_SHADER")
            .unwrap_or("target/rvm-dx12-quarter/round-rne.cso".into()),
        &raw,
        &[f16::ZERO; 2],
        &[f16::ZERO; 2],
        &[0u8; 4],
        w as u32,
        h as u32,
    )?;
    probe.shader(1)?;
    let actual = probe.read(w * h)?;
    for (i, (bits, input)) in actual.iter().zip(&values).enumerate() {
        let expected = f16::from_f32(*input);
        for output in [
            f16::from_bits(bits.to_bits() as u16),
            f16::from_bits((bits.to_bits() >> 16) as u16),
        ] {
            anyhow::ensure!(
                if expected.is_nan() {
                    output.is_nan()
                } else {
                    output.to_bits() == expected.to_bits()
                },
                "FP16 mismatch at {i}: input {input:?}, expected {expected:?}, actual {output:?}"
            );
        }
    }
    eprintln!("GPU_HALF_RNE cases={tested} both_conversion_paths_match=true");
    Ok(())
}

fn valid_packed_mask(bytes: &[u8]) -> bool {
    if bytes.len() % 4 != 0 {
        return false;
    }
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        return unsafe { valid_packed_mask_avx2(bytes) };
    }
    bytes.chunks_exact(4).all(|p| p[3] == 255)
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn valid_packed_mask_avx2(bytes: &[u8]) -> bool {
    use std::arch::x86_64::*;
    let ff = _mm256_set1_epi8(-1);
    let mut chunks = bytes.chunks_exact(32);
    for chunk in &mut chunks {
        let data = _mm256_loadu_si256(chunk.as_ptr().cast());
        if (_mm256_movemask_epi8(_mm256_cmpeq_epi8(data, ff)) as u32) & 0x88888888 != 0x88888888 {
            return false;
        }
    }
    chunks.remainder().chunks_exact(4).all(|p| p[3] == 255)
}
#[test]
fn packed_mask_guard_checks_every_alpha_byte_and_tail() {
    for length in [4usize, 12, 28, 32, 36, 60, 64, 68, 508] {
        let mut bytes: Vec<u8> = (0..length).map(|i| (i * 73) as u8).collect();
        for p in bytes.chunks_exact_mut(4) {
            p[3] = 255;
        }
        assert!(valid_packed_mask(&bytes));
        for i in (3..length).step_by(4) {
            for value in 0..255u8 {
                bytes[i] = value;
                assert!(!valid_packed_mask(&bytes));
            }
            bytes[i] = 255;
        }
    }
    assert!(!valid_packed_mask(&[255; 3]));
}

struct FixedBinding {
    binding: ort::io_binding::IoBinding,
    _inputs: Vec<DynTensor>,
    resources: std::collections::HashMap<String, Resource>,
    values: std::collections::HashMap<String, DynTensor>,
    bytes: std::collections::HashMap<String, usize>,
    _allocator: ort::memory::Allocator,
}
fn fixed_binding(session: &Session) -> anyhow::Result<FixedBinding> {
    let allocator = ort::memory::Allocator::new(session, gpu_memory()?)?;
    let mut binding = session.create_binding()?;
    let mut inputs = Vec::new();
    let mut resources = std::collections::HashMap::new();
    let mut bytes = std::collections::HashMap::new();
    let mut values = std::collections::HashMap::new();
    for (input, name, value_type) in session
        .inputs
        .iter()
        .map(|v| (true, &v.name, &v.input_type))
        .chain(
            session
                .outputs
                .iter()
                .map(|v| (false, &v.name, &v.output_type)),
        )
    {
        let ValueType::Tensor { ty, shape, .. } = value_type else {
            anyhow::bail!("not tensor")
        };
        anyhow::ensure!(
            *ty == ort::tensor::TensorElementType::Float16 && shape.iter().all(|v| *v > 0),
            "expected static FP16 tensor"
        );
        let tensor = DynTensor::new(&allocator, *ty, shape.to_vec())?;
        bytes.insert(name.clone(), shape.iter().product::<i64>() as usize * 2);
        resources.insert(name.clone(), resource(&allocator, &tensor)?);
        values.insert(
            name.clone(),
            tensor
                .view()
                .try_upgrade()
                .map_err(|_| anyhow::anyhow!("could not retain owned tensor"))?,
        );
        if input {
            binding.bind_input(name, &tensor)?;
            inputs.push(tensor);
        } else {
            binding.bind_output(name, tensor)?;
        }
    }
    Ok(FixedBinding {
        binding,
        _inputs: inputs,
        resources,
        values,
        bytes,
        _allocator: allocator,
    })
}

#[test]
#[ignore = "completed temporal batching: GPU encoder batch, sequential decoder, packed masks"]
fn dx12_temporal_video() -> anyhow::Result<()> {
    use crate::decoder::{open_video_decoder, DecodeOptions};
    let root = std::env::var("SEGMENTER_SHADER_DIR").unwrap_or("target/rvm-dx12-quarter".into());
    let graphs =
        std::env::var("SEGMENTER_TEMPORAL_DIR").unwrap_or("target/rvm-quarter-temporal".into());
    let pack = std::env::var_os("SEGMENTER_TEMPORAL_PACK").is_some();
    let fused = std::env::var_os("SEGMENTER_TEMPORAL_FUSED").is_some();
    let ping_pong = std::env::var_os("SEGMENTER_TEMPORAL_PINGPONG").is_some();
    let batch: usize = std::env::var("SEGMENTER_DX12_BATCH")
        .unwrap_or("4".into())
        .parse()?;
    anyhow::ensure!(
        [1, 2, 4, 8, 16].contains(&batch),
        "unsupported encoder batch"
    );
    anyhow::ensure!(
        !fused || batch == 1,
        "fused recurrent graph requires batch one"
    );
    let repeats: usize = std::env::var("SEGMENTER_HYBRID_REPEATS")
        .unwrap_or("1".into())
        .parse()?;
    anyhow::ensure!(repeats > 0, "invalid repeats");
    let warmup: usize = std::env::var("SEGMENTER_BENCH_WARMUP")
        .unwrap_or("30".into())
        .parse()?;
    let frame_limit: usize = std::env::var("SEGMENTER_BENCH_FRAMES")
        .unwrap_or("150".into())
        .parse()?;
    anyhow::ensure!(frame_limit > 10, "need more than 10 frames");
    let mut decoder = open_video_decoder(
        &PathBuf::from(std::env::var("SEGMENTER_BENCH_VIDEO")?),
        DecodeOptions {
            max_dimension: Some(480),
        },
    )?;
    let mut frames = Vec::new();
    while frames.len() < frame_limit {
        let Some(f) = decoder.read_frame()? else {
            break;
        };
        frames.push(f);
    }
    anyhow::ensure!(
        frames.len() > 10 && warmup < frames.len() * repeats,
        "too few frames or excessive warmup"
    );
    let w = frames[0].width;
    let h = frames[0].height;
    let n = (w * h) as usize;
    let reference = fs::read(std::env::var("SEGMENTER_BENCH_REFERENCE")?)?;
    anyhow::ensure!(
        reference.len() == frames.len() * n,
        "reference size mismatch"
    );
    let probe = Probe::new()?;
    let suffix = if pack { "-packed" } else { "" };
    let mut encoder = if fused {
        None
    } else {
        Some(probe.session(&PathBuf::from(format!(
            "{graphs}/encoder{batch:02}{suffix}.onnx"
        )))?)
    };
    let network_path = if fused {
        format!("{root}/resident-source.onnx")
    } else {
        format!("{graphs}/decoder{suffix}.onnx")
    };
    let mut network = probe.session(&PathBuf::from(network_path))?;
    let source = encoder
        .as_ref()
        .unwrap_or(&network)
        .inputs
        .iter()
        .find(|i| i.name == "src")
        .context("encoder is missing src")?;
    let ValueType::Tensor { shape, .. } = &source.input_type else {
        anyhow::bail!("encoder src is not a tensor")
    };
    anyhow::ensure!(
        shape.as_ref() == [batch as i64, 3, h as i64, w as i64],
        "encoder shape does not match video and batch"
    );
    let enc = encoder.as_ref().map(fixed_binding).transpose()?;
    let dec = fixed_binding(&network)?;
    let mut alternate = network.create_binding()?;
    if ping_pong {
        for input in &network.inputs {
            let source = match input.name.as_str() {
                "r1i" => "r1o",
                "r2i" => "r2o",
                "r3i" => "r3o",
                "r4i" => "r4o",
                name => name,
            };
            alternate.bind_input(&input.name, &dec.values[source])?;
        }
        for output in &network.outputs {
            let destination = match output.name.as_str() {
                "r1o" => "r1i",
                "r2o" => "r2i",
                "r3o" => "r3i",
                "r4o" => "r4i",
                name => name,
            };
            let value = dec.values[destination]
                .view()
                .try_upgrade()
                .map_err(|_| anyhow::anyhow!("could not retain tensor"))?;
            alternate.bind_output(&output.name, value)?;
        }
    }
    let weights = fs::read(format!("{root}/refiner-weights.f32"))?;
    probe.setup(
        &format!("{root}/refiner-shared.cso"),
        &[f16::ZERO; 2],
        &[f16::ZERO; 2],
        &[f16::ZERO; 2],
        &weights,
        w,
        h,
    )?;
    checked(unsafe {
        rvm_probe_attach(
            probe.0,
            dec.resources[if pack && !fused { "packed" } else { "389" }].0,
            dec.resources["751"].0,
            dec.resources["752"].0,
        )
    })?;
    let prepare = CString::new(format!("{root}/preprocess-source.cso"))?;
    checked(unsafe {
        rvm_probe_prepare_setup(
            probe.0,
            prepare.as_ptr(),
            enc.as_ref().unwrap_or(&dec).resources["src"].0,
            1,
            batch as u32,
            1,
        )
    })?;
    let post = CString::new(format!("{root}/refiner-bgra.cso"))?;
    checked(unsafe { rvm_probe_post(probe.0, post.as_ptr(), rvm_probe_bgra(probe.0)) })?;
    let state_in = [R1_INPUT, R2_INPUT, R3_INPUT, R4_INPUT].map(|s| dec.resources[s].0);
    let state_out = [R1_OUTPUT, R2_OUTPUT, R3_OUTPUT, R4_OUTPUT].map(|s| dec.resources[s].0);
    let state_bytes = [R1_INPUT, R2_INPUT, R3_INPUT, R4_INPUT].map(|s| dec.bytes[s]);
    checked(unsafe {
        rvm_probe_recurrence(
            probe.0,
            state_in.as_ptr(),
            state_out.as_ptr(),
            state_bytes.as_ptr(),
            4,
        )
    })?;
    if ping_pong {
        checked(unsafe { rvm_probe_ping_pong(probe.0) })?;
    }
    if let Some(enc) = &enc {
        let names = if pack {
            vec!["packed"]
        } else {
            vec!["389", "403", "420", "464", "601", "602", "603", "604"]
        };
        let src: Vec<_> = names.iter().map(|s| enc.resources[*s].0).collect();
        let dst: Vec<_> = names.iter().map(|s| dec.resources[*s].0).collect();
        let bytes: Vec<_> = names.iter().map(|s| dec.bytes[*s]).collect();
        checked(unsafe {
            rvm_probe_feature_setup(
                probe.0,
                src.as_ptr(),
                dst.as_ptr(),
                bytes.as_ptr(),
                names.len() as u32,
                batch as u32,
            )
        })?;
    }
    let mut options = ort::session::RunOptions::new()?;
    options.disable_device_sync()?;
    let mut alternate_options = ort::session::RunOptions::new()?;
    alternate_options.disable_device_sync()?;
    alternate_options.add_config_entry("gpu_graph_id", "1")?;
    let mut masks = Vec::new();
    let mut times = Vec::new();
    let mut batches = Vec::new();
    let mut index = 0;
    for _ in 0..repeats {
        checked(unsafe { rvm_probe_reset_recurrence(probe.0) })?;
        masks.clear();
        for chunk in frames.chunks(batch) {
            let start = Instant::now();
            // Pad the final encoder batch only. Padded frames never update a GRU
            // or produce an output mask, and their compute is included in timing.
            for slot in 0..batch {
                let frame = &chunk[slot.min(chunk.len() - 1)];
                anyhow::ensure!(frame.width == w && frame.height == h, "changed dimensions");
                crate::frame::validate_bgra_buffer(w, h, frame.bytes_per_row, frame.data.len())?;
                checked(unsafe {
                    rvm_probe_prepare_slot(
                        probe.0,
                        frame.data.as_ptr().cast(),
                        frame.bytes_per_row as usize,
                        slot as u32,
                    )
                })?;
            }
            if let (Some(encoder), Some(enc)) = (&mut encoder, &enc) {
                let _encoded = encoder.run_binding_with_options(&enc.binding, &options)?;
            }
            for slot in 0..chunk.len() {
                if !fused {
                    checked(unsafe { rvm_probe_feature_slice(probe.0, slot as u32) })?;
                }
                let phase_one = ping_pong && ((index + slot) % frames.len()) % 2 == 1;
                let _decoded = network.run_binding_with_options(
                    if phase_one { &alternate } else { &dec.binding },
                    if phase_one {
                        &alternate_options
                    } else {
                        &options
                    },
                )?;
                checked(unsafe { rvm_probe_dispatch_slot(probe.0, slot as u32) })?;
            }
            checked(unsafe { rvm_probe_wait(probe.0) })?;
            let mut results = Vec::with_capacity(chunk.len());
            for slot in 0..chunk.len() {
                let mut result = Vec::<u8>::with_capacity(n * 4);
                checked(unsafe {
                    rvm_probe_read_slot(probe.0, slot as u32, result.as_mut_ptr().cast(), n * 4)
                })?;
                // Success guarantees native memcpy initialized all n*4 bytes.
                // Keep length zero on errors; never expose uninitialized bytes.
                unsafe {
                    result.set_len(n * 4);
                }
                anyhow::ensure!(valid_packed_mask(&result), "nonfinite shader output");
                results.push(result);
            }
            let elapsed = start.elapsed().as_secs_f64() * 1000.;
            if index >= warmup {
                times.extend(std::iter::repeat_n(
                    elapsed / chunk.len() as f64,
                    chunk.len(),
                ));
                batches.push(elapsed);
            }
            index += chunk.len();
            for result in results {
                masks.extend(result.chunks_exact(4).map(|p| p[0]));
            }
        }
    }
    anyhow::ensure!(
        !times.is_empty(),
        "warmup leaves no complete measured batches"
    );
    let (mut sum, mut max, mut over4) = (0u64, 0u8, 0usize);
    for (a, b) in masks.iter().zip(&reference) {
        let d = a.abs_diff(*b);
        sum += u64::from(d);
        max = max.max(d);
        over4 += usize::from(d > 4);
    }
    let mae = sum as f64 / masks.len() as f64;
    times.sort_by(f64::total_cmp);
    batches.sort_by(f64::total_cmp);
    eprintln!("DX12_TEMPORAL fused={fused} ping_pong={ping_pong} packed={pack} batch={batch} frames={} measured_frames={} completed_ms_per_frame={:.4} median_amortized_ms={:.4} p95_batch_ms={:.4} mae_u8={mae:.6} max_u8={max} fraction_over4={:.8}",frames.len()*repeats,times.len(),times.iter().sum::<f64>()/times.len()as f64,times[times.len()/2],batches[batches.len()*95/100],over4 as f64/masks.len()as f64);
    if let Ok(save) = std::env::var("SEGMENTER_BENCH_SAVE") {
        fs::write(save, &masks)?;
    }
    if let Ok(path) = std::env::var("SEGMENTER_BENCH_PREVIEWS") {
        let worst = masks
            .chunks_exact(n)
            .zip(reference.chunks_exact(n))
            .enumerate()
            .max_by_key(|(_, (a, b))| {
                a.iter()
                    .zip(*b)
                    .map(|(a, b)| u64::from(a.abs_diff(*b)))
                    .sum::<u64>()
            })
            .map(|(i, _)| i)
            .unwrap();
        for i in [
            0,
            frames.len() / 5,
            frames.len() / 2,
            frames.len() - 1,
            worst,
        ] {
            super::super::benchmark::preview(
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
        "temporal GPU sequence quality failed"
    );
    Ok(())
}

#[test]
#[ignore = "quality-only temporal encoder batching with sequential original decoder/refiner"]
fn dx12_temporal_quality() -> anyhow::Result<()> {
    use crate::decoder::{open_video_decoder, DecodeOptions};
    let root =
        std::env::var("SEGMENTER_TEMPORAL_DIR").unwrap_or("target/rvm-quarter-temporal".into());
    let batch: usize = std::env::var("SEGMENTER_DX12_BATCH")
        .unwrap_or("8".into())
        .parse()?;
    anyhow::ensure!(
        std::env::var_os("SEGMENTER_DX12_CAPTURE").is_none(),
        "quality-only CPU transfer path cannot use capture"
    );
    let mut decoder = open_video_decoder(
        &PathBuf::from(std::env::var("SEGMENTER_BENCH_VIDEO")?),
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
    let w = frames[0].width as usize;
    let h = frames[0].height as usize;
    let n = w * h;
    let reference = fs::read(std::env::var("SEGMENTER_BENCH_REFERENCE")?)?;
    anyhow::ensure!(reference.len() == 150 * n, "reference size mismatch");
    let probe = Probe::new()?;
    let mut encoder = probe.session(&PathBuf::from(format!("{root}/encoder{batch:02}.onnx")))?;
    let mut network = probe.session(&PathBuf::from(format!("{root}/decoder-full.onnx")))?;
    let mut binding = network.create_binding()?;
    let mut recurrent = Vec::new();
    for name in [R1_INPUT, R2_INPUT, R3_INPUT, R4_INPUT] {
        let v = network
            .inputs
            .iter()
            .find(|v| v.name == name)
            .context("missing state")?;
        let ValueType::Tensor { shape, .. } = &v.input_type else {
            anyhow::bail!("not tensor")
        };
        recurrent.push(
            Tensor::from_array((
                shape.to_vec(),
                vec![f16::ZERO; shape.iter().product::<i64>() as usize],
            ))?
            .upcast(),
        );
    }
    binding.bind_output_to_device(
        "pha",
        &MemoryInfo::new(
            AllocationDevice::CPU,
            0,
            AllocatorType::Device,
            MemoryType::Default,
        )?,
    )?;
    let mut masks = Vec::new();
    for chunk in frames.chunks(batch) {
        let mut pixels = vec![f16::ZERO; batch * 3 * n];
        for i in 0..batch {
            super::super::tensor::preprocess_f16(
                &chunk[i.min(chunk.len() - 1)],
                &mut pixels[i * 3 * n..(i + 1) * 3 * n],
            )?;
        }
        let source = Tensor::from_array(([batch, 3, h, w], pixels))?;
        let encoded = encoder.run(ort::inputs!["src"=>&source])?;
        for (i, frame) in chunk.iter().enumerate() {
            binding.clear_inputs();
            let mut keep = Vec::new();
            for input in &network.inputs {
                if [R1_INPUT, R2_INPUT, R3_INPUT, R4_INPUT].contains(&input.name.as_str()) {
                    continue;
                }
                let ValueType::Tensor { shape, .. } = &input.input_type else {
                    anyhow::bail!("not tensor")
                };
                let data = if input.name == "src" {
                    source.try_extract_tensor::<f16>()?.1
                } else {
                    encoded[input.name.as_str()].try_extract_tensor::<f16>()?.1
                };
                let size = shape.iter().product::<i64>() as usize;
                let tensor =
                    Tensor::from_array((shape.to_vec(), data[i * size..(i + 1) * size].to_vec()))?;
                binding.bind_input(&input.name, &tensor)?;
                keep.push(tensor);
            }
            for (name, value) in [R1_INPUT, R2_INPUT, R3_INPUT, R4_INPUT]
                .into_iter()
                .zip(&recurrent)
            {
                binding.bind_input(name, value)?;
            }
            for name in [R1_OUTPUT, R2_OUTPUT, R3_OUTPUT, R4_OUTPUT] {
                binding.bind_output_to_device(name, &gpu_memory()?)?;
            }
            let mut output = network.run_binding(&binding)?;
            binding.synchronize_outputs()?;
            let alpha = output["pha"]
                .try_extract_array::<f16>()?
                .into_dimensionality::<ndarray::Ix4>()?;
            let mask =
                super::super::tensor::alpha_to_mask_f16(alpha, w as u32, h as u32, frame.time)?;
            masks.extend(mask.data.chunks_exact(4).map(|p| p[0]));
            recurrent = take_recurrent_outputs(&mut output)?;
        }
    }
    let (mut sum, mut max, mut over4) = (0u64, 0u8, 0usize);
    for (a, b) in masks.iter().zip(&reference) {
        let d = a.abs_diff(*b);
        sum += u64::from(d);
        max = max.max(d);
        over4 += usize::from(d > 4);
    }
    let mae = sum as f64 / masks.len() as f64;
    eprintln!(
        "TEMPORAL_QUALITY batch={batch} mae_u8={mae:.6} max_u8={max} fraction_over4={:.8}",
        over4 as f64 / masks.len() as f64
    );
    if let Ok(save) = std::env::var("SEGMENTER_BENCH_SAVE") {
        fs::write(save, &masks)?;
    }
    anyhow::ensure!(
        mae <= 0.05 && max <= 8 && over4 as f64 / masks.len() as f64 <= 0.0001,
        "temporal batch quality failed"
    );
    Ok(())
}

#[test]
#[ignore = "ratio-.25 GPU preprocessing/capture experiment; real video and original masks required"]
fn dx12_quarter_resident_video() -> anyhow::Result<()> {
    use crate::decoder::{open_video_decoder, DecodeOptions};
    use std::collections::HashMap;
    let root = std::env::var("SEGMENTER_SHADER_DIR").unwrap_or("target/rvm-dx12-quarter".into());
    let gpu_resize = std::env::var_os("SEGMENTER_DX12_GPU_RESIZE").is_some();
    let slots: usize = std::env::var("SEGMENTER_DX12_BATCH")
        .unwrap_or("1".into())
        .parse()?;
    anyhow::ensure!((1..=16).contains(&slots), "batch must be 1..16");
    let mut decoder = open_video_decoder(
        &PathBuf::from(std::env::var("SEGMENTER_BENCH_VIDEO")?),
        DecodeOptions {
            max_dimension: Some(480),
        },
    )?;
    let mut frames = Vec::new();
    while frames.len() < 150 {
        let Some(frame) = decoder.read_frame()? else {
            break;
        };
        frames.push(frame);
    }
    anyhow::ensure!(frames.len() == 150, "expected 150 frames");
    let w = frames[0].width;
    let h = frames[0].height;
    let n = (w * h) as usize;
    anyhow::ensure!(w % 8 == 0 && h >= 4, "unsupported preprocessing dimensions");
    let reference = fs::read(std::env::var("SEGMENTER_BENCH_REFERENCE")?)?;
    anyhow::ensure!(
        reference.len() == frames.len() * n,
        "reference size mismatch"
    );
    let repeats: usize = std::env::var("SEGMENTER_HYBRID_REPEATS")
        .unwrap_or("1".into())
        .parse()?;
    anyhow::ensure!(repeats > 0, "invalid repeat count");
    let probe = Probe::new()?;
    let graph = if gpu_resize {
        "resident-prefix"
    } else {
        "resident-source"
    };
    let mut session = probe
        .session(&PathBuf::from(format!("{root}/{graph}.onnx")))
        .context("create resident DML session")?;
    let allocator = ort::memory::Allocator::new(&session, gpu_memory()?)?;
    let mut binding = session.create_binding()?;
    let mut resources = HashMap::new();
    let mut sizes = HashMap::new();
    let mut inputs = Vec::new();
    for input in &session.inputs {
        let ValueType::Tensor { ty, shape, .. } = &input.input_type else {
            anyhow::bail!("not tensor")
        };
        anyhow::ensure!(
            *ty == ort::tensor::TensorElementType::Float16 && shape.iter().all(|v| *v > 0),
            "expected static FP16 input"
        );
        if input.name == "src" || input.name == "389" {
            let scale = if gpu_resize { 4 } else { 1 };
            anyhow::ensure!(
                shape.as_ref() == [1, 3, (h / scale) as i64, (w / scale) as i64],
                "resident graph shape does not match video"
            );
        }
        let tensor = DynTensor::new(&allocator, *ty, shape.to_vec())?;
        sizes.insert(
            input.name.clone(),
            shape.iter().product::<i64>() as usize * 2,
        );
        resources.insert(input.name.clone(), resource(&allocator, &tensor)?);
        binding.bind_input(&input.name, &tensor)?;
        inputs.push(tensor);
    }
    for output in &session.outputs {
        let ValueType::Tensor { ty, shape, .. } = &output.output_type else {
            anyhow::bail!("not tensor")
        };
        anyhow::ensure!(
            *ty == ort::tensor::TensorElementType::Float16 && shape.iter().all(|v| *v > 0),
            "expected static FP16 output"
        );
        let tensor = DynTensor::new(&allocator, *ty, shape.to_vec())?;
        resources.insert(output.name.clone(), resource(&allocator, &tensor)?);
        binding.bind_output(&output.name, tensor)?;
    }
    let weights = fs::read(format!("{root}/refiner-weights.f32"))?;
    probe
        .setup(
            &format!("{root}/refiner-shared.cso"),
            &[f16::ZERO; 2],
            &[f16::ZERO; 2],
            &[f16::ZERO; 2],
            &weights,
            w,
            h,
        )
        .context("setup coefficient shader")?;
    checked(unsafe {
        rvm_probe_attach(
            probe.0,
            resources["389"].0,
            resources["751"].0,
            resources["752"].0,
        )
    })
    .context("attach fixed GPU tensors")?;
    let (shader, input, scale) = if gpu_resize {
        ("preprocess", "389", 4)
    } else {
        ("preprocess-source", "src", 1)
    };
    let prepare = CString::new(format!("{root}/{shader}.cso"))?;
    checked(unsafe {
        rvm_probe_prepare_setup(
            probe.0,
            prepare.as_ptr(),
            resources[input].0,
            scale,
            slots as u32,
            0,
        )
    })
    .context("setup GPU preprocessing")?;
    let post = CString::new(format!("{root}/refiner-bgra.cso"))?;
    checked(unsafe { rvm_probe_post(probe.0, post.as_ptr(), rvm_probe_bgra(probe.0)) })
        .context("setup BGRA postprocessing")?;
    let state_in = [R1_INPUT, R2_INPUT, R3_INPUT, R4_INPUT].map(|s| resources[s].0);
    let state_out = [R1_OUTPUT, R2_OUTPUT, R3_OUTPUT, R4_OUTPUT].map(|s| resources[s].0);
    let state_bytes = [R1_INPUT, R2_INPUT, R3_INPUT, R4_INPUT].map(|s| sizes[s]);
    checked(unsafe {
        rvm_probe_recurrence(
            probe.0,
            state_in.as_ptr(),
            state_out.as_ptr(),
            state_bytes.as_ptr(),
            4,
        )
    })
    .context("setup GPU recurrent copies")?;
    if std::env::var_os("SEGMENTER_DX12_CHECK_INPUT").is_some() {
        anyhow::ensure!(
            !gpu_resize,
            "input diagnostic requires original-size GPU normalization"
        );
        let f = &frames[0];
        let mut expected = vec![f16::ZERO; 3 * n];
        let mut actual = expected.clone();
        super::super::tensor::preprocess_f16(f, &mut expected)?;
        checked(unsafe {
            rvm_probe_prepare(probe.0, f.data.as_ptr().cast(), f.bytes_per_row as usize)
        })?;
        checked(unsafe {
            rvm_probe_copy_tensor(
                probe.0,
                resources["src"].0,
                actual.as_mut_ptr().cast(),
                actual.len() * 2,
            )
        })?;
        let different: Vec<_> = actual
            .iter()
            .zip(&expected)
            .enumerate()
            .filter(|(_, (a, b))| a.to_bits() != b.to_bits())
            .collect();
        eprintln!(
            "GPU_INPUT different={} examples={:?}",
            different.len(),
            &different[..different.len().min(12)]
        );
        anyhow::ensure!(
            different.is_empty(),
            "GPU normalized input differs from CPU"
        );
    }
    let mut options = ort::session::RunOptions::new()?;
    options.disable_device_sync()?;
    let mut times = Vec::new();
    let mut stages = [0.; 3];
    let mut masks = Vec::new();
    let mut pending = 0;
    let mut batch_start = Instant::now();
    let mut batch_first = 0;
    let mut batch_stages = [0.; 2];
    for (i, frame) in frames
        .iter()
        .cycle()
        .take(frames.len() * repeats)
        .enumerate()
    {
        anyhow::ensure!(frame.width == w && frame.height == h, "changed dimensions");
        crate::frame::validate_bgra_buffer(w, h, frame.bytes_per_row, frame.data.len())?;
        if i % frames.len() == 0 {
            checked(unsafe { rvm_probe_reset_recurrence(probe.0) })?;
            masks.clear();
        }
        if pending == 0 {
            batch_start = Instant::now();
            batch_first = i;
            batch_stages = [0.; 2];
        }
        let start = Instant::now();
        checked(unsafe {
            rvm_probe_prepare_slot(
                probe.0,
                frame.data.as_ptr().cast(),
                frame.bytes_per_row as usize,
                pending as u32,
            )
        })?;
        let uploaded = Instant::now();
        let _outputs = session.run_binding_with_options(&binding, &options)?;
        let submitted = Instant::now();
        checked(unsafe { rvm_probe_dispatch(probe.0) })?;
        checked(unsafe { rvm_probe_queue_read(probe.0, pending as u32) })?;
        batch_stages[0] += uploaded.duration_since(start).as_secs_f64() * 1000.;
        batch_stages[1] += submitted.duration_since(uploaded).as_secs_f64() * 1000.;
        pending += 1;
        if pending == slots || i % frames.len() == frames.len() - 1 {
            checked(unsafe { rvm_probe_wait(probe.0) })?;
            let mut results = Vec::with_capacity(pending);
            for slot in 0..pending {
                let mut mask = vec![0u8; n * 4];
                checked(unsafe {
                    rvm_probe_read_slot(probe.0, slot as u32, mask.as_mut_ptr().cast(), mask.len())
                })?;
                anyhow::ensure!(
                    mask.chunks_exact(4).all(|p| p[3] == 255),
                    "nonfinite packed shader output"
                );
                results.push(mask);
            }
            let elapsed = batch_start.elapsed().as_secs_f64() * 1000.;
            if batch_first >= 30 {
                times.extend(std::iter::repeat_n(elapsed / pending as f64, pending));
                stages[0] += batch_stages[0];
                stages[1] += batch_stages[1];
                stages[2] += elapsed - batch_stages[0] - batch_stages[1];
            }
            for mask in results {
                masks.extend(mask.chunks_exact(4).map(|p| p[0]));
            }
            pending = 0;
        }
    }
    let (mut sum, mut max, mut over4) = (0u64, 0u8, 0usize);
    for (a, b) in masks.iter().zip(&reference) {
        let d = a.abs_diff(*b);
        sum += u64::from(d);
        max = max.max(d);
        over4 += usize::from(d > 4);
    }
    let mae = sum as f64 / masks.len() as f64;
    times.sort_by(f64::total_cmp);
    eprintln!("DX12_RESIDENT batch={slots} gpu_resize={gpu_resize} capture={} calls={} completed_ms_per_frame={:.4} median_amortized_ms={:.4} p95_amortized_ms={:.4} upload_cpu_ms={:.4} submit_cpu_ms={:.4} finish_ms={:.4} mae_u8={mae:.6} max_u8={max} fraction_over4={:.8}",std::env::var_os("SEGMENTER_DX12_CAPTURE").is_some(),times.len(),times.iter().sum::<f64>()/times.len()as f64,times[times.len()/2],times[times.len()*95/100],stages[0]/times.len()as f64,stages[1]/times.len()as f64,stages[2]/times.len()as f64,over4 as f64/masks.len()as f64);
    if let Ok(save) = std::env::var("SEGMENTER_BENCH_SAVE") {
        fs::write(save, &masks)?;
    }
    anyhow::ensure!(
        mae <= 0.05 && max <= 8 && over4 as f64 / masks.len() as f64 <= 0.0001,
        "resident sequence quality failed"
    );
    Ok(())
}
