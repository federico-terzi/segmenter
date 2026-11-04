//! Opt-in native DirectX probes. Not part of the deployed inference path.
#[path = "dx12_resident_probe.rs"]
mod resident;
use super::*;
use ort::{
    memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType},
    session::{builder::GraphOptimizationLevel, Session},
    value::{DynTensor, ValueType},
    AsPointer,
};
use std::{
    ffi::{c_char, c_void, CStr, CString},
    time::Instant,
};

extern "C" {
    fn rvm_probe_error() -> *const c_char;
    fn rvm_probe_create(adapter: i32) -> *mut c_void;
    fn rvm_probe_destroy(p: *mut c_void);
    fn rvm_probe_device(p: *mut c_void) -> *mut c_void;
    fn rvm_probe_queue(p: *mut c_void) -> *mut c_void;
    fn rvm_probe_begin(p: *mut c_void) -> i32;
    fn rvm_probe_end(p: *mut c_void, ms: *mut f64) -> i32;
    fn rvm_probe_setup(
        p: *mut c_void,
        code: *const c_char,
        src: *const c_void,
        src_size: usize,
        hid: *const c_void,
        hid_size: usize,
        proj: *const c_void,
        proj_size: usize,
        weights: *const c_void,
        weights_size: usize,
        w: u32,
        h: u32,
        channels: u32,
    ) -> i32;
    fn rvm_probe_shader(p: *mut c_void, repeats: u32, ms: *mut f64) -> i32;
    fn rvm_probe_read(p: *mut c_void, out: *mut f32, count: usize) -> i32;
    fn rvm_probe_attach(
        p: *mut c_void,
        src: *mut c_void,
        hid: *mut c_void,
        proj: *mut c_void,
    ) -> i32;
    fn rvm_probe_dispatch(p: *mut c_void) -> i32;
    fn rvm_probe_release(p: *mut c_void);
    fn rvm_probe_dispatch_depth(p: *mut c_void, depth: u32) -> i32;
    fn rvm_probe_group_coverage(p: *mut c_void, width: u32, height: u32) -> i32;
    fn rvm_probe_post(p: *mut c_void, path: *const c_char, fine: *mut c_void) -> i32;
    fn rvm_probe_prepare_setup(
        p: *mut c_void,
        path: *const c_char,
        low: *mut c_void,
        scale: u32,
        slots: u32,
        temporal: u32,
    ) -> i32;
    fn rvm_probe_prepare(p: *mut c_void, data: *const c_void, stride: usize) -> i32;
    fn rvm_probe_bgra(p: *mut c_void) -> *mut c_void;
    fn rvm_probe_recurrence(
        p: *mut c_void,
        ins: *const *mut c_void,
        outs: *const *mut c_void,
        bytes: *const usize,
        count: u32,
    ) -> i32;
    fn rvm_probe_reset_recurrence(p: *mut c_void) -> i32;
    fn rvm_probe_ping_pong(p: *mut c_void) -> i32;
    fn rvm_probe_copy_tensor(
        p: *mut c_void,
        tensor: *mut c_void,
        data: *mut c_void,
        bytes: usize,
    ) -> i32;
    fn rvm_probe_prepare_slot(p: *mut c_void, data: *const c_void, stride: usize, slot: u32)
        -> i32;
    fn rvm_probe_queue_read(p: *mut c_void, slot: u32) -> i32;
    fn rvm_probe_wait(p: *mut c_void) -> i32;
    fn rvm_probe_read_slot(p: *mut c_void, slot: u32, data: *mut c_void, bytes: usize) -> i32;
    fn rvm_probe_feature_setup(
        p: *mut c_void,
        src: *const *mut c_void,
        dst: *const *mut c_void,
        bytes: *const usize,
        count: u32,
        slots: u32,
    ) -> i32;
    fn rvm_probe_feature_slice(p: *mut c_void, slot: u32) -> i32;
    fn rvm_probe_dispatch_slot(p: *mut c_void, slot: u32) -> i32;
}
#[repr(C)]
struct DmlApiPrefix {
    append_default: *const c_void,
    append_shared: unsafe extern "system" fn(
        *mut ort::sys::OrtSessionOptions,
        *mut c_void,
        *mut c_void,
    ) -> ort::sys::OrtStatusPtr,
    create_allocation: *const c_void,
    free_allocation: *const c_void,
    get_resource: unsafe extern "system" fn(
        *mut ort::sys::OrtAllocator,
        *mut c_void,
        *mut *mut c_void,
    ) -> ort::sys::OrtStatusPtr,
}
struct Resource(*mut c_void);
impl Drop for Resource {
    fn drop(&mut self) {
        unsafe { rvm_probe_release(self.0) }
    }
}
fn resource(allocator: &ort::memory::Allocator, tensor: &DynTensor) -> anyhow::Result<Resource> {
    anyhow::ensure!(
        tensor.memory_info().allocation_device() == AllocationDevice::DIRECTML,
        "expected a DML allocation"
    );
    let mut api = std::ptr::null();
    let mut result = std::ptr::null_mut();
    unsafe {
        ort::error::status_to_result((ort::api().GetExecutionProviderApi)(
            c"DML".as_ptr(),
            ort::MINOR_VERSION,
            &mut api,
        ))?;
        let api = &*(api as *const DmlApiPrefix);
        ort::error::status_to_result((api.get_resource)(
            allocator.ptr().cast_mut(),
            tensor.data_ptr().cast_mut(),
            &mut result,
        ))?;
    }
    anyhow::ensure!(!result.is_null(), "DML returned no resource");
    Ok(Resource(result))
}
struct Probe(*mut c_void);
impl Drop for Probe {
    fn drop(&mut self) {
        unsafe {
            rvm_probe_destroy(self.0);
        }
    }
}
fn checked(status: i32) -> anyhow::Result<()> {
    anyhow::ensure!(
        status == 0,
        "{}",
        unsafe { CStr::from_ptr(rvm_probe_error()) }.to_string_lossy()
    );
    Ok(())
}
fn gpu_memory() -> ort::Result<MemoryInfo> {
    MemoryInfo::new(
        AllocationDevice::DIRECTML,
        0,
        AllocatorType::Device,
        MemoryType::Default,
    )
}
impl Probe {
    fn new() -> anyhow::Result<Self> {
        let adapter = std::env::var("SEGMENTER_BENCH_DEVICE")
            .unwrap_or("0".into())
            .parse()?;
        let p = unsafe { rvm_probe_create(adapter) };
        if p.is_null() {
            checked(-1)?;
        }
        Ok(Self(p))
    }
    fn session(&self, path: &std::path::Path) -> anyhow::Result<Session> {
        let mut builder = Session::builder()?
            .with_memory_pattern(false)?
            .with_parallel_execution(false)?
            .with_intra_threads(1)?
            .with_intra_op_spinning(false)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?;
        if std::env::var_os("SEGMENTER_DX12_CAPTURE").is_some() {
            builder = builder.with_config_entry("ep.dml.enable_graph_capture", "true")?;
        }
        let mut api = std::ptr::null();
        // ORT's stable DML API prefix contains these two entry points. Both COM
        // objects share the same D3D12 device and live through session creation.
        unsafe {
            ort::error::status_to_result((ort::api().GetExecutionProviderApi)(
                c"DML".as_ptr(),
                ort::MINOR_VERSION,
                &mut api,
            ))?;
            anyhow::ensure!(!api.is_null(), "DML API unavailable");
            let api = &*(api as *const DmlApiPrefix);
            ort::error::status_to_result((api.append_shared)(
                builder.ptr_mut(),
                rvm_probe_device(self.0),
                rvm_probe_queue(self.0),
            ))?;
        }
        Ok(builder.commit_from_file(path)?)
    }
    fn begin(&self) -> anyhow::Result<()> {
        checked(unsafe { rvm_probe_begin(self.0) })
    }
    fn end(&self) -> anyhow::Result<f64> {
        let mut ms = 0.;
        checked(unsafe { rvm_probe_end(self.0, &mut ms) })?;
        Ok(ms)
    }
    fn setup(
        &self,
        shader: &str,
        src: &[f16],
        hid: &[f16],
        proj: &[f16],
        weights: &[u8],
        w: u32,
        h: u32,
    ) -> anyhow::Result<()> {
        self.setup_channels(shader, src, hid, proj, weights, w, h, 1)
    }
    fn setup_channels(
        &self,
        shader: &str,
        src: &[f16],
        hid: &[f16],
        proj: &[f16],
        weights: &[u8],
        w: u32,
        h: u32,
        channels: u32,
    ) -> anyhow::Result<()> {
        let shader = CString::new(shader)?;
        checked(unsafe {
            rvm_probe_setup(
                self.0,
                shader.as_ptr(),
                src.as_ptr().cast(),
                std::mem::size_of_val(src),
                hid.as_ptr().cast(),
                std::mem::size_of_val(hid),
                proj.as_ptr().cast(),
                std::mem::size_of_val(proj),
                weights.as_ptr().cast(),
                weights.len(),
                w,
                h,
                channels,
            )
        })
    }
    fn shader(&self, count: u32) -> anyhow::Result<f64> {
        anyhow::ensure!(count > 0, "empty shader batch");
        let mut ms = 0.;
        checked(unsafe { rvm_probe_shader(self.0, count, &mut ms) })?;
        Ok(ms)
    }
    fn read(&self, n: usize) -> anyhow::Result<Vec<f32>> {
        let mut data = vec![0.; n];
        checked(unsafe { rvm_probe_read(self.0, data.as_mut_ptr(), n) })?;
        Ok(data)
    }
    fn read_packed(&self, n: usize) -> anyhow::Result<Vec<u8>> {
        let mut data = vec![0u8; n * 4];
        // Native read only memcpy's bytes and never dereferences the float pointer.
        checked(unsafe { rvm_probe_read(self.0, data.as_mut_ptr().cast(), n) })?;
        anyhow::ensure!(
            data.chunks_exact(4).all(|p| p[3] == 255),
            "nonfinite packed shader output"
        );
        Ok(data)
    }
}

#[test]
#[ignore = "native DirectX queue-span timing; requires research-dx12 and fixed model"]
fn dx12_network_timing() -> anyhow::Result<()> {
    let probe = Probe::new()?;
    let path = PathBuf::from(
        std::env::var("SEGMENTER_BENCH_GRAPH").unwrap_or("target/rvm-blocks/00_full.onnx".into()),
    );
    let mut session = probe.session(&path)?;
    let mut binding = session.create_binding()?;
    let mut inputs = Vec::new();
    for input in &session.inputs {
        let ValueType::Tensor { shape, .. } = &input.input_type else {
            anyhow::bail!("not tensor");
        };
        anyhow::ensure!(shape.iter().all(|d| *d > 0), "dynamic input");
        let tensor = Tensor::from_array((
            shape.to_vec(),
            vec![f16::from_f32(0.1); shape.iter().product::<i64>() as usize],
        ))?;
        binding.bind_input(&input.name, &tensor)?;
        inputs.push(tensor);
    }
    for output in &session.outputs {
        binding.bind_output_to_device(&output.name, &gpu_memory()?)?;
    }
    let mut options = ort::session::RunOptions::new()?;
    options.disable_device_sync()?;
    for _ in 0..100 {
        session.run_binding_with_options(&binding, &options)?;
        binding.synchronize_outputs()?;
    }
    for batch in [1, 16, 64] {
        let mut gpu = 0.;
        let mut wall = 0.;
        for _ in 0..20 {
            let start = Instant::now();
            probe.begin()?;
            for _ in 0..batch {
                session.run_binding_with_options(&binding, &options)?;
            }
            gpu += probe.end()? / f64::from(batch);
            wall += start.elapsed().as_secs_f64() * 1000. / f64::from(batch);
        }
        eprintln!(
            "DX12_NETWORK {} batch={batch} queue_span_ms={:.4} wall_ms={:.4}",
            path.display(),
            gpu / 20.,
            wall / 20.
        );
    }
    Ok(())
}

#[test]
#[ignore = "real-video custom refiner comparison; requires exported probe artifacts"]
fn dx12_refiner_video() -> anyhow::Result<()> {
    use crate::decoder::{open_video_decoder, DecodeOptions};
    let probe = Probe::new()?;
    let root = std::env::var("SEGMENTER_SHADER_DIR").unwrap_or("target/rvm-dx12".into());
    let model =
        std::env::var("SEGMENTER_SHADER_MODEL").unwrap_or(format!("{root}/intermediates.onnx"));
    let mut session = probe.session(&PathBuf::from(model))?;
    let shader = std::env::var("SEGMENTER_SHADER").unwrap_or(format!("{root}/refiner-shared.cso"));
    let weights = fs::read(format!("{root}/refiner-weights.f32"))?;
    anyhow::ensure!(weights.len() == 725 * 4, "unexpected weights");
    let path = PathBuf::from(std::env::var("SEGMENTER_BENCH_VIDEO")?);
    let mut decoder = open_video_decoder(
        &path,
        DecodeOptions {
            max_dimension: Some(480),
        },
    )?;
    let mut binding = session.create_binding()?;
    let cpu = MemoryInfo::new(
        AllocationDevice::CPU,
        0,
        AllocatorType::Device,
        MemoryType::Default,
    )?;
    for name in ["pha", "389", "751", "752"] {
        binding.bind_output_to_device(name, &cpu)?;
    }
    for output in &session.outputs {
        if ["745", "748"].contains(&output.name.as_str()) {
            binding.bind_output_to_device(&output.name, &cpu)?;
        }
    }
    let mut recurrent: Vec<DynTensor> = Vec::new();
    for name in [R1_INPUT, R2_INPUT, R3_INPUT, R4_INPUT] {
        let input = session.inputs.iter().find(|i| i.name == name).unwrap();
        let ValueType::Tensor { shape, .. } = &input.input_type else {
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
    let (mut frames, mut total, mut err, mut max, mut over4) = (0, 0u64, 0u64, 0u8, 0u64);
    let mut perf = Vec::new();
    let reference = std::env::var("SEGMENTER_BENCH_REFERENCE")
        .ok()
        .map(fs::read)
        .transpose()?;
    let save = std::env::var("SEGMENTER_BENCH_SAVE").ok();
    let mut masks = Vec::new();
    let mut export_max = 0u8;
    while frames < 150 {
        let Some(frame) = decoder.read_frame()? else {
            break;
        };
        let n = (frame.width * frame.height) as usize;
        let mut pixels = vec![f16::ZERO; n * 3];
        super::tensor::preprocess_f16(&frame, &mut pixels)?;
        let src =
            Tensor::from_array(([1, 3, frame.height as usize, frame.width as usize], pixels))?;
        binding.clear_inputs();
        binding.bind_input(SRC_INPUT, &src)?;
        for (name, value) in [R1_INPUT, R2_INPUT, R3_INPUT, R4_INPUT]
            .into_iter()
            .zip(&recurrent)
        {
            binding.bind_input(name, value)?;
        }
        for name in [R1_OUTPUT, R2_OUTPUT, R3_OUTPUT, R4_OUTPUT] {
            binding.bind_output_to_device(name, &gpu_memory()?)?;
        }
        let mut outputs = session.run_binding(&binding)?;
        probe.setup(
            &shader,
            outputs["389"].try_extract_tensor::<f16>()?.1,
            outputs["751"].try_extract_tensor::<f16>()?.1,
            outputs["752"].try_extract_tensor::<f16>()?.1,
            &weights,
            frame.width,
            frame.height,
        )?;
        probe.shader(1)?;
        let actual = probe.read(n)?;
        let expected = outputs["pha"].try_extract_tensor::<f16>()?.1;
        for (pixel, (a, b)) in actual.iter().zip(expected).enumerate() {
            anyhow::ensure!(a.is_finite(), "nonfinite shader output");
            let delta = (a.clamp(0., 1.) * 255.).round() as u8;
            if save.is_some() {
                masks.push(delta);
            }
            let teacher = (b.to_f32().clamp(0., 1.) * 255.).round() as u8;
            let reference = if let Some(reference) = &reference {
                let r = *reference
                    .get(frames * n + pixel)
                    .context("reference too short")?;
                export_max = export_max.max(r.abs_diff(teacher));
                r
            } else {
                teacher
            };
            let delta = delta.abs_diff(reference);
            total += 1;
            err += u64::from(delta);
            max = max.max(delta);
            over4 += u64::from(delta > 4);
        }
        if frames == 75 {
            probe.shader(50)?;
            for _ in 0..20 {
                perf.push(probe.shader(50)?);
            }
            for (name, count) in [("src", 3 * n), ("hidden", 16 * n), ("projection", 4 * n)] {
                let key = match name {
                    "src" => "389",
                    "hidden" => "751",
                    _ => "752",
                };
                let data = outputs[key].try_extract_tensor::<f16>()?.1;
                anyhow::ensure!(data.len() == count, "unexpected intermediate length");
                let bytes: Vec<u8> = data
                    .iter()
                    .flat_map(|v| v.to_bits().to_le_bytes())
                    .collect();
                fs::write(format!("{root}/frame75-{name}.f16"), bytes)?;
            }
            for key in ["745", "748"] {
                if let Some(value) = outputs.get(key) {
                    let data = value.try_extract_tensor::<f16>()?.1;
                    fs::write(
                        format!("{root}/frame75-{key}.f16"),
                        data.iter()
                            .flat_map(|v| v.to_bits().to_le_bytes())
                            .collect::<Vec<_>>(),
                    )?;
                }
            }
        }
        recurrent = take_recurrent_outputs(&mut outputs)?;
        frames += 1;
    }
    anyhow::ensure!(frames > 75, "video too short");
    if let Some(reference) = reference {
        anyhow::ensure!(
            reference.len() == total as usize,
            "reference has extra data"
        );
    }
    if let Some(save) = save {
        fs::write(save, masks)?;
    }
    eprintln!("DX12_SHADER {shader} frames={frames} pixels={total} mae_u8={:.6} max_u8={max} fraction_over4={:.8} gpu_ms={:.4} teacher_export_max_u8={export_max}",err as f64/total as f64,over4 as f64/total as f64,perf.iter().sum::<f64>()/perf.len() as f64);
    anyhow::ensure!(
        err as f64 / total as f64 <= 0.05 && max <= 8 && over4 as f64 / total as f64 <= 0.0001,
        "custom shader quality threshold exceeded"
    );
    Ok(())
}

fn read_half(path: String) -> anyhow::Result<Vec<f16>> {
    let bytes = fs::read(path)?;
    anyhow::ensure!(bytes.len() % 2 == 0, "bad half file");
    Ok(bytes
        .chunks_exact(2)
        .map(|v| f16::from_bits(u16::from_le_bytes([v[0], v[1]])))
        .collect())
}

#[test]
#[ignore = "bounded convolution-v2 correctness and timing; starts at 64x64"]
fn dx12_convolution_v2() -> anyhow::Result<()> {
    let root = "target/rvm-dx12";
    let original = read_half(format!("{root}/frame75-745.f16"))?;
    let winograd = std::env::var_os("SEGMENTER_CONV_WINOGRAD").is_some();
    let (shader_name, weight_name, weight_count) = if winograd {
        ("conv-winograd", "conv290-winograd", 8976)
    } else {
        ("conv-v2", "conv290-weights", 5056)
    };
    let weights = fs::read(format!("{root}/{weight_name}.f32"))?;
    anyhow::ensure!(
        original.len() == 35 * 480 * 360 && weights.len() == weight_count * 4,
        "wrong convolution artifacts"
    );
    let probe = Probe::new()?;
    let sizes = if std::env::var_os("SEGMENTER_CONV_V2_FULL").is_some() {
        vec![(64usize, 64usize), (480, 360)]
    } else {
        vec![(64, 64)]
    };
    for (w, h) in sizes {
        let mut input = Vec::with_capacity(35 * w * h);
        for c in 0..35 {
            for y in 0..h {
                input.extend_from_slice(
                    &original[c * 480 * 360 + y * 480..c * 480 * 360 + y * 480 + w],
                );
            }
        }
        let model = if w == 64 {
            format!("{root}/conv290-small.onnx")
        } else {
            format!("{root}/conv290.onnx")
        };
        let mut session = probe.session(&PathBuf::from(model))?;
        let mut binding = session.create_binding()?;
        let tensor = Tensor::from_array(([1, 35, h, w], input.clone()))?;
        binding.bind_input("745", &tensor)?;
        binding.bind_output_to_device(
            "748",
            &MemoryInfo::new(
                AllocationDevice::CPU,
                0,
                AllocatorType::Device,
                MemoryType::Default,
            )?,
        )?;
        let outputs = session.run_binding(&binding)?;
        let expected = outputs["748"].try_extract_tensor::<f16>()?.1.to_vec();
        drop(outputs);
        binding.bind_output_to_device("748", &gpu_memory()?)?;
        for _ in 0..20 {
            session.run_binding(&binding)?;
            binding.synchronize_outputs()?;
        }
        let mut dml = 0.;
        for _ in 0..50 {
            probe.begin()?;
            session.run_binding(&binding)?;
            dml += probe.end()?;
        }
        probe.setup_channels(
            &format!("{root}/{shader_name}.cso"),
            &input,
            &[f16::ZERO],
            &[f16::ZERO],
            &weights,
            w as u32,
            h as u32,
            16,
        )?;
        checked(unsafe { rvm_probe_dispatch_depth(probe.0, 4) })?;
        if winograd {
            checked(unsafe { rvm_probe_group_coverage(probe.0, 32, 16) })?;
        }
        let first = probe.shader(1)?;
        anyhow::ensure!(
            first < 100.,
            "bounded convolution probe was too slow: {first} ms"
        );
        let actual = probe.read(w * h * 16)?;
        let mut max = 0f32;
        let mut sum = 0.;
        for (a, b) in actual.iter().zip(&expected) {
            anyhow::ensure!(a.is_finite(), "nonfinite convolution");
            let d = (a - b.to_f32()).abs();
            sum += f64::from(d);
            max = max.max(d);
        }
        let mut shader = 0.;
        for _ in 0..20 {
            shader += probe.shader(1)?;
        }
        eprintln!("CONV_V2 variant={shader_name} {w}x{h} DirectML_ms={:.4} HLSL_ms={:.4} mae={:.8} max={max:.8}",dml/50.,shader/20.,sum/actual.len() as f64);
        anyhow::ensure!(max <= 0.05, "convolution-v2 output mismatch");
        if w == 64 {
            anyhow::ensure!(
                shader / 20. < 0.1,
                "small kernel too slow to justify full-frame dispatch"
            );
        }
    }
    Ok(())
}

#[test]
#[ignore = "same-queue refiner timing using saved real frame 75"]
fn dx12_refiner_timing() -> anyhow::Result<()> {
    let probe = Probe::new()?;
    let root = std::env::var("SEGMENTER_SHADER_DIR").unwrap_or("target/rvm-dx12".into());
    let w: usize = std::env::var("SEGMENTER_BENCH_WIDTH")
        .unwrap_or("480".into())
        .parse()?;
    let h: usize = std::env::var("SEGMENTER_BENCH_HEIGHT")
        .unwrap_or("360".into())
        .parse()?;
    let n = w * h;
    let src = read_half(format!("{root}/frame75-src.f16"))?;
    let hid = read_half(format!("{root}/frame75-hidden.f16"))?;
    let proj = read_half(format!("{root}/frame75-projection.f16"))?;
    anyhow::ensure!(
        src.len() == 3 * n && hid.len() == 16 * n && proj.len() == 4 * n,
        "incorrect frame shapes"
    );
    let weights = fs::read(format!("{root}/refiner-weights.f32"))?;
    let model = std::env::var("SEGMENTER_BENCH_GRAPH").unwrap_or(format!("{root}/refiner.onnx"));
    let mut session = probe.session(&PathBuf::from(model))?;
    let mut binding = session.create_binding()?;
    let mut inputs = Vec::new();
    for (name, c, data) in [
        ("389", 3, &src[..]),
        ("src", 3, &src[..]),
        ("751", 16, &hid[..]),
        ("753", 3, &proj[..3 * n]),
        ("754", 1, &proj[3 * n..]),
    ] {
        let value = Tensor::from_array(([1, c, h, w], data.to_vec()))?;
        binding.bind_input(name, &value)?;
        inputs.push(value);
    }
    binding.bind_output_to_device("pha", &gpu_memory()?)?;
    let mut options = ort::session::RunOptions::new()?;
    options.disable_device_sync()?;
    for _ in 0..100 {
        session.run_binding_with_options(&binding, &options)?;
        binding.synchronize_outputs()?;
    }
    for batch in [1, 50] {
        let mut times = Vec::new();
        for _ in 0..20 {
            probe.begin()?;
            for _ in 0..batch {
                session.run_binding_with_options(&binding, &options)?;
            }
            times.push(probe.end()? / f64::from(batch));
        }
        eprintln!(
            "DX12_REFINER DirectML batch={batch} queue_span_ms={:.4}",
            times.iter().sum::<f64>() / times.len() as f64
        );
    }
    for variant in ["direct", "shared"] {
        probe.setup(
            &format!("{root}/refiner-{variant}.cso"),
            &src,
            &hid,
            &proj,
            &weights,
            w as u32,
            h as u32,
        )?;
        probe.shader(100)?;
        for batch in [1, 50] {
            let mut times = Vec::new();
            for _ in 0..20 {
                times.push(probe.shader(batch)?);
            }
            eprintln!(
                "DX12_REFINER HLSL-{variant} batch={batch} queue_span_ms={:.4}",
                times.iter().sum::<f64>() / times.len() as f64
            );
        }
    }
    Ok(())
}

#[test]
#[ignore = "dense convolution shader vs DirectML on saved real activations"]
fn dx12_convolution_timing() -> anyhow::Result<()> {
    anyhow::ensure!(std::env::var_os("SEGMENTER_CONV_GROUP_ONLY").is_some()
        || std::env::var_os("SEGMENTER_RUN_REJECTED_CONV").is_some(),
        "full-size tiled candidate was rejected after device removal; only the single-group probe is normally enabled");
    let probe = Probe::new()?;
    let root = std::env::var("SEGMENTER_SHADER_DIR").unwrap_or("target/rvm-dx12".into());
    let w = 480usize;
    let h = 360usize;
    let n = w * h;
    let src = read_half(format!("{root}/frame75-745.f16"))?;
    anyhow::ensure!(src.len() == 35 * n, "unexpected convolution input");
    let weights = fs::read(format!("{root}/conv290-weights.f32"))?;
    anyhow::ensure!(weights.len() == 5056 * 4, "unexpected weights");
    if std::env::var_os("SEGMENTER_CONV_GROUP_ONLY").is_some() {
        probe.setup_channels(
            &format!("{root}/conv290.cso"),
            &src,
            &[f16::ZERO],
            &[f16::ZERO],
            &weights,
            16,
            8,
            16,
        )?;
        eprintln!("DX12_CONV single_group_ms={:.4}", probe.shader(1)?);
        return Ok(());
    }
    let mut session = probe.session(&PathBuf::from(format!("{root}/conv290.onnx")))?;
    let mut binding = session.create_binding()?;
    let input = Tensor::from_array(([1, 35, h, w], src.clone()))?;
    binding.bind_input("745", &input)?;
    let cpu = MemoryInfo::new(
        AllocationDevice::CPU,
        0,
        AllocatorType::Device,
        MemoryType::Default,
    )?;
    binding.bind_output_to_device("748", &cpu)?;
    let outputs = session.run_binding(&binding)?;
    let expected = outputs["748"].try_extract_tensor::<f16>()?.1.to_vec();
    drop(outputs);
    binding.bind_output_to_device("748", &gpu_memory()?)?;
    let mut options = ort::session::RunOptions::new()?;
    options.disable_device_sync()?;
    for _ in 0..100 {
        session.run_binding_with_options(&binding, &options)?;
        binding.synchronize_outputs()?;
    }
    let mut total = 0.;
    for _ in 0..20 {
        probe.begin()?;
        for _ in 0..50 {
            session.run_binding_with_options(&binding, &options)?;
        }
        total += probe.end()? / 50.;
    }
    eprintln!("DX12_CONV DirectML queue_span_ms={:.4}", total / 20.);
    probe.setup_channels(
        &format!("{root}/conv290.cso"),
        &src,
        &[f16::ZERO],
        &[f16::ZERO],
        &weights,
        w as u32,
        h as u32,
        16,
    )?;
    // Rejected: even one full-size dispatch returned DEVICE_REMOVED on the
    // tested NVIDIA driver. The exact cause has not been established.
    probe.shader(1)?;
    let mut total = 0.;
    for _ in 0..5 {
        total += probe.shader(1)?;
    }
    let actual = probe.read(16 * n)?;
    let mut abs = 0.;
    let mut max = 0f32;
    for (a, b) in actual.iter().zip(&expected) {
        anyhow::ensure!(a.is_finite(), "nonfinite convolution");
        let d = (a - b.to_f32()).abs();
        abs += f64::from(d);
        max = max.max(d);
    }
    eprintln!(
        "DX12_CONV HLSL-tiled queue_span_ms={:.4} feature_mae={:.8} feature_max={max:.8}",
        total / 5.,
        abs / actual.len() as f64
    );
    anyhow::ensure!(max <= 0.05, "convolution mismatch");
    Ok(())
}

#[test]
#[ignore = "GPU buffer handoff with actual recurrent video; exported prefix required"]
fn dx12_hybrid_video() -> anyhow::Result<()> {
    use crate::decoder::{open_video_decoder, DecodeOptions};
    let root = std::env::var("SEGMENTER_SHADER_DIR").unwrap_or("target/rvm-dx12".into());
    let quarter = std::env::var_os("SEGMENTER_HYBRID_QUARTER").is_some();
    let packed = std::env::var_os("SEGMENTER_HYBRID_PACKED").is_some();
    anyhow::ensure!(
        !packed || quarter,
        "packed shader requires quarter-ratio export"
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
    anyhow::ensure!(frames.len() == 150, "expected 150 evaluation frames");
    let w = frames[0].width;
    let h = frames[0].height;
    let n = (w * h) as usize;
    anyhow::ensure!(
        frames.iter().all(|f| f.width == w && f.height == h),
        "frame dimensions changed"
    );
    let mut baseline = Vec::new();
    let repeats: usize = std::env::var("SEGMENTER_HYBRID_REPEATS")
        .unwrap_or("1".into())
        .parse()?;
    anyhow::ensure!(repeats > 0, "empty repeat count");
    let modes = if std::env::var_os("SEGMENTER_HYBRID_REVERSE").is_some() {
        [true, false]
    } else {
        [false, true]
    };
    let weights = fs::read(format!("{root}/refiner-weights.f32"))?;
    let reference = fs::read(std::env::var("SEGMENTER_BENCH_REFERENCE")?)?;
    anyhow::ensure!(
        reference.len() == frames.len() * n,
        "reference length mismatch"
    );
    for hybrid in modes {
        let probe = Probe::new()?;
        let full = if std::env::var_os("SEGMENTER_HYBRID_GATHER").is_some() {
            "full-gather"
        } else {
            "full"
        };
        let name = if hybrid { "prefix" } else { full };
        let mut session = probe.session(&PathBuf::from(format!("{root}/{name}.onnx")))?;
        let mut binding = session.create_binding()?;
        let allocator = ort::memory::Allocator::new(&session, gpu_memory()?)?;
        let mut stable = Vec::new();
        if hybrid {
            let names = if quarter {
                vec!["389", "751", "752", "fine_src"]
            } else {
                vec!["389", "751", "752"]
            };
            for name in names {
                let info = session
                    .outputs
                    .iter()
                    .find(|o| o.name == name)
                    .context("missing prefix output")?;
                let ValueType::Tensor { ty, shape, .. } = &info.output_type else {
                    anyhow::bail!("not tensor")
                };
                anyhow::ensure!(shape.iter().all(|v| *v > 0), "dynamic prefix output");
                let tensor = DynTensor::new(&allocator, *ty, shape.to_vec())?;
                let res = resource(&allocator, &tensor)?;
                binding.bind_output(name, tensor)?;
                stable.push(res);
            }
            // Setup/compilation and initial allocation are excluded from steady-state timing.
            probe.setup(
                &format!("{root}/refiner-shared.cso"),
                &vec![f16::ZERO; 3 * n],
                &vec![f16::ZERO; 16 * n],
                &vec![f16::ZERO; 4 * n],
                &weights,
                w,
                h,
            )?;
            checked(unsafe { rvm_probe_attach(probe.0, stable[0].0, stable[1].0, stable[2].0) })?;
            if quarter {
                let file = if packed {
                    "refiner-packed"
                } else {
                    "refiner-upsample"
                };
                let path = CString::new(format!("{root}/{file}.cso"))?;
                checked(unsafe { rvm_probe_post(probe.0, path.as_ptr(), stable[3].0) })?;
            }
        } else {
            binding.bind_output_to_device(
                "pha",
                &ort::memory::MemoryInfo::new(
                    AllocationDevice::CPU,
                    0,
                    AllocatorType::Device,
                    MemoryType::Default,
                )?,
            )?;
        }
        let mut shapes = Vec::new();
        for name in [R1_INPUT, R2_INPUT, R3_INPUT, R4_INPUT] {
            let info = session
                .inputs
                .iter()
                .find(|i| i.name == name)
                .context("missing recurrent input")?;
            let ValueType::Tensor { shape, .. } = &info.input_type else {
                anyhow::bail!("not tensor")
            };
            shapes.push(shape.to_vec());
        }
        let mut options = ort::session::RunOptions::new()?;
        options.disable_device_sync()?;
        let mut masks = Vec::with_capacity(n * frames.len());
        let mut times = Vec::new();
        let mut recurrent = Vec::new();
        let mut stable_src =
            Tensor::from_array(([1, 3, h as usize, w as usize], vec![f16::ZERO; 3 * n]))?;
        let mut stage_ms = [0.; 4];
        for (i, frame) in frames
            .iter()
            .cycle()
            .take(frames.len() * repeats)
            .enumerate()
        {
            if i % frames.len() == 0 {
                // Each repeat is an independent complete sequence, including its
                // initial recurrent state. The final sequence is quality-scored.
                recurrent = shapes
                    .iter()
                    .map(|s| {
                        Tensor::from_array((
                            s.clone(),
                            vec![f16::ZERO; s.iter().product::<i64>() as usize],
                        ))
                        .map(|v| v.upcast())
                    })
                    .collect::<ort::Result<Vec<_>>>()?;
                masks.clear();
            }
            let start = Instant::now();
            binding.clear_inputs();
            super::tensor::preprocess_f16(frame, stable_src.try_extract_tensor_mut::<f16>()?.1)?;
            let preprocessed = Instant::now();
            binding.bind_input(SRC_INPUT, &stable_src)?;
            for (name, value) in [R1_INPUT, R2_INPUT, R3_INPUT, R4_INPUT]
                .into_iter()
                .zip(&recurrent)
            {
                binding.bind_input(name, value)?;
            }
            for name in [R1_OUTPUT, R2_OUTPUT, R3_OUTPUT, R4_OUTPUT] {
                binding.bind_output_to_device(name, &gpu_memory()?)?;
            }
            let mut outputs = session.run_binding_with_options(&binding, &options)?;
            let submitted = Instant::now();
            let mut bgra = None;
            if hybrid && packed {
                checked(unsafe { rvm_probe_dispatch(probe.0) })?;
                bgra = Some(probe.read_packed(n)?);
            } else if hybrid {
                checked(unsafe { rvm_probe_dispatch(probe.0) })?;
                let alpha = probe.read(n)?; // One fence, after both DirectML and HLSL plus readback.
                anyhow::ensure!(
                    alpha.iter().all(|a| a.is_finite()),
                    "nonfinite hybrid alpha"
                );
                masks.extend(
                    alpha
                        .into_iter()
                        .map(|a| (a.clamp(0., 1.) * 255.).round() as u8),
                );
            } else {
                binding.synchronize_outputs()?;
                if packed {
                    let a = outputs["pha"].try_extract_array::<f16>()?;
                    bgra = Some(
                        super::tensor::alpha_to_mask_f16(
                            a.into_dimensionality::<ndarray::Ix4>()?,
                            w,
                            h,
                            frame.time,
                        )?
                        .data,
                    );
                } else {
                    masks.extend(
                        outputs["pha"]
                            .try_extract_tensor::<f16>()?
                            .1
                            .iter()
                            .map(|a| (a.to_f32().clamp(0., 1.) * 255.).round() as u8),
                    );
                }
            }
            let returned = Instant::now();
            recurrent = take_recurrent_outputs(&mut outputs)?;
            if i >= 30 {
                times.push(start.elapsed().as_secs_f64() * 1000.);
                stage_ms[0] += preprocessed.duration_since(start).as_secs_f64() * 1000.;
                stage_ms[1] += submitted.duration_since(preprocessed).as_secs_f64() * 1000.;
                stage_ms[2] += returned.duration_since(submitted).as_secs_f64() * 1000.;
                stage_ms[3] += start.elapsed().as_secs_f64() * 1000.;
            }
            // Like inference_benchmark, extracting quality-only alpha bytes is
            // outside timing once a complete BGRA result has been returned.
            if let Some(bgra) = bgra {
                masks.extend(bgra.chunks_exact(4).map(|p| p[0]));
            }
        }
        let (mut error, mut max, mut over4) = (0u64, 0u8, 0usize);
        for (a, b) in masks.iter().zip(&reference) {
            let d = a.abs_diff(*b);
            error += u64::from(d);
            max = max.max(d);
            over4 += usize::from(d > 4);
        }
        let mae = error as f64 / masks.len() as f64;
        eprintln!("DX12_HYBRID_STAGES packed={packed} preprocess={:.4} bind_submit={:.4} finish_read_convert={:.4}",stage_ms[0]/times.len() as f64,stage_ms[1]/times.len() as f64,stage_ms[2]/times.len() as f64);
        times.sort_by(f64::total_cmp);
        eprintln!("DX12_HYBRID mode={name} frames={} measured_calls={} mean_ms={:.4} median_ms={:.4} p95_ms={:.4} mae_u8={mae:.6} max_u8={max} fraction_over4={:.8}",frames.len(),times.len(),times.iter().sum::<f64>()/times.len() as f64,times[times.len()/2],times[times.len()*95/100],over4 as f64/masks.len() as f64);
        if hybrid {
            if !baseline.is_empty() {
                let paired = masks
                    .iter()
                    .zip(&baseline)
                    .map(|(a, b)| u64::from(a.abs_diff(*b)))
                    .sum::<u64>() as f64
                    / masks.len() as f64;
                eprintln!("DX12_HYBRID paired_mae_u8={paired:.6}");
            }
            if let Ok(save) = std::env::var("SEGMENTER_BENCH_SAVE") {
                fs::write(save, &masks)?;
            }
            if let Ok(directory) = std::env::var("SEGMENTER_BENCH_PREVIEWS") {
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
                for i in [0, 30, 75, 149, worst] {
                    super::benchmark::preview(
                        std::path::Path::new(&directory),
                        i,
                        &frames[i],
                        &reference[i * n..(i + 1) * n],
                        &masks[i * n..(i + 1) * n],
                    )?;
                }
            }
        } else {
            baseline = masks;
        }
        anyhow::ensure!(
            mae <= 0.05 && max <= 8 && over4 as f64 / reference.len() as f64 <= 0.0001,
            "hybrid quality threshold exceeded"
        );
    }
    Ok(())
}
