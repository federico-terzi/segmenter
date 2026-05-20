use std::path::PathBuf;

use anyhow::{bail, Context};
use half::f16;
use ndarray::{Array1, Array4, ArrayView4};
use ort::{
    execution_providers::{CPUExecutionProvider, ExecutionProviderDispatch},
    inputs,
    logging::LogLevel,
    session::{builder::GraphOptimizationLevel, Session},
    tensor::TensorElementType,
    value::{DynTensor, DynTensorValueType, Tensor, TensorRef, ValueType},
};

use crate::frame::{MediaTime, PixelFormat, VideoFrame};

const SRC_INPUT: &str = "src";
const R1_INPUT: &str = "r1i";
const R2_INPUT: &str = "r2i";
const R3_INPUT: &str = "r3i";
const R4_INPUT: &str = "r4i";
const DOWNSAMPLE_RATIO_INPUT: &str = "downsample_ratio";
const PHA_OUTPUT: &str = "pha";
const R1_OUTPUT: &str = "r1o";
const R2_OUTPUT: &str = "r2o";
const R3_OUTPUT: &str = "r3o";
const R4_OUTPUT: &str = "r4o";
const RECURRENT_STATE_SHAPE: [usize; 4] = [1, 1, 1, 1];

pub struct RvmSegmenter {
    session: Session,
    precision: RvmPrecision,
    recurrent: Vec<DynTensor>,
    downsample_ratio: Array1<f32>,
}

impl RvmSegmenter {
    pub fn new(model_path: PathBuf, downsample_ratio: f32) -> anyhow::Result<Self> {
        if !model_path.exists() {
            bail!("RVM model path does not exist: {}", model_path.display());
        }
        if !downsample_ratio.is_finite() || downsample_ratio <= 0.0 || downsample_ratio > 1.0 {
            bail!(
                "RVM downsample ratio must be finite and within (0, 1], got {}",
                downsample_ratio
            );
        }

        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_execution_providers(execution_providers())?
            .with_parallel_execution(false)?
            .with_log_level(LogLevel::Error)?
            .commit_from_file(&model_path)
            .with_context(|| format!("failed to load RVM model: {}", model_path.display()))?;

        validate_io_names(&session)?;
        let precision = model_precision(&session)?;
        let recurrent = initial_recurrent_states(precision)?;

        Ok(Self {
            session,
            precision,
            recurrent,
            downsample_ratio: Array1::from_vec(vec![downsample_ratio]),
        })
    }

    pub fn segment(&mut self, frame: &VideoFrame) -> anyhow::Result<VideoFrame> {
        if frame.format != PixelFormat::Bgra {
            bail!("RVM segmenter only accepts BGRA frames");
        }

        let mut outputs = match self.precision {
            RvmPrecision::Float16 => {
                let src = preprocess_f16(frame)?;
                self.session.run(inputs![
                    SRC_INPUT => TensorRef::from_array_view(src.view())?,
                    R1_INPUT => &self.recurrent[0],
                    R2_INPUT => &self.recurrent[1],
                    R3_INPUT => &self.recurrent[2],
                    R4_INPUT => &self.recurrent[3],
                    DOWNSAMPLE_RATIO_INPUT => TensorRef::from_array_view(
                        self.downsample_ratio.view()
                    )?,
                ])?
            }
            RvmPrecision::Float32 => {
                let src = preprocess_f32(frame)?;
                self.session.run(inputs![
                    SRC_INPUT => TensorRef::from_array_view(src.view())?,
                    R1_INPUT => &self.recurrent[0],
                    R2_INPUT => &self.recurrent[1],
                    R3_INPUT => &self.recurrent[2],
                    R4_INPUT => &self.recurrent[3],
                    DOWNSAMPLE_RATIO_INPUT => TensorRef::from_array_view(
                        self.downsample_ratio.view()
                    )?,
                ])?
            }
        };

        let mask = match self.precision {
            RvmPrecision::Float16 => {
                let alpha = outputs
                    .get(PHA_OUTPUT)
                    .context("RVM did not return its alpha output")?
                    .try_extract_array::<f16>()
                    .context("failed to extract RVM fp16 alpha output")?;
                let alpha = alpha
                    .view()
                    .into_dimensionality::<ndarray::Ix4>()
                    .context("RVM alpha output was not a 4D tensor")?;
                alpha_to_mask_f16(alpha, frame.width, frame.height, frame.time)?
            }
            RvmPrecision::Float32 => {
                let alpha = outputs
                    .get(PHA_OUTPUT)
                    .context("RVM did not return its alpha output")?
                    .try_extract_array::<f32>()
                    .context("failed to extract RVM fp32 alpha output")?;
                let alpha = alpha
                    .view()
                    .into_dimensionality::<ndarray::Ix4>()
                    .context("RVM alpha output was not a 4D tensor")?;
                alpha_to_mask_f32(alpha, frame.width, frame.height, frame.time)?
            }
        };

        self.recurrent = take_recurrent_outputs(&mut outputs)?;
        Ok(mask)
    }
}

fn preprocess_f16(frame: &VideoFrame) -> anyhow::Result<Array4<f16>> {
    let width = frame.width as usize;
    let height = frame.height as usize;
    let mut tensor = Array4::<f16>::from_elem([1, 3, height, width], f16::from_f32(0.0));

    for y in 0..height {
        for x in 0..width {
            let source = frame.checked_pixel_offset(x as u32, y as u32)?;
            tensor[[0, 0, y, x]] = f16::from_f32(frame.data[source + 2] as f32 / 255.0);
            tensor[[0, 1, y, x]] = f16::from_f32(frame.data[source + 1] as f32 / 255.0);
            tensor[[0, 2, y, x]] = f16::from_f32(frame.data[source] as f32 / 255.0);
        }
    }

    Ok(tensor)
}

fn preprocess_f32(frame: &VideoFrame) -> anyhow::Result<Array4<f32>> {
    let width = frame.width as usize;
    let height = frame.height as usize;
    let mut tensor = Array4::<f32>::zeros([1, 3, height, width]);

    for y in 0..height {
        for x in 0..width {
            let source = frame.checked_pixel_offset(x as u32, y as u32)?;
            tensor[[0, 0, y, x]] = frame.data[source + 2] as f32 / 255.0;
            tensor[[0, 1, y, x]] = frame.data[source + 1] as f32 / 255.0;
            tensor[[0, 2, y, x]] = frame.data[source] as f32 / 255.0;
        }
    }

    Ok(tensor)
}

fn alpha_to_mask_f16(
    alpha: ArrayView4<'_, f16>,
    width: u32,
    height: u32,
    time: MediaTime,
) -> anyhow::Result<VideoFrame> {
    let shape = alpha.shape();
    validate_alpha_shape(shape)?;

    let source_x = sample_coordinates(width as usize, shape[3]);
    let source_y = sample_coordinates(height as usize, shape[2]);
    let bytes_per_row = width.checked_mul(4).context("mask row width overflowed")?;
    let mut data = vec![0_u8; bytes_per_row as usize * height as usize];

    for (target_y, sample_y) in source_y.into_iter().enumerate() {
        for (target_x, sample_x) in source_x.iter().copied().enumerate() {
            let value = sample_alpha_f16(alpha, sample_x, sample_y);
            write_mask_pixel(&mut data, bytes_per_row, target_x, target_y, value)?;
        }
    }

    VideoFrame::new_bgra(width, height, bytes_per_row, time, data)
}

fn alpha_to_mask_f32(
    alpha: ArrayView4<'_, f32>,
    width: u32,
    height: u32,
    time: MediaTime,
) -> anyhow::Result<VideoFrame> {
    let shape = alpha.shape();
    validate_alpha_shape(shape)?;

    let source_x = sample_coordinates(width as usize, shape[3]);
    let source_y = sample_coordinates(height as usize, shape[2]);
    let bytes_per_row = width.checked_mul(4).context("mask row width overflowed")?;
    let mut data = vec![0_u8; bytes_per_row as usize * height as usize];

    for (target_y, sample_y) in source_y.into_iter().enumerate() {
        for (target_x, sample_x) in source_x.iter().copied().enumerate() {
            let value = sample_alpha_f32(alpha, sample_x, sample_y);
            write_mask_pixel(&mut data, bytes_per_row, target_x, target_y, value)?;
        }
    }

    VideoFrame::new_bgra(width, height, bytes_per_row, time, data)
}

fn write_mask_pixel(
    data: &mut [u8],
    bytes_per_row: u32,
    x: usize,
    y: usize,
    value: f32,
) -> anyhow::Result<()> {
    if !value.is_finite() {
        bail!("RVM alpha output contained a non-finite value");
    }

    let alpha = (value.clamp(0.0, 1.0) * 255.0).round() as u8;
    let offset = y * bytes_per_row as usize + x * 4;
    data[offset] = alpha;
    data[offset + 1] = alpha;
    data[offset + 2] = alpha;
    data[offset + 3] = 255;

    Ok(())
}

fn take_recurrent_outputs(
    outputs: &mut ort::session::SessionOutputs<'_>,
) -> anyhow::Result<Vec<DynTensor>> {
    let mut recurrent = Vec::with_capacity(4);
    for name in [R1_OUTPUT, R2_OUTPUT, R3_OUTPUT, R4_OUTPUT] {
        recurrent.push(
            outputs
                .remove(name)
                .with_context(|| format!("RVM did not return recurrent state output {name}"))?
                .downcast::<DynTensorValueType>()
                .with_context(|| format!("RVM recurrent state output {name} was not a tensor"))?,
        );
    }

    Ok(recurrent)
}

fn initial_recurrent_states(precision: RvmPrecision) -> anyhow::Result<Vec<DynTensor>> {
    let mut recurrent = Vec::with_capacity(4);
    for _ in 0..4 {
        recurrent.push(match precision {
            RvmPrecision::Float16 => Tensor::from_array((
                RECURRENT_STATE_SHAPE,
                vec![f16::from_f32(0.0)].into_boxed_slice(),
            ))?
            .upcast(),
            RvmPrecision::Float32 => {
                Tensor::from_array((RECURRENT_STATE_SHAPE, vec![0.0_f32].into_boxed_slice()))?
                    .upcast()
            }
        });
    }

    Ok(recurrent)
}

fn model_precision(session: &Session) -> anyhow::Result<RvmPrecision> {
    match input_tensor_type(session, SRC_INPUT)? {
        TensorElementType::Float16 => Ok(RvmPrecision::Float16),
        TensorElementType::Float32 => Ok(RvmPrecision::Float32),
        ty => bail!("RVM src input must be f16 or f32, got {ty}"),
    }
}

fn validate_io_names(session: &Session) -> anyhow::Result<()> {
    for name in [
        SRC_INPUT,
        R1_INPUT,
        R2_INPUT,
        R3_INPUT,
        R4_INPUT,
        DOWNSAMPLE_RATIO_INPUT,
    ] {
        input_tensor_type(session, name)?;
    }

    for name in [PHA_OUTPUT, R1_OUTPUT, R2_OUTPUT, R3_OUTPUT, R4_OUTPUT] {
        output_tensor_type(session, name)?;
    }

    if input_tensor_type(session, DOWNSAMPLE_RATIO_INPUT)? != TensorElementType::Float32 {
        bail!("RVM downsample_ratio input must be f32");
    }

    Ok(())
}

fn input_tensor_type(session: &Session, name: &str) -> anyhow::Result<TensorElementType> {
    let input = session
        .inputs
        .iter()
        .find(|input| input.name == name)
        .with_context(|| format!("RVM model did not define input {name}"))?;
    tensor_type(&input.input_type).with_context(|| format!("RVM input {name} was not a tensor"))
}

fn output_tensor_type(session: &Session, name: &str) -> anyhow::Result<TensorElementType> {
    let output = session
        .outputs
        .iter()
        .find(|output| output.name == name)
        .with_context(|| format!("RVM model did not define output {name}"))?;
    tensor_type(&output.output_type).with_context(|| format!("RVM output {name} was not a tensor"))
}

fn tensor_type(value_type: &ValueType) -> Option<TensorElementType> {
    match value_type {
        ValueType::Tensor { ty, .. } => Some(*ty),
        _ => None,
    }
}

fn validate_alpha_shape(shape: &[usize]) -> anyhow::Result<()> {
    if shape.len() != 4 || shape[0] != 1 || shape[1] != 1 || shape[2] == 0 || shape[3] == 0 {
        bail!("RVM alpha output had invalid shape {:?}", shape);
    }

    Ok(())
}

fn execution_providers() -> Vec<ExecutionProviderDispatch> {
    vec![CPUExecutionProvider::default().build()]
}

fn sample_coordinates(target_len: usize, source_len: usize) -> Vec<SampleCoordinate> {
    (0..target_len)
        .map(|target| source_coordinate(target, target_len, source_len))
        .collect()
}

fn source_coordinate(target: usize, target_len: usize, source_len: usize) -> SampleCoordinate {
    if source_len <= 1 || target_len <= 1 {
        return SampleCoordinate {
            lower: 0,
            upper: 0,
            weight: 0.0,
        };
    }

    let source = ((target as f32 + 0.5) * source_len as f32 / target_len as f32 - 0.5)
        .clamp(0.0, (source_len - 1) as f32);
    let lower = source.floor() as usize;
    let upper = (lower + 1).min(source_len - 1);

    SampleCoordinate {
        lower,
        upper,
        weight: source - lower as f32,
    }
}

fn sample_alpha_f16(alpha: ArrayView4<'_, f16>, x: SampleCoordinate, y: SampleCoordinate) -> f32 {
    let top = mix(
        alpha[[0, 0, y.lower, x.lower]].to_f32(),
        alpha[[0, 0, y.lower, x.upper]].to_f32(),
        x.weight,
    );
    let bottom = mix(
        alpha[[0, 0, y.upper, x.lower]].to_f32(),
        alpha[[0, 0, y.upper, x.upper]].to_f32(),
        x.weight,
    );

    mix(top, bottom, y.weight)
}

fn sample_alpha_f32(alpha: ArrayView4<'_, f32>, x: SampleCoordinate, y: SampleCoordinate) -> f32 {
    let top = mix(
        alpha[[0, 0, y.lower, x.lower]],
        alpha[[0, 0, y.lower, x.upper]],
        x.weight,
    );
    let bottom = mix(
        alpha[[0, 0, y.upper, x.lower]],
        alpha[[0, 0, y.upper, x.upper]],
        x.weight,
    );

    mix(top, bottom, y.weight)
}

fn mix(from: f32, to: f32, amount: f32) -> f32 {
    from + (to - from) * amount
}

#[derive(Debug, Clone, Copy)]
enum RvmPrecision {
    Float16,
    Float32,
}

#[derive(Debug, Clone, Copy)]
struct SampleCoordinate {
    lower: usize,
    upper: usize,
    weight: f32,
}
