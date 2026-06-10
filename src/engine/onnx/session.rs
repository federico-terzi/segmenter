use std::path::Path;

use anyhow::{bail, Context};
use half::f16;
#[cfg(target_os = "windows")]
use ort::execution_providers::DirectMLExecutionProvider;
use ort::{
    execution_providers::{CPUExecutionProvider, ExecutionProviderDispatch},
    session::{builder::GraphOptimizationLevel, Session},
    tensor::TensorElementType,
    value::{DynTensor, DynTensorValueType, Tensor, ValueType},
};

use super::constants::{
    DOWNSAMPLE_RATIO_INPUT, PHA_OUTPUT, R1_INPUT, R1_OUTPUT, R2_INPUT, R2_OUTPUT, R3_INPUT,
    R3_OUTPUT, R4_INPUT, R4_OUTPUT, RECURRENT_STATE_SHAPE, SRC_INPUT,
};

pub(super) enum ModelSource<'a> {
    File(&'a Path),
    Memory(&'a [u8]),
}

pub(super) struct OnnxSessionState {
    pub(super) session: Session,
    pub(super) precision: RvmPrecision,
    pub(super) recurrent: Vec<DynTensor>,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) uses_downsample_ratio: bool,
}

pub(super) fn build_session_state(
    source: ModelSource<'_>,
    model_path: &Path,
    width: u32,
    height: u32,
    uses_downsample_ratio: bool,
) -> anyhow::Result<OnnxSessionState> {
    let builder = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_execution_providers(execution_providers())?
        .with_parallel_execution(false)?
        .with_memory_pattern(false)?;

    let session = match source {
        ModelSource::File(path) => builder.commit_from_file(path),
        ModelSource::Memory(bytes) => builder.commit_from_memory(bytes),
    }
    .with_context(|| format!("failed to load RVM model: {}", model_path.display()))?;

    validate_io_names(&session, uses_downsample_ratio)?;
    let precision = model_precision(&session)?;
    let recurrent = initial_recurrent_states(precision)?;

    Ok(OnnxSessionState {
        session,
        precision,
        recurrent,
        width,
        height,
        uses_downsample_ratio,
    })
}

pub(super) fn take_recurrent_outputs(
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

fn validate_io_names(session: &Session, uses_downsample_ratio: bool) -> anyhow::Result<()> {
    for name in [SRC_INPUT, R1_INPUT, R2_INPUT, R3_INPUT, R4_INPUT] {
        input_tensor_type(session, name)?;
    }
    if uses_downsample_ratio {
        input_tensor_type(session, DOWNSAMPLE_RATIO_INPUT)?;
    }

    for name in [PHA_OUTPUT, R1_OUTPUT, R2_OUTPUT, R3_OUTPUT, R4_OUTPUT] {
        output_tensor_type(session, name)?;
    }

    if uses_downsample_ratio
        && input_tensor_type(session, DOWNSAMPLE_RATIO_INPUT)? != TensorElementType::Float32
    {
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

#[cfg(target_os = "windows")]
fn execution_providers() -> Vec<ExecutionProviderDispatch> {
    vec![
        DirectMLExecutionProvider::default().build(),
        CPUExecutionProvider::default().build(),
    ]
}

#[cfg(not(target_os = "windows"))]
fn execution_providers() -> Vec<ExecutionProviderDispatch> {
    vec![CPUExecutionProvider::default().build()]
}

#[derive(Debug, Clone, Copy)]
pub(super) enum RvmPrecision {
    Float16,
    Float32,
}
