mod constants;
mod model_patch;
mod proto;
mod session;
mod tensor;

use std::{fs, path::PathBuf};

use anyhow::{bail, Context};
use half::f16;
use ndarray::Array1;
use ort::{inputs, value::TensorRef};

use self::{
    constants::{
        DOWNSAMPLE_RATIO_INPUT, PHA_OUTPUT, R1_INPUT, R2_INPUT, R3_INPUT, R4_INPUT,
        RESIZE_IDENTITY_NODE_NAME, SRC_INPUT,
    },
    model_patch::{
        prepare_identity_retry_model, prepare_primary_model, should_patch_identity_resize,
        PreparedModel, PreparedModelSource,
    },
    session::{
        build_session_state, take_recurrent_outputs, ModelSource, OnnxSessionState, RvmPrecision,
    },
    tensor::{alpha_to_mask_f16, alpha_to_mask_f32, preprocess_f16, preprocess_f32},
};
use crate::{
    engine::Engine,
    frame::{PixelFormat, VideoFrame},
};

pub struct OnnxEngine {
    model_path: PathBuf,
    state: Option<OnnxSessionState>,
    requested_downsample_ratio: f32,
    downsample_ratio: Array1<f32>,
}

impl OnnxEngine {
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

        let effective_downsample_ratio = if should_patch_identity_resize(downsample_ratio) {
            1.0
        } else {
            downsample_ratio
        };

        Ok(Self {
            model_path,
            state: None,
            requested_downsample_ratio: downsample_ratio,
            downsample_ratio: Array1::from_vec(vec![effective_downsample_ratio]),
        })
    }

    fn ensure_initialized(&mut self, width: u32, height: u32) -> anyhow::Result<()> {
        if let Some(state) = &self.state {
            if state.width != width || state.height != height {
                bail!(
                    "RVM ONNX session was initialized for {}x{} frames, but received {}x{}",
                    state.width,
                    state.height,
                    width,
                    height
                );
            }
            return Ok(());
        }

        eprintln!(
            "INFO: Initializing ONNX session with fixed dimensions width={} height={}",
            width, height
        );
        eprintln!("INFO: ONNX Runtime memory pattern is disabled");

        let original_model_bytes = fs::read(&self.model_path)
            .with_context(|| format!("failed to read RVM model: {}", self.model_path.display()))?;
        let prepared = prepare_primary_model(
            &original_model_bytes,
            width,
            height,
            self.requested_downsample_ratio,
        )?;
        log_messages(&prepared.messages);
        self.downsample_ratio = Array1::from_vec(vec![prepared.effective_downsample_ratio]);

        let state = match self.build_prepared_session(&prepared, width, height) {
            Ok(state) => state,
            Err(primary_error) if should_patch_identity_resize(self.requested_downsample_ratio) => {
                eprintln!(
                    "INFO: ONNX session creation failed after primary {} identity path: {primary_error:#}",
                    RESIZE_IDENTITY_NODE_NAME
                );
                eprintln!(
                    "INFO: Retrying ONNX session with {} identity patch and dynamic src dimensions",
                    RESIZE_IDENTITY_NODE_NAME
                );

                let retry = match prepare_identity_retry_model(
                    &original_model_bytes,
                    prepared.src_element_type,
                ) {
                    Ok(retry) => retry,
                    Err(retry_error) => {
                        return Err(primary_error).with_context(|| {
                            format!("identity-patch retry preparation failed: {retry_error:#}")
                        });
                    }
                };
                log_messages(&retry.messages);
                self.downsample_ratio = Array1::from_vec(vec![retry.effective_downsample_ratio]);
                self.build_prepared_session(&retry, width, height)
                    .with_context(|| "identity-patch retry with dynamic src dimensions failed")?
            }
            Err(error) => return Err(error),
        };

        self.state = Some(state);
        Ok(())
    }

    fn build_prepared_session(
        &self,
        prepared: &PreparedModel,
        width: u32,
        height: u32,
    ) -> anyhow::Result<OnnxSessionState> {
        let source = match &prepared.source {
            PreparedModelSource::OriginalFile => ModelSource::File(&self.model_path),
            PreparedModelSource::Bytes(bytes) => ModelSource::Memory(bytes),
        };

        build_session_state(
            source,
            &self.model_path,
            width,
            height,
            prepared.uses_downsample_ratio,
        )
    }
}

impl Engine for OnnxEngine {
    fn segment(&mut self, frame: &VideoFrame) -> anyhow::Result<VideoFrame> {
        if frame.format != PixelFormat::Bgra {
            bail!("RVM segmenter only accepts BGRA frames");
        }
        self.ensure_initialized(frame.width, frame.height)?;

        let downsample_ratio = self.downsample_ratio.view();
        let state = self
            .state
            .as_mut()
            .context("RVM ONNX session was not initialized")?;
        let precision = state.precision;

        let mut outputs = match precision {
            RvmPrecision::Float16 => {
                let src = preprocess_f16(frame)?;
                if state.uses_downsample_ratio {
                    state.session.run(inputs![
                        SRC_INPUT => TensorRef::from_array_view(src.view())?,
                        R1_INPUT => &state.recurrent[0],
                        R2_INPUT => &state.recurrent[1],
                        R3_INPUT => &state.recurrent[2],
                        R4_INPUT => &state.recurrent[3],
                        DOWNSAMPLE_RATIO_INPUT => TensorRef::from_array_view(downsample_ratio)?,
                    ])?
                } else {
                    state.session.run(inputs![
                        SRC_INPUT => TensorRef::from_array_view(src.view())?,
                        R1_INPUT => &state.recurrent[0],
                        R2_INPUT => &state.recurrent[1],
                        R3_INPUT => &state.recurrent[2],
                        R4_INPUT => &state.recurrent[3],
                    ])?
                }
            }
            RvmPrecision::Float32 => {
                let src = preprocess_f32(frame)?;
                if state.uses_downsample_ratio {
                    state.session.run(inputs![
                        SRC_INPUT => TensorRef::from_array_view(src.view())?,
                        R1_INPUT => &state.recurrent[0],
                        R2_INPUT => &state.recurrent[1],
                        R3_INPUT => &state.recurrent[2],
                        R4_INPUT => &state.recurrent[3],
                        DOWNSAMPLE_RATIO_INPUT => TensorRef::from_array_view(downsample_ratio)?,
                    ])?
                } else {
                    state.session.run(inputs![
                        SRC_INPUT => TensorRef::from_array_view(src.view())?,
                        R1_INPUT => &state.recurrent[0],
                        R2_INPUT => &state.recurrent[1],
                        R3_INPUT => &state.recurrent[2],
                        R4_INPUT => &state.recurrent[3],
                    ])?
                }
            }
        };

        let mask = match precision {
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

        state.recurrent = take_recurrent_outputs(&mut outputs)?;
        Ok(mask)
    }
}

fn log_messages(messages: &[String]) {
    for message in messages {
        eprintln!("INFO: {message}");
    }
}
