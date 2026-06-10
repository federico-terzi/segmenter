pub(super) const SRC_INPUT: &str = "src";
pub(super) const R1_INPUT: &str = "r1i";
pub(super) const R2_INPUT: &str = "r2i";
pub(super) const R3_INPUT: &str = "r3i";
pub(super) const R4_INPUT: &str = "r4i";
pub(super) const DOWNSAMPLE_RATIO_INPUT: &str = "downsample_ratio";

pub(super) const PHA_OUTPUT: &str = "pha";
pub(super) const R1_OUTPUT: &str = "r1o";
pub(super) const R2_OUTPUT: &str = "r2o";
pub(super) const R3_OUTPUT: &str = "r3o";
pub(super) const R4_OUTPUT: &str = "r4o";

pub(super) const RECURRENT_STATE_SHAPE: [usize; 4] = [1, 1, 1, 1];

pub(super) const DOWNSAMPLE_RATIO_IDENTITY_EPSILON: f32 = 1.0e-5;
pub(super) const RESIZE_IDENTITY_NODE_NAME: &str = "Resize_3";
pub(super) const RESIZE_OP_TYPE: &str = "Resize";
pub(super) const ARITHMETIC_IDENTITY_OP_TYPE: &str = "Mul";
pub(super) const RESIZE_IDENTITY_INITIALIZER: &str = "__segmenter_resize_3_identity_scale";

pub(super) const ONNX_TENSOR_FLOAT: u64 = 1;
pub(super) const ONNX_TENSOR_FLOAT16: u64 = 10;
