use anyhow::{bail, Context};

use super::{
    constants::{
        ARITHMETIC_IDENTITY_OP_TYPE, DOWNSAMPLE_RATIO_IDENTITY_EPSILON, DOWNSAMPLE_RATIO_INPUT,
        ONNX_TENSOR_FLOAT, ONNX_TENSOR_FLOAT16, RESIZE_IDENTITY_INITIALIZER,
        RESIZE_IDENTITY_NODE_NAME, RESIZE_OP_TYPE, SRC_INPUT,
    },
    proto::{
        encode_length_delimited_field, encode_string_field, encode_varint_field,
        parse_proto_fields, read_proto_string, read_varint, ProtoField,
    },
};

pub(super) enum PreparedModelSource {
    OriginalFile,
    Bytes(Vec<u8>),
}

pub(super) struct PreparedModel {
    pub(super) source: PreparedModelSource,
    pub(super) uses_downsample_ratio: bool,
    pub(super) effective_downsample_ratio: f32,
    pub(super) src_element_type: Option<u64>,
    pub(super) messages: Vec<String>,
}

pub(super) fn should_patch_identity_resize(downsample_ratio: f32) -> bool {
    downsample_ratio.is_finite()
        && (1.0 - downsample_ratio).abs() <= DOWNSAMPLE_RATIO_IDENTITY_EPSILON
}

pub(super) fn prepare_primary_model(
    original_model_bytes: &[u8],
    width: u32,
    height: u32,
    requested_downsample_ratio: f32,
) -> anyhow::Result<PreparedModel> {
    let source_patch = patch_src_input_dimensions(original_model_bytes, width, height)?;
    let mut model_bytes = source_patch.model_bytes;
    let mut changed_model = source_patch.applied || source_patch.removed_value_info > 0;
    let mut uses_downsample_ratio = true;
    let mut effective_downsample_ratio = requested_downsample_ratio;
    let mut messages = Vec::new();

    if source_patch.applied {
        messages.push(format!(
            "Patched ONNX src input to fixed dimensions width={width} height={height}"
        ));
    } else {
        messages.push(format!(
            "Skipped ONNX src fixed-dimension patch: {}",
            source_patch
                .skip_reason
                .as_deref()
                .unwrap_or("src input was not patched")
        ));
    }
    if source_patch.removed_value_info > 0 {
        messages.push(format!(
            "Removed {} ONNX graph value_info entries before session creation",
            source_patch.removed_value_info
        ));
    }

    if should_patch_identity_resize(requested_downsample_ratio) {
        effective_downsample_ratio = 1.0;
        messages.push(format!(
            "RVM downsample ratio {} is within {} of 1.0; feeding 1.0 and attempting {} identity patch",
            requested_downsample_ratio,
            DOWNSAMPLE_RATIO_IDENTITY_EPSILON,
            RESIZE_IDENTITY_NODE_NAME
        ));
        match patch_resize_3_to_identity(&model_bytes, source_patch.src_element_type)? {
            IdentityPatchResult::Applied(patched) => {
                model_bytes = patched;
                changed_model = true;
                uses_downsample_ratio = false;
                messages.push(format!(
                    "Applied ONNX arithmetic identity patch for {}",
                    RESIZE_IDENTITY_NODE_NAME
                ));
            }
            IdentityPatchResult::Skipped(reason) => {
                messages.push(format!(
                    "Skipped ONNX identity patch for {}: {}",
                    RESIZE_IDENTITY_NODE_NAME, reason
                ));
            }
        }
    } else {
        messages.push(format!(
            "RVM downsample ratio {} is outside {} identity tolerance; loading model without {} patch",
            requested_downsample_ratio,
            DOWNSAMPLE_RATIO_IDENTITY_EPSILON,
            RESIZE_IDENTITY_NODE_NAME
        ));
    }

    Ok(PreparedModel {
        source: if changed_model {
            PreparedModelSource::Bytes(model_bytes)
        } else {
            PreparedModelSource::OriginalFile
        },
        uses_downsample_ratio,
        effective_downsample_ratio,
        src_element_type: source_patch.src_element_type,
        messages,
    })
}

pub(super) fn prepare_identity_retry_model(
    original_model_bytes: &[u8],
    src_element_type: Option<u64>,
) -> anyhow::Result<PreparedModel> {
    let prune = prune_graph_value_info(original_model_bytes)?;
    let mut retry_bytes = prune.model_bytes;
    let mut messages = Vec::new();

    if prune.removed_value_info > 0 {
        messages.push(format!(
            "Removed {} ONNX graph value_info entries for identity-patch retry",
            prune.removed_value_info
        ));
    }

    match patch_resize_3_to_identity(&retry_bytes, src_element_type)? {
        IdentityPatchResult::Applied(patched) => {
            retry_bytes = patched;
            messages.push(format!(
                "Applied ONNX arithmetic identity patch for {} on retry",
                RESIZE_IDENTITY_NODE_NAME
            ));
        }
        IdentityPatchResult::Skipped(reason) => {
            bail!(
                "identity-patch retry could not patch {}: {}",
                RESIZE_IDENTITY_NODE_NAME,
                reason
            );
        }
    }

    Ok(PreparedModel {
        source: PreparedModelSource::Bytes(retry_bytes),
        uses_downsample_ratio: false,
        effective_downsample_ratio: 1.0,
        src_element_type,
        messages,
    })
}

struct SourceDimensionPatchResult {
    model_bytes: Vec<u8>,
    applied: bool,
    skip_reason: Option<String>,
    removed_value_info: usize,
    src_element_type: Option<u64>,
}

struct PruneValueInfoResult {
    model_bytes: Vec<u8>,
    removed_value_info: usize,
}

fn prune_graph_value_info(model_bytes: &[u8]) -> anyhow::Result<PruneValueInfoResult> {
    let model_fields =
        parse_proto_fields(model_bytes).context("failed to parse ONNX ModelProto fields")?;
    let graph_fields: Vec<_> = model_fields
        .iter()
        .filter(|field| field.number == 7 && field.wire_type == 2)
        .copied()
        .collect();

    let graph_field = match graph_fields.as_slice() {
        [] => {
            return Ok(PruneValueInfoResult {
                model_bytes: model_bytes.to_vec(),
                removed_value_info: 0,
            });
        }
        [graph_field] => *graph_field,
        _ => {
            return Ok(PruneValueInfoResult {
                model_bytes: model_bytes.to_vec(),
                removed_value_info: 0,
            });
        }
    };

    let graph_bytes = &model_bytes[graph_field.data_start..graph_field.data_end];
    let graph_fields =
        parse_proto_fields(graph_bytes).context("failed to parse ONNX GraphProto fields")?;
    let mut patched_graph = Vec::with_capacity(graph_bytes.len());
    let mut removed_value_info = 0;

    for field in graph_fields {
        if field.number == 13 && field.wire_type == 2 {
            removed_value_info += 1;
        } else {
            patched_graph.extend_from_slice(&graph_bytes[field.start..field.end]);
        }
    }

    let mut patched = Vec::with_capacity(model_bytes.len());
    patched.extend_from_slice(&model_bytes[..graph_field.start]);
    encode_length_delimited_field(7, &patched_graph, &mut patched);
    patched.extend_from_slice(&model_bytes[graph_field.end..]);

    Ok(PruneValueInfoResult {
        model_bytes: patched,
        removed_value_info,
    })
}

fn patch_src_input_dimensions(
    model_bytes: &[u8],
    width: u32,
    height: u32,
) -> anyhow::Result<SourceDimensionPatchResult> {
    let model_fields =
        parse_proto_fields(model_bytes).context("failed to parse ONNX ModelProto fields")?;
    let graph_fields: Vec<_> = model_fields
        .iter()
        .filter(|field| field.number == 7 && field.wire_type == 2)
        .copied()
        .collect();

    let graph_field = match graph_fields.as_slice() {
        [] => {
            return Ok(SourceDimensionPatchResult {
                model_bytes: model_bytes.to_vec(),
                applied: false,
                skip_reason: Some("ONNX ModelProto graph field was not found".to_string()),
                removed_value_info: 0,
                src_element_type: None,
            });
        }
        [graph_field] => *graph_field,
        _ => {
            return Ok(SourceDimensionPatchResult {
                model_bytes: model_bytes.to_vec(),
                applied: false,
                skip_reason: Some("ONNX ModelProto had multiple graph fields".to_string()),
                removed_value_info: 0,
                src_element_type: None,
            });
        }
    };

    let graph_bytes = &model_bytes[graph_field.data_start..graph_field.data_end];
    let graph_patch = patch_graph_src_input_dimensions(graph_bytes, width, height)?;

    let mut patched = Vec::with_capacity(model_bytes.len());
    patched.extend_from_slice(&model_bytes[..graph_field.start]);
    encode_length_delimited_field(7, &graph_patch.graph_bytes, &mut patched);
    patched.extend_from_slice(&model_bytes[graph_field.end..]);

    Ok(SourceDimensionPatchResult {
        model_bytes: patched,
        applied: graph_patch.applied,
        skip_reason: graph_patch.skip_reason,
        removed_value_info: graph_patch.removed_value_info,
        src_element_type: graph_patch.src_element_type,
    })
}

struct GraphSourceDimensionPatchResult {
    graph_bytes: Vec<u8>,
    applied: bool,
    skip_reason: Option<String>,
    removed_value_info: usize,
    src_element_type: Option<u64>,
}

fn patch_graph_src_input_dimensions(
    graph_bytes: &[u8],
    width: u32,
    height: u32,
) -> anyhow::Result<GraphSourceDimensionPatchResult> {
    let graph_fields =
        parse_proto_fields(graph_bytes).context("failed to parse ONNX GraphProto fields")?;
    let mut patched = Vec::with_capacity(graph_bytes.len());
    let mut applied = false;
    let mut saw_src = false;
    let mut skip_reason = None;
    let mut removed_value_info = 0;
    let mut src_element_type = None;

    for field in graph_fields {
        if field.number == 13 && field.wire_type == 2 {
            removed_value_info += 1;
            continue;
        }

        if field.number == 11 && field.wire_type == 2 {
            let value_info = &graph_bytes[field.data_start..field.data_end];
            match patch_src_value_info_dimensions(value_info, width, height)? {
                ValueInfoDimensionPatch::NotSrc => {
                    patched.extend_from_slice(&graph_bytes[field.start..field.end]);
                }
                ValueInfoDimensionPatch::Applied {
                    bytes: patched_value_info,
                    element_type,
                } => {
                    saw_src = true;
                    applied = true;
                    src_element_type = Some(element_type);
                    encode_length_delimited_field(11, &patched_value_info, &mut patched);
                }
                ValueInfoDimensionPatch::Invalid(reason) => {
                    saw_src = true;
                    if skip_reason.is_none() {
                        skip_reason = Some(reason);
                    }
                    patched.extend_from_slice(&graph_bytes[field.start..field.end]);
                }
            }
        } else {
            patched.extend_from_slice(&graph_bytes[field.start..field.end]);
        }
    }

    if !saw_src {
        skip_reason = Some("src graph input was not found".to_string());
    }

    Ok(GraphSourceDimensionPatchResult {
        graph_bytes: patched,
        applied,
        skip_reason,
        removed_value_info,
        src_element_type,
    })
}

enum ValueInfoDimensionPatch {
    NotSrc,
    Applied { bytes: Vec<u8>, element_type: u64 },
    Invalid(String),
}

fn patch_src_value_info_dimensions(
    value_info_bytes: &[u8],
    width: u32,
    height: u32,
) -> anyhow::Result<ValueInfoDimensionPatch> {
    let fields = parse_proto_fields(value_info_bytes)?;
    let name = fields
        .iter()
        .find(|field| field.number == 1)
        .map(|field| read_proto_string(value_info_bytes, *field))
        .transpose()?;
    if name.as_deref() != Some(SRC_INPUT) {
        return Ok(ValueInfoDimensionPatch::NotSrc);
    }

    let type_field = match fields
        .iter()
        .find(|field| field.number == 2 && field.wire_type == 2)
        .copied()
    {
        Some(type_field) => type_field,
        None => {
            return Ok(ValueInfoDimensionPatch::Invalid(
                "src ValueInfoProto did not contain a tensor type".to_string(),
            ));
        }
    };

    let patched_type = match patch_type_proto_dimensions(
        &value_info_bytes[type_field.data_start..type_field.data_end],
        width,
        height,
    )? {
        Some(patched_type) => patched_type,
        None => {
            return Ok(ValueInfoDimensionPatch::Invalid(
                "src TypeProto did not contain a patchable tensor shape".to_string(),
            ));
        }
    };

    let mut patched = Vec::with_capacity(value_info_bytes.len());
    for field in fields {
        if field.start == type_field.start {
            encode_length_delimited_field(2, &patched_type.bytes, &mut patched);
        } else {
            patched.extend_from_slice(&value_info_bytes[field.start..field.end]);
        }
    }
    Ok(ValueInfoDimensionPatch::Applied {
        bytes: patched,
        element_type: patched_type.element_type,
    })
}

struct TypeDimensionPatch {
    bytes: Vec<u8>,
    element_type: u64,
}

fn patch_type_proto_dimensions(
    type_bytes: &[u8],
    width: u32,
    height: u32,
) -> anyhow::Result<Option<TypeDimensionPatch>> {
    let fields = parse_proto_fields(type_bytes)?;
    let tensor_field = match fields
        .iter()
        .find(|field| field.number == 1 && field.wire_type == 2)
        .copied()
    {
        Some(tensor_field) => tensor_field,
        None => return Ok(None),
    };

    let Some(patched_tensor) = patch_tensor_type_dimensions(
        &type_bytes[tensor_field.data_start..tensor_field.data_end],
        width,
        height,
    )?
    else {
        return Ok(None);
    };

    let mut patched = Vec::with_capacity(type_bytes.len());
    for field in fields {
        if field.start == tensor_field.start {
            encode_length_delimited_field(1, &patched_tensor.bytes, &mut patched);
        } else {
            patched.extend_from_slice(&type_bytes[field.start..field.end]);
        }
    }
    Ok(Some(TypeDimensionPatch {
        bytes: patched,
        element_type: patched_tensor.element_type,
    }))
}

struct TensorTypeDimensionPatch {
    bytes: Vec<u8>,
    element_type: u64,
}

fn patch_tensor_type_dimensions(
    tensor_bytes: &[u8],
    width: u32,
    height: u32,
) -> anyhow::Result<Option<TensorTypeDimensionPatch>> {
    let fields = parse_proto_fields(tensor_bytes)?;
    let element_type = fields
        .iter()
        .find(|field| field.number == 1 && field.wire_type == 0)
        .map(|field| read_varint(tensor_bytes, field.data_start).map(|(value, _)| value))
        .transpose()?;
    let Some(element_type) = element_type else {
        return Ok(None);
    };

    let shape_field = match fields
        .iter()
        .find(|field| field.number == 2 && field.wire_type == 2)
        .copied()
    {
        Some(shape_field) => shape_field,
        None => return Ok(None),
    };

    let Some(patched_shape) = patch_tensor_shape_dimensions(
        &tensor_bytes[shape_field.data_start..shape_field.data_end],
        width,
        height,
    )?
    else {
        return Ok(None);
    };

    let mut patched = Vec::with_capacity(tensor_bytes.len());
    for field in fields {
        if field.start == shape_field.start {
            encode_length_delimited_field(2, &patched_shape, &mut patched);
        } else {
            patched.extend_from_slice(&tensor_bytes[field.start..field.end]);
        }
    }
    Ok(Some(TensorTypeDimensionPatch {
        bytes: patched,
        element_type,
    }))
}

fn patch_tensor_shape_dimensions(
    shape_bytes: &[u8],
    width: u32,
    height: u32,
) -> anyhow::Result<Option<Vec<u8>>> {
    let fields = parse_proto_fields(shape_bytes)?;
    let dim_count = fields
        .iter()
        .filter(|field| field.number == 1 && field.wire_type == 2)
        .count();
    if dim_count < 4 {
        return Ok(None);
    }

    let mut dim_index = 0;
    let mut patched = Vec::with_capacity(shape_bytes.len());
    for field in fields {
        if field.number == 1 && field.wire_type == 2 {
            match dim_index {
                2 => encode_length_delimited_field(1, &dim_value_proto(height), &mut patched),
                3 => encode_length_delimited_field(1, &dim_value_proto(width), &mut patched),
                _ => patched.extend_from_slice(&shape_bytes[field.start..field.end]),
            }
            dim_index += 1;
        } else {
            patched.extend_from_slice(&shape_bytes[field.start..field.end]);
        }
    }
    Ok(Some(patched))
}

fn dim_value_proto(value: u32) -> Vec<u8> {
    let mut dim = Vec::new();
    encode_varint_field(1, u64::from(value), &mut dim);
    dim
}

fn patch_resize_3_to_identity(
    model_bytes: &[u8],
    src_element_type: Option<u64>,
) -> anyhow::Result<IdentityPatchResult> {
    let Some(src_element_type) = src_element_type else {
        return Ok(IdentityPatchResult::Skipped(
            "src element type was not available for arithmetic identity initializer".to_string(),
        ));
    };

    let model_fields =
        parse_proto_fields(model_bytes).context("failed to parse ONNX ModelProto fields")?;
    let graph_fields: Vec<_> = model_fields
        .iter()
        .filter(|field| field.number == 7 && field.wire_type == 2)
        .copied()
        .collect();

    let graph_field = match graph_fields.as_slice() {
        [] => {
            return Ok(IdentityPatchResult::Skipped(
                "ONNX ModelProto graph field was not found".to_string(),
            ));
        }
        [graph_field] => *graph_field,
        _ => {
            return Ok(IdentityPatchResult::Skipped(
                "ONNX ModelProto had multiple graph fields".to_string(),
            ));
        }
    };

    let graph_bytes = &model_bytes[graph_field.data_start..graph_field.data_end];
    match patch_graph_resize_3_to_identity(graph_bytes, src_element_type)? {
        IdentityPatchResult::Applied(patched_graph) => {
            let mut patched = Vec::with_capacity(model_bytes.len());
            patched.extend_from_slice(&model_bytes[..graph_field.start]);
            encode_length_delimited_field(7, &patched_graph, &mut patched);
            patched.extend_from_slice(&model_bytes[graph_field.end..]);
            Ok(IdentityPatchResult::Applied(patched))
        }
        IdentityPatchResult::Skipped(reason) => Ok(IdentityPatchResult::Skipped(reason)),
    }
}

fn patch_graph_resize_3_to_identity(
    graph_bytes: &[u8],
    src_element_type: u64,
) -> anyhow::Result<IdentityPatchResult> {
    let Some(identity_initializer) = encode_identity_initializer(src_element_type) else {
        return Ok(IdentityPatchResult::Skipped(format!(
            "unsupported src tensor element type {src_element_type} for arithmetic identity patch"
        )));
    };

    let graph_fields =
        parse_proto_fields(graph_bytes).context("failed to parse ONNX GraphProto fields")?;
    let mut match_to_patch = None;
    let mut nodes = Vec::new();

    if graph_fields
        .iter()
        .filter(|field| field.number == 5 && field.wire_type == 2)
        .any(|field| {
            tensor_initializer_name(&graph_bytes[field.data_start..field.data_end])
                .is_ok_and(|name| name.as_deref() == Some(RESIZE_IDENTITY_INITIALIZER))
        })
    {
        return Ok(IdentityPatchResult::Skipped(format!(
            "initializer {RESIZE_IDENTITY_INITIALIZER} already exists"
        )));
    }

    for field in graph_fields
        .iter()
        .filter(|field| field.number == 1 && field.wire_type == 2)
    {
        let node_bytes = &graph_bytes[field.data_start..field.data_end];
        let info = read_node_info(node_bytes)?;
        match inspect_resize_3_node(&info) {
            ResizeNodeInspection::NotResize3 => {}
            ResizeNodeInspection::Invalid(reason) => {
                return Ok(IdentityPatchResult::Skipped(reason));
            }
            ResizeNodeInspection::Valid {
                output,
                dead_inputs,
            } => {
                if match_to_patch.is_some() {
                    return Ok(IdentityPatchResult::Skipped(format!(
                        "found multiple {RESIZE_IDENTITY_NODE_NAME} nodes"
                    )));
                }
                match_to_patch = Some((*field, output, dead_inputs));
            }
        }
        nodes.push((*field, info));
    }

    let Some((node_field, output, dead_inputs)) = match_to_patch else {
        return Ok(IdentityPatchResult::Skipped(format!(
            "{RESIZE_IDENTITY_NODE_NAME} node was not found"
        )));
    };

    let removed_node_starts = dead_producer_node_starts(&nodes, node_field.start, dead_inputs);
    let identity_node = encode_identity_node(&output);
    let mut patched = Vec::with_capacity(graph_bytes.len());
    for field in graph_fields {
        if field.number == 1 && field.wire_type == 2 {
            if field.start == node_field.start {
                encode_length_delimited_field(1, &identity_node, &mut patched);
            } else if removed_node_starts.contains(&field.start) {
                continue;
            } else {
                patched.extend_from_slice(&graph_bytes[field.start..field.end]);
            }
        } else if field.number == 11
            && field.wire_type == 2
            && value_info_name(&graph_bytes[field.data_start..field.data_end])?.as_deref()
                == Some(DOWNSAMPLE_RATIO_INPUT)
        {
            continue;
        } else {
            patched.extend_from_slice(&graph_bytes[field.start..field.end]);
        }
    }
    encode_length_delimited_field(5, &identity_initializer, &mut patched);
    Ok(IdentityPatchResult::Applied(patched))
}

enum IdentityPatchResult {
    Applied(Vec<u8>),
    Skipped(String),
}

enum ResizeNodeInspection {
    NotResize3,
    Valid {
        output: String,
        dead_inputs: Vec<String>,
    },
    Invalid(String),
}

#[derive(Default)]
struct NodeInfo {
    inputs: Vec<String>,
    outputs: Vec<String>,
    name: Option<String>,
    op_type: Option<String>,
    domain: Option<String>,
    attribute_names: Vec<String>,
}

fn inspect_resize_3_node(info: &NodeInfo) -> ResizeNodeInspection {
    if info.name.as_deref() != Some(RESIZE_IDENTITY_NODE_NAME) {
        return ResizeNodeInspection::NotResize3;
    }

    if info.op_type.as_deref() != Some(RESIZE_OP_TYPE) {
        return ResizeNodeInspection::Invalid(format!(
            "{RESIZE_IDENTITY_NODE_NAME} op_type was {:?}, expected {RESIZE_OP_TYPE}",
            info.op_type
        ));
    }
    if info
        .domain
        .as_deref()
        .is_some_and(|domain| !domain.is_empty())
    {
        return ResizeNodeInspection::Invalid(format!(
            "{RESIZE_IDENTITY_NODE_NAME} used non-default ONNX domain {:?}",
            info.domain
        ));
    }
    if info.inputs.len() < 3 {
        return ResizeNodeInspection::Invalid(format!(
            "{RESIZE_IDENTITY_NODE_NAME} had {} inputs, expected at least 3",
            info.inputs.len()
        ));
    }
    if info.inputs.first().map(String::as_str) != Some(SRC_INPUT) {
        return ResizeNodeInspection::Invalid(format!(
            "{RESIZE_IDENTITY_NODE_NAME} first input was {:?}, expected {SRC_INPUT}",
            info.inputs.first()
        ));
    }
    if info.outputs.len() != 1 {
        return ResizeNodeInspection::Invalid(format!(
            "{RESIZE_IDENTITY_NODE_NAME} had {} outputs, expected 1",
            info.outputs.len()
        ));
    }
    for attribute in ["coordinate_transformation_mode", "mode", "nearest_mode"] {
        if !info
            .attribute_names
            .iter()
            .any(|attribute_name| attribute_name == attribute)
        {
            return ResizeNodeInspection::Invalid(format!(
                "{RESIZE_IDENTITY_NODE_NAME} was missing expected Resize attribute {attribute}"
            ));
        }
    }

    ResizeNodeInspection::Valid {
        output: info.outputs[0].clone(),
        dead_inputs: info.inputs.iter().skip(1).cloned().collect(),
    }
}

fn dead_producer_node_starts(
    nodes: &[(ProtoField, NodeInfo)],
    resize_node_start: usize,
    initial_dead_values: Vec<String>,
) -> Vec<usize> {
    let mut dead_values = initial_dead_values;
    let mut removed_starts = Vec::new();
    let mut changed = true;

    while changed {
        changed = false;
        for (field, info) in nodes {
            if field.start == resize_node_start || removed_starts.contains(&field.start) {
                continue;
            }
            if info
                .outputs
                .iter()
                .any(|output| dead_values.iter().any(|value| value == output))
            {
                removed_starts.push(field.start);
                for input in &info.inputs {
                    if input != SRC_INPUT
                        && input != DOWNSAMPLE_RATIO_INPUT
                        && !dead_values.iter().any(|value| value == input)
                    {
                        dead_values.push(input.clone());
                    }
                }
                changed = true;
            }
        }
    }

    removed_starts
}

fn read_node_info(node_bytes: &[u8]) -> anyhow::Result<NodeInfo> {
    let mut info = NodeInfo::default();
    for field in parse_proto_fields(node_bytes)? {
        match field.number {
            1 => info.inputs.push(read_proto_string(node_bytes, field)?),
            2 => info.outputs.push(read_proto_string(node_bytes, field)?),
            3 => info.name = Some(read_proto_string(node_bytes, field)?),
            4 => info.op_type = Some(read_proto_string(node_bytes, field)?),
            5 => {
                if field.wire_type != 2 {
                    bail!("ONNX NodeProto attribute field was not length-delimited");
                }
                if let Some(name) =
                    read_attribute_name(&node_bytes[field.data_start..field.data_end])?
                {
                    info.attribute_names.push(name);
                }
            }
            7 => info.domain = Some(read_proto_string(node_bytes, field)?),
            _ => {}
        }
    }
    Ok(info)
}

fn read_attribute_name(attribute_bytes: &[u8]) -> anyhow::Result<Option<String>> {
    for field in parse_proto_fields(attribute_bytes)? {
        if field.number == 1 {
            return Ok(Some(read_proto_string(attribute_bytes, field)?));
        }
    }
    Ok(None)
}

fn encode_identity_node(output: &str) -> Vec<u8> {
    let mut node = Vec::new();
    encode_string_field(1, SRC_INPUT, &mut node);
    encode_string_field(1, RESIZE_IDENTITY_INITIALIZER, &mut node);
    encode_string_field(2, output, &mut node);
    encode_string_field(3, RESIZE_IDENTITY_NODE_NAME, &mut node);
    encode_string_field(4, ARITHMETIC_IDENTITY_OP_TYPE, &mut node);
    node
}

fn encode_identity_initializer(element_type: u64) -> Option<Vec<u8>> {
    let raw_data: &[u8] = match element_type {
        ONNX_TENSOR_FLOAT => &[0x00, 0x00, 0x80, 0x3f],
        ONNX_TENSOR_FLOAT16 => &[0x00, 0x3c],
        _ => return None,
    };

    let mut tensor = Vec::new();
    encode_varint_field(1, 1, &mut tensor);
    encode_varint_field(2, element_type, &mut tensor);
    encode_string_field(8, RESIZE_IDENTITY_INITIALIZER, &mut tensor);
    encode_length_delimited_field(9, raw_data, &mut tensor);
    Some(tensor)
}

fn tensor_initializer_name(tensor_bytes: &[u8]) -> anyhow::Result<Option<String>> {
    for field in parse_proto_fields(tensor_bytes)? {
        if field.number == 8 {
            return Ok(Some(read_proto_string(tensor_bytes, field)?));
        }
    }
    Ok(None)
}

fn value_info_name(value_info_bytes: &[u8]) -> anyhow::Result<Option<String>> {
    for field in parse_proto_fields(value_info_bytes)? {
        if field.number == 1 {
            return Ok(Some(read_proto_string(value_info_bytes, field)?));
        }
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::super::proto::{
        encode_length_delimited_field, encode_string_field, encode_varint_field,
    };
    use super::*;

    #[test]
    fn identity_tolerance_accepts_exact_and_near_one_ratios() {
        assert!(should_patch_identity_resize(1.0));
        assert!(should_patch_identity_resize(0.999_999));
        assert!(!should_patch_identity_resize(0.999));
        assert!(!should_patch_identity_resize(0.0));
    }

    #[test]
    fn resize_3_patch_rewrites_node_to_identity_and_preserves_output() {
        let model = model_with_node(resize_node(
            RESIZE_IDENTITY_NODE_NAME,
            RESIZE_OP_TYPE,
            "389",
        ));
        let patched = match patch_resize_3_to_identity(&model, Some(ONNX_TENSOR_FLOAT16)).unwrap() {
            IdentityPatchResult::Applied(patched) => patched,
            IdentityPatchResult::Skipped(reason) => panic!("patch skipped: {reason}"),
        };

        let info = only_node_info(&patched);
        assert_eq!(info.name.as_deref(), Some(RESIZE_IDENTITY_NODE_NAME));
        assert_eq!(info.op_type.as_deref(), Some(ARITHMETIC_IDENTITY_OP_TYPE));
        assert_eq!(info.inputs, vec![SRC_INPUT, RESIZE_IDENTITY_INITIALIZER]);
        assert_eq!(info.outputs, vec!["389"]);
        assert!(info.attribute_names.is_empty());
    }

    #[test]
    fn resize_3_patch_skips_when_node_is_not_found() {
        let model = model_with_node(resize_node("Resize_9", RESIZE_OP_TYPE, "389"));

        match patch_resize_3_to_identity(&model, Some(ONNX_TENSOR_FLOAT16)).unwrap() {
            IdentityPatchResult::Applied(_) => panic!("unexpected patch"),
            IdentityPatchResult::Skipped(reason) => {
                assert!(reason.contains("Resize_3 node was not found"));
            }
        }
    }

    #[test]
    fn resize_3_patch_skips_when_node_shape_is_unexpected() {
        let model = model_with_node(resize_node(
            RESIZE_IDENTITY_NODE_NAME,
            ARITHMETIC_IDENTITY_OP_TYPE,
            "389",
        ));

        match patch_resize_3_to_identity(&model, Some(ONNX_TENSOR_FLOAT16)).unwrap() {
            IdentityPatchResult::Applied(_) => panic!("unexpected patch"),
            IdentityPatchResult::Skipped(reason) => {
                assert!(reason.contains("op_type"));
            }
        }
    }

    #[test]
    fn runtime_patches_static_src_and_remove_dead_resize_scale_chain() {
        let model = model_with_graph(test_graph_with_resize_scale_chain());
        let source_patch = patch_src_input_dimensions(&model, 480, 270).unwrap();

        assert!(source_patch.applied);
        assert_eq!(source_patch.src_element_type, Some(ONNX_TENSOR_FLOAT16));
        assert_eq!(source_patch.removed_value_info, 1);

        let patched = match patch_resize_3_to_identity(
            &source_patch.model_bytes,
            source_patch.src_element_type,
        )
        .unwrap()
        {
            IdentityPatchResult::Applied(patched) => patched,
            IdentityPatchResult::Skipped(reason) => panic!("patch skipped: {reason}"),
        };

        let graph = only_graph_bytes(&patched);
        let input_names = graph_input_names(graph);
        assert!(input_names.contains(&SRC_INPUT.to_string()));
        assert!(!input_names.contains(&DOWNSAMPLE_RATIO_INPUT.to_string()));

        let nodes = graph_node_infos(graph);
        assert!(!nodes
            .iter()
            .any(|node| node.name.as_deref() == Some("Constant_0")));
        assert!(!nodes
            .iter()
            .any(|node| node.name.as_deref() == Some("Constant_1")));
        assert!(!nodes
            .iter()
            .any(|node| node.name.as_deref() == Some("Concat_2")));
        assert!(nodes.iter().any(|node| {
            node.name.as_deref() == Some(RESIZE_IDENTITY_NODE_NAME)
                && node.op_type.as_deref() == Some(ARITHMETIC_IDENTITY_OP_TYPE)
        }));

        assert!(graph_initializer_names(graph).contains(&RESIZE_IDENTITY_INITIALIZER.to_string()));
    }

    fn model_with_node(node: Vec<u8>) -> Vec<u8> {
        let mut graph = Vec::new();
        encode_length_delimited_field(1, &node, &mut graph);

        model_with_graph(graph)
    }

    fn model_with_graph(graph: Vec<u8>) -> Vec<u8> {
        let mut model = Vec::new();
        encode_length_delimited_field(7, &graph, &mut model);
        model
    }

    fn resize_node(name: &str, op_type: &str, output: &str) -> Vec<u8> {
        let mut node = Vec::new();
        encode_string_field(1, SRC_INPUT, &mut node);
        encode_string_field(1, "386", &mut node);
        encode_string_field(1, "388", &mut node);
        encode_string_field(2, output, &mut node);
        encode_string_field(3, name, &mut node);
        encode_string_field(4, op_type, &mut node);
        for attribute in ["coordinate_transformation_mode", "mode", "nearest_mode"] {
            encode_length_delimited_field(5, &attribute_proto(attribute), &mut node);
        }
        node
    }

    fn attribute_proto(name: &str) -> Vec<u8> {
        let mut attribute = Vec::new();
        encode_string_field(1, name, &mut attribute);
        attribute
    }

    fn test_graph_with_resize_scale_chain() -> Vec<u8> {
        let mut graph = Vec::new();
        encode_length_delimited_field(
            1,
            &node("Constant_0", "Constant", &[], &["386"]),
            &mut graph,
        );
        encode_length_delimited_field(
            1,
            &node("Constant_1", "Constant", &[], &["387"]),
            &mut graph,
        );
        encode_length_delimited_field(
            1,
            &node(
                "Concat_2",
                "Concat",
                &["387", DOWNSAMPLE_RATIO_INPUT, DOWNSAMPLE_RATIO_INPUT],
                &["388"],
            ),
            &mut graph,
        );
        encode_length_delimited_field(
            1,
            &resize_node(RESIZE_IDENTITY_NODE_NAME, RESIZE_OP_TYPE, "389"),
            &mut graph,
        );
        encode_length_delimited_field(11, &src_value_info(), &mut graph);
        encode_length_delimited_field(
            11,
            &value_info_with_name(DOWNSAMPLE_RATIO_INPUT),
            &mut graph,
        );
        encode_length_delimited_field(13, &value_info_with_name("stale_value_info"), &mut graph);
        graph
    }

    fn node(name: &str, op_type: &str, inputs: &[&str], outputs: &[&str]) -> Vec<u8> {
        let mut node = Vec::new();
        for input in inputs {
            encode_string_field(1, input, &mut node);
        }
        for output in outputs {
            encode_string_field(2, output, &mut node);
        }
        encode_string_field(3, name, &mut node);
        encode_string_field(4, op_type, &mut node);
        node
    }

    fn src_value_info() -> Vec<u8> {
        let mut shape = Vec::new();
        encode_length_delimited_field(1, &dim_value_proto(1), &mut shape);
        encode_length_delimited_field(1, &dim_value_proto(3), &mut shape);
        encode_length_delimited_field(1, &dim_param_proto("height"), &mut shape);
        encode_length_delimited_field(1, &dim_param_proto("width"), &mut shape);

        let mut tensor = Vec::new();
        encode_varint_field(1, ONNX_TENSOR_FLOAT16, &mut tensor);
        encode_length_delimited_field(2, &shape, &mut tensor);

        let mut type_proto = Vec::new();
        encode_length_delimited_field(1, &tensor, &mut type_proto);

        let mut value_info = value_info_with_name(SRC_INPUT);
        encode_length_delimited_field(2, &type_proto, &mut value_info);
        value_info
    }

    fn dim_param_proto(value: &str) -> Vec<u8> {
        let mut dim = Vec::new();
        encode_string_field(2, value, &mut dim);
        dim
    }

    fn value_info_with_name(name: &str) -> Vec<u8> {
        let mut value_info = Vec::new();
        encode_string_field(1, name, &mut value_info);
        value_info
    }

    fn only_node_info(model: &[u8]) -> NodeInfo {
        let graph = only_graph_bytes(model);
        let graph_fields = parse_proto_fields(graph).unwrap();
        let node_field = graph_fields
            .iter()
            .find(|field| field.number == 1 && field.wire_type == 2)
            .unwrap();
        read_node_info(&graph[node_field.data_start..node_field.data_end]).unwrap()
    }

    fn only_graph_bytes(model: &[u8]) -> &[u8] {
        let model_fields = parse_proto_fields(model).unwrap();
        let graph_field = model_fields
            .iter()
            .find(|field| field.number == 7 && field.wire_type == 2)
            .unwrap();
        &model[graph_field.data_start..graph_field.data_end]
    }

    fn graph_input_names(graph: &[u8]) -> Vec<String> {
        let graph_fields = parse_proto_fields(graph).unwrap();
        graph_fields
            .iter()
            .filter(|field| field.number == 11 && field.wire_type == 2)
            .filter_map(|field| value_info_name(&graph[field.data_start..field.data_end]).unwrap())
            .collect()
    }

    fn graph_node_infos(graph: &[u8]) -> Vec<NodeInfo> {
        let graph_fields = parse_proto_fields(graph).unwrap();
        graph_fields
            .iter()
            .filter(|field| field.number == 1 && field.wire_type == 2)
            .map(|field| read_node_info(&graph[field.data_start..field.data_end]).unwrap())
            .collect()
    }

    fn graph_initializer_names(graph: &[u8]) -> Vec<String> {
        let graph_fields = parse_proto_fields(graph).unwrap();
        graph_fields
            .iter()
            .filter(|field| field.number == 5 && field.wire_type == 2)
            .filter_map(|field| {
                tensor_initializer_name(&graph[field.data_start..field.data_end]).unwrap()
            })
            .collect()
    }
}
