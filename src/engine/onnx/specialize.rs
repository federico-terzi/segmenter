//! Specialize only input metadata and the constant run configuration. Weights
//! and mathematical operations are unchanged; ORT can fold shape computations.
use super::proto::*;
use anyhow::{ensure, Context};

pub(super) fn specialize(
    model: &[u8],
    shapes: &[(&str, Vec<i64>)],
    ratio: f32,
) -> anyhow::Result<Vec<u8>> {
    ensure!(
        ratio.is_finite() && ratio > 0.0 && ratio <= 1.0,
        "invalid downsample ratio"
    );
    let fields = parse_proto_fields(model)?;
    ensure!(
        fields.iter().filter(|f| f.number == 7).count() == 1,
        "expected one ONNX graph"
    );
    let graph = fields
        .iter()
        .find(|f| f.number == 7 && f.wire_type == 2)
        .context("missing ONNX graph")?;
    let bytes = &model[graph.data_start..graph.data_end];
    let mut patched = Vec::with_capacity(bytes.len());
    let mut found = vec![false; shapes.len()];
    let mut freeze_ratio = false;
    let graph_fields = parse_proto_fields(bytes)?;
    let has_ratio_input =
        graph_fields
            .iter()
            .filter(|f| f.number == 11)
            .try_fold(false, |found, f| {
                Ok::<_, anyhow::Error>(
                    found || name(&bytes[f.data_start..f.data_end], 1)? == "downsample_ratio",
                )
            })?;
    for f in graph_fields {
        if f.number == 13 {
            continue;
        } // stale symbolic intermediate metadata
        if has_ratio_input
            && f.number == 5
            && name(&bytes[f.data_start..f.data_end], 8)? == "downsample_ratio"
        {
            continue; // Replace an overridable initializer, rather than duplicate it.
        }
        if f.number == 12 && f.wire_type == 2 && name(&bytes[f.data_start..f.data_end], 1)? == "fgr"
        {
            continue; // Segmenter only consumes alpha and recurrent state.
        }
        if f.number == 11 && f.wire_type == 2 {
            let value = &bytes[f.data_start..f.data_end];
            let name = name(value, 1)?;
            if name == "downsample_ratio" {
                ensure!(!freeze_ratio, "duplicate downsample_ratio input");
                let tensor = nested(value, &[2, 1])?;
                let fields = parse_proto_fields(tensor)?;
                let ty = fields
                    .iter()
                    .find(|f| f.number == 1 && f.wire_type == 0)
                    .context("missing ratio type")?;
                ensure!(
                    read_varint(tensor, ty.data_start)?.0 == 1,
                    "downsample_ratio must be FLOAT"
                );
                freeze_ratio = true;
                continue;
            }
            if let Some((i, (_, dims))) = shapes.iter().enumerate().find(|(_, (n, _))| *n == name) {
                ensure!(!found[i], "duplicate ONNX input {name}");
                ensure!(
                    dims.len() == 4 && dims.iter().all(|d| *d > 0),
                    "invalid shape for {name}"
                );
                ensure!(
                    parse_proto_fields(nested(value, &[2, 1, 2])?)?
                        .iter()
                        .filter(|f| f.number == 1)
                        .count()
                        == 4,
                    "expected rank four for {name}"
                );
                let mut shape = Vec::new();
                for &d in dims {
                    let mut dim = Vec::new();
                    encode_varint_field(1, d as u64, &mut dim);
                    encode_length_delimited_field(1, &dim, &mut shape);
                }
                let value = replace_nested(value, &[2, 1, 2], &shape)?;
                encode_length_delimited_field(11, &value, &mut patched);
                found[i] = true;
                continue;
            }
        }
        patched.extend_from_slice(&bytes[f.start..f.end]);
    }
    ensure!(found.iter().all(|v| *v), "missing input to specialize");
    if freeze_ratio {
        let mut tensor = Vec::new();
        encode_varint_field(1, 1, &mut tensor);
        encode_varint_field(2, 1, &mut tensor); // FLOAT
        encode_string_field(8, "downsample_ratio", &mut tensor);
        encode_length_delimited_field(9, &ratio.to_le_bytes(), &mut tensor);
        encode_length_delimited_field(5, &tensor, &mut patched);
    }
    let mut result = model[..graph.start].to_vec();
    encode_length_delimited_field(7, &patched, &mut result);
    result.extend_from_slice(&model[graph.end..]);
    Ok(result)
}

fn nested<'a>(mut bytes: &'a [u8], path: &[u32]) -> anyhow::Result<&'a [u8]> {
    for number in path {
        let field = parse_proto_fields(bytes)?
            .into_iter()
            .find(|f| f.number == *number && f.wire_type == 2)
            .context("missing tensor metadata")?;
        bytes = &bytes[field.data_start..field.data_end];
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(name: &str, dims: usize, dtype: u64) -> Vec<u8> {
        let mut shape = Vec::new();
        for _ in 0..dims {
            let mut d = Vec::new();
            encode_string_field(2, "dynamic", &mut d);
            encode_length_delimited_field(1, &d, &mut shape);
        }
        let mut tensor = Vec::new();
        encode_varint_field(1, dtype, &mut tensor);
        encode_length_delimited_field(2, &shape, &mut tensor);
        let mut ty = Vec::new();
        encode_length_delimited_field(1, &tensor, &mut ty);
        let mut value = Vec::new();
        encode_string_field(1, name, &mut value);
        encode_length_delimited_field(2, &ty, &mut value);
        value
    }

    fn fixture(rank: usize) -> Vec<u8> {
        let mut graph = Vec::new();
        encode_length_delimited_field(11, &input("src", rank, 10), &mut graph);
        encode_length_delimited_field(11, &input("downsample_ratio", 1, 1), &mut graph);
        encode_length_delimited_field(12, &input("fgr", 4, 10), &mut graph);
        encode_length_delimited_field(12, &input("pha", 4, 10), &mut graph);
        let mut weight = Vec::new();
        encode_string_field(8, "weights", &mut weight);
        encode_length_delimited_field(9, &[1, 2, 3, 4], &mut weight);
        encode_length_delimited_field(5, &weight, &mut graph);
        let mut model = Vec::new();
        encode_length_delimited_field(7, &graph, &mut model);
        model
    }

    #[test]
    fn fixes_shapes_freezes_ratio_preserves_weights_and_alpha() {
        let model = fixture(4);
        let output = specialize(&model, &[("src", vec![1, 3, 271, 481])], 0.25).unwrap();
        let graph = nested(&output, &[7]).unwrap();
        let fields = parse_proto_fields(graph).unwrap();
        let inputs: Vec<_> = fields.iter().filter(|f| f.number == 11).collect();
        assert_eq!(inputs.len(), 1);
        let shape = nested(&graph[inputs[0].data_start..inputs[0].data_end], &[2, 1, 2]).unwrap();
        let dims: Vec<_> = parse_proto_fields(shape)
            .unwrap()
            .iter()
            .map(|f| read_varint(shape, f.data_start + 1).unwrap().0)
            .collect();
        assert_eq!(dims, [1, 3, 271, 481]);
        let outputs: Vec<_> = fields
            .iter()
            .filter(|f| f.number == 12)
            .map(|f| name(&graph[f.data_start..f.data_end], 1).unwrap())
            .collect();
        assert_eq!(outputs, ["pha"]);
        let tensors: Vec<_> = fields
            .iter()
            .filter(|f| f.number == 5)
            .map(|f| &graph[f.data_start..f.data_end])
            .collect();
        assert_eq!(nested(tensors[0], &[9]).unwrap(), [1, 2, 3, 4]);
        assert_eq!(nested(tensors[1], &[9]).unwrap(), 0.25f32.to_le_bytes());
        // Re-applying specialization must not add another ratio initializer.
        assert_eq!(
            specialize(&output, &[("src", vec![1, 3, 271, 481])], 0.25).unwrap(),
            output
        );
    }

    #[test]
    fn rejects_incomplete_or_malformed_specializations() {
        assert!(specialize(&fixture(3), &[("src", vec![1; 4])], 1.).is_err());
        assert!(specialize(&fixture(4), &[("missing", vec![1; 4])], 1.).is_err());
        assert!(specialize(&fixture(4), &[("src", vec![1, 3, 0, 480])], 1.).is_err());
        assert!(specialize(&fixture(4), &[("src", vec![1; 4])], f32::NAN).is_err());
        assert!(specialize(&[0xff], &[], 1.).is_err());
    }
}

fn name(bytes: &[u8], number: u32) -> anyhow::Result<String> {
    let field = parse_proto_fields(bytes)?
        .into_iter()
        .find(|f| f.number == number)
        .context("missing name")?;
    read_proto_string(bytes, field)
}

fn replace_nested(bytes: &[u8], path: &[u32], replacement: &[u8]) -> anyhow::Result<Vec<u8>> {
    if path.is_empty() {
        return Ok(replacement.to_vec());
    }
    let fields = parse_proto_fields(bytes)?;
    let field = fields
        .iter()
        .find(|f| f.number == path[0] && f.wire_type == 2)
        .context("missing tensor shape metadata")?;
    let child = replace_nested(
        &bytes[field.data_start..field.data_end],
        &path[1..],
        replacement,
    )?;
    let mut result = bytes[..field.start].to_vec();
    encode_length_delimited_field(field.number, &child, &mut result);
    result.extend_from_slice(&bytes[field.end..]);
    Ok(result)
}
