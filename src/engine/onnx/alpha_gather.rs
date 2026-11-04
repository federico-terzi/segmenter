//! Avoid a DirectML graph-compiler failure for a Split whose sole live output
//! is one channel (the final RVM alpha). Gather retains that dimension and
//! selects the identical element range, without changing any model weights.
use super::proto::*;
use anyhow::{ensure, Context};
use std::collections::{HashMap, HashSet};

struct Node<'a> {
    field: ProtoField,
    bytes: &'a [u8],
    fields: Vec<ProtoField>,
    inputs: Vec<String>,
    outputs: Vec<String>,
}

fn strings(bytes: &[u8], fields: &[ProtoField], number: u32) -> anyhow::Result<Vec<String>> {
    fields
        .iter()
        .filter(|f| f.number == number)
        .map(|f| read_proto_string(bytes, *f))
        .collect()
}

pub(super) fn rewrite_model(model: &[u8]) -> anyhow::Result<Vec<u8>> {
    let fields = parse_proto_fields(model)?;
    let graph = fields
        .iter()
        .find(|f| f.number == 7 && f.wire_type == 2)
        .context("missing ONNX graph")?;
    let bytes = &model[graph.data_start..graph.data_end];
    let rewritten = rewrite(bytes)?;
    if rewritten == bytes {
        return Ok(model.to_vec());
    }
    let mut result = model[..graph.start].to_vec();
    encode_length_delimited_field(7, &rewritten, &mut result);
    result.extend_from_slice(&model[graph.end..]);
    Ok(result)
}

/// GraphProto in, GraphProto out. Graphs with nested subgraphs are left alone:
/// their implicit outer-scope captures need a more general liveness analysis.
pub(super) fn rewrite(graph: &[u8]) -> anyhow::Result<Vec<u8>> {
    let fields = parse_proto_fields(graph)?;
    let mut nodes = Vec::new();
    let mut live = HashSet::new();
    let mut names = HashSet::new();
    for field in &fields {
        let bytes = &graph[field.data_start..field.data_end];
        if field.number == 1 {
            let nf = parse_proto_fields(bytes)?;
            for attribute in nf.iter().filter(|f| f.number == 5) {
                let af = parse_proto_fields(&bytes[attribute.data_start..attribute.data_end])?;
                if af.iter().any(|f| f.number == 6 || f.number == 11) {
                    return Ok(graph.to_vec());
                }
            }
            let inputs = strings(bytes, &nf, 1)?;
            let outputs = strings(bytes, &nf, 2)?;
            names.extend(inputs.iter().chain(&outputs).cloned());
            nodes.push(Node {
                field: *field,
                bytes,
                fields: nf,
                inputs,
                outputs,
            });
        } else if field.number == 12 {
            live.extend(strings(bytes, &parse_proto_fields(bytes)?, 1)?);
        } else if field.number == 5 {
            names.extend(strings(bytes, &parse_proto_fields(bytes)?, 8)?);
        } else if field.number == 11 {
            names.extend(strings(bytes, &parse_proto_fields(bytes)?, 1)?);
        }
    }
    let producers: HashMap<_, _> = nodes
        .iter()
        .enumerate()
        .flat_map(|(i, n)| n.outputs.iter().map(move |o| (o.as_str(), i)))
        .collect();
    let mut stack: Vec<_> = live.iter().cloned().collect();
    let mut needed = HashSet::new();
    while let Some(name) = stack.pop() {
        if let Some(&i) = producers.get(name.as_str()) {
            if needed.insert(i) {
                for input in &nodes[i].inputs {
                    if !input.is_empty() && live.insert(input.clone()) {
                        stack.push(input.clone());
                    }
                }
            }
        }
    }
    let mut replacements = HashMap::new();
    let mut initializers = Vec::new();
    for (i, node) in nodes.iter().enumerate().filter(|(i, _)| needed.contains(i)) {
        if strings(node.bytes, &node.fields, 4)? != ["Split"]
            || strings(node.bytes, &node.fields, 7)?
                .iter()
                .any(|d| !d.is_empty())
            || node.inputs.len() != 1
        {
            continue;
        }
        let used: Vec<_> = node
            .outputs
            .iter()
            .enumerate()
            .filter(|(_, o)| live.contains(*o))
            .collect();
        if used.len() != 1 {
            continue;
        }
        let (selected, output) = used[0];
        let mut axis = 0;
        let mut splits = Vec::new();
        for f in node.fields.iter().filter(|f| f.number == 5) {
            let attr = &node.bytes[f.data_start..f.data_end];
            let af = parse_proto_fields(attr)?;
            let name = strings(attr, &af, 1)?;
            if name == ["axis"] {
                let f = af
                    .iter()
                    .find(|f| f.number == 3)
                    .context("missing Split axis")?;
                axis = read_varint(attr, f.data_start)?.0;
            } else if name == ["split"] {
                for f in af.iter().filter(|f| f.number == 8) {
                    if f.wire_type == 0 {
                        splits.push(read_varint(attr, f.data_start)?.0);
                    } else if f.wire_type == 2 {
                        let mut pos = f.data_start;
                        while pos < f.data_end {
                            let (v, end) = read_varint(attr, pos)?;
                            ensure!(end <= f.data_end, "invalid packed Split lengths");
                            splits.push(v);
                            pos = end;
                        }
                    }
                }
            }
        }
        if splits.len() != node.outputs.len()
            || splits[selected] != 1
            || splits.iter().any(|v| *v == 0 || *v > i64::MAX as u64)
        {
            continue;
        }
        let index = splits[..selected]
            .iter()
            .try_fold(0u64, |a, b| a.checked_add(*b))
            .filter(|v| *v <= i64::MAX as u64)
            .context("Split index overflow")?;
        let mut suffix = i;
        let index_name = loop {
            let name = format!("__segmenter_alpha_index_{suffix}");
            if names.insert(name.clone()) {
                break name;
            }
            suffix += 1;
        };
        let mut tensor = Vec::new();
        encode_varint_field(1, 1, &mut tensor);
        encode_varint_field(2, 7, &mut tensor); // INT64
        encode_string_field(8, &index_name, &mut tensor);
        encode_length_delimited_field(9, &(index as i64).to_le_bytes(), &mut tensor);
        encode_length_delimited_field(5, &tensor, &mut initializers);
        let mut replacement = Vec::new();
        encode_string_field(1, &node.inputs[0], &mut replacement);
        encode_string_field(1, &index_name, &mut replacement);
        encode_string_field(2, output, &mut replacement);
        for f in node.fields.iter().filter(|f| f.number == 3) {
            replacement.extend_from_slice(&node.bytes[f.start..f.end]);
        }
        encode_string_field(4, "Gather", &mut replacement);
        let mut attribute = Vec::new();
        encode_string_field(1, "axis", &mut attribute);
        encode_varint_field(3, axis, &mut attribute);
        encode_varint_field(20, 2, &mut attribute); // INT
        encode_length_delimited_field(5, &attribute, &mut replacement);
        replacements.insert(node.field.start, replacement);
    }
    if replacements.is_empty() {
        return Ok(graph.to_vec());
    }
    let dead: HashSet<_> = nodes
        .iter()
        .enumerate()
        .filter(|(i, _)| !needed.contains(i))
        .map(|(_, n)| n.field.start)
        .collect();
    let mut result = Vec::with_capacity(graph.len());
    for f in fields {
        if f.number == 1 && dead.contains(&f.start) {
            continue;
        }
        if let Some(bytes) = replacements.get(&f.start) {
            encode_length_delimited_field(1, bytes, &mut result);
        } else {
            result.extend_from_slice(&graph[f.start..f.end]);
        }
    }
    result.extend_from_slice(&initializers);
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn graph(packed: bool, foreground_output: bool, nested: bool) -> Vec<u8> {
        let mut node = Vec::new();
        encode_string_field(1, "x", &mut node);
        encode_string_field(2, "color", &mut node);
        encode_string_field(2, "alpha", &mut node);
        encode_string_field(4, "Split", &mut node);
        let mut attr = Vec::new();
        encode_string_field(1, "split", &mut attr);
        if packed {
            encode_length_delimited_field(8, &[3, 1], &mut attr);
        } else {
            encode_varint_field(8, 3, &mut attr);
            encode_varint_field(8, 1, &mut attr);
        }
        encode_length_delimited_field(5, &attr, &mut node);
        let mut attr = Vec::new();
        encode_string_field(1, "axis", &mut attr);
        encode_varint_field(3, 1, &mut attr);
        if nested {
            encode_length_delimited_field(6, &[], &mut attr);
        }
        encode_length_delimited_field(5, &attr, &mut node);
        let mut graph = Vec::new();
        encode_length_delimited_field(1, &node, &mut graph);
        let mut dead_node = Vec::new();
        encode_string_field(1, "color", &mut dead_node);
        encode_string_field(2, "fgr", &mut dead_node);
        encode_string_field(4, "Identity", &mut dead_node);
        encode_length_delimited_field(1, &dead_node, &mut graph);
        let mut output = Vec::new();
        encode_string_field(1, "alpha", &mut output);
        encode_length_delimited_field(12, &output, &mut graph);
        if foreground_output {
            let mut output = Vec::new();
            encode_string_field(1, "fgr", &mut output);
            encode_length_delimited_field(12, &output, &mut graph);
        }
        graph
    }

    #[test]
    fn sole_alpha_output_becomes_dimension_preserving_gather() {
        for packed in [false, true] {
            let result = rewrite(&graph(packed, false, false)).unwrap();
            let fields = parse_proto_fields(&result).unwrap();
            assert_eq!(fields.iter().filter(|f| f.number == 1).count(), 1);
            let n = fields.iter().find(|f| f.number == 1).unwrap();
            let node = &result[n.data_start..n.data_end];
            let nf = parse_proto_fields(node).unwrap();
            assert_eq!(strings(node, &nf, 4).unwrap(), ["Gather"]);
            assert_eq!(strings(node, &nf, 2).unwrap(), ["alpha"]);
            let t = fields.iter().find(|f| f.number == 5).unwrap();
            let tensor = &result[t.data_start..t.data_end];
            let tf = parse_proto_fields(tensor).unwrap();
            let data = tf.iter().find(|f| f.number == 9).unwrap();
            assert_eq!(
                &tensor[data.data_start..data.data_end],
                &3_i64.to_le_bytes()
            );
            assert_eq!(
                read_varint(
                    tensor,
                    tf.iter().find(|f| f.number == 1).unwrap().data_start
                )
                .unwrap()
                .0,
                1
            );
        }
    }

    #[test]
    fn preserves_split_when_both_outputs_are_live() {
        let model = graph(true, true, false);
        assert_eq!(rewrite(&model).unwrap(), model);
    }

    #[test]
    fn leaves_subgraphs_with_implicit_captures_untouched() {
        let model = graph(true, false, true);
        assert_eq!(rewrite(&model).unwrap(), model);
    }
}
