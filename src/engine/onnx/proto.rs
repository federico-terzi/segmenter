use anyhow::{bail, Context};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ProtoField {
    pub(super) number: u32,
    pub(super) wire_type: u8,
    pub(super) start: usize,
    pub(super) data_start: usize,
    pub(super) data_end: usize,
    pub(super) end: usize,
}

pub(super) fn parse_proto_fields(bytes: &[u8]) -> anyhow::Result<Vec<ProtoField>> {
    let mut fields = Vec::new();
    let mut position = 0;
    while position < bytes.len() {
        let start = position;
        let (key, key_end) = read_varint(bytes, position)?;
        position = key_end;

        let number = (key >> 3) as u32;
        let wire_type = (key & 0x07) as u8;
        if number == 0 {
            bail!("protobuf field number must be non-zero");
        }

        let (data_start, data_end, end) = match wire_type {
            0 => {
                let (_, end) = read_varint(bytes, position)?;
                (position, end, end)
            }
            1 => {
                let end = position
                    .checked_add(8)
                    .context("protobuf fixed64 field offset overflowed")?;
                if end > bytes.len() {
                    bail!("protobuf fixed64 field exceeded message length");
                }
                (position, end, end)
            }
            2 => {
                let (len, data_start) = read_varint(bytes, position)?;
                let len = usize::try_from(len)
                    .context("protobuf length-delimited field was too large")?;
                let data_end = data_start
                    .checked_add(len)
                    .context("protobuf length-delimited field offset overflowed")?;
                if data_end > bytes.len() {
                    bail!("protobuf length-delimited field exceeded message length");
                }
                (data_start, data_end, data_end)
            }
            5 => {
                let end = position
                    .checked_add(4)
                    .context("protobuf fixed32 field offset overflowed")?;
                if end > bytes.len() {
                    bail!("protobuf fixed32 field exceeded message length");
                }
                (position, end, end)
            }
            wire_type => bail!("unsupported protobuf wire type {wire_type}"),
        };

        fields.push(ProtoField {
            number,
            wire_type,
            start,
            data_start,
            data_end,
            end,
        });
        position = end;
    }
    Ok(fields)
}

pub(super) fn read_varint(bytes: &[u8], mut position: usize) -> anyhow::Result<(u64, usize)> {
    let mut value = 0_u64;
    for shift in (0..64).step_by(7) {
        if position >= bytes.len() {
            bail!("protobuf varint ended unexpectedly");
        }
        let byte = bytes[position];
        position += 1;
        value |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return Ok((value, position));
        }
    }
    bail!("protobuf varint exceeded 64 bits")
}

pub(super) fn read_proto_string(bytes: &[u8], field: ProtoField) -> anyhow::Result<String> {
    if field.wire_type != 2 {
        bail!("protobuf field {} was not length-delimited", field.number);
    }
    std::str::from_utf8(&bytes[field.data_start..field.data_end])
        .with_context(|| format!("protobuf field {} was not valid UTF-8", field.number))
        .map(str::to_owned)
}

pub(super) fn encode_length_delimited_field(number: u32, payload: &[u8], out: &mut Vec<u8>) {
    encode_key(number, 2, out);
    encode_varint(payload.len() as u64, out);
    out.extend_from_slice(payload);
}

pub(super) fn encode_string_field(number: u32, value: &str, out: &mut Vec<u8>) {
    encode_length_delimited_field(number, value.as_bytes(), out);
}

pub(super) fn encode_varint_field(number: u32, value: u64, out: &mut Vec<u8>) {
    encode_key(number, 0, out);
    encode_varint(value, out);
}

fn encode_key(number: u32, wire_type: u8, out: &mut Vec<u8>) {
    encode_varint((u64::from(number) << 3) | u64::from(wire_type), out);
}

fn encode_varint(mut value: u64, out: &mut Vec<u8>) {
    while value >= 0x80 {
        out.push((value as u8 & 0x7f) | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}
