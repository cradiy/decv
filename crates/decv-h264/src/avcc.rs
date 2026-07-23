//! AVCDecoderConfigurationRecord and length-prefixed NAL parsing.

use crate::{H264Error, NalHeader, NalUnit, NalUnitType, Result};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AvcDecoderConfiguration {
    pub length_size: u8,
    pub parameter_sets: Vec<Vec<u8>>,
}

pub(crate) fn parse_avcc(data: &[u8]) -> Result<AvcDecoderConfiguration> {
    if data.len() < 7 || data[0] != 1 {
        return Err(H264Error::InvalidSyntax(
            "invalid AVCDecoderConfigurationRecord header",
        ));
    }
    if data[4] & 0xfc != 0xfc || data[5] & 0xe0 != 0xe0 {
        return Err(H264Error::InvalidSyntax(
            "avcC reserved bits do not have their required values",
        ));
    }
    let length_size = (data[4] & 3) + 1;
    let mut offset = 6usize;
    let sps_count = usize::from(data[5] & 0x1f);
    if sps_count == 0 {
        return Err(H264Error::InvalidSyntax("avcC contains no SPS"));
    }
    let mut parameter_sets = Vec::new();
    parse_parameter_sets(
        data,
        &mut offset,
        sps_count,
        NalUnitType::Sps,
        &mut parameter_sets,
    )?;
    let pps_count = usize::from(*data.get(offset).ok_or(H264Error::UnexpectedEof)?);
    offset += 1;
    if pps_count == 0 {
        return Err(H264Error::InvalidSyntax("avcC contains no PPS"));
    }
    parse_parameter_sets(
        data,
        &mut offset,
        pps_count,
        NalUnitType::Pps,
        &mut parameter_sets,
    )?;
    // High-profile avcC records may append chroma/bit-depth and SPS-ext
    // fields. The SPS already carries the normative values needed here.
    Ok(AvcDecoderConfiguration {
        length_size,
        parameter_sets,
    })
}

fn parse_parameter_sets(
    data: &[u8],
    offset: &mut usize,
    count: usize,
    expected_type: NalUnitType,
    output: &mut Vec<Vec<u8>>,
) -> Result<()> {
    for _ in 0..count {
        let length_bytes = data
            .get(*offset..offset.checked_add(2).ok_or(H264Error::IntegerOverflow)?)
            .ok_or(H264Error::UnexpectedEof)?;
        *offset += 2;
        let length = usize::from(u16::from_be_bytes([length_bytes[0], length_bytes[1]]));
        if length == 0 {
            return Err(H264Error::InvalidSyntax(
                "avcC parameter-set NAL must not be empty",
            ));
        }
        let end = offset
            .checked_add(length)
            .ok_or(H264Error::IntegerOverflow)?;
        let nal = data.get(*offset..end).ok_or(H264Error::UnexpectedEof)?;
        let header = NalHeader::parse(nal[0])?;
        if header.unit_type != expected_type {
            return Err(H264Error::InvalidSyntax(
                "avcC parameter-set array contains the wrong NAL type",
            ));
        }
        output.push(nal.to_vec());
        *offset = end;
    }
    Ok(())
}

#[derive(Debug, Clone)]
pub(crate) struct LengthPrefixedNalReader<'a> {
    data: &'a [u8],
    length_size: usize,
    offset: usize,
}

impl<'a> LengthPrefixedNalReader<'a> {
    pub(crate) fn new(data: &'a [u8], length_size: u8) -> Self {
        Self {
            data,
            length_size: usize::from(length_size),
            offset: 0,
        }
    }
}

impl<'a> Iterator for LengthPrefixedNalReader<'a> {
    type Item = Result<NalUnit<'a>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.offset == self.data.len() {
            return None;
        }
        let prefix_end = match self.offset.checked_add(self.length_size) {
            Some(end) if end <= self.data.len() => end,
            _ => {
                self.offset = self.data.len();
                return Some(Err(H264Error::UnexpectedEof));
            }
        };
        let mut length = 0usize;
        for &byte in &self.data[self.offset..prefix_end] {
            length = (length << 8) | usize::from(byte);
        }
        let nal_offset = prefix_end;
        let nal_end = match nal_offset.checked_add(length) {
            Some(end) if length != 0 && end <= self.data.len() => end,
            Some(_) if length == 0 => {
                self.offset = self.data.len();
                return Some(Err(H264Error::InvalidSyntax(
                    "length-prefixed NAL unit must not be empty",
                )));
            }
            _ => {
                self.offset = self.data.len();
                return Some(Err(H264Error::UnexpectedEof));
            }
        };
        self.offset = nal_end;
        let nal = &self.data[nal_offset..nal_end];
        let result = NalHeader::parse(nal[0]).map(|header| NalUnit {
            header,
            ebsp: &nal[1..],
            stream_offset: nal_offset,
        });
        Some(result)
    }
}

#[cfg(test)]
mod tests {
    use super::{LengthPrefixedNalReader, parse_avcc};
    use crate::{H264Error, NalUnitType};

    #[test]
    fn parses_avcc_parameter_sets_and_length_size() {
        let data = [
            1, 100, 0, 40, 0xff, 0xe1, 0, 3, 0x67, 1, 2, 1, 0, 2, 0x68, 3,
        ];
        let config = parse_avcc(&data).unwrap();
        assert_eq!(config.length_size, 4);
        assert_eq!(config.parameter_sets, [vec![0x67, 1, 2], vec![0x68, 3]]);
    }

    #[test]
    fn splits_big_endian_length_prefixed_nals() {
        let data = [0, 2, 0x67, 1, 0, 3, 0x68, 2, 3];
        let units = LengthPrefixedNalReader::new(&data, 2)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(units[0].header.unit_type, NalUnitType::Sps);
        assert_eq!(units[0].ebsp, [1]);
        assert_eq!(units[1].header.unit_type, NalUnitType::Pps);
        assert_eq!(units[1].ebsp, [2, 3]);
    }

    #[test]
    fn rejects_empty_and_truncated_length_prefixed_nals() {
        assert!(matches!(
            LengthPrefixedNalReader::new(&[0, 0], 2).next(),
            Some(Err(H264Error::InvalidSyntax(_)))
        ));
        assert!(matches!(
            LengthPrefixedNalReader::new(&[0, 3, 0x67], 2).next(),
            Some(Err(H264Error::UnexpectedEof))
        ));
    }
}
