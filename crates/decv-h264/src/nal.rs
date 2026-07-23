//! NAL-header parsing and NAL-unit classification.

use crate::{AnnexBNalUnit, H264Error, Result};

/// Semantically meaningful H.264 NAL-unit types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum NalUnitType {
    NonIdrSlice,
    SliceDataPartitionA,
    SliceDataPartitionB,
    SliceDataPartitionC,
    IdrSlice,
    Sei,
    Sps,
    Pps,
    AccessUnitDelimiter,
    EndOfSequence,
    EndOfStream,
    FillerData,
    SpsExtension,
    Prefix,
    SubsetSps,
    AuxiliaryCodedPicture,
    Extension,
    DepthExtension,
    Unknown(u8),
}

impl NalUnitType {
    #[inline]
    pub const fn from_value(value: u8) -> Self {
        match value {
            1 => Self::NonIdrSlice,
            2 => Self::SliceDataPartitionA,
            3 => Self::SliceDataPartitionB,
            4 => Self::SliceDataPartitionC,
            5 => Self::IdrSlice,
            6 => Self::Sei,
            7 => Self::Sps,
            8 => Self::Pps,
            9 => Self::AccessUnitDelimiter,
            10 => Self::EndOfSequence,
            11 => Self::EndOfStream,
            12 => Self::FillerData,
            13 => Self::SpsExtension,
            14 => Self::Prefix,
            15 => Self::SubsetSps,
            19 => Self::AuxiliaryCodedPicture,
            20 => Self::Extension,
            21 => Self::DepthExtension,
            value => Self::Unknown(value),
        }
    }

    #[inline]
    pub const fn value(self) -> u8 {
        match self {
            Self::NonIdrSlice => 1,
            Self::SliceDataPartitionA => 2,
            Self::SliceDataPartitionB => 3,
            Self::SliceDataPartitionC => 4,
            Self::IdrSlice => 5,
            Self::Sei => 6,
            Self::Sps => 7,
            Self::Pps => 8,
            Self::AccessUnitDelimiter => 9,
            Self::EndOfSequence => 10,
            Self::EndOfStream => 11,
            Self::FillerData => 12,
            Self::SpsExtension => 13,
            Self::Prefix => 14,
            Self::SubsetSps => 15,
            Self::AuxiliaryCodedPicture => 19,
            Self::Extension => 20,
            Self::DepthExtension => 21,
            Self::Unknown(value) => value,
        }
    }

    #[inline]
    pub const fn is_vcl(self) -> bool {
        matches!(
            self,
            Self::NonIdrSlice
                | Self::SliceDataPartitionA
                | Self::SliceDataPartitionB
                | Self::SliceDataPartitionC
                | Self::IdrSlice
                | Self::AuxiliaryCodedPicture
                | Self::Extension
                | Self::DepthExtension
        )
    }
}

/// The one-byte header shared by ordinary H.264 NAL units.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NalHeader {
    pub nal_ref_idc: u8,
    pub unit_type: NalUnitType,
}

impl NalHeader {
    #[inline]
    pub fn parse(byte: u8) -> Result<Self> {
        if byte & 0x80 != 0 {
            return Err(H264Error::InvalidNalHeader);
        }

        Ok(Self {
            nal_ref_idc: (byte >> 5) & 0b11,
            unit_type: NalUnitType::from_value(byte & 0x1f),
        })
    }
}

/// A parsed NAL header and its borrowed EBSP payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NalUnit<'a> {
    pub header: NalHeader,
    pub ebsp: &'a [u8],
    pub stream_offset: usize,
}

impl<'a> TryFrom<AnnexBNalUnit<'a>> for NalUnit<'a> {
    type Error = H264Error;

    fn try_from(unit: AnnexBNalUnit<'a>) -> Result<Self> {
        let (&header, ebsp) = unit
            .bytes()
            .split_first()
            .ok_or(H264Error::InvalidNalHeader)?;

        Ok(Self {
            header: NalHeader::parse(header)?,
            ebsp,
            stream_offset: unit.stream_offset(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{NalHeader, NalUnit, NalUnitType};
    use crate::{AnnexBReader, H264Error};

    #[test]
    fn parses_common_nal_headers() {
        assert_eq!(
            NalHeader::parse(0x67),
            Ok(NalHeader {
                nal_ref_idc: 3,
                unit_type: NalUnitType::Sps,
            })
        );
        assert_eq!(
            NalHeader::parse(0x65),
            Ok(NalHeader {
                nal_ref_idc: 3,
                unit_type: NalUnitType::IdrSlice,
            })
        );
        assert!(NalUnitType::IdrSlice.is_vcl());
        assert!(!NalUnitType::Sps.is_vcl());
    }

    #[test]
    fn rejects_a_set_forbidden_zero_bit() {
        assert_eq!(NalHeader::parse(0xe7), Err(H264Error::InvalidNalHeader));
    }

    #[test]
    fn retains_unknown_type_values() {
        let unit_type = NalUnitType::from_value(31);

        assert_eq!(unit_type, NalUnitType::Unknown(31));
        assert_eq!(unit_type.value(), 31);
    }

    #[test]
    fn separates_the_header_from_ebsp() {
        let data = [0, 0, 1, 0x67, 0x42, 0x00, 0x1f];
        let annex_b_unit = AnnexBReader::new(&data).next().unwrap().unwrap();
        let nal = NalUnit::try_from(annex_b_unit).unwrap();

        assert_eq!(nal.header.unit_type, NalUnitType::Sps);
        assert_eq!(nal.ebsp, &[0x42, 0x00, 0x1f]);
        assert_eq!(nal.stream_offset, 3);
    }
}
