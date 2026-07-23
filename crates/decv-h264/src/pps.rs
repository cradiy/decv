//! Picture Parameter Set syntax, slice-group maps, and validation.

use bit_readers::BitReader;

use crate::{
    H264Error, Result, ScalingMatrices, SequenceParameterSet, consume_rbsp_trailing_bits,
    rbsp::more_rbsp_data, sps::parse_scaling_matrices,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntropyCodingMode {
    Cavlc,
    Cabac,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightedBiprediction {
    Default,
    Explicit,
    Implicit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SliceGroupRectangle {
    pub top_left: u32,
    pub bottom_right: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SliceGroupMap {
    /// Map type 0.
    Interleaved { run_lengths: Vec<u32> },
    /// Map type 1. The map is derived from macroblock coordinates.
    Dispersed,
    /// Map type 2.
    Foreground {
        rectangles: Vec<SliceGroupRectangle>,
    },
    /// Map types 3, 4, and 5.
    Changing {
        map_type: u8,
        change_direction: bool,
        change_rate: u32,
    },
    /// Map type 6.
    Explicit { slice_group_ids: Vec<u8> },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PictureParameterSet {
    pub id: u32,
    pub sequence_parameter_set_id: u32,
    pub entropy_coding_mode: EntropyCodingMode,
    pub bottom_field_pic_order_in_frame_present: bool,

    pub num_slice_groups: u8,
    pub slice_group_map: Option<SliceGroupMap>,

    pub num_ref_idx_l0_default_active: u8,
    pub num_ref_idx_l1_default_active: u8,
    pub weighted_prediction: bool,
    pub weighted_biprediction: WeightedBiprediction,

    pub pic_init_qp: i8,
    pub pic_init_qs: i8,
    pub chroma_qp_index_offset: i8,
    pub deblocking_filter_control_present: bool,
    pub constrained_intra_prediction: bool,
    pub redundant_pic_count_present: bool,

    pub transform_8x8_mode: bool,
    pub scaling_matrices: Option<ScalingMatrices>,
    pub second_chroma_qp_index_offset: i8,
}

impl PictureParameterSet {
    /// Parses one unescaped PPS RBSP against the SPS it references.
    ///
    /// Passing the referenced SPS allows FMO map sizes and profile-dependent
    /// extension syntax to be validated before allocating storage.
    pub fn parse(rbsp: &[u8], sps: &SequenceParameterSet) -> Result<Self> {
        let mut reader = BitReader::new(rbsp);

        let id = read_ue(&mut reader)?;
        if id > 255 {
            return Err(H264Error::InvalidSyntax("pic_parameter_set_id exceeds 255"));
        }

        let sequence_parameter_set_id = read_ue(&mut reader)?;
        if sequence_parameter_set_id > 31 {
            return Err(H264Error::InvalidSyntax(
                "seq_parameter_set_id in PPS exceeds 31",
            ));
        }
        if sequence_parameter_set_id != sps.id {
            return Err(H264Error::MissingSps(sequence_parameter_set_id));
        }

        let entropy_coding_mode = if read_flag(&mut reader)? {
            EntropyCodingMode::Cabac
        } else {
            EntropyCodingMode::Cavlc
        };
        let bottom_field_pic_order_in_frame_present = read_flag(&mut reader)?;

        let num_slice_groups_minus1 = read_ue(&mut reader)?;
        if num_slice_groups_minus1 > 7 {
            return Err(H264Error::InvalidSyntax(
                "num_slice_groups_minus1 exceeds 7",
            ));
        }
        let num_slice_groups = (num_slice_groups_minus1 + 1) as u8;
        let pic_size_in_map_units = sps
            .pic_width_in_mbs
            .checked_mul(sps.pic_height_in_map_units)
            .ok_or(H264Error::IntegerOverflow)?;
        let slice_group_map = if num_slice_groups_minus1 == 0 {
            None
        } else {
            Some(parse_slice_group_map(
                &mut reader,
                num_slice_groups,
                pic_size_in_map_units,
                sps.pic_width_in_mbs,
            )?)
        };

        let num_ref_idx_l0_default_active = parse_reference_count(&mut reader)?;
        let num_ref_idx_l1_default_active = parse_reference_count(&mut reader)?;
        let weighted_prediction = read_flag(&mut reader)?;
        let weighted_biprediction = match read_bits::<2>(&mut reader)? {
            0 => WeightedBiprediction::Default,
            1 => WeightedBiprediction::Explicit,
            2 => WeightedBiprediction::Implicit,
            _ => {
                return Err(H264Error::InvalidSyntax(
                    "weighted_bipred_idc has reserved value 3",
                ));
            }
        };

        let pic_init_qp = parse_initial_qp(read_se(&mut reader)?, "pic_init_qp_minus26")?;
        let pic_init_qs = parse_initial_qp(read_se(&mut reader)?, "pic_init_qs_minus26")?;
        let chroma_qp_index_offset =
            parse_chroma_qp_offset(read_se(&mut reader)?, "chroma_qp_index_offset")?;
        let deblocking_filter_control_present = read_flag(&mut reader)?;
        let constrained_intra_prediction = read_flag(&mut reader)?;
        let redundant_pic_count_present = read_flag(&mut reader)?;

        let mut transform_8x8_mode = false;
        let mut scaling_matrices = None;
        let mut second_chroma_qp_index_offset = chroma_qp_index_offset;

        if more_rbsp_data(&reader) {
            transform_8x8_mode = read_flag(&mut reader)?;
            if read_flag(&mut reader)? {
                let list_count = if transform_8x8_mode { 8 } else { 6 };
                scaling_matrices = Some(parse_scaling_matrices(&mut reader, list_count)?);
            }
            second_chroma_qp_index_offset =
                parse_chroma_qp_offset(read_se(&mut reader)?, "second_chroma_qp_index_offset")?;
        }

        consume_rbsp_trailing_bits(&mut reader)?;

        Ok(Self {
            id,
            sequence_parameter_set_id,
            entropy_coding_mode,
            bottom_field_pic_order_in_frame_present,
            num_slice_groups,
            slice_group_map,
            num_ref_idx_l0_default_active,
            num_ref_idx_l1_default_active,
            weighted_prediction,
            weighted_biprediction,
            pic_init_qp,
            pic_init_qs,
            chroma_qp_index_offset,
            deblocking_filter_control_present,
            constrained_intra_prediction,
            redundant_pic_count_present,
            transform_8x8_mode,
            scaling_matrices,
            second_chroma_qp_index_offset,
        })
    }
}

fn parse_slice_group_map(
    reader: &mut BitReader<'_>,
    num_slice_groups: u8,
    pic_size_in_map_units: u32,
    pic_width_in_mbs: u32,
) -> Result<SliceGroupMap> {
    let map_type = read_ue(reader)?;
    match map_type {
        0 => {
            let mut run_lengths = Vec::with_capacity(num_slice_groups as usize);
            for _ in 0..num_slice_groups {
                let run_length = read_ue(reader)?
                    .checked_add(1)
                    .ok_or(H264Error::IntegerOverflow)?;
                if run_length > pic_size_in_map_units {
                    return Err(H264Error::InvalidSyntax(
                        "slice-group run length exceeds picture size",
                    ));
                }
                run_lengths.push(run_length);
            }
            Ok(SliceGroupMap::Interleaved { run_lengths })
        }
        1 => Ok(SliceGroupMap::Dispersed),
        2 => {
            let mut rectangles = Vec::with_capacity(num_slice_groups as usize - 1);
            for _ in 0..num_slice_groups - 1 {
                let rectangle = SliceGroupRectangle {
                    top_left: read_ue(reader)?,
                    bottom_right: read_ue(reader)?,
                };
                validate_slice_group_rectangle(rectangle, pic_size_in_map_units, pic_width_in_mbs)?;
                rectangles.push(rectangle);
            }
            Ok(SliceGroupMap::Foreground { rectangles })
        }
        3..=5 => {
            if num_slice_groups != 2 {
                return Err(H264Error::InvalidSyntax(
                    "slice-group map types 3..=5 require exactly two groups",
                ));
            }
            let change_direction = read_flag(reader)?;
            let change_rate = read_ue(reader)?
                .checked_add(1)
                .ok_or(H264Error::IntegerOverflow)?;
            if change_rate > pic_size_in_map_units {
                return Err(H264Error::InvalidSyntax(
                    "slice_group_change_rate exceeds picture size",
                ));
            }
            Ok(SliceGroupMap::Changing {
                map_type: map_type as u8,
                change_direction,
                change_rate,
            })
        }
        6 => {
            let signalled_pic_size = read_ue(reader)?
                .checked_add(1)
                .ok_or(H264Error::IntegerOverflow)?;
            if signalled_pic_size != pic_size_in_map_units {
                return Err(H264Error::InvalidSyntax(
                    "explicit slice-group map size does not match SPS",
                ));
            }

            let id_bits = u32::from(num_slice_groups).ilog2()
                + u32::from(!num_slice_groups.is_power_of_two());
            let map_len =
                usize::try_from(signalled_pic_size).map_err(|_| H264Error::IntegerOverflow)?;
            let mut slice_group_ids = Vec::new();
            slice_group_ids
                .try_reserve_exact(map_len)
                .map_err(|_| H264Error::InvalidSyntax("explicit slice-group map is too large"))?;
            for _ in 0..signalled_pic_size {
                let group_id = reader.read_bits(id_bits).ok_or(H264Error::UnexpectedEof)?;
                if group_id >= u32::from(num_slice_groups) {
                    return Err(H264Error::InvalidSyntax(
                        "slice_group_id exceeds num_slice_groups",
                    ));
                }
                slice_group_ids.push(group_id as u8);
            }
            Ok(SliceGroupMap::Explicit { slice_group_ids })
        }
        _ => Err(H264Error::InvalidSyntax(
            "slice_group_map_type must be in 0..=6",
        )),
    }
}

fn validate_slice_group_rectangle(
    rectangle: SliceGroupRectangle,
    pic_size_in_map_units: u32,
    pic_width_in_mbs: u32,
) -> Result<()> {
    if rectangle.top_left >= pic_size_in_map_units
        || rectangle.bottom_right >= pic_size_in_map_units
        || rectangle.top_left > rectangle.bottom_right
        || rectangle.top_left % pic_width_in_mbs > rectangle.bottom_right % pic_width_in_mbs
    {
        return Err(H264Error::InvalidSyntax(
            "invalid foreground slice-group rectangle",
        ));
    }
    Ok(())
}

fn parse_reference_count(reader: &mut BitReader<'_>) -> Result<u8> {
    let minus1 = read_ue(reader)?;
    if minus1 > 31 {
        return Err(H264Error::InvalidSyntax(
            "num_ref_idx_default_active_minus1 exceeds 31",
        ));
    }
    Ok((minus1 + 1) as u8)
}

fn parse_initial_qp(minus26: i32, field: &'static str) -> Result<i8> {
    if !(-26..=25).contains(&minus26) {
        return Err(H264Error::InvalidSyntax(field));
    }
    i8::try_from(minus26 + 26).map_err(|_| H264Error::IntegerOverflow)
}

fn parse_chroma_qp_offset(offset: i32, field: &'static str) -> Result<i8> {
    if !(-12..=12).contains(&offset) {
        return Err(H264Error::InvalidSyntax(field));
    }
    i8::try_from(offset).map_err(|_| H264Error::IntegerOverflow)
}

#[inline]
fn read_flag(reader: &mut BitReader<'_>) -> Result<bool> {
    reader
        .read_bit()
        .map(|value| value != 0)
        .ok_or(H264Error::UnexpectedEof)
}

#[inline]
fn read_bits<const COUNT: u32>(reader: &mut BitReader<'_>) -> Result<u32> {
    reader
        .read_bits_const::<COUNT>()
        .ok_or(H264Error::UnexpectedEof)
}

#[inline]
fn read_ue(reader: &mut BitReader<'_>) -> Result<u32> {
    reader.read_ue().ok_or(H264Error::UnexpectedEof)
}

#[inline]
fn read_se(reader: &mut BitReader<'_>) -> Result<i32> {
    reader.read_se().ok_or(H264Error::UnexpectedEof)
}

#[cfg(test)]
mod tests {
    use super::{
        EntropyCodingMode, PictureParameterSet, SliceGroupMap, SliceGroupRectangle,
        WeightedBiprediction,
    };
    use crate::{H264Error, SequenceParameterSet};

    #[test]
    fn parses_a_basic_cavlc_pps_and_its_defaults() {
        let sps = test_sps(4, 3);
        let mut writer = BitWriter::default();
        write_pps_prefix(&mut writer, 0, 0, false, false, 0);
        write_pps_suffix(&mut writer, 0, 1, true, 0, -2, true, true, false);

        let pps = PictureParameterSet::parse(&writer.finish_rbsp(), &sps).unwrap();

        assert_eq!(pps.id, 0);
        assert_eq!(pps.entropy_coding_mode, EntropyCodingMode::Cavlc);
        assert_eq!(pps.num_slice_groups, 1);
        assert_eq!(pps.slice_group_map, None);
        assert_eq!(pps.num_ref_idx_l0_default_active, 1);
        assert_eq!(pps.num_ref_idx_l1_default_active, 2);
        assert!(pps.weighted_prediction);
        assert_eq!(pps.weighted_biprediction, WeightedBiprediction::Default);
        assert_eq!(pps.pic_init_qp, 26);
        assert_eq!(pps.pic_init_qs, 27);
        assert_eq!(pps.chroma_qp_index_offset, -2);
        assert_eq!(pps.second_chroma_qp_index_offset, -2);
        assert!(!pps.transform_8x8_mode);
    }

    #[test]
    fn parses_all_algorithmic_slice_group_map_shapes() {
        let sps = test_sps(4, 3);

        let mut interleaved = BitWriter::default();
        write_pps_prefix(&mut interleaved, 1, 0, false, false, 2);
        interleaved.write_ue(0);
        interleaved.write_ue(1);
        interleaved.write_ue(2);
        interleaved.write_ue(3);
        write_pps_suffix(&mut interleaved, 0, 0, false, 0, 0, false, false, false);
        let parsed = PictureParameterSet::parse(&interleaved.finish_rbsp(), &sps).unwrap();
        assert_eq!(
            parsed.slice_group_map,
            Some(SliceGroupMap::Interleaved {
                run_lengths: vec![2, 3, 4]
            })
        );

        let mut dispersed = BitWriter::default();
        write_pps_prefix(&mut dispersed, 2, 0, false, false, 1);
        dispersed.write_ue(1);
        write_pps_suffix(&mut dispersed, 0, 0, false, 0, 0, false, false, false);
        let parsed = PictureParameterSet::parse(&dispersed.finish_rbsp(), &sps).unwrap();
        assert_eq!(parsed.slice_group_map, Some(SliceGroupMap::Dispersed));

        for map_type in 3..=5 {
            let mut changing = BitWriter::default();
            write_pps_prefix(&mut changing, 3 + map_type, 0, false, false, 1);
            changing.write_ue(map_type);
            changing.write_flag(true);
            changing.write_ue(3);
            write_pps_suffix(&mut changing, 0, 0, false, 0, 0, false, false, false);
            let parsed = PictureParameterSet::parse(&changing.finish_rbsp(), &sps).unwrap();
            assert_eq!(
                parsed.slice_group_map,
                Some(SliceGroupMap::Changing {
                    map_type: map_type as u8,
                    change_direction: true,
                    change_rate: 4
                })
            );
        }
    }

    #[test]
    fn parses_foreground_and_explicit_slice_group_maps() {
        let sps = test_sps(4, 3);

        let mut foreground = BitWriter::default();
        write_pps_prefix(&mut foreground, 7, 0, false, false, 2);
        foreground.write_ue(2);
        foreground.write_ue(0);
        foreground.write_ue(5);
        foreground.write_ue(6);
        foreground.write_ue(11);
        write_pps_suffix(&mut foreground, 0, 0, false, 0, 0, false, false, false);
        let parsed = PictureParameterSet::parse(&foreground.finish_rbsp(), &sps).unwrap();
        assert_eq!(
            parsed.slice_group_map,
            Some(SliceGroupMap::Foreground {
                rectangles: vec![
                    SliceGroupRectangle {
                        top_left: 0,
                        bottom_right: 5
                    },
                    SliceGroupRectangle {
                        top_left: 6,
                        bottom_right: 11
                    }
                ]
            })
        );

        let mut explicit = BitWriter::default();
        write_pps_prefix(&mut explicit, 8, 0, false, false, 2);
        explicit.write_ue(6);
        explicit.write_ue(11);
        let ids = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2];
        for id in ids {
            explicit.write_bits(id, 2);
        }
        write_pps_suffix(&mut explicit, 0, 0, false, 0, 0, false, false, false);
        let parsed = PictureParameterSet::parse(&explicit.finish_rbsp(), &sps).unwrap();
        assert_eq!(
            parsed.slice_group_map,
            Some(SliceGroupMap::Explicit {
                slice_group_ids: ids.map(|id| id as u8).to_vec()
            })
        );
    }

    #[test]
    fn parses_cabac_and_high_profile_pps_extensions() {
        let sps = test_sps(4, 3);
        let mut writer = BitWriter::default();
        write_pps_prefix(&mut writer, 9, 0, true, true, 0);
        write_pps_suffix(&mut writer, 2, 0, false, 2, 3, true, false, true);
        writer.write_flag(true); // transform_8x8_mode
        writer.write_flag(true); // scaling matrix present
        writer.write_flag(true); // first scaling list
        for _ in 0..16 {
            writer.write_se(0);
        }
        for _ in 1..8 {
            writer.write_flag(false);
        }
        writer.write_se(-3);

        let pps = PictureParameterSet::parse(&writer.finish_rbsp(), &sps).unwrap();

        assert_eq!(pps.entropy_coding_mode, EntropyCodingMode::Cabac);
        assert!(pps.bottom_field_pic_order_in_frame_present);
        assert_eq!(pps.weighted_biprediction, WeightedBiprediction::Implicit);
        assert_eq!(pps.pic_init_qp, 28);
        assert_eq!(pps.chroma_qp_index_offset, 3);
        assert!(pps.transform_8x8_mode);
        assert_eq!(pps.scaling_matrices.unwrap().lists.len(), 8);
        assert_eq!(pps.second_chroma_qp_index_offset, -3);
    }

    #[test]
    fn rejects_invalid_pps_values_before_allocating_maps() {
        let sps = test_sps(4, 3);

        let mut wrong_sps = BitWriter::default();
        write_pps_prefix(&mut wrong_sps, 0, 1, false, false, 0);
        assert_eq!(
            PictureParameterSet::parse(&wrong_sps.finish_rbsp(), &sps),
            Err(H264Error::MissingSps(1))
        );

        let mut too_many_groups = BitWriter::default();
        write_pps_prefix(&mut too_many_groups, 0, 0, false, false, 8);
        assert!(matches!(
            PictureParameterSet::parse(&too_many_groups.finish_rbsp(), &sps),
            Err(H264Error::InvalidSyntax(_))
        ));

        let mut wrong_map_size = BitWriter::default();
        write_pps_prefix(&mut wrong_map_size, 0, 0, false, false, 1);
        wrong_map_size.write_ue(6);
        wrong_map_size.write_ue(10);
        assert!(matches!(
            PictureParameterSet::parse(&wrong_map_size.finish_rbsp(), &sps),
            Err(H264Error::InvalidSyntax(_))
        ));

        let mut reserved_bipred = BitWriter::default();
        write_pps_prefix(&mut reserved_bipred, 0, 0, false, false, 0);
        write_pps_suffix(&mut reserved_bipred, 0, 0, false, 3, 0, false, false, false);
        assert!(matches!(
            PictureParameterSet::parse(&reserved_bipred.finish_rbsp(), &sps),
            Err(H264Error::InvalidSyntax(_))
        ));
    }

    fn test_sps(width_in_mbs: u32, height_in_map_units: u32) -> SequenceParameterSet {
        let mut writer = BitWriter::default();
        writer.write_bits(66, 8);
        writer.write_bits(0, 8);
        writer.write_bits(30, 8);
        writer.write_ue(0);
        writer.write_ue(0);
        writer.write_ue(0);
        writer.write_ue(0);
        writer.write_ue(1);
        writer.write_flag(false);
        writer.write_ue(width_in_mbs - 1);
        writer.write_ue(height_in_map_units - 1);
        writer.write_flag(true);
        writer.write_flag(true);
        writer.write_flag(false);
        writer.write_flag(false);
        SequenceParameterSet::parse(&writer.finish_rbsp()).unwrap()
    }

    #[allow(clippy::too_many_arguments)]
    fn write_pps_prefix(
        writer: &mut BitWriter,
        pps_id: u32,
        sps_id: u32,
        cabac: bool,
        bottom_field_poc: bool,
        num_slice_groups_minus1: u32,
    ) {
        writer.write_ue(pps_id);
        writer.write_ue(sps_id);
        writer.write_flag(cabac);
        writer.write_flag(bottom_field_poc);
        writer.write_ue(num_slice_groups_minus1);
    }

    #[allow(clippy::too_many_arguments)]
    fn write_pps_suffix(
        writer: &mut BitWriter,
        pic_init_qp_minus26: i32,
        pic_init_qs_minus26: i32,
        weighted_prediction: bool,
        weighted_biprediction: u32,
        chroma_qp_index_offset: i32,
        deblocking_filter_control_present: bool,
        constrained_intra_prediction: bool,
        redundant_pic_count_present: bool,
    ) {
        writer.write_ue(0);
        writer.write_ue(1);
        writer.write_flag(weighted_prediction);
        writer.write_bits(u64::from(weighted_biprediction), 2);
        writer.write_se(pic_init_qp_minus26);
        writer.write_se(pic_init_qs_minus26);
        writer.write_se(chroma_qp_index_offset);
        writer.write_flag(deblocking_filter_control_present);
        writer.write_flag(constrained_intra_prediction);
        writer.write_flag(redundant_pic_count_present);
    }

    #[derive(Default)]
    struct BitWriter {
        bytes: Vec<u8>,
        current: u8,
        bits: u8,
    }

    impl BitWriter {
        fn write_flag(&mut self, value: bool) {
            self.write_bits(u64::from(value), 1);
        }

        fn write_bits(&mut self, value: u64, count: u8) {
            for shift in (0..count).rev() {
                self.current = (self.current << 1) | ((value >> shift) as u8 & 1);
                self.bits += 1;
                if self.bits == 8 {
                    self.bytes.push(self.current);
                    self.current = 0;
                    self.bits = 0;
                }
            }
        }

        fn write_ue(&mut self, value: u32) {
            let code_num = u64::from(value) + 1;
            let width = 64 - code_num.leading_zeros() as u8;
            self.write_bits(0, width - 1);
            self.write_bits(code_num, width);
        }

        fn write_se(&mut self, value: i32) {
            let code_num = if value <= 0 {
                u32::try_from(-i64::from(value) * 2).unwrap()
            } else {
                u32::try_from(i64::from(value) * 2 - 1).unwrap()
            };
            self.write_ue(code_num);
        }

        fn finish_rbsp(mut self) -> Vec<u8> {
            self.write_flag(true);
            if self.bits != 0 {
                self.current <<= 8 - self.bits;
                self.bytes.push(self.current);
            }
            self.bytes
        }
    }
}
