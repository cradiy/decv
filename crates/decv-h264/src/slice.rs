//! Slice-header parsing and slice-data dispatch.

use bit_readers::BitReader;

use crate::{
    ActiveParameterSets, EntropyCodingMode, H264Error, NalHeader, NalUnitType, ParameterSetStore,
    PicOrderCount, Result, SliceGroupMap, WeightedBiprediction,
};

const MAX_LIST_OPERATIONS: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SliceType {
    P,
    B,
    I,
    Sp,
    Si,
}

impl SliceType {
    fn parse(value: u32) -> Result<Self> {
        match value {
            0 | 5 => Ok(Self::P),
            1 | 6 => Ok(Self::B),
            2 | 7 => Ok(Self::I),
            3 | 8 => Ok(Self::Sp),
            4 | 9 => Ok(Self::Si),
            _ => Err(H264Error::InvalidSyntax("slice_type exceeds 9")),
        }
    }

    pub const fn is_intra(self) -> bool {
        matches!(self, Self::I | Self::Si)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlicePictureOrder {
    pub pic_order_cnt_lsb: Option<u32>,
    pub delta_pic_order_bottom: Option<i32>,
    pub delta_pic_order: [Option<i32>; 2],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReferenceListModification {
    SubtractPicNum { abs_diff_pic_num: u32 },
    AddPicNum { abs_diff_pic_num: u32 },
    LongTerm { long_term_pic_num: u32 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WeightOffset {
    pub weight: i32,
    pub offset: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PredictionWeight {
    pub luma: Option<WeightOffset>,
    pub chroma: Option<[WeightOffset; 2]>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PredictionWeightTable {
    pub luma_log2_weight_denom: u8,
    pub chroma_log2_weight_denom: u8,
    pub list0: Vec<PredictionWeight>,
    pub list1: Vec<PredictionWeight>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryManagementOperation {
    ForgetShortTerm {
        difference_of_pic_nums: u32,
    },
    ForgetLongTerm {
        long_term_pic_num: u32,
    },
    AssignLongTerm {
        difference_of_pic_nums: u32,
        long_term_frame_idx: u32,
    },
    LimitLongTerm {
        max_long_term_frame_idx_plus1: u32,
    },
    ForgetAll,
    MarkCurrentLongTerm {
        long_term_frame_idx: u32,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReferencePictureMarking {
    None,
    Idr {
        no_output_of_prior_pictures: bool,
        long_term_reference: bool,
    },
    SlidingWindow,
    Adaptive(Vec<MemoryManagementOperation>),
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct DeblockingFilter {
    /// Zero enables filtering, one disables it, and two enables filtering
    /// except across slice boundaries.
    pub idc: u8,
    pub alpha_c0_offset_div2: i8,
    pub beta_offset_div2: i8,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SliceHeader {
    pub first_mb_in_slice: u32,
    pub slice_type: SliceType,
    /// True for the duplicate slice-type codes 5 through 9.
    pub all_slices_same_type: bool,
    pub picture_parameter_set_id: u32,
    pub frame_num: u32,
    pub field_picture: bool,
    pub bottom_field: bool,
    pub idr_pic_id: Option<u32>,
    pub picture_order: SlicePictureOrder,
    pub redundant_pic_count: Option<u32>,
    pub direct_spatial_mv_prediction: Option<bool>,
    pub num_ref_idx_l0_active: u8,
    pub num_ref_idx_l1_active: u8,
    pub ref_pic_list_modifications_l0: Vec<ReferenceListModification>,
    pub ref_pic_list_modifications_l1: Vec<ReferenceListModification>,
    pub prediction_weights: Option<PredictionWeightTable>,
    pub reference_picture_marking: ReferencePictureMarking,
    pub cabac_init_idc: Option<u8>,
    pub slice_qp_y: u8,
    pub sp_for_switch: Option<bool>,
    pub slice_qs_y: Option<u8>,
    pub deblocking_filter: Option<DeblockingFilter>,
    pub slice_group_change_cycle: Option<u32>,
    /// Number of RBSP bits consumed by the header. Slice data starts here.
    pub bit_size: usize,
}

#[derive(Debug, Clone)]
pub struct ParsedSliceHeader {
    pub header: SliceHeader,
    pub parameter_sets: ActiveParameterSets,
}

impl ParsedSliceHeader {
    /// Parses an ordinary non-partitioned AVC slice header.
    ///
    /// The input must be an already unescaped RBSP payload. Slice extensions
    /// and data-partition NAL units are deliberately rejected at this layer.
    pub fn parse(
        rbsp: &[u8],
        nal_header: NalHeader,
        parameter_sets: &ParameterSetStore,
    ) -> Result<Self> {
        let is_idr = match nal_header.unit_type {
            NalUnitType::NonIdrSlice => false,
            NalUnitType::IdrSlice => true,
            _ => {
                return Err(H264Error::UnsupportedFeature(
                    "only ordinary non-IDR and IDR slice NAL units are supported",
                ));
            }
        };
        if is_idr && nal_header.nal_ref_idc == 0 {
            return Err(H264Error::InvalidSyntax(
                "IDR slice must have non-zero nal_ref_idc",
            ));
        }

        let mut reader = BitReader::new(rbsp);
        let first_mb_in_slice = read_ue(&mut reader)?;
        let slice_type_code = read_ue(&mut reader)?;
        let slice_type = SliceType::parse(slice_type_code)?;
        let all_slices_same_type = slice_type_code >= 5;
        if is_idr && !slice_type.is_intra() {
            return Err(H264Error::InvalidSyntax(
                "IDR NAL unit must contain an intra slice",
            ));
        }
        let picture_parameter_set_id = read_ue(&mut reader)?;
        if picture_parameter_set_id > 255 {
            return Err(H264Error::InvalidSyntax(
                "pic_parameter_set_id in slice exceeds 255",
            ));
        }
        let active = parameter_sets.resolve(picture_parameter_set_id)?;
        let sps = &active.sequence;
        let pps = &active.picture;

        let frame_num = read_variable_bits(&mut reader, u32::from(sps.log2_max_frame_num))?;
        if is_idr && frame_num != 0 {
            return Err(H264Error::InvalidSyntax(
                "frame_num in an IDR slice must be zero",
            ));
        }
        let (field_picture, bottom_field) = if sps.frame_mbs_only {
            (false, false)
        } else {
            let field_picture = read_flag(&mut reader)?;
            let bottom_field = field_picture && read_flag(&mut reader)?;
            (field_picture, bottom_field)
        };
        validate_first_macroblock(first_mb_in_slice, sps, field_picture)?;

        let idr_pic_id = if is_idr {
            let id = read_ue(&mut reader)?;
            if id > 65_535 {
                return Err(H264Error::InvalidSyntax("idr_pic_id exceeds 65535"));
            }
            Some(id)
        } else {
            None
        };

        let picture_order = parse_picture_order(&mut reader, sps, pps, field_picture)?;
        let redundant_pic_count = if pps.redundant_pic_count_present {
            let count = read_ue(&mut reader)?;
            if count > 127 {
                return Err(H264Error::InvalidSyntax("redundant_pic_cnt exceeds 127"));
            }
            Some(count)
        } else {
            None
        };

        let direct_spatial_mv_prediction = if slice_type == SliceType::B {
            Some(read_flag(&mut reader)?)
        } else {
            None
        };

        let (num_ref_idx_l0_active, num_ref_idx_l1_active) =
            parse_reference_counts(&mut reader, slice_type, pps)?;
        let (ref_pic_list_modifications_l0, ref_pic_list_modifications_l1) =
            parse_reference_list_modifications(&mut reader, slice_type)?;

        let prediction_weights = if needs_prediction_weight_table(slice_type, pps) {
            Some(parse_prediction_weight_table(
                &mut reader,
                slice_type,
                num_ref_idx_l0_active,
                num_ref_idx_l1_active,
            )?)
        } else {
            None
        };

        let reference_picture_marking =
            parse_reference_picture_marking(&mut reader, nal_header.nal_ref_idc, is_idr)?;

        let cabac_init_idc =
            if pps.entropy_coding_mode == EntropyCodingMode::Cabac && !slice_type.is_intra() {
                let idc = read_ue(&mut reader)?;
                if idc > 2 {
                    return Err(H264Error::InvalidSyntax("cabac_init_idc exceeds 2"));
                }
                Some(idc as u8)
            } else {
                None
            };

        let slice_qp_y = derive_slice_quantizer(pps.pic_init_qp, read_se(&mut reader)?)?;
        let (sp_for_switch, slice_qs_y) = match slice_type {
            SliceType::Sp => (
                Some(read_flag(&mut reader)?),
                Some(derive_slice_quantizer(
                    pps.pic_init_qs,
                    read_se(&mut reader)?,
                )?),
            ),
            SliceType::Si => (
                None,
                Some(derive_slice_quantizer(
                    pps.pic_init_qs,
                    read_se(&mut reader)?,
                )?),
            ),
            _ => (None, None),
        };

        let deblocking_filter = if pps.deblocking_filter_control_present {
            Some(parse_deblocking_filter(&mut reader)?)
        } else {
            None
        };
        let slice_group_change_cycle = parse_slice_group_change_cycle(&mut reader, sps, pps)?;

        Ok(Self {
            header: SliceHeader {
                first_mb_in_slice,
                slice_type,
                all_slices_same_type,
                picture_parameter_set_id,
                frame_num,
                field_picture,
                bottom_field,
                idr_pic_id,
                picture_order,
                redundant_pic_count,
                direct_spatial_mv_prediction,
                num_ref_idx_l0_active,
                num_ref_idx_l1_active,
                ref_pic_list_modifications_l0,
                ref_pic_list_modifications_l1,
                prediction_weights,
                reference_picture_marking,
                cabac_init_idc,
                slice_qp_y,
                sp_for_switch,
                slice_qs_y,
                deblocking_filter,
                slice_group_change_cycle,
                bit_size: reader.bit_position(),
            },
            parameter_sets: active,
        })
    }
}

fn validate_first_macroblock(
    first_mb_in_slice: u32,
    sps: &crate::SequenceParameterSet,
    field_picture: bool,
) -> Result<()> {
    let height_in_mbs = sps
        .pic_height_in_map_units
        .checked_mul(if sps.frame_mbs_only { 1 } else { 2 })
        .ok_or(H264Error::IntegerOverflow)?;
    let height_in_mbs = if field_picture {
        height_in_mbs / 2
    } else {
        height_in_mbs
    };
    let picture_size_in_mbs = sps
        .pic_width_in_mbs
        .checked_mul(height_in_mbs)
        .ok_or(H264Error::IntegerOverflow)?;
    let mbaff_factor = if sps.mb_adaptive_frame_field && !field_picture {
        2
    } else {
        1
    };
    let first_mb_address = first_mb_in_slice
        .checked_mul(mbaff_factor)
        .ok_or(H264Error::IntegerOverflow)?;
    if first_mb_address >= picture_size_in_mbs {
        return Err(H264Error::InvalidSyntax(
            "first_mb_in_slice exceeds picture size",
        ));
    }
    Ok(())
}

fn parse_picture_order(
    reader: &mut BitReader<'_>,
    sps: &crate::SequenceParameterSet,
    pps: &crate::PictureParameterSet,
    field_picture: bool,
) -> Result<SlicePictureOrder> {
    let mut order = SlicePictureOrder {
        pic_order_cnt_lsb: None,
        delta_pic_order_bottom: None,
        delta_pic_order: [None, None],
    };

    match &sps.pic_order_count {
        PicOrderCount::Type0 {
            log2_max_pic_order_cnt_lsb,
        } => {
            order.pic_order_cnt_lsb = Some(read_variable_bits(
                reader,
                u32::from(*log2_max_pic_order_cnt_lsb),
            )?);
            if pps.bottom_field_pic_order_in_frame_present && !field_picture {
                order.delta_pic_order_bottom = Some(read_se(reader)?);
            }
        }
        PicOrderCount::Type1 {
            delta_pic_order_always_zero,
            ..
        } if !delta_pic_order_always_zero => {
            order.delta_pic_order[0] = Some(read_se(reader)?);
            if pps.bottom_field_pic_order_in_frame_present && !field_picture {
                order.delta_pic_order[1] = Some(read_se(reader)?);
            }
        }
        PicOrderCount::Type1 { .. } | PicOrderCount::Type2 => {}
    }
    Ok(order)
}

fn parse_reference_counts(
    reader: &mut BitReader<'_>,
    slice_type: SliceType,
    pps: &crate::PictureParameterSet,
) -> Result<(u8, u8)> {
    let mut list0 = pps.num_ref_idx_l0_default_active;
    let mut list1 = pps.num_ref_idx_l1_default_active;

    if matches!(slice_type, SliceType::P | SliceType::Sp | SliceType::B) && read_flag(reader)? {
        list0 = parse_active_reference_count(reader)?;
        if slice_type == SliceType::B {
            list1 = parse_active_reference_count(reader)?;
        }
    }
    if slice_type != SliceType::B {
        list1 = 0;
    }
    if slice_type.is_intra() {
        list0 = 0;
    }
    Ok((list0, list1))
}

fn parse_active_reference_count(reader: &mut BitReader<'_>) -> Result<u8> {
    let minus1 = read_ue(reader)?;
    if minus1 > 31 {
        return Err(H264Error::InvalidSyntax(
            "num_ref_idx_active_minus1 exceeds 31",
        ));
    }
    Ok((minus1 + 1) as u8)
}

fn parse_reference_list_modifications(
    reader: &mut BitReader<'_>,
    slice_type: SliceType,
) -> Result<(
    Vec<ReferenceListModification>,
    Vec<ReferenceListModification>,
)> {
    let list0 = if slice_type.is_intra() {
        Vec::new()
    } else {
        parse_reference_list_modification(reader)?
    };
    let list1 = if slice_type == SliceType::B {
        parse_reference_list_modification(reader)?
    } else {
        Vec::new()
    };
    Ok((list0, list1))
}

fn parse_reference_list_modification(
    reader: &mut BitReader<'_>,
) -> Result<Vec<ReferenceListModification>> {
    if !read_flag(reader)? {
        return Ok(Vec::new());
    }

    let mut operations = Vec::new();
    loop {
        match read_ue(reader)? {
            0 => push_bounded(
                &mut operations,
                ReferenceListModification::SubtractPicNum {
                    abs_diff_pic_num: add_one(read_ue(reader)?)?,
                },
                "too many reference-list modifications",
            )?,
            1 => push_bounded(
                &mut operations,
                ReferenceListModification::AddPicNum {
                    abs_diff_pic_num: add_one(read_ue(reader)?)?,
                },
                "too many reference-list modifications",
            )?,
            2 => push_bounded(
                &mut operations,
                ReferenceListModification::LongTerm {
                    long_term_pic_num: read_ue(reader)?,
                },
                "too many reference-list modifications",
            )?,
            3 => break,
            _ => {
                return Err(H264Error::InvalidSyntax(
                    "invalid modification_of_pic_nums_idc",
                ));
            }
        }
    }
    Ok(operations)
}

fn needs_prediction_weight_table(slice_type: SliceType, pps: &crate::PictureParameterSet) -> bool {
    (pps.weighted_prediction && matches!(slice_type, SliceType::P | SliceType::Sp))
        || (pps.weighted_biprediction == WeightedBiprediction::Explicit
            && slice_type == SliceType::B)
}

fn parse_prediction_weight_table(
    reader: &mut BitReader<'_>,
    slice_type: SliceType,
    list0_count: u8,
    list1_count: u8,
) -> Result<PredictionWeightTable> {
    let luma_log2_weight_denom = parse_weight_denom(reader)?;
    let chroma_log2_weight_denom = parse_weight_denom(reader)?;
    let list0 = parse_prediction_weights(reader, list0_count)?;
    let list1 = if slice_type == SliceType::B {
        parse_prediction_weights(reader, list1_count)?
    } else {
        Vec::new()
    };
    Ok(PredictionWeightTable {
        luma_log2_weight_denom,
        chroma_log2_weight_denom,
        list0,
        list1,
    })
}

fn parse_weight_denom(reader: &mut BitReader<'_>) -> Result<u8> {
    let value = read_ue(reader)?;
    if value > 7 {
        return Err(H264Error::InvalidSyntax(
            "prediction weight denominator exceeds 7",
        ));
    }
    Ok(value as u8)
}

fn parse_prediction_weights(
    reader: &mut BitReader<'_>,
    count: u8,
) -> Result<Vec<PredictionWeight>> {
    let mut weights = Vec::with_capacity(count as usize);
    for _ in 0..count {
        let luma = if read_flag(reader)? {
            Some(parse_weight_offset(reader)?)
        } else {
            None
        };
        let chroma = if read_flag(reader)? {
            Some([parse_weight_offset(reader)?, parse_weight_offset(reader)?])
        } else {
            None
        };
        weights.push(PredictionWeight { luma, chroma });
    }
    Ok(weights)
}

fn parse_weight_offset(reader: &mut BitReader<'_>) -> Result<WeightOffset> {
    let weight = read_se(reader)?;
    let offset = read_se(reader)?;
    if !(-128..=127).contains(&weight) || !(-128..=127).contains(&offset) {
        return Err(H264Error::InvalidSyntax(
            "prediction weight or offset is outside -128..=127",
        ));
    }
    Ok(WeightOffset { weight, offset })
}

fn parse_reference_picture_marking(
    reader: &mut BitReader<'_>,
    nal_ref_idc: u8,
    is_idr: bool,
) -> Result<ReferencePictureMarking> {
    if nal_ref_idc == 0 {
        return Ok(ReferencePictureMarking::None);
    }
    if is_idr {
        return Ok(ReferencePictureMarking::Idr {
            no_output_of_prior_pictures: read_flag(reader)?,
            long_term_reference: read_flag(reader)?,
        });
    }
    if !read_flag(reader)? {
        return Ok(ReferencePictureMarking::SlidingWindow);
    }

    let mut operations = Vec::new();
    loop {
        let operation = match read_ue(reader)? {
            0 => break,
            1 => MemoryManagementOperation::ForgetShortTerm {
                difference_of_pic_nums: add_one(read_ue(reader)?)?,
            },
            2 => MemoryManagementOperation::ForgetLongTerm {
                long_term_pic_num: read_ue(reader)?,
            },
            3 => MemoryManagementOperation::AssignLongTerm {
                difference_of_pic_nums: add_one(read_ue(reader)?)?,
                long_term_frame_idx: read_ue(reader)?,
            },
            4 => MemoryManagementOperation::LimitLongTerm {
                max_long_term_frame_idx_plus1: read_ue(reader)?,
            },
            5 => MemoryManagementOperation::ForgetAll,
            6 => MemoryManagementOperation::MarkCurrentLongTerm {
                long_term_frame_idx: read_ue(reader)?,
            },
            _ => {
                return Err(H264Error::InvalidSyntax(
                    "memory_management_control_operation exceeds 6",
                ));
            }
        };
        push_bounded(
            &mut operations,
            operation,
            "too many memory-management operations",
        )?;
    }
    Ok(ReferencePictureMarking::Adaptive(operations))
}

fn derive_slice_quantizer(initial: i8, delta: i32) -> Result<u8> {
    let value = i32::from(initial)
        .checked_add(delta)
        .ok_or(H264Error::IntegerOverflow)?;
    if !(0..=51).contains(&value) {
        return Err(H264Error::InvalidSyntax(
            "derived slice quantizer is outside 0..=51",
        ));
    }
    Ok(value as u8)
}

fn parse_deblocking_filter(reader: &mut BitReader<'_>) -> Result<DeblockingFilter> {
    let idc = read_ue(reader)?;
    if idc > 2 {
        return Err(H264Error::InvalidSyntax(
            "disable_deblocking_filter_idc exceeds 2",
        ));
    }
    let (alpha_c0_offset_div2, beta_offset_div2) = if idc == 1 {
        (0, 0)
    } else {
        (
            parse_deblocking_offset(read_se(reader)?)?,
            parse_deblocking_offset(read_se(reader)?)?,
        )
    };
    Ok(DeblockingFilter {
        idc: idc as u8,
        alpha_c0_offset_div2,
        beta_offset_div2,
    })
}

fn parse_deblocking_offset(value: i32) -> Result<i8> {
    if !(-6..=6).contains(&value) {
        return Err(H264Error::InvalidSyntax(
            "slice deblocking offset is outside -6..=6",
        ));
    }
    Ok(value as i8)
}

fn parse_slice_group_change_cycle(
    reader: &mut BitReader<'_>,
    sps: &crate::SequenceParameterSet,
    pps: &crate::PictureParameterSet,
) -> Result<Option<u32>> {
    let Some(SliceGroupMap::Changing { change_rate, .. }) = &pps.slice_group_map else {
        return Ok(None);
    };
    let picture_size = sps
        .pic_width_in_mbs
        .checked_mul(sps.pic_height_in_map_units)
        .ok_or(H264Error::IntegerOverflow)?;
    let quotient = picture_size / change_rate;
    let bit_count = quotient.ilog2() + 1;
    let cycle = read_variable_bits(reader, bit_count)?;
    if cycle > picture_size.div_ceil(*change_rate) {
        return Err(H264Error::InvalidSyntax(
            "slice_group_change_cycle exceeds picture size",
        ));
    }
    Ok(Some(cycle))
}

#[inline]
fn read_flag(reader: &mut BitReader<'_>) -> Result<bool> {
    reader
        .read_bit()
        .map(|value| value != 0)
        .ok_or(H264Error::UnexpectedEof)
}

#[inline]
fn read_variable_bits(reader: &mut BitReader<'_>, count: u32) -> Result<u32> {
    reader.read_bits(count).ok_or(H264Error::UnexpectedEof)
}

#[inline]
fn read_ue(reader: &mut BitReader<'_>) -> Result<u32> {
    reader.read_ue().ok_or(H264Error::UnexpectedEof)
}

#[inline]
fn read_se(reader: &mut BitReader<'_>) -> Result<i32> {
    reader.read_se().ok_or(H264Error::UnexpectedEof)
}

#[inline]
fn add_one(value: u32) -> Result<u32> {
    value.checked_add(1).ok_or(H264Error::IntegerOverflow)
}

fn push_bounded<T>(values: &mut Vec<T>, value: T, error: &'static str) -> Result<()> {
    if values.len() == MAX_LIST_OPERATIONS {
        return Err(H264Error::InvalidSyntax(error));
    }
    values.push(value);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        DeblockingFilter, MemoryManagementOperation, ParsedSliceHeader, ReferenceListModification,
        ReferencePictureMarking, SliceType, WeightOffset,
    };
    use crate::{H264Error, NalHeader, NalUnitType, ParameterSetStore};

    #[test]
    fn parses_an_idr_i_slice_and_resolves_parameter_sets() {
        let mut store = parameter_sets(PpsOptions::default());
        let mut writer = BitWriter::default();
        writer.write_ue(0);
        writer.write_ue(2);
        writer.write_ue(0);
        writer.write_bits(0, 4);
        writer.write_ue(7);
        writer.write_bits(2, 4);
        writer.write_flag(false); // no_output_of_prior_pics
        writer.write_flag(true); // long_term_reference
        writer.write_se(0); // slice_qp_delta
        writer.write_ue(0); // deblocking enabled
        writer.write_se(0);
        writer.write_se(0);

        let parsed = ParsedSliceHeader::parse(
            &writer.finish_bytes(),
            NalHeader {
                nal_ref_idc: 3,
                unit_type: NalUnitType::IdrSlice,
            },
            &store,
        )
        .unwrap();
        let header = parsed.header;

        assert_eq!(header.slice_type, SliceType::I);
        assert_eq!(header.frame_num, 0);
        assert_eq!(header.idr_pic_id, Some(7));
        assert_eq!(header.picture_order.pic_order_cnt_lsb, Some(2));
        assert_eq!(header.num_ref_idx_l0_active, 0);
        assert_eq!(header.num_ref_idx_l1_active, 0);
        assert_eq!(header.slice_qp_y, 26);
        assert_eq!(
            header.reference_picture_marking,
            ReferencePictureMarking::Idr {
                no_output_of_prior_pictures: false,
                long_term_reference: true
            }
        );
        assert_eq!(
            header.deblocking_filter,
            Some(DeblockingFilter {
                idc: 0,
                alpha_c0_offset_div2: 0,
                beta_offset_div2: 0
            })
        );
        assert_eq!(parsed.parameter_sets.picture.id, 0);
        assert!(header.bit_size > 0);

        store.clear();
    }

    #[test]
    fn parses_weighted_p_slice_modifications_mmco_and_cabac() {
        let store = parameter_sets(PpsOptions {
            id: 1,
            cabac: true,
            weighted_prediction: true,
            ..PpsOptions::default()
        });
        let mut writer = BitWriter::default();
        write_slice_prefix(&mut writer, 0, 0, 1, 5, 4, 6);
        writer.write_flag(true); // override active refs
        writer.write_ue(1); // two L0 references
        writer.write_flag(true); // modify L0
        writer.write_ue(0);
        writer.write_ue(2);
        writer.write_ue(2);
        writer.write_ue(4);
        writer.write_ue(3);
        writer.write_ue(0); // luma denominator
        writer.write_ue(0); // chroma denominator
        writer.write_flag(true);
        writer.write_se(1);
        writer.write_se(-1);
        writer.write_flag(false);
        writer.write_flag(false);
        writer.write_flag(false);
        writer.write_flag(true); // adaptive marking
        writer.write_ue(1);
        writer.write_ue(0);
        writer.write_ue(6);
        writer.write_ue(2);
        writer.write_ue(0);
        writer.write_ue(2); // cabac_init_idc
        writer.write_se(-2);
        writer.write_ue(2); // filter except across slice boundaries
        writer.write_se(-1);
        writer.write_se(1);

        let parsed = ParsedSliceHeader::parse(
            &writer.finish_bytes(),
            NalHeader {
                nal_ref_idc: 2,
                unit_type: NalUnitType::NonIdrSlice,
            },
            &store,
        )
        .unwrap();
        let header = parsed.header;

        assert_eq!(header.slice_type, SliceType::P);
        assert_eq!(header.num_ref_idx_l0_active, 2);
        assert_eq!(
            header.ref_pic_list_modifications_l0,
            vec![
                ReferenceListModification::SubtractPicNum {
                    abs_diff_pic_num: 3
                },
                ReferenceListModification::LongTerm {
                    long_term_pic_num: 4
                }
            ]
        );
        let weights = header.prediction_weights.unwrap();
        assert_eq!(
            weights.list0[0].luma,
            Some(WeightOffset {
                weight: 1,
                offset: -1
            })
        );
        assert_eq!(weights.list0[1].luma, None);
        assert_eq!(
            header.reference_picture_marking,
            ReferencePictureMarking::Adaptive(vec![
                MemoryManagementOperation::ForgetShortTerm {
                    difference_of_pic_nums: 1
                },
                MemoryManagementOperation::MarkCurrentLongTerm {
                    long_term_frame_idx: 2
                }
            ])
        );
        assert_eq!(header.cabac_init_idc, Some(2));
        assert_eq!(header.slice_qp_y, 24);
        assert_eq!(
            header.deblocking_filter,
            Some(DeblockingFilter {
                idc: 2,
                alpha_c0_offset_div2: -1,
                beta_offset_div2: 1
            })
        );
    }

    #[test]
    fn parses_an_explicitly_weighted_b_slice() {
        let store = parameter_sets(PpsOptions {
            id: 2,
            weighted_biprediction: 1,
            ..PpsOptions::default()
        });
        let mut writer = BitWriter::default();
        write_slice_prefix(&mut writer, 0, 1, 2, 1, 4, 3);
        writer.write_flag(true); // direct spatial prediction
        writer.write_flag(false); // default reference counts
        writer.write_flag(false); // no L0 modification
        writer.write_flag(false); // no L1 modification
        writer.write_ue(0);
        writer.write_ue(0);
        writer.write_flag(false);
        writer.write_flag(false);
        writer.write_flag(true);
        writer.write_se(2);
        writer.write_se(0);
        writer.write_flag(true);
        writer.write_se(1);
        writer.write_se(-1);
        writer.write_se(2);
        writer.write_se(-2);
        writer.write_se(0); // slice_qp_delta
        writer.write_ue(1); // disable deblocking

        let parsed = ParsedSliceHeader::parse(
            &writer.finish_bytes(),
            NalHeader {
                nal_ref_idc: 0,
                unit_type: NalUnitType::NonIdrSlice,
            },
            &store,
        )
        .unwrap();
        let header = parsed.header;

        assert_eq!(header.slice_type, SliceType::B);
        assert_eq!(header.direct_spatial_mv_prediction, Some(true));
        assert_eq!(header.num_ref_idx_l0_active, 1);
        assert_eq!(header.num_ref_idx_l1_active, 1);
        assert_eq!(
            header.reference_picture_marking,
            ReferencePictureMarking::None
        );
        let weights = header.prediction_weights.unwrap();
        assert_eq!(weights.list0[0].luma, None);
        assert_eq!(
            weights.list1[0].chroma,
            Some([
                WeightOffset {
                    weight: 1,
                    offset: -1
                },
                WeightOffset {
                    weight: 2,
                    offset: -2
                }
            ])
        );
    }

    #[test]
    fn parses_type_one_poc_sp_and_dynamic_slice_groups() {
        let mut store = ParameterSetStore::new();
        store.parse_sps(&interlaced_sps_rbsp()).unwrap();
        store.parse_pps(&dynamic_slice_group_pps_rbsp()).unwrap();

        let mut writer = BitWriter::default();
        writer.write_ue(0);
        writer.write_ue(3); // SP
        writer.write_ue(3);
        writer.write_bits(2, 4); // frame_num
        writer.write_flag(false); // frame picture using MBAFF
        writer.write_se(1);
        writer.write_se(-1);
        writer.write_ue(2); // redundant_pic_cnt
        writer.write_flag(false); // default reference count
        writer.write_flag(false); // no list modification
        writer.write_se(0); // slice_qp_delta
        writer.write_flag(true); // sp_for_switch
        writer.write_se(-1); // slice_qs_delta
        writer.write_ue(1); // disable deblocking
        writer.write_bits(3, 4); // slice_group_change_cycle

        let parsed = ParsedSliceHeader::parse(
            &writer.finish_bytes(),
            NalHeader {
                nal_ref_idc: 0,
                unit_type: NalUnitType::NonIdrSlice,
            },
            &store,
        )
        .unwrap();
        let header = parsed.header;

        assert_eq!(header.slice_type, SliceType::Sp);
        assert_eq!(header.picture_order.delta_pic_order, [Some(1), Some(-1)]);
        assert_eq!(header.redundant_pic_count, Some(2));
        assert_eq!(header.sp_for_switch, Some(true));
        assert_eq!(header.slice_qs_y, Some(25));
        assert_eq!(header.slice_group_change_cycle, Some(3));
    }

    #[test]
    fn rejects_missing_sets_invalid_idr_and_out_of_picture_addresses() {
        let empty = ParameterSetStore::new();
        let mut missing = BitWriter::default();
        missing.write_ue(0);
        missing.write_ue(2);
        missing.write_ue(9);
        assert!(matches!(
            ParsedSliceHeader::parse(
                &missing.finish_bytes(),
                NalHeader {
                    nal_ref_idc: 3,
                    unit_type: NalUnitType::IdrSlice
                },
                &empty
            ),
            Err(H264Error::MissingPps(9))
        ));

        let store = parameter_sets(PpsOptions::default());
        assert!(matches!(
            ParsedSliceHeader::parse(
                &[],
                NalHeader {
                    nal_ref_idc: 0,
                    unit_type: NalUnitType::IdrSlice
                },
                &store
            ),
            Err(H264Error::InvalidSyntax(_))
        ));

        let mut outside = BitWriter::default();
        write_slice_prefix(&mut outside, 12, 2, 0, 0, 4, 0);
        assert!(matches!(
            ParsedSliceHeader::parse(
                &outside.finish_bytes(),
                NalHeader {
                    nal_ref_idc: 3,
                    unit_type: NalUnitType::IdrSlice
                },
                &store
            ),
            Err(H264Error::InvalidSyntax(_))
        ));
    }

    fn parameter_sets(options: PpsOptions) -> ParameterSetStore {
        let mut store = ParameterSetStore::new();
        store.parse_sps(&sps_rbsp()).unwrap();
        store.parse_pps(&pps_rbsp(options)).unwrap();
        store
    }

    fn sps_rbsp() -> Vec<u8> {
        let mut writer = BitWriter::default();
        writer.write_bits(66, 8);
        writer.write_bits(0, 8);
        writer.write_bits(30, 8);
        writer.write_ue(0);
        writer.write_ue(0);
        writer.write_ue(0);
        writer.write_ue(0);
        writer.write_ue(2);
        writer.write_flag(false);
        writer.write_ue(3); // four macroblocks wide
        writer.write_ue(2); // three macroblocks high
        writer.write_flag(true);
        writer.write_flag(true);
        writer.write_flag(false);
        writer.write_flag(false);
        writer.finish_rbsp()
    }

    fn interlaced_sps_rbsp() -> Vec<u8> {
        let mut writer = BitWriter::default();
        writer.write_bits(77, 8);
        writer.write_bits(0, 8);
        writer.write_bits(32, 8);
        writer.write_ue(0);
        writer.write_ue(0);
        writer.write_ue(1);
        writer.write_flag(false);
        writer.write_se(-1);
        writer.write_se(2);
        writer.write_ue(0);
        writer.write_ue(2);
        writer.write_flag(false);
        writer.write_ue(3);
        writer.write_ue(1);
        writer.write_flag(false);
        writer.write_flag(true);
        writer.write_flag(true);
        writer.write_flag(false);
        writer.write_flag(false);
        writer.finish_rbsp()
    }

    #[derive(Default, Clone, Copy)]
    struct PpsOptions {
        id: u32,
        cabac: bool,
        weighted_prediction: bool,
        weighted_biprediction: u64,
    }

    fn pps_rbsp(options: PpsOptions) -> Vec<u8> {
        let mut writer = BitWriter::default();
        writer.write_ue(options.id);
        writer.write_ue(0);
        writer.write_flag(options.cabac);
        writer.write_flag(false);
        writer.write_ue(0);
        writer.write_ue(0);
        writer.write_ue(0);
        writer.write_flag(options.weighted_prediction);
        writer.write_bits(options.weighted_biprediction, 2);
        writer.write_se(0);
        writer.write_se(0);
        writer.write_se(0);
        writer.write_flag(true);
        writer.write_flag(false);
        writer.write_flag(false);
        writer.finish_rbsp()
    }

    fn dynamic_slice_group_pps_rbsp() -> Vec<u8> {
        let mut writer = BitWriter::default();
        writer.write_ue(3);
        writer.write_ue(0);
        writer.write_flag(false);
        writer.write_flag(true);
        writer.write_ue(1);
        writer.write_ue(3);
        writer.write_flag(false);
        writer.write_ue(0);
        writer.write_ue(0);
        writer.write_ue(0);
        writer.write_flag(false);
        writer.write_bits(0, 2);
        writer.write_se(0);
        writer.write_se(0);
        writer.write_se(0);
        writer.write_flag(true);
        writer.write_flag(false);
        writer.write_flag(true);
        writer.finish_rbsp()
    }

    fn write_slice_prefix(
        writer: &mut BitWriter,
        first_mb: u32,
        slice_type: u32,
        pps_id: u32,
        frame_num: u64,
        frame_num_bits: u8,
        pic_order_cnt_lsb: u64,
    ) {
        writer.write_ue(first_mb);
        writer.write_ue(slice_type);
        writer.write_ue(pps_id);
        writer.write_bits(frame_num, frame_num_bits);
        writer.write_bits(pic_order_cnt_lsb, 4);
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

        fn finish_bytes(mut self) -> Vec<u8> {
            if self.bits != 0 {
                self.current <<= 8 - self.bits;
                self.bytes.push(self.current);
            }
            self.bytes
        }

        fn finish_rbsp(mut self) -> Vec<u8> {
            self.write_flag(true);
            self.finish_bytes()
        }
    }
}
