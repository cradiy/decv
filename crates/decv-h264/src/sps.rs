//! Sequence Parameter Set syntax, validation, and derived image metadata.

use bit_readers::BitReader;
use decv_core::{ColorInfo, ColorMatrix, ColorPrimaries, ColorRange, Rect, Size, TransferFunction};

use crate::{H264Error, Result, consume_rbsp_trailing_bits};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Profile {
    Baseline,
    Main,
    High,
}

impl Profile {
    fn from_idc(profile_idc: u8) -> Result<Self> {
        match profile_idc {
            66 => Ok(Self::Baseline),
            77 => Ok(Self::Main),
            100 => Ok(Self::High),
            profile => Err(H264Error::UnsupportedProfile(profile)),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScalingList {
    pub values: Vec<u8>,
    pub use_default: bool,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ScalingMatrices {
    /// Six 4x4 lists followed by two 8x8 lists for 4:2:0 streams.
    ///
    /// A `None` entry means `seq_scaling_list_present_flag` was zero and the
    /// normative fallback rule must be applied when the matrix is used.
    pub lists: Vec<Option<ScalingList>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PicOrderCount {
    Type0 {
        log2_max_pic_order_cnt_lsb: u8,
    },
    Type1 {
        delta_pic_order_always_zero: bool,
        offset_for_non_ref_pic: i32,
        offset_for_top_to_bottom_field: i32,
        offset_for_ref_frame: Vec<i32>,
    },
    Type2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SampleAspectRatio {
    pub width: u16,
    pub height: u16,
}

impl Default for SampleAspectRatio {
    fn default() -> Self {
        Self {
            width: 1,
            height: 1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimingInfo {
    pub num_units_in_tick: u32,
    pub time_scale: u32,
    pub fixed_frame_rate: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BitstreamRestrictions {
    pub motion_vectors_over_pic_boundaries: bool,
    pub max_bytes_per_pic_denom: u32,
    pub max_bits_per_mb_denom: u32,
    pub log2_max_mv_length_horizontal: u32,
    pub log2_max_mv_length_vertical: u32,
    pub max_num_reorder_frames: u32,
    pub max_dec_frame_buffering: u32,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct VuiParameters {
    pub sample_aspect_ratio: Option<SampleAspectRatio>,
    pub overscan_appropriate: Option<bool>,
    pub video_format: Option<u8>,
    pub color: ColorInfo,
    pub chroma_sample_loc_type_top_field: Option<u32>,
    pub chroma_sample_loc_type_bottom_field: Option<u32>,
    pub timing: Option<TimingInfo>,
    pub pic_struct_present: bool,
    pub bitstream_restrictions: Option<BitstreamRestrictions>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SequenceParameterSet {
    pub id: u32,
    pub profile: Profile,
    pub profile_idc: u8,
    /// Constraint-set flags in bits 7 through 2, exactly as signalled.
    pub constraint_flags: u8,
    pub level_idc: u8,

    pub bit_depth_luma: u8,
    pub bit_depth_chroma: u8,
    pub qpprime_y_zero_transform_bypass: bool,
    pub scaling_matrices: Option<ScalingMatrices>,

    pub log2_max_frame_num: u8,
    pub pic_order_count: PicOrderCount,
    pub max_num_ref_frames: u32,
    pub gaps_in_frame_num_value_allowed: bool,

    pub pic_width_in_mbs: u32,
    pub pic_height_in_map_units: u32,
    pub frame_mbs_only: bool,
    pub mb_adaptive_frame_field: bool,
    pub direct_8x8_inference: bool,

    pub coded_size: Size,
    pub visible_rect: Rect,
    pub display_size: Size,
    pub vui: Option<VuiParameters>,
}

impl SequenceParameterSet {
    /// Parses one unescaped SPS RBSP, including its trailing bits.
    pub fn parse(rbsp: &[u8]) -> Result<Self> {
        let mut reader = BitReader::new(rbsp);

        let profile_idc = read_u8(&mut reader)?;
        let profile = Profile::from_idc(profile_idc)?;
        let constraint_and_reserved = read_u8(&mut reader)?;
        if constraint_and_reserved & 0b11 != 0 {
            return Err(H264Error::InvalidSyntax(
                "reserved_zero_2bits in SPS must be zero",
            ));
        }
        let constraint_flags = constraint_and_reserved & 0b1111_1100;
        let level_idc = read_u8(&mut reader)?;
        let id = read_ue(&mut reader)?;
        if id > 31 {
            return Err(H264Error::InvalidSyntax("seq_parameter_set_id exceeds 31"));
        }

        let mut bit_depth_luma = 8;
        let mut bit_depth_chroma = 8;
        let mut qpprime_y_zero_transform_bypass = false;
        let mut scaling_matrices = None;

        if profile == Profile::High {
            let chroma_format_idc = read_ue(&mut reader)?;
            if chroma_format_idc != 1 {
                return Err(H264Error::UnsupportedFeature(
                    "only 4:2:0 chroma_format_idc is currently supported",
                ));
            }

            let bit_depth_luma_minus8 = read_ue(&mut reader)?;
            let bit_depth_chroma_minus8 = read_ue(&mut reader)?;
            bit_depth_luma = add_u8(bit_depth_luma_minus8, 8)?;
            bit_depth_chroma = add_u8(bit_depth_chroma_minus8, 8)?;
            if bit_depth_luma != 8 || bit_depth_chroma != 8 {
                return Err(H264Error::UnsupportedFeature(
                    "only 8-bit H.264 pictures are currently supported",
                ));
            }

            qpprime_y_zero_transform_bypass = read_flag(&mut reader)?;
            if read_flag(&mut reader)? {
                scaling_matrices = Some(parse_scaling_matrices(&mut reader, 8)?);
            }
        }

        let log2_max_frame_num_minus4 = read_ue(&mut reader)?;
        if log2_max_frame_num_minus4 > 12 {
            return Err(H264Error::InvalidSyntax(
                "log2_max_frame_num_minus4 exceeds 12",
            ));
        }
        let log2_max_frame_num = add_u8(log2_max_frame_num_minus4, 4)?;

        let pic_order_count = match read_ue(&mut reader)? {
            0 => {
                let minus4 = read_ue(&mut reader)?;
                if minus4 > 12 {
                    return Err(H264Error::InvalidSyntax(
                        "log2_max_pic_order_cnt_lsb_minus4 exceeds 12",
                    ));
                }
                PicOrderCount::Type0 {
                    log2_max_pic_order_cnt_lsb: add_u8(minus4, 4)?,
                }
            }
            1 => {
                let delta_pic_order_always_zero = read_flag(&mut reader)?;
                let offset_for_non_ref_pic = read_se(&mut reader)?;
                let offset_for_top_to_bottom_field = read_se(&mut reader)?;
                let cycle_length = read_ue(&mut reader)?;
                if cycle_length > 255 {
                    return Err(H264Error::InvalidSyntax(
                        "num_ref_frames_in_pic_order_cnt_cycle exceeds 255",
                    ));
                }

                let mut offset_for_ref_frame = Vec::with_capacity(cycle_length as usize);
                for _ in 0..cycle_length {
                    offset_for_ref_frame.push(read_se(&mut reader)?);
                }

                PicOrderCount::Type1 {
                    delta_pic_order_always_zero,
                    offset_for_non_ref_pic,
                    offset_for_top_to_bottom_field,
                    offset_for_ref_frame,
                }
            }
            2 => PicOrderCount::Type2,
            _ => {
                return Err(H264Error::InvalidSyntax(
                    "pic_order_cnt_type must be 0, 1, or 2",
                ));
            }
        };

        let max_num_ref_frames = read_ue(&mut reader)?;
        let gaps_in_frame_num_value_allowed = read_flag(&mut reader)?;
        let pic_width_in_mbs = add_u32(read_ue(&mut reader)?, 1)?;
        let pic_height_in_map_units = add_u32(read_ue(&mut reader)?, 1)?;
        let frame_mbs_only = read_flag(&mut reader)?;
        let mb_adaptive_frame_field = if frame_mbs_only {
            false
        } else {
            read_flag(&mut reader)?
        };
        let direct_8x8_inference = read_flag(&mut reader)?;

        let crop_offsets = if read_flag(&mut reader)? {
            Some([
                read_ue(&mut reader)?,
                read_ue(&mut reader)?,
                read_ue(&mut reader)?,
                read_ue(&mut reader)?,
            ])
        } else {
            None
        };

        let vui = if read_flag(&mut reader)? {
            Some(parse_vui(&mut reader)?)
        } else {
            None
        };

        consume_rbsp_trailing_bits(&mut reader)?;

        let coded_size =
            derive_coded_size(pic_width_in_mbs, pic_height_in_map_units, frame_mbs_only)?;
        let visible_rect = derive_visible_rect(coded_size, frame_mbs_only, crop_offsets)?;
        let display_size = derive_display_size(
            visible_rect.size(),
            vui.as_ref()
                .and_then(|vui| vui.sample_aspect_ratio)
                .unwrap_or_default(),
        )?;

        Ok(Self {
            id,
            profile,
            profile_idc,
            constraint_flags,
            level_idc,
            bit_depth_luma,
            bit_depth_chroma,
            qpprime_y_zero_transform_bypass,
            scaling_matrices,
            log2_max_frame_num,
            pic_order_count,
            max_num_ref_frames,
            gaps_in_frame_num_value_allowed,
            pic_width_in_mbs,
            pic_height_in_map_units,
            frame_mbs_only,
            mb_adaptive_frame_field,
            direct_8x8_inference,
            coded_size,
            visible_rect,
            display_size,
            vui,
        })
    }
}

pub(crate) fn parse_scaling_matrices(
    reader: &mut BitReader<'_>,
    list_count: usize,
) -> Result<ScalingMatrices> {
    let mut lists = Vec::with_capacity(list_count);
    for index in 0..list_count {
        if read_flag(reader)? {
            let size = if index < 6 { 16 } else { 64 };
            lists.push(Some(parse_scaling_list(reader, size)?));
        } else {
            lists.push(None);
        }
    }
    Ok(ScalingMatrices { lists })
}

fn parse_scaling_list(reader: &mut BitReader<'_>, size: usize) -> Result<ScalingList> {
    let mut values = Vec::with_capacity(size);
    let mut last_scale = 8i32;
    let mut next_scale = 8i32;
    let mut use_default = false;

    for index in 0..size {
        if next_scale != 0 {
            let delta_scale = read_se(reader)?;
            if !(-128..=127).contains(&delta_scale) {
                return Err(H264Error::InvalidSyntax(
                    "scaling-list delta_scale is outside -128..127",
                ));
            }
            next_scale = (last_scale + delta_scale + 256).rem_euclid(256);
            if index == 0 && next_scale == 0 {
                use_default = true;
            }
        }

        let scale = if next_scale == 0 {
            last_scale
        } else {
            next_scale
        };
        values.push(u8::try_from(scale).map_err(|_| H264Error::IntegerOverflow)?);
        last_scale = scale;
    }

    Ok(ScalingList {
        values,
        use_default,
    })
}

fn parse_vui(reader: &mut BitReader<'_>) -> Result<VuiParameters> {
    let sample_aspect_ratio = if read_flag(reader)? {
        parse_sample_aspect_ratio(reader)?
    } else {
        None
    };

    let overscan_appropriate = if read_flag(reader)? {
        Some(read_flag(reader)?)
    } else {
        None
    };

    let mut video_format = None;
    let mut color = ColorInfo::default();
    if read_flag(reader)? {
        video_format = Some(read_bits::<3>(reader)? as u8);
        color.range = if read_flag(reader)? {
            ColorRange::Full
        } else {
            ColorRange::Limited
        };

        if read_flag(reader)? {
            color.primaries = map_color_primaries(read_u8(reader)?);
            color.transfer = map_transfer_function(read_u8(reader)?);
            color.matrix = map_color_matrix(read_u8(reader)?);
        }
    }

    let (chroma_sample_loc_type_top_field, chroma_sample_loc_type_bottom_field) =
        if read_flag(reader)? {
            (Some(read_ue(reader)?), Some(read_ue(reader)?))
        } else {
            (None, None)
        };

    let timing = if read_flag(reader)? {
        let num_units_in_tick = read_bits::<32>(reader)?;
        let time_scale = read_bits::<32>(reader)?;
        if num_units_in_tick == 0 || time_scale == 0 {
            return Err(H264Error::InvalidSyntax(
                "VUI timing values must be non-zero",
            ));
        }
        Some(TimingInfo {
            num_units_in_tick,
            time_scale,
            fixed_frame_rate: read_flag(reader)?,
        })
    } else {
        None
    };

    let nal_hrd_present = read_flag(reader)?;
    if nal_hrd_present {
        parse_hrd_parameters(reader)?;
    }
    let vcl_hrd_present = read_flag(reader)?;
    if vcl_hrd_present {
        parse_hrd_parameters(reader)?;
    }
    if nal_hrd_present || vcl_hrd_present {
        read_flag(reader)?;
    }

    let pic_struct_present = read_flag(reader)?;
    let bitstream_restrictions = if read_flag(reader)? {
        Some(BitstreamRestrictions {
            motion_vectors_over_pic_boundaries: read_flag(reader)?,
            max_bytes_per_pic_denom: read_ue(reader)?,
            max_bits_per_mb_denom: read_ue(reader)?,
            log2_max_mv_length_horizontal: read_ue(reader)?,
            log2_max_mv_length_vertical: read_ue(reader)?,
            max_num_reorder_frames: read_ue(reader)?,
            max_dec_frame_buffering: read_ue(reader)?,
        })
    } else {
        None
    };

    Ok(VuiParameters {
        sample_aspect_ratio,
        overscan_appropriate,
        video_format,
        color,
        chroma_sample_loc_type_top_field,
        chroma_sample_loc_type_bottom_field,
        timing,
        pic_struct_present,
        bitstream_restrictions,
    })
}

fn parse_sample_aspect_ratio(reader: &mut BitReader<'_>) -> Result<Option<SampleAspectRatio>> {
    let aspect_ratio_idc = read_u8(reader)?;
    let (width, height) = match aspect_ratio_idc {
        0 => return Ok(None),
        1 => (1, 1),
        2 => (12, 11),
        3 => (10, 11),
        4 => (16, 11),
        5 => (40, 33),
        6 => (24, 11),
        7 => (20, 11),
        8 => (32, 11),
        9 => (80, 33),
        10 => (18, 11),
        11 => (15, 11),
        12 => (64, 33),
        13 => (160, 99),
        14 => (4, 3),
        15 => (3, 2),
        16 => (2, 1),
        255 => (
            read_bits::<16>(reader)? as u16,
            read_bits::<16>(reader)? as u16,
        ),
        17..=254 => {
            return Err(H264Error::InvalidSyntax("reserved aspect_ratio_idc"));
        }
    };

    if width == 0 || height == 0 {
        return Err(H264Error::InvalidSyntax(
            "sample aspect ratio must be non-zero",
        ));
    }

    Ok(Some(SampleAspectRatio { width, height }))
}

fn parse_hrd_parameters(reader: &mut BitReader<'_>) -> Result<()> {
    let cpb_count_minus1 = read_ue(reader)?;
    if cpb_count_minus1 > 31 {
        return Err(H264Error::InvalidSyntax("cpb_cnt_minus1 exceeds 31"));
    }

    read_bits::<4>(reader)?;
    read_bits::<4>(reader)?;
    for _ in 0..=cpb_count_minus1 {
        read_ue(reader)?;
        read_ue(reader)?;
        read_flag(reader)?;
    }
    read_bits::<5>(reader)?;
    read_bits::<5>(reader)?;
    read_bits::<5>(reader)?;
    read_bits::<5>(reader)?;
    Ok(())
}

fn derive_coded_size(
    width_in_mbs: u32,
    height_in_map_units: u32,
    frame_mbs_only: bool,
) -> Result<Size> {
    let width = width_in_mbs
        .checked_mul(16)
        .ok_or(H264Error::IntegerOverflow)?;
    let frame_height_factor = if frame_mbs_only { 1 } else { 2 };
    let height = height_in_map_units
        .checked_mul(frame_height_factor)
        .and_then(|value| value.checked_mul(16))
        .ok_or(H264Error::IntegerOverflow)?;
    Ok(Size::new(width, height))
}

fn derive_visible_rect(
    coded_size: Size,
    frame_mbs_only: bool,
    crop_offsets: Option<[u32; 4]>,
) -> Result<Rect> {
    let Some([left, right, top, bottom]) = crop_offsets else {
        return Ok(Rect::new(0, 0, coded_size.width, coded_size.height));
    };

    // This decoder currently accepts only 4:2:0 SPS data.
    let crop_unit_x = 2u32;
    let crop_unit_y = if frame_mbs_only { 2u32 } else { 4u32 };
    let x = left
        .checked_mul(crop_unit_x)
        .ok_or(H264Error::IntegerOverflow)?;
    let y = top
        .checked_mul(crop_unit_y)
        .ok_or(H264Error::IntegerOverflow)?;
    let horizontal_crop = left
        .checked_add(right)
        .and_then(|value| value.checked_mul(crop_unit_x))
        .ok_or(H264Error::IntegerOverflow)?;
    let vertical_crop = top
        .checked_add(bottom)
        .and_then(|value| value.checked_mul(crop_unit_y))
        .ok_or(H264Error::IntegerOverflow)?;
    let width = coded_size
        .width
        .checked_sub(horizontal_crop)
        .ok_or(H264Error::InvalidSyntax("SPS crop exceeds coded width"))?;
    let height = coded_size
        .height
        .checked_sub(vertical_crop)
        .ok_or(H264Error::InvalidSyntax("SPS crop exceeds coded height"))?;

    if width == 0 || height == 0 {
        return Err(H264Error::InvalidSyntax(
            "SPS crop produces an empty visible rectangle",
        ));
    }
    Ok(Rect::new(x, y, width, height))
}

fn derive_display_size(visible_size: Size, sar: SampleAspectRatio) -> Result<Size> {
    let numerator = u64::from(visible_size.width)
        .checked_mul(u64::from(sar.width))
        .ok_or(H264Error::IntegerOverflow)?;
    let denominator = u64::from(sar.height);
    let display_width = numerator
        .checked_add(denominator / 2)
        .ok_or(H264Error::IntegerOverflow)?
        / denominator;
    let display_width = u32::try_from(display_width).map_err(|_| H264Error::IntegerOverflow)?;
    if display_width == 0 {
        return Err(H264Error::InvalidSyntax(
            "sample aspect ratio produces an empty display size",
        ));
    }
    Ok(Size::new(display_width, visible_size.height))
}

fn map_color_primaries(value: u8) -> ColorPrimaries {
    match value {
        2 => ColorPrimaries::Unspecified,
        1 => ColorPrimaries::Bt709,
        5 => ColorPrimaries::Bt601_625,
        6 => ColorPrimaries::Bt601_525,
        9 => ColorPrimaries::Bt2020,
        value => ColorPrimaries::Other(value),
    }
}

fn map_transfer_function(value: u8) -> TransferFunction {
    match value {
        2 => TransferFunction::Unspecified,
        1 => TransferFunction::Bt709,
        5 => TransferFunction::Bt470Bg,
        6 => TransferFunction::Smpte170M,
        8 => TransferFunction::Linear,
        13 => TransferFunction::Srgb,
        14 => TransferFunction::Bt2020TenBit,
        15 => TransferFunction::Bt2020TwelveBit,
        value => TransferFunction::Other(value),
    }
}

fn map_color_matrix(value: u8) -> ColorMatrix {
    match value {
        2 => ColorMatrix::Unspecified,
        0 => ColorMatrix::Identity,
        1 => ColorMatrix::Bt709,
        5 => ColorMatrix::Bt470Bg,
        6 => ColorMatrix::Smpte170M,
        9 => ColorMatrix::Bt2020NonConstantLuminance,
        10 => ColorMatrix::Bt2020ConstantLuminance,
        value => ColorMatrix::Other(value),
    }
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
fn read_u8(reader: &mut BitReader<'_>) -> Result<u8> {
    read_bits::<8>(reader).map(|value| value as u8)
}

#[inline]
fn read_ue(reader: &mut BitReader<'_>) -> Result<u32> {
    reader.read_ue().ok_or(H264Error::UnexpectedEof)
}

#[inline]
fn read_se(reader: &mut BitReader<'_>) -> Result<i32> {
    reader.read_se().ok_or(H264Error::UnexpectedEof)
}

fn add_u8(value: u32, addend: u8) -> Result<u8> {
    value
        .checked_add(u32::from(addend))
        .and_then(|value| u8::try_from(value).ok())
        .ok_or(H264Error::IntegerOverflow)
}

fn add_u32(value: u32, addend: u32) -> Result<u32> {
    value.checked_add(addend).ok_or(H264Error::IntegerOverflow)
}

#[cfg(test)]
mod tests {
    use bit_readers::BitReader;
    use decv_core::{ColorMatrix, ColorPrimaries, ColorRange, Size, TransferFunction};

    use super::{
        PicOrderCount, Profile, SampleAspectRatio, SequenceParameterSet, parse_sample_aspect_ratio,
    };

    #[test]
    fn parses_a_baseline_640_by_480_sps() {
        let mut writer = BitWriter::default();
        write_common_header(&mut writer, 66, 30, 0);
        writer.write_ue(0); // log2_max_frame_num_minus4
        writer.write_ue(0); // pic_order_cnt_type
        writer.write_ue(0); // log2_max_pic_order_cnt_lsb_minus4
        writer.write_ue(1); // max_num_ref_frames
        writer.write_flag(false); // gaps
        writer.write_ue(39); // 40 macroblocks wide
        writer.write_ue(29); // 30 map units high
        writer.write_flag(true); // frame_mbs_only
        writer.write_flag(true); // direct_8x8_inference
        writer.write_flag(false); // cropping
        writer.write_flag(false); // VUI
        let rbsp = writer.finish_rbsp();

        let sps = SequenceParameterSet::parse(&rbsp).unwrap();

        assert_eq!(sps.profile, Profile::Baseline);
        assert_eq!(sps.id, 0);
        assert_eq!(sps.coded_size, Size::new(640, 480));
        assert_eq!(sps.visible_rect.size(), Size::new(640, 480));
        assert_eq!(sps.display_size, Size::new(640, 480));
        assert_eq!(sps.log2_max_frame_num, 4);
        assert_eq!(
            sps.pic_order_count,
            PicOrderCount::Type0 {
                log2_max_pic_order_cnt_lsb: 4
            }
        );
    }

    #[test]
    fn parses_high_profile_crop_color_and_timing() {
        let mut writer = BitWriter::default();
        write_common_header(&mut writer, 100, 40, 3);
        writer.write_ue(1); // chroma_format_idc = 4:2:0
        writer.write_ue(0); // bit_depth_luma_minus8
        writer.write_ue(0); // bit_depth_chroma_minus8
        writer.write_flag(false); // transform bypass
        writer.write_flag(false); // scaling matrices
        writer.write_ue(0); // log2_max_frame_num_minus4
        writer.write_ue(0); // pic_order_cnt_type
        writer.write_ue(2); // log2_max_pic_order_cnt_lsb_minus4
        writer.write_ue(4); // max_num_ref_frames
        writer.write_flag(false); // gaps
        writer.write_ue(119); // 1920 / 16 - 1
        writer.write_ue(67); // 1088 / 16 - 1
        writer.write_flag(true); // frame_mbs_only
        writer.write_flag(true); // direct_8x8_inference
        writer.write_flag(true); // cropping
        writer.write_ue(0);
        writer.write_ue(0);
        writer.write_ue(0);
        writer.write_ue(4); // crop eight luma rows
        writer.write_flag(true); // VUI
        writer.write_flag(true); // aspect ratio
        writer.write_bits(1, 8); // square pixels
        writer.write_flag(false); // overscan
        writer.write_flag(true); // video signal
        writer.write_bits(5, 3); // unspecified video format
        writer.write_flag(false); // limited range
        writer.write_flag(true); // colour description
        writer.write_bits(1, 8); // BT.709 primaries
        writer.write_bits(1, 8); // BT.709 transfer
        writer.write_bits(1, 8); // BT.709 matrix
        writer.write_flag(false); // chroma location
        writer.write_flag(true); // timing
        writer.write_bits(1_001, 32);
        writer.write_bits(60_000, 32);
        writer.write_flag(true);
        writer.write_flag(false); // NAL HRD
        writer.write_flag(false); // VCL HRD
        writer.write_flag(false); // pic_struct_present
        writer.write_flag(false); // bitstream restrictions
        let rbsp = writer.finish_rbsp();

        let sps = SequenceParameterSet::parse(&rbsp).unwrap();
        let vui = sps.vui.unwrap();

        assert_eq!(sps.profile, Profile::High);
        assert_eq!(sps.coded_size, Size::new(1920, 1088));
        assert_eq!(sps.visible_rect.size(), Size::new(1920, 1080));
        assert_eq!(sps.display_size, Size::new(1920, 1080));
        assert_eq!(vui.sample_aspect_ratio, Some(SampleAspectRatio::default()));
        assert_eq!(vui.color.range, ColorRange::Limited);
        assert_eq!(vui.color.matrix, ColorMatrix::Bt709);
        assert_eq!(vui.color.primaries, ColorPrimaries::Bt709);
        assert_eq!(vui.color.transfer, TransferFunction::Bt709);
        assert_eq!(vui.timing.unwrap().num_units_in_tick, 1_001);
        assert_eq!(vui.timing.unwrap().time_scale, 60_000);
    }

    #[test]
    fn parses_scaling_lists_and_non_square_sar() {
        let mut writer = BitWriter::default();
        write_common_header(&mut writer, 100, 31, 1);
        writer.write_ue(1);
        writer.write_ue(0);
        writer.write_ue(0);
        writer.write_flag(false);
        writer.write_flag(true); // sequence scaling matrix
        writer.write_flag(true); // first 4x4 list
        for _ in 0..16 {
            writer.write_se(0);
        }
        for _ in 1..8 {
            writer.write_flag(false);
        }
        writer.write_ue(0);
        writer.write_ue(2); // POC type 2
        writer.write_ue(1);
        writer.write_flag(false);
        writer.write_ue(44); // 720
        writer.write_ue(35); // 576
        writer.write_flag(true);
        writer.write_flag(true);
        writer.write_flag(false);
        writer.write_flag(true); // VUI
        writer.write_flag(true); // SAR
        writer.write_bits(14, 8); // 4:3 sample aspect ratio
        writer.write_flag(false); // overscan
        writer.write_flag(false); // video signal
        writer.write_flag(false); // chroma location
        writer.write_flag(false); // timing
        writer.write_flag(false); // NAL HRD
        writer.write_flag(false); // VCL HRD
        writer.write_flag(false); // pic struct
        writer.write_flag(false); // restrictions
        let rbsp = writer.finish_rbsp();

        let sps = SequenceParameterSet::parse(&rbsp).unwrap();
        let matrices = sps.scaling_matrices.unwrap();

        assert_eq!(matrices.lists.len(), 8);
        assert_eq!(matrices.lists[0].as_ref().unwrap().values, vec![8; 16]);
        assert_eq!(sps.visible_rect.size(), Size::new(720, 576));
        assert_eq!(sps.display_size, Size::new(960, 576));
        assert_eq!(
            sps.vui.unwrap().sample_aspect_ratio,
            Some(SampleAspectRatio {
                width: 4,
                height: 3
            })
        );
    }

    #[test]
    fn preserves_unspecified_sample_aspect_ratio() {
        let mut reader = BitReader::new(&[0]);

        assert_eq!(parse_sample_aspect_ratio(&mut reader), Ok(None));
    }

    #[test]
    fn parses_type_one_poc_interlacing_and_hrd() {
        let mut writer = BitWriter::default();
        write_common_header(&mut writer, 77, 32, 2);
        writer.write_ue(0); // log2_max_frame_num_minus4
        writer.write_ue(1); // POC type 1
        writer.write_flag(false); // delta_pic_order_always_zero
        writer.write_se(-1); // non-reference offset
        writer.write_se(2); // top-to-bottom offset
        writer.write_ue(2); // reference cycle
        writer.write_se(1);
        writer.write_se(-1);
        writer.write_ue(2); // max_num_ref_frames
        writer.write_flag(true); // gaps allowed
        writer.write_ue(19); // 320 pixels
        writer.write_ue(14); // 15 map units * two fields = 480 pixels
        writer.write_flag(false); // frame_mbs_only
        writer.write_flag(true); // MBAFF
        writer.write_flag(true); // direct_8x8
        writer.write_flag(false); // crop
        writer.write_flag(true); // VUI
        writer.write_flag(false); // SAR unspecified
        writer.write_flag(true); // overscan info
        writer.write_flag(false); // overscan not appropriate
        writer.write_flag(false); // video signal unspecified
        writer.write_flag(true); // chroma location
        writer.write_ue(2);
        writer.write_ue(3);
        writer.write_flag(false); // timing
        writer.write_flag(true); // NAL HRD
        writer.write_ue(0); // one CPB
        writer.write_bits(0, 4); // bit-rate scale
        writer.write_bits(0, 4); // CPB-size scale
        writer.write_ue(0); // bit-rate value
        writer.write_ue(0); // CPB-size value
        writer.write_flag(true); // CBR
        for _ in 0..4 {
            writer.write_bits(23, 5);
        }
        writer.write_flag(false); // VCL HRD
        writer.write_flag(true); // low-delay HRD
        writer.write_flag(true); // pic_struct_present
        writer.write_flag(true); // restrictions
        writer.write_flag(true); // MVs over boundaries
        writer.write_ue(2);
        writer.write_ue(1);
        writer.write_ue(16);
        writer.write_ue(16);
        writer.write_ue(1);
        writer.write_ue(2);
        let rbsp = writer.finish_rbsp();

        let sps = SequenceParameterSet::parse(&rbsp).unwrap();
        let vui = sps.vui.unwrap();

        assert_eq!(sps.coded_size, Size::new(320, 480));
        assert!(sps.mb_adaptive_frame_field);
        assert_eq!(vui.sample_aspect_ratio, None);
        assert_eq!(vui.chroma_sample_loc_type_top_field, Some(2));
        assert_eq!(vui.chroma_sample_loc_type_bottom_field, Some(3));
        assert!(vui.pic_struct_present);
        assert_eq!(
            sps.pic_order_count,
            PicOrderCount::Type1 {
                delta_pic_order_always_zero: false,
                offset_for_non_ref_pic: -1,
                offset_for_top_to_bottom_field: 2,
                offset_for_ref_frame: vec![1, -1],
            }
        );
        assert_eq!(
            vui.bitstream_restrictions.unwrap().max_num_reorder_frames,
            1
        );
    }

    #[test]
    fn rejects_unsupported_or_invalid_sps_values() {
        let mut unsupported = BitWriter::default();
        write_common_header(&mut unsupported, 110, 40, 0);
        assert!(SequenceParameterSet::parse(&unsupported.finish_rbsp()).is_err());

        let mut invalid_reserved = BitWriter::default();
        invalid_reserved.write_bits(66, 8);
        invalid_reserved.write_bits(1, 8);
        invalid_reserved.write_bits(30, 8);
        invalid_reserved.write_ue(0);
        assert!(SequenceParameterSet::parse(&invalid_reserved.finish_rbsp()).is_err());
    }

    fn write_common_header(writer: &mut BitWriter, profile_idc: u8, level_idc: u8, sps_id: u32) {
        writer.write_bits(u64::from(profile_idc), 8);
        writer.write_bits(0, 8);
        writer.write_bits(u64::from(level_idc), 8);
        writer.write_ue(sps_id);
    }

    #[derive(Default)]
    struct BitWriter {
        bits: Vec<u8>,
    }

    impl BitWriter {
        fn write_flag(&mut self, value: bool) {
            self.bits.push(u8::from(value));
        }

        fn write_bits(&mut self, value: u64, count: u32) {
            for shift in (0..count).rev() {
                self.bits.push(((value >> shift) & 1) as u8);
            }
        }

        fn write_ue(&mut self, value: u32) {
            let code_num = u64::from(value) + 1;
            let width = 64 - code_num.leading_zeros();
            self.bits.extend(std::iter::repeat_n(0, width as usize - 1));
            self.write_bits(code_num, width);
        }

        fn write_se(&mut self, value: i32) {
            let code_num = if value <= 0 {
                value.unsigned_abs() * 2
            } else {
                value as u32 * 2 - 1
            };
            self.write_ue(code_num);
        }

        fn finish_rbsp(mut self) -> Vec<u8> {
            self.bits.push(1);
            while !self.bits.len().is_multiple_of(8) {
                self.bits.push(0);
            }

            let mut bytes = vec![0; self.bits.len() / 8];
            for (position, bit) in self.bits.into_iter().enumerate() {
                bytes[position / 8] |= bit << (7 - position % 8);
            }
            bytes
        }
    }
}
