//! Picture-order derivation and decoded-picture metadata.

use crate::{
    H264Error, MemoryManagementOperation, NalHeader, NalUnitType, ParsedSliceHeader, PicOrderCount,
    ReferencePictureMarking, Result,
};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct FieldOrderCount {
    pub top: Option<i32>,
    pub bottom: Option<i32>,
}

impl FieldOrderCount {
    pub fn picture_order_count(self) -> i32 {
        match (self.top, self.bottom) {
            (Some(top), Some(bottom)) => top.min(bottom),
            (Some(top), None) => top,
            (None, Some(bottom)) => bottom,
            (None, None) => unreachable!("a coded picture has at least one field"),
        }
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct PictureOrderCount {
    /// Values used while decoding the current picture.
    pub decoding: FieldOrderCount,
    /// Values retained after MMCO 5 normalization for later pictures.
    pub stored: FieldOrderCount,
}

#[derive(Debug, Default, Clone)]
pub struct PictureOrderCountState {
    prev_pic_order_cnt_msb: i32,
    prev_pic_order_cnt_lsb: u32,
    prev_reference_had_mmco5: bool,
    prev_reference_was_bottom_field: bool,
    prev_reference_top_field_order_cnt: Option<i32>,
    prev_frame_num: u32,
    prev_frame_num_offset: u64,
}

impl PictureOrderCountState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Derives and commits POC state for the first slice of one coded picture.
    pub fn derive(
        &mut self,
        parsed: &ParsedSliceHeader,
        nal_header: NalHeader,
    ) -> Result<PictureOrderCount> {
        let is_idr = nal_header.unit_type == NalUnitType::IdrSlice;
        let has_mmco5 = includes_mmco5(&parsed.header.reference_picture_marking);
        let sps = &parsed.parameter_sets.sequence;

        let (decoding, type0_msb) = match &sps.pic_order_count {
            PicOrderCount::Type0 {
                log2_max_pic_order_cnt_lsb,
            } => {
                let (order, msb) =
                    self.derive_type0(parsed, is_idr, *log2_max_pic_order_cnt_lsb)?;
                (order, Some(msb))
            }
            PicOrderCount::Type1 {
                offset_for_non_ref_pic,
                offset_for_top_to_bottom_field,
                offset_for_ref_frame,
                ..
            } => (
                self.derive_type1(
                    parsed,
                    nal_header.nal_ref_idc,
                    is_idr,
                    *offset_for_non_ref_pic,
                    *offset_for_top_to_bottom_field,
                    offset_for_ref_frame,
                )?,
                None,
            ),
            PicOrderCount::Type2 => (
                self.derive_type2(parsed, nal_header.nal_ref_idc, is_idr)?,
                None,
            ),
        };

        let mut stored = decoding;
        if has_mmco5 {
            let adjustment = stored.picture_order_count();
            stored.top = stored
                .top
                .map(|value| {
                    value
                        .checked_sub(adjustment)
                        .ok_or(H264Error::IntegerOverflow)
                })
                .transpose()?;
            stored.bottom = stored
                .bottom
                .map(|value| {
                    value
                        .checked_sub(adjustment)
                        .ok_or(H264Error::IntegerOverflow)
                })
                .transpose()?;
        }

        match &sps.pic_order_count {
            PicOrderCount::Type0 { .. } => {
                if nal_header.nal_ref_idc != 0 {
                    self.prev_reference_had_mmco5 = has_mmco5;
                    self.prev_reference_was_bottom_field =
                        parsed.header.field_picture && parsed.header.bottom_field;
                    self.prev_reference_top_field_order_cnt = stored.top;
                    if !has_mmco5 {
                        self.prev_pic_order_cnt_msb =
                            type0_msb.expect("type 0 derivation returns its MSB");
                        self.prev_pic_order_cnt_lsb = parsed
                            .header
                            .picture_order
                            .pic_order_cnt_lsb
                            .expect("type 0 slice has pic_order_cnt_lsb");
                    }
                }
            }
            PicOrderCount::Type1 { .. } | PicOrderCount::Type2 => {
                if has_mmco5 {
                    self.prev_frame_num = 0;
                    self.prev_frame_num_offset = 0;
                } else {
                    self.prev_frame_num = parsed.header.frame_num;
                    self.prev_frame_num_offset = self.frame_num_offset(
                        parsed.header.frame_num,
                        is_idr,
                        sps.log2_max_frame_num,
                    )?;
                }
            }
        }

        Ok(PictureOrderCount { decoding, stored })
    }

    pub fn reset(&mut self) {
        *self = Self::default();
    }

    fn derive_type0(
        &self,
        parsed: &ParsedSliceHeader,
        is_idr: bool,
        log2_max_pic_order_cnt_lsb: u8,
    ) -> Result<(FieldOrderCount, i32)> {
        let lsb = parsed
            .header
            .picture_order
            .pic_order_cnt_lsb
            .ok_or(H264Error::InvalidSyntax(
                "missing pic_order_cnt_lsb for POC type 0",
            ))?;
        let max_lsb = 1i64 << log2_max_pic_order_cnt_lsb;
        let (prev_msb, prev_lsb) = if is_idr {
            (0i64, 0i64)
        } else if self.prev_reference_had_mmco5 {
            let lsb = if self.prev_reference_was_bottom_field {
                0
            } else {
                i64::from(self.prev_reference_top_field_order_cnt.unwrap_or(0))
            };
            (0, lsb)
        } else {
            (
                i64::from(self.prev_pic_order_cnt_msb),
                i64::from(self.prev_pic_order_cnt_lsb),
            )
        };
        let lsb = i64::from(lsb);
        let msb = if lsb < prev_lsb && prev_lsb - lsb >= max_lsb / 2 {
            prev_msb
                .checked_add(max_lsb)
                .ok_or(H264Error::IntegerOverflow)?
        } else if lsb > prev_lsb && lsb - prev_lsb > max_lsb / 2 {
            prev_msb
                .checked_sub(max_lsb)
                .ok_or(H264Error::IntegerOverflow)?
        } else {
            prev_msb
        };

        let field_value = checked_poc(msb.checked_add(lsb).ok_or(H264Error::IntegerOverflow)?)?;
        let order = if !parsed.header.field_picture {
            let delta = parsed
                .header
                .picture_order
                .delta_pic_order_bottom
                .unwrap_or(0);
            FieldOrderCount {
                top: Some(field_value),
                bottom: Some(
                    field_value
                        .checked_add(delta)
                        .ok_or(H264Error::IntegerOverflow)?,
                ),
            }
        } else if parsed.header.bottom_field {
            FieldOrderCount {
                top: None,
                bottom: Some(field_value),
            }
        } else {
            FieldOrderCount {
                top: Some(field_value),
                bottom: None,
            }
        };
        Ok((order, checked_poc(msb)?))
    }

    #[allow(clippy::too_many_arguments)]
    fn derive_type1(
        &self,
        parsed: &ParsedSliceHeader,
        nal_ref_idc: u8,
        is_idr: bool,
        offset_for_non_ref_pic: i32,
        offset_for_top_to_bottom_field: i32,
        offset_for_ref_frame: &[i32],
    ) -> Result<FieldOrderCount> {
        let frame_num_offset = self.frame_num_offset(
            parsed.header.frame_num,
            is_idr,
            parsed.parameter_sets.sequence.log2_max_frame_num,
        )?;
        let mut abs_frame_num = if offset_for_ref_frame.is_empty() {
            0
        } else {
            frame_num_offset
                .checked_add(u64::from(parsed.header.frame_num))
                .ok_or(H264Error::IntegerOverflow)?
        };
        if nal_ref_idc == 0 && abs_frame_num > 0 {
            abs_frame_num -= 1;
        }

        let mut expected = 0i64;
        if abs_frame_num > 0 {
            let cycle_length = offset_for_ref_frame.len() as u64;
            let cycle_count = (abs_frame_num - 1) / cycle_length;
            let frame_in_cycle = ((abs_frame_num - 1) % cycle_length) as usize;
            let delta_per_cycle = offset_for_ref_frame.iter().try_fold(0i64, |sum, &offset| {
                sum.checked_add(i64::from(offset))
                    .ok_or(H264Error::IntegerOverflow)
            })?;
            expected = i64::try_from(cycle_count)
                .ok()
                .and_then(|count| count.checked_mul(delta_per_cycle))
                .ok_or(H264Error::IntegerOverflow)?;
            for &offset in &offset_for_ref_frame[..=frame_in_cycle] {
                expected = expected
                    .checked_add(i64::from(offset))
                    .ok_or(H264Error::IntegerOverflow)?;
            }
        }
        if nal_ref_idc == 0 {
            expected = expected
                .checked_add(i64::from(offset_for_non_ref_pic))
                .ok_or(H264Error::IntegerOverflow)?;
        }

        let delta0 = i64::from(parsed.header.picture_order.delta_pic_order[0].unwrap_or(0));
        let delta1 = i64::from(parsed.header.picture_order.delta_pic_order[1].unwrap_or(0));
        let top = checked_poc(
            expected
                .checked_add(delta0)
                .ok_or(H264Error::IntegerOverflow)?,
        )?;
        if !parsed.header.field_picture {
            let bottom = i64::from(top)
                .checked_add(i64::from(offset_for_top_to_bottom_field))
                .and_then(|value| value.checked_add(delta1))
                .ok_or(H264Error::IntegerOverflow)?;
            Ok(FieldOrderCount {
                top: Some(top),
                bottom: Some(checked_poc(bottom)?),
            })
        } else if parsed.header.bottom_field {
            let bottom = expected
                .checked_add(i64::from(offset_for_top_to_bottom_field))
                .and_then(|value| value.checked_add(delta0))
                .ok_or(H264Error::IntegerOverflow)?;
            Ok(FieldOrderCount {
                top: None,
                bottom: Some(checked_poc(bottom)?),
            })
        } else {
            Ok(FieldOrderCount {
                top: Some(top),
                bottom: None,
            })
        }
    }

    fn derive_type2(
        &self,
        parsed: &ParsedSliceHeader,
        nal_ref_idc: u8,
        is_idr: bool,
    ) -> Result<FieldOrderCount> {
        let frame_num_offset = self.frame_num_offset(
            parsed.header.frame_num,
            is_idr,
            parsed.parameter_sets.sequence.log2_max_frame_num,
        )?;
        let absolute_frame_num = frame_num_offset
            .checked_add(u64::from(parsed.header.frame_num))
            .ok_or(H264Error::IntegerOverflow)?;
        let value = if is_idr {
            0
        } else {
            let doubled = absolute_frame_num
                .checked_mul(2)
                .ok_or(H264Error::IntegerOverflow)?;
            if nal_ref_idc == 0 {
                doubled.checked_sub(1).ok_or(H264Error::IntegerOverflow)?
            } else {
                doubled
            }
        };
        let value = i64::try_from(value).map_err(|_| H264Error::IntegerOverflow)?;
        let value = checked_poc(value)?;
        if !parsed.header.field_picture {
            Ok(FieldOrderCount {
                top: Some(value),
                bottom: Some(value),
            })
        } else if parsed.header.bottom_field {
            Ok(FieldOrderCount {
                top: None,
                bottom: Some(value),
            })
        } else {
            Ok(FieldOrderCount {
                top: Some(value),
                bottom: None,
            })
        }
    }

    fn frame_num_offset(
        &self,
        frame_num: u32,
        is_idr: bool,
        log2_max_frame_num: u8,
    ) -> Result<u64> {
        if is_idr {
            return Ok(0);
        }
        if self.prev_frame_num > frame_num {
            self.prev_frame_num_offset
                .checked_add(1u64 << log2_max_frame_num)
                .ok_or(H264Error::IntegerOverflow)
        } else {
            Ok(self.prev_frame_num_offset)
        }
    }
}

fn includes_mmco5(marking: &ReferencePictureMarking) -> bool {
    matches!(
        marking,
        ReferencePictureMarking::Adaptive(operations)
            if operations.contains(&MemoryManagementOperation::ForgetAll)
    )
}

fn checked_poc(value: i64) -> Result<i32> {
    i32::try_from(value)
        .map_err(|_| H264Error::InvalidSyntax("picture order count is outside the i32 range"))
}
