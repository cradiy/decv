//! Decoded Picture Buffer and reference-picture management.

use std::sync::Arc;

use crate::{
    H264Error, MemoryManagementOperation, ReferenceListModification, Result, Yuv420Picture,
};

pub type ReferencePicture = Arc<Yuv420Picture>;
pub type DefaultReferenceList = Vec<ReferencePicture>;
pub type ActiveReferenceList = Vec<Option<ReferencePicture>>;
pub type DefaultBReferenceLists = (DefaultReferenceList, DefaultReferenceList);
pub type ActiveBReferenceLists = (ActiveReferenceList, ActiveReferenceList);
pub type ActiveReferenceInfoList = Vec<Option<ActiveReferenceInfo>>;
pub type ActiveBReferenceInfoLists = (ActiveReferenceInfoList, ActiveReferenceInfoList);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ReferenceId(pub(crate) u64);

impl ReferenceId {
    #[inline]
    pub const fn get(self) -> u64 {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReferenceKind {
    ShortTerm,
    LongTerm { frame_index: u32 },
}

#[derive(Debug, Clone)]
pub struct DpbReference {
    pub id: ReferenceId,
    pub frame_num: u32,
    pub picture_order_count: i32,
    pub kind: ReferenceKind,
    pub picture: ReferencePicture,
    pub motion: Arc<crate::ReferenceMotionField>,
}

#[derive(Debug, Clone)]
pub struct ActiveReferenceInfo {
    pub id: ReferenceId,
    pub picture_order_count: i32,
    pub kind: ReferenceKind,
    pub picture: ReferencePicture,
    pub motion: Arc<crate::ReferenceMotionField>,
}

/// Reference-picture subset of the decoded picture buffer.
///
/// This DPB layer covers progressive frame pictures, IDR reset, default P/B
/// reference-list ordering, list modification, and reference-picture marking.
/// Display reordering is intentionally a separate later stage.
#[derive(Debug, Clone)]
pub struct DecodedPictureBuffer {
    max_num_ref_frames: usize,
    max_frame_num: u32,
    max_long_term_frame_idx: Option<u32>,
    next_reference_id: u64,
    references: Vec<DpbReference>,
}

impl DecodedPictureBuffer {
    pub fn new(max_num_ref_frames: u32, log2_max_frame_num: u8) -> Result<Self> {
        if !(4..=16).contains(&log2_max_frame_num) {
            return Err(H264Error::InvalidSyntax(
                "DPB log2_max_frame_num is outside 4..=16",
            ));
        }
        let max_num_ref_frames =
            usize::try_from(max_num_ref_frames).map_err(|_| H264Error::IntegerOverflow)?;
        Ok(Self {
            max_num_ref_frames,
            max_frame_num: 1u32 << log2_max_frame_num,
            max_long_term_frame_idx: None,
            next_reference_id: 1,
            references: Vec::with_capacity(max_num_ref_frames.max(1)),
        })
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.references.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.references.is_empty()
    }

    pub fn clear(&mut self) {
        self.references.clear();
        self.max_long_term_frame_idx = None;
        self.next_reference_id = 1;
    }

    /// Builds the default reference picture List 0 for a progressive P slice:
    /// short-term pictures by descending PicNum, then long-term pictures by
    /// ascending long-term frame index.
    pub fn default_p_list0(&self, current_frame_num: u32) -> Result<DefaultReferenceList> {
        self.ensure_frame_num(current_frame_num)?;
        Ok(self
            .ordered_p_references(current_frame_num)
            .into_iter()
            .map(|reference| reference.picture.clone())
            .collect())
    }

    /// Builds and modifies an active P List 0. Missing initial entries are
    /// retained as `None`, so a legal stream may declare more active indices
    /// than it actually selects without weakening validation at use sites.
    pub fn p_list0(
        &self,
        current_frame_num: u32,
        active_count: u8,
        modifications: &[ReferenceListModification],
    ) -> Result<ActiveReferenceList> {
        self.ensure_frame_num(current_frame_num)?;
        let ordered = self.ordered_p_references(current_frame_num);
        self.active_reference_list(current_frame_num, active_count, modifications, ordered)
    }

    pub fn p_reference_info_list(
        &self,
        current_frame_num: u32,
        active_count: u8,
        modifications: &[ReferenceListModification],
    ) -> Result<ActiveReferenceInfoList> {
        let list = self.p_list0(current_frame_num, active_count, modifications)?;
        self.reference_info_for_active_list(list)
    }

    /// Builds the default reference lists for a progressive B slice.
    ///
    /// List 0 starts with short-term pictures whose POC is no later than the
    /// current picture in descending order, then later pictures in ascending
    /// order. List 1 uses the opposite groups. Long-term pictures follow in
    /// ascending frame-index order. When both lists would be identical, the
    /// first two List-1 entries are swapped.
    pub fn default_b_lists(
        &self,
        current_picture_order_count: i32,
    ) -> Result<DefaultBReferenceLists> {
        let (list0, list1) = self.ordered_b_references(current_picture_order_count)?;
        Ok((
            list0
                .into_iter()
                .map(|reference| reference.picture.clone())
                .collect(),
            list1
                .into_iter()
                .map(|reference| reference.picture.clone())
                .collect(),
        ))
    }

    /// Builds and independently modifies both active progressive B lists.
    pub fn b_lists(
        &self,
        current_frame_num: u32,
        current_picture_order_count: i32,
        active_count_l0: u8,
        modifications_l0: &[ReferenceListModification],
        active_count_l1: u8,
        modifications_l1: &[ReferenceListModification],
    ) -> Result<ActiveBReferenceLists> {
        self.ensure_frame_num(current_frame_num)?;
        let (ordered_l0, ordered_l1) = self.ordered_b_references(current_picture_order_count)?;
        Ok((
            self.active_reference_list(
                current_frame_num,
                active_count_l0,
                modifications_l0,
                ordered_l0,
            )?,
            self.active_reference_list(
                current_frame_num,
                active_count_l1,
                modifications_l1,
                ordered_l1,
            )?,
        ))
    }

    /// Builds active B lists while retaining stable DPB identity, POC, and
    /// short/long-term classification for Direct and implicit weighting.
    pub fn b_reference_info_lists(
        &self,
        current_frame_num: u32,
        current_picture_order_count: i32,
        active_count_l0: u8,
        modifications_l0: &[ReferenceListModification],
        active_count_l1: u8,
        modifications_l1: &[ReferenceListModification],
    ) -> Result<ActiveBReferenceInfoLists> {
        let (list0, list1) = self.b_lists(
            current_frame_num,
            current_picture_order_count,
            active_count_l0,
            modifications_l0,
            active_count_l1,
            modifications_l1,
        )?;
        Ok((
            self.reference_info_for_active_list(list0)?,
            self.reference_info_for_active_list(list1)?,
        ))
    }

    fn reference_info_for_active_list(
        &self,
        list: ActiveReferenceList,
    ) -> Result<ActiveReferenceInfoList> {
        list.into_iter()
            .map(|picture| {
                picture
                    .map(|picture| {
                        let reference = self
                            .references
                            .iter()
                            .find(|reference| Arc::ptr_eq(&reference.picture, &picture))
                            .ok_or(H264Error::InvalidSyntax(
                                "active reference is not present in the DPB",
                            ))?;
                        Ok(ActiveReferenceInfo {
                            id: reference.id,
                            picture_order_count: reference.picture_order_count,
                            kind: reference.kind,
                            picture,
                            motion: reference.motion.clone(),
                        })
                    })
                    .transpose()
            })
            .collect()
    }

    fn active_reference_list<'a>(
        &'a self,
        current_frame_num: u32,
        active_count: u8,
        modifications: &[ReferenceListModification],
        ordered: Vec<&'a DpbReference>,
    ) -> Result<ActiveReferenceList> {
        if active_count == 0 || active_count > 32 {
            return Err(H264Error::InvalidSyntax(
                "reference-list active count is outside 1..=32",
            ));
        }
        let active_count = usize::from(active_count);
        let mut list: Vec<Option<&DpbReference>> =
            ordered.into_iter().take(active_count).map(Some).collect();
        list.resize(active_count + 1, None);

        let mut reference_index = 0usize;
        let mut predicted_pic_num = i64::from(current_frame_num);
        for modification in modifications {
            if reference_index >= active_count {
                return Err(H264Error::InvalidSyntax(
                    "reference-list modifications exceed the active list",
                ));
            }
            let selected = match *modification {
                ReferenceListModification::SubtractPicNum { abs_diff_pic_num } => {
                    predicted_pic_num = (predicted_pic_num - i64::from(abs_diff_pic_num))
                        .rem_euclid(i64::from(self.max_frame_num));
                    self.short_term_by_modified_pic_num(current_frame_num, predicted_pic_num)?
                }
                ReferenceListModification::AddPicNum { abs_diff_pic_num } => {
                    predicted_pic_num = (predicted_pic_num + i64::from(abs_diff_pic_num))
                        .rem_euclid(i64::from(self.max_frame_num));
                    self.short_term_by_modified_pic_num(current_frame_num, predicted_pic_num)?
                }
                ReferenceListModification::LongTerm { long_term_pic_num } => self
                    .references
                    .iter()
                    .find(|reference| {
                        reference.kind
                            == ReferenceKind::LongTerm {
                                frame_index: long_term_pic_num,
                            }
                    })
                    .ok_or(H264Error::InvalidSyntax(
                        "P List 0 modification names a missing long-term reference",
                    ))?,
            };
            for index in (reference_index + 1..=active_count).rev() {
                list[index] = list[index - 1];
            }
            list[reference_index] = Some(selected);
            reference_index += 1;

            let mut write = reference_index;
            for read in reference_index..=active_count {
                if list[read].is_some_and(|reference| !std::ptr::eq(reference, selected)) {
                    list[write] = list[read];
                    write += 1;
                }
            }
            list[write..=active_count].fill(None);
        }
        list.truncate(active_count);
        Ok(list
            .into_iter()
            .map(|reference| reference.map(|reference| reference.picture.clone()))
            .collect())
    }

    /// Clears prior references and stores an IDR picture as short-term or
    /// long-term frame index zero.
    pub fn store_idr(
        &mut self,
        picture_order_count: i32,
        picture: ReferencePicture,
        long_term_reference: bool,
    ) -> Result<()> {
        let motion = Arc::new(crate::ReferenceMotionField::all_intra(
            picture.coded_size(),
        )?);
        self.store_idr_with_motion(picture_order_count, picture, motion, long_term_reference)
    }

    pub fn store_idr_with_motion(
        &mut self,
        picture_order_count: i32,
        picture: ReferencePicture,
        motion: Arc<crate::ReferenceMotionField>,
        long_term_reference: bool,
    ) -> Result<()> {
        let kind = if long_term_reference {
            self.max_long_term_frame_idx = Some(0);
            ReferenceKind::LongTerm { frame_index: 0 }
        } else {
            self.max_long_term_frame_idx = None;
            ReferenceKind::ShortTerm
        };
        self.references.clear();
        self.next_reference_id = 1;
        let id = self.allocate_reference_id()?;
        self.references.push(DpbReference {
            id,
            frame_num: 0,
            picture_order_count,
            kind,
            picture,
            motion,
        });
        Ok(())
    }

    /// Applies sliding-window marking and stores the current reference frame
    /// as short-term.
    pub fn store_short_term(
        &mut self,
        frame_num: u32,
        picture_order_count: i32,
        picture: ReferencePicture,
    ) -> Result<()> {
        let motion = Arc::new(crate::ReferenceMotionField::all_intra(
            picture.coded_size(),
        )?);
        self.store_short_term_with_motion(frame_num, picture_order_count, picture, motion)
    }

    pub fn store_short_term_with_motion(
        &mut self,
        frame_num: u32,
        picture_order_count: i32,
        picture: ReferencePicture,
        motion: Arc<crate::ReferenceMotionField>,
    ) -> Result<()> {
        self.ensure_frame_num(frame_num)?;
        self.ensure_can_store(&picture)?;
        if self.references.iter().any(|reference| {
            reference.kind == ReferenceKind::ShortTerm && reference.frame_num == frame_num
        }) {
            return Err(H264Error::InvalidSyntax(
                "DPB already contains this short-term frame_num",
            ));
        }
        if self.references.len() == self.reference_limit() {
            let oldest = self
                .references
                .iter()
                .enumerate()
                .filter(|(_, reference)| reference.kind == ReferenceKind::ShortTerm)
                .min_by_key(|(_, reference)| {
                    frame_num_wrap(reference.frame_num, frame_num, self.max_frame_num)
                })
                .map(|(index, _)| index)
                .ok_or(H264Error::InvalidSyntax(
                    "sliding-window DPB has no short-term reference to remove",
                ))?;
            self.references.remove(oldest);
        }
        let id = self.allocate_reference_id()?;
        self.references.push(DpbReference {
            id,
            frame_num,
            picture_order_count,
            kind: ReferenceKind::ShortTerm,
            picture,
            motion,
        });
        Ok(())
    }

    /// Applies progressive-frame MMCO 1 through 6 and stores the current
    /// decoded reference picture. The complete command sequence is atomic.
    pub fn store_adaptive(
        &mut self,
        frame_num: u32,
        picture_order_count: i32,
        picture: ReferencePicture,
        operations: &[MemoryManagementOperation],
    ) -> Result<()> {
        let motion = Arc::new(crate::ReferenceMotionField::all_intra(
            picture.coded_size(),
        )?);
        self.store_adaptive_with_motion(frame_num, picture_order_count, picture, motion, operations)
    }

    pub fn store_adaptive_with_motion(
        &mut self,
        frame_num: u32,
        picture_order_count: i32,
        picture: ReferencePicture,
        motion: Arc<crate::ReferenceMotionField>,
        operations: &[MemoryManagementOperation],
    ) -> Result<()> {
        self.ensure_frame_num(frame_num)?;
        self.ensure_can_store(&picture)?;
        let mut references = self.references.clone();
        let mut max_long_term_frame_idx = self.max_long_term_frame_idx;
        let mut current_kind = ReferenceKind::ShortTerm;

        for operation in operations {
            match *operation {
                MemoryManagementOperation::ForgetShortTerm {
                    difference_of_pic_nums,
                } => {
                    let index = short_term_operation_index(
                        &references,
                        frame_num,
                        self.max_frame_num,
                        difference_of_pic_nums,
                    )?;
                    references.remove(index);
                }
                MemoryManagementOperation::ForgetLongTerm { long_term_pic_num } => {
                    let index = long_term_index(&references, long_term_pic_num).ok_or(
                        H264Error::InvalidSyntax("MMCO 2 names a missing long-term reference"),
                    )?;
                    references.remove(index);
                }
                MemoryManagementOperation::AssignLongTerm {
                    difference_of_pic_nums,
                    long_term_frame_idx,
                } => {
                    ensure_long_term_index(max_long_term_frame_idx, long_term_frame_idx)?;
                    if let Some(index) = long_term_index(&references, long_term_frame_idx) {
                        references.remove(index);
                    }
                    let index = short_term_operation_index(
                        &references,
                        frame_num,
                        self.max_frame_num,
                        difference_of_pic_nums,
                    )?;
                    references[index].kind = ReferenceKind::LongTerm {
                        frame_index: long_term_frame_idx,
                    };
                }
                MemoryManagementOperation::LimitLongTerm {
                    max_long_term_frame_idx_plus1,
                } => {
                    references.retain(|reference| match reference.kind {
                        ReferenceKind::ShortTerm => true,
                        ReferenceKind::LongTerm { frame_index } => {
                            frame_index < max_long_term_frame_idx_plus1
                        }
                    });
                    max_long_term_frame_idx = max_long_term_frame_idx_plus1.checked_sub(1);
                }
                MemoryManagementOperation::ForgetAll => {
                    references.clear();
                    max_long_term_frame_idx = None;
                }
                MemoryManagementOperation::MarkCurrentLongTerm {
                    long_term_frame_idx,
                } => {
                    ensure_long_term_index(max_long_term_frame_idx, long_term_frame_idx)?;
                    if let Some(index) = long_term_index(&references, long_term_frame_idx) {
                        references.remove(index);
                    }
                    current_kind = ReferenceKind::LongTerm {
                        frame_index: long_term_frame_idx,
                    };
                }
            }
        }

        if current_kind == ReferenceKind::ShortTerm
            && references.iter().any(|reference| {
                reference.kind == ReferenceKind::ShortTerm && reference.frame_num == frame_num
            })
        {
            return Err(H264Error::InvalidSyntax(
                "adaptive DPB already contains current short-term frame_num",
            ));
        }
        if references.len() >= self.reference_limit() {
            return Err(H264Error::InvalidSyntax(
                "adaptive memory control exceeds max_num_ref_frames",
            ));
        }
        let id = ReferenceId(self.next_reference_id);
        let next_reference_id = self
            .next_reference_id
            .checked_add(1)
            .ok_or(H264Error::IntegerOverflow)?;
        references.push(DpbReference {
            id,
            frame_num,
            picture_order_count,
            kind: current_kind,
            picture,
            motion,
        });
        self.references = references;
        self.max_long_term_frame_idx = max_long_term_frame_idx;
        self.next_reference_id = next_reference_id;
        Ok(())
    }

    fn allocate_reference_id(&mut self) -> Result<ReferenceId> {
        let id = ReferenceId(self.next_reference_id);
        self.next_reference_id = self
            .next_reference_id
            .checked_add(1)
            .ok_or(H264Error::IntegerOverflow)?;
        Ok(id)
    }

    fn ensure_frame_num(&self, frame_num: u32) -> Result<()> {
        if frame_num >= self.max_frame_num {
            return Err(H264Error::InvalidSyntax("frame_num exceeds MaxFrameNum"));
        }
        Ok(())
    }

    fn ordered_p_references(&self, current_frame_num: u32) -> Vec<&DpbReference> {
        let mut short: Vec<&DpbReference> = self
            .references
            .iter()
            .filter(|reference| reference.kind == ReferenceKind::ShortTerm)
            .collect();
        short.sort_unstable_by_key(|reference| {
            std::cmp::Reverse(frame_num_wrap(
                reference.frame_num,
                current_frame_num,
                self.max_frame_num,
            ))
        });
        let mut long: Vec<&DpbReference> = self
            .references
            .iter()
            .filter(|reference| matches!(reference.kind, ReferenceKind::LongTerm { .. }))
            .collect();
        long.sort_unstable_by_key(|reference| match reference.kind {
            ReferenceKind::LongTerm { frame_index } => frame_index,
            ReferenceKind::ShortTerm => unreachable!(),
        });
        short.extend(long);
        short
    }

    fn ordered_b_references(
        &self,
        current_picture_order_count: i32,
    ) -> Result<(Vec<&DpbReference>, Vec<&DpbReference>)> {
        let mut earlier = Vec::new();
        let mut later = Vec::new();
        let mut long = Vec::new();
        for reference in &self.references {
            match reference.kind {
                ReferenceKind::ShortTerm
                    if reference.picture_order_count <= current_picture_order_count =>
                {
                    earlier.push(reference);
                }
                ReferenceKind::ShortTerm => later.push(reference),
                ReferenceKind::LongTerm { .. } => long.push(reference),
            }
        }
        earlier.sort_unstable_by_key(|reference| std::cmp::Reverse(reference.picture_order_count));
        later.sort_unstable_by_key(|reference| reference.picture_order_count);
        long.sort_unstable_by_key(|reference| match reference.kind {
            ReferenceKind::LongTerm { frame_index } => frame_index,
            ReferenceKind::ShortTerm => unreachable!(),
        });

        let mut list0 = Vec::with_capacity(self.references.len());
        list0.extend(earlier.iter().copied());
        list0.extend(later.iter().copied());
        list0.extend(long.iter().copied());

        let mut list1 = Vec::with_capacity(self.references.len());
        list1.extend(later);
        list1.extend(earlier);
        list1.extend(long);
        if list0.len() > 1
            && list0
                .iter()
                .zip(&list1)
                .all(|(left, right)| std::ptr::eq(*left, *right))
        {
            list1.swap(0, 1);
        }
        Ok((list0, list1))
    }

    fn short_term_by_modified_pic_num(
        &self,
        current_frame_num: u32,
        pic_num_no_wrap: i64,
    ) -> Result<&DpbReference> {
        let pic_num = if pic_num_no_wrap > i64::from(current_frame_num) {
            pic_num_no_wrap - i64::from(self.max_frame_num)
        } else {
            pic_num_no_wrap
        };
        self.references
            .iter()
            .find(|reference| {
                reference.kind == ReferenceKind::ShortTerm
                    && frame_num_wrap(reference.frame_num, current_frame_num, self.max_frame_num)
                        == pic_num
            })
            .ok_or(H264Error::InvalidSyntax(
                "reference-list modification names a missing short-term reference",
            ))
    }

    fn ensure_can_store(&self, picture: &Yuv420Picture) -> Result<()> {
        if self
            .references
            .first()
            .is_some_and(|reference| reference.picture.coded_size() != picture.coded_size())
        {
            return Err(H264Error::InvalidSyntax(
                "DPB reference picture coded size changed without reset",
            ));
        }
        Ok(())
    }

    #[inline]
    fn reference_limit(&self) -> usize {
        self.max_num_ref_frames.max(1)
    }
}

fn short_term_operation_index(
    references: &[DpbReference],
    current_frame_num: u32,
    max_frame_num: u32,
    difference_of_pic_nums: u32,
) -> Result<usize> {
    if difference_of_pic_nums == 0 {
        return Err(H264Error::InvalidSyntax(
            "MMCO short-term picture difference must be non-zero",
        ));
    }
    let target_pic_num = i64::from(current_frame_num) - i64::from(difference_of_pic_nums);
    references
        .iter()
        .position(|reference| {
            reference.kind == ReferenceKind::ShortTerm
                && frame_num_wrap(reference.frame_num, current_frame_num, max_frame_num)
                    == target_pic_num
        })
        .ok_or(H264Error::InvalidSyntax(
            "MMCO names a missing short-term reference",
        ))
}

fn long_term_index(references: &[DpbReference], frame_index: u32) -> Option<usize> {
    references
        .iter()
        .position(|reference| reference.kind == ReferenceKind::LongTerm { frame_index })
}

fn ensure_long_term_index(maximum: Option<u32>, frame_index: u32) -> Result<()> {
    if maximum.is_none_or(|maximum| frame_index > maximum) {
        return Err(H264Error::InvalidSyntax(
            "MMCO long-term frame index exceeds MaxLongTermFrameIdx",
        ));
    }
    Ok(())
}

#[inline]
fn frame_num_wrap(frame_num: u32, current_frame_num: u32, max_frame_num: u32) -> i64 {
    if frame_num > current_frame_num {
        i64::from(frame_num) - i64::from(max_frame_num)
    } else {
        i64::from(frame_num)
    }
}

#[cfg(test)]
mod tests {
    use decv_core::Size;

    use super::*;

    fn picture(value: u8) -> Arc<Yuv420Picture> {
        let mut picture = Yuv420Picture::new(Size::new(16, 16)).unwrap();
        picture.planes_mut().0.fill(value);
        Arc::new(picture)
    }

    fn luma_value(picture: &Yuv420Picture) -> u8 {
        picture.planes().0[0]
    }

    #[test]
    fn orders_short_term_references_across_frame_num_wrap() {
        let mut dpb = DecodedPictureBuffer::new(4, 4).unwrap();
        dpb.store_short_term(14, 0, picture(14)).unwrap();
        dpb.store_short_term(15, 1, picture(15)).unwrap();
        dpb.store_short_term(0, 2, picture(0)).unwrap();
        let list = dpb.default_p_list0(1).unwrap();
        assert_eq!(
            list.iter()
                .map(|picture| luma_value(picture))
                .collect::<Vec<_>>(),
            [0, 15, 14]
        );
    }

    #[test]
    fn orders_default_b_lists_around_the_current_poc() {
        let mut dpb = DecodedPictureBuffer::new(4, 4).unwrap();
        dpb.store_short_term(0, 2, picture(10)).unwrap();
        dpb.store_short_term(1, 6, picture(11)).unwrap();
        dpb.store_short_term(2, 10, picture(12)).unwrap();

        let (list0, list1) = dpb.default_b_lists(8).unwrap();
        assert_eq!(
            list0
                .iter()
                .map(|picture| luma_value(picture))
                .collect::<Vec<_>>(),
            [11, 10, 12]
        );
        assert_eq!(
            list1
                .iter()
                .map(|picture| luma_value(picture))
                .collect::<Vec<_>>(),
            [12, 11, 10]
        );
    }

    #[test]
    fn swaps_default_b_list1_when_both_lists_match() {
        let mut dpb = DecodedPictureBuffer::new(4, 4).unwrap();
        dpb.store_short_term(0, 2, picture(10)).unwrap();
        dpb.store_short_term(1, 6, picture(11)).unwrap();
        dpb.store_short_term(2, 10, picture(12)).unwrap();

        let (list0, list1) = dpb.default_b_lists(12).unwrap();
        assert_eq!(
            list0
                .iter()
                .map(|picture| luma_value(picture))
                .collect::<Vec<_>>(),
            [12, 11, 10]
        );
        assert_eq!(
            list1
                .iter()
                .map(|picture| luma_value(picture))
                .collect::<Vec<_>>(),
            [11, 12, 10]
        );
    }

    #[test]
    fn modifies_b_lists_independently_and_preserves_missing_entries() {
        let mut dpb = DecodedPictureBuffer::new(4, 4).unwrap();
        dpb.store_idr(0, picture(9), true).unwrap();
        dpb.store_short_term(1, 2, picture(10)).unwrap();
        dpb.store_short_term(2, 10, picture(12)).unwrap();

        let (list0, list1) = dpb
            .b_lists(
                3,
                6,
                3,
                &[ReferenceListModification::LongTerm {
                    long_term_pic_num: 0,
                }],
                4,
                &[ReferenceListModification::SubtractPicNum {
                    abs_diff_pic_num: 2,
                }],
            )
            .unwrap();
        assert_eq!(
            list0
                .iter()
                .map(|picture| picture.as_deref().map(luma_value))
                .collect::<Vec<_>>(),
            [Some(9), Some(10), Some(12)]
        );
        assert_eq!(
            list1
                .iter()
                .map(|picture| picture.as_deref().map(luma_value))
                .collect::<Vec<_>>(),
            [Some(10), Some(12), Some(9), None]
        );
    }

    #[test]
    fn retains_reference_identity_poc_and_kind_in_active_b_lists() {
        let mut dpb = DecodedPictureBuffer::new(3, 4).unwrap();
        dpb.store_idr(0, picture(9), true).unwrap();
        dpb.store_short_term(1, 4, picture(10)).unwrap();

        let (list0, list1) = dpb.b_reference_info_lists(2, 2, 2, &[], 2, &[]).unwrap();
        let list0 = list0.into_iter().flatten().collect::<Vec<_>>();
        let list1 = list1.into_iter().flatten().collect::<Vec<_>>();
        assert_eq!(
            list0
                .iter()
                .map(|reference| (reference.picture_order_count, reference.kind))
                .collect::<Vec<_>>(),
            [
                (4, ReferenceKind::ShortTerm),
                (0, ReferenceKind::LongTerm { frame_index: 0 })
            ]
        );
        assert_eq!(
            list1
                .iter()
                .map(|reference| reference.picture_order_count)
                .collect::<Vec<_>>(),
            [0, 4]
        );
        assert_eq!(list0[0].id, list1[1].id);
        assert_eq!(list0[1].id, list1[0].id);
        assert_ne!(list0[0].id, list0[1].id);
    }

    #[test]
    fn includes_equal_poc_short_term_b_reference_in_the_earlier_group() {
        let mut dpb = DecodedPictureBuffer::new(1, 4).unwrap();
        dpb.store_short_term(0, 4, picture(1)).unwrap();
        let (list0, list1) = dpb.default_b_lists(4).unwrap();
        assert_eq!(luma_value(&list0[0]), 1);
        assert_eq!(luma_value(&list1[0]), 1);
    }

    #[test]
    fn sliding_window_removes_the_oldest_short_term_picture() {
        let mut dpb = DecodedPictureBuffer::new(2, 4).unwrap();
        dpb.store_short_term(14, 0, picture(14)).unwrap();
        dpb.store_short_term(15, 1, picture(15)).unwrap();
        dpb.store_short_term(0, 2, picture(0)).unwrap();
        let list = dpb.default_p_list0(1).unwrap();
        assert_eq!(
            list.iter()
                .map(|picture| luma_value(picture))
                .collect::<Vec<_>>(),
            [0, 15]
        );
    }

    #[test]
    fn idr_clears_prior_references_and_can_be_long_term() {
        let mut dpb = DecodedPictureBuffer::new(2, 4).unwrap();
        dpb.store_short_term(3, 0, picture(3)).unwrap();
        dpb.store_idr(0, picture(9), true).unwrap();
        assert_eq!(dpb.len(), 1);
        dpb.store_short_term(1, 2, picture(1)).unwrap();
        let list = dpb.default_p_list0(2).unwrap();
        assert_eq!(
            list.iter()
                .map(|picture| luma_value(picture))
                .collect::<Vec<_>>(),
            [1, 9]
        );
    }

    #[test]
    fn rejected_store_is_atomic() {
        let mut dpb = DecodedPictureBuffer::new(1, 4).unwrap();
        dpb.store_short_term(0, 0, picture(1)).unwrap();
        let different = Arc::new(Yuv420Picture::new(Size::new(32, 16)).unwrap());
        assert!(matches!(
            dpb.store_short_term(1, 1, different),
            Err(H264Error::InvalidSyntax(_))
        ));
        assert_eq!(dpb.len(), 1);
        assert_eq!(luma_value(&dpb.default_p_list0(1).unwrap()[0]), 1);
    }

    #[test]
    fn applies_short_and_long_term_list_modifications_in_order() {
        let mut dpb = DecodedPictureBuffer::new(4, 4).unwrap();
        dpb.store_idr(0, picture(9), true).unwrap();
        dpb.store_short_term(1, 1, picture(1)).unwrap();
        dpb.store_short_term(2, 2, picture(2)).unwrap();
        dpb.store_short_term(3, 3, picture(3)).unwrap();

        let list = dpb
            .p_list0(
                4,
                4,
                &[
                    ReferenceListModification::SubtractPicNum {
                        abs_diff_pic_num: 3,
                    },
                    ReferenceListModification::AddPicNum {
                        abs_diff_pic_num: 1,
                    },
                    ReferenceListModification::LongTerm {
                        long_term_pic_num: 0,
                    },
                ],
            )
            .unwrap();
        assert_eq!(
            list.iter()
                .map(|picture| luma_value(picture.as_ref().unwrap()))
                .collect::<Vec<_>>(),
            [1, 2, 9, 3]
        );
    }

    #[test]
    fn preserves_missing_active_entries_and_rejects_missing_targets() {
        let mut dpb = DecodedPictureBuffer::new(2, 4).unwrap();
        dpb.store_short_term(0, 0, picture(7)).unwrap();
        let list = dpb.p_list0(1, 3, &[]).unwrap();
        assert_eq!(list.len(), 3);
        assert!(list[0].is_some());
        assert!(list[1..].iter().all(Option::is_none));
        assert!(matches!(
            dpb.p_list0(
                1,
                1,
                &[ReferenceListModification::LongTerm {
                    long_term_pic_num: 0
                }]
            ),
            Err(H264Error::InvalidSyntax(_))
        ));
    }

    #[test]
    fn applies_adaptive_memory_control_atomically() {
        let mut dpb = DecodedPictureBuffer::new(4, 4).unwrap();
        dpb.store_short_term(0, 0, picture(10)).unwrap();
        dpb.store_short_term(1, 1, picture(11)).unwrap();
        dpb.store_short_term(2, 2, picture(12)).unwrap();
        dpb.store_adaptive(
            3,
            3,
            picture(13),
            &[
                MemoryManagementOperation::LimitLongTerm {
                    max_long_term_frame_idx_plus1: 2,
                },
                MemoryManagementOperation::AssignLongTerm {
                    difference_of_pic_nums: 2,
                    long_term_frame_idx: 1,
                },
                MemoryManagementOperation::ForgetShortTerm {
                    difference_of_pic_nums: 1,
                },
                MemoryManagementOperation::MarkCurrentLongTerm {
                    long_term_frame_idx: 0,
                },
            ],
        )
        .unwrap();
        let list = dpb.default_p_list0(4).unwrap();
        assert_eq!(
            list.iter()
                .map(|picture| luma_value(picture))
                .collect::<Vec<_>>(),
            [10, 13, 11]
        );
    }

    #[test]
    fn failed_adaptive_sequence_preserves_existing_references() {
        let mut dpb = DecodedPictureBuffer::new(2, 4).unwrap();
        dpb.store_short_term(0, 0, picture(10)).unwrap();
        let before = dpb.default_p_list0(1).unwrap();
        assert!(matches!(
            dpb.store_adaptive(
                1,
                1,
                picture(11),
                &[
                    MemoryManagementOperation::ForgetAll,
                    MemoryManagementOperation::MarkCurrentLongTerm {
                        long_term_frame_idx: 0
                    }
                ]
            ),
            Err(H264Error::InvalidSyntax(_))
        ));
        let after = dpb.default_p_list0(1).unwrap();
        assert_eq!(luma_value(&before[0]), luma_value(&after[0]));
        assert_eq!(dpb.len(), 1);
    }

    #[test]
    fn max_num_ref_frames_zero_still_allows_one_reference() {
        let mut dpb = DecodedPictureBuffer::new(0, 4).unwrap();
        dpb.store_idr(0, picture(5), false).unwrap();
        assert_eq!(dpb.len(), 1);
        dpb.store_short_term(1, 1, picture(6)).unwrap();
        assert_eq!(dpb.len(), 1);
        assert_eq!(luma_value(&dpb.default_p_list0(2).unwrap()[0]), 6);
    }
}
