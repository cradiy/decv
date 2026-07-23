//! Decoded Picture Buffer and reference-picture management.

use std::sync::Arc;

use crate::{H264Error, ReferenceListModification, Result, Yuv420Picture};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReferenceKind {
    ShortTerm,
    LongTerm { frame_index: u32 },
}

#[derive(Debug, Clone)]
pub struct DpbReference {
    pub frame_num: u32,
    pub picture_order_count: i32,
    pub kind: ReferenceKind,
    pub picture: Arc<Yuv420Picture>,
}

/// Reference-picture subset of the decoded picture buffer.
///
/// This first DPB layer covers progressive frame pictures, IDR reset, default
/// P List-0 ordering, and sliding-window marking. Adaptive MMCO and output
/// reordering are intentionally separate later stages.
#[derive(Debug, Clone)]
pub struct DecodedPictureBuffer {
    max_num_ref_frames: usize,
    max_frame_num: u32,
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
            references: Vec::with_capacity(max_num_ref_frames),
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
    }

    /// Builds the default reference picture List 0 for a progressive P slice:
    /// short-term pictures by descending PicNum, then long-term pictures by
    /// ascending long-term frame index.
    pub fn default_p_list0(&self, current_frame_num: u32) -> Result<Vec<Arc<Yuv420Picture>>> {
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
    ) -> Result<Vec<Option<Arc<Yuv420Picture>>>> {
        self.ensure_frame_num(current_frame_num)?;
        if active_count == 0 || active_count > 32 {
            return Err(H264Error::InvalidSyntax(
                "P List 0 active count is outside 1..=32",
            ));
        }
        let active_count = usize::from(active_count);
        let ordered = self.ordered_p_references(current_frame_num);
        let mut list: Vec<Option<&DpbReference>> =
            ordered.into_iter().take(active_count).map(Some).collect();
        list.resize(active_count + 1, None);

        let mut reference_index = 0usize;
        let mut predicted_pic_num = i64::from(current_frame_num);
        for modification in modifications {
            if reference_index >= active_count {
                return Err(H264Error::InvalidSyntax(
                    "P List 0 modifications exceed the active list",
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
        picture: Arc<Yuv420Picture>,
        long_term_reference: bool,
    ) -> Result<()> {
        if self.max_num_ref_frames == 0 {
            return Err(H264Error::InvalidSyntax(
                "SPS does not permit storing reference frames",
            ));
        }
        let kind = if long_term_reference {
            ReferenceKind::LongTerm { frame_index: 0 }
        } else {
            ReferenceKind::ShortTerm
        };
        self.references.clear();
        self.references.push(DpbReference {
            frame_num: 0,
            picture_order_count,
            kind,
            picture,
        });
        Ok(())
    }

    /// Applies sliding-window marking and stores the current reference frame
    /// as short-term.
    pub fn store_short_term(
        &mut self,
        frame_num: u32,
        picture_order_count: i32,
        picture: Arc<Yuv420Picture>,
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
        if self.references.len() == self.max_num_ref_frames {
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
        self.references.push(DpbReference {
            frame_num,
            picture_order_count,
            kind: ReferenceKind::ShortTerm,
            picture,
        });
        Ok(())
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
                "P List 0 modification names a missing short-term reference",
            ))
    }

    fn ensure_can_store(&self, picture: &Yuv420Picture) -> Result<()> {
        if self.max_num_ref_frames == 0 {
            return Err(H264Error::InvalidSyntax(
                "SPS does not permit storing reference frames",
            ));
        }
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
}
