//! Decoded Picture Buffer and reference-picture management.

use std::sync::Arc;

use crate::{H264Error, Result, Yuv420Picture};

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
        Ok(short
            .into_iter()
            .chain(long)
            .map(|reference| reference.picture.clone())
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
}
