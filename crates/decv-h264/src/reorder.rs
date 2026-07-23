//! Bounded picture-order-count output reordering.
//!
//! H.264 references are decoded before pictures that depend on them, so decode
//! order can differ from display order. This buffer is deliberately separate
//! from the decoded picture buffer: the DPB owns prediction references, while
//! this component owns pictures waiting to be presented.

use crate::{H264Error, Result};

#[derive(Debug)]
struct PendingOutput<T> {
    picture_order_count: i32,
    decode_index: u64,
    value: T,
}

/// Holds at most the signalled reorder depth before releasing the lowest POC.
#[derive(Debug)]
pub(crate) struct PictureReorderBuffer<T> {
    max_num_reorder_frames: usize,
    next_decode_index: u64,
    pending: Vec<PendingOutput<T>>,
}

impl<T> PictureReorderBuffer<T> {
    pub(crate) fn new(max_num_reorder_frames: usize) -> Self {
        Self {
            max_num_reorder_frames,
            next_decode_index: 0,
            pending: Vec::new(),
        }
    }

    /// Adds one decoded picture and returns the next displayable value once
    /// the configured reorder depth is exceeded.
    pub(crate) fn push(&mut self, picture_order_count: i32, value: T) -> Result<Option<T>> {
        let decode_index = self.next_decode_index;
        self.next_decode_index = self
            .next_decode_index
            .checked_add(1)
            .ok_or(H264Error::IntegerOverflow)?;
        self.pending.push(PendingOutput {
            picture_order_count,
            decode_index,
            value,
        });
        if self.pending.len() <= self.max_num_reorder_frames {
            return Ok(None);
        }
        Ok(Some(self.remove_lowest_poc()))
    }

    /// Releases every delayed picture in display order.
    pub(crate) fn drain(&mut self) -> Vec<T> {
        self.pending
            .sort_unstable_by_key(|entry| (entry.picture_order_count, entry.decode_index));
        self.pending.drain(..).map(|entry| entry.value).collect()
    }

    /// Drops delayed output after a discontinuity or an IDR instruction that
    /// suppresses prior pictures.
    pub(crate) fn clear(&mut self) {
        self.pending.clear();
        self.next_decode_index = 0;
    }

    fn remove_lowest_poc(&mut self) -> T {
        let index = self
            .pending
            .iter()
            .enumerate()
            .min_by_key(|(_, entry)| (entry.picture_order_count, entry.decode_index))
            .map(|(index, _)| index)
            .expect("push inserts an entry before output selection");
        self.pending.swap_remove(index).value
    }
}

#[cfg(test)]
mod tests {
    use super::PictureReorderBuffer;

    #[test]
    fn releases_decode_order_immediately_when_reordering_is_disabled() {
        let mut buffer = PictureReorderBuffer::new(0);
        assert_eq!(buffer.push(0, "I"), Ok(Some("I")));
        assert_eq!(buffer.push(2, "P"), Ok(Some("P")));
        assert!(buffer.drain().is_empty());
    }

    #[test]
    fn releases_b_pictures_in_poc_order_with_bounded_delay() {
        let mut buffer = PictureReorderBuffer::new(2);
        assert_eq!(buffer.push(0, "I"), Ok(None));
        assert_eq!(buffer.push(6, "P"), Ok(None));
        assert_eq!(buffer.push(2, "B0"), Ok(Some("I")));
        assert_eq!(buffer.push(4, "B1"), Ok(Some("B0")));
        assert_eq!(buffer.drain(), ["B1", "P"]);
    }

    #[test]
    fn preserves_decode_order_for_equal_picture_order_counts() {
        let mut buffer = PictureReorderBuffer::new(3);
        assert_eq!(buffer.push(5, 0), Ok(None));
        assert_eq!(buffer.push(5, 1), Ok(None));
        assert_eq!(buffer.push(5, 2), Ok(None));
        assert_eq!(buffer.drain(), [0, 1, 2]);
    }

    #[test]
    fn clear_discards_delayed_pictures_and_resets_the_sequence() {
        let mut buffer = PictureReorderBuffer::new(1);
        assert_eq!(buffer.push(4, "old"), Ok(None));
        buffer.clear();
        assert!(buffer.drain().is_empty());
        assert_eq!(buffer.push(0, "new"), Ok(None));
        assert_eq!(buffer.drain(), ["new"]);
    }
}
