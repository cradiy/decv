use std::ops::Range;

use crate::{Result, Vp9Error};

const SUPERFRAME_MARKER: u8 = 0xc0;
const SUPERFRAME_MARKER_MASK: u8 = 0xe0;
const MAX_FRAMES: usize = 8;

/// A packet split into its one to eight coded VP9 frames.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Superframe {
    ranges: Vec<Range<usize>>,
}

impl Superframe {
    pub fn parse(packet: &[u8]) -> Result<Self> {
        let Some(&marker) = packet.last() else {
            return Err(Vp9Error::Truncated("packet"));
        };
        if marker & SUPERFRAME_MARKER_MASK != SUPERFRAME_MARKER {
            #[allow(clippy::single_range_in_vec_init)]
            return Ok(Self {
                ranges: vec![0..packet.len()],
            });
        }

        let frame_count = usize::from(marker & 7) + 1;
        let magnitude = usize::from(marker >> 3 & 3) + 1;
        debug_assert!(frame_count <= MAX_FRAMES);
        let index_size = 2usize
            .checked_add(
                frame_count
                    .checked_mul(magnitude)
                    .ok_or(Vp9Error::IntegerOverflow)?,
            )
            .ok_or(Vp9Error::IntegerOverflow)?;
        if packet.len() < index_size {
            return Err(Vp9Error::Truncated("superframe index"));
        }

        let index_start = packet.len() - index_size;
        if packet[index_start] != marker {
            return Err(Vp9Error::InvalidData(
                "superframe index markers do not match",
            ));
        }

        let mut cursor = index_start + 1;
        let mut payload_cursor = 0usize;
        let mut ranges = Vec::with_capacity(frame_count);
        for _ in 0..frame_count {
            let mut size = 0usize;
            for shift in 0..magnitude {
                size |= usize::from(packet[cursor]) << (shift * 8);
                cursor += 1;
            }
            if size == 0 {
                return Err(Vp9Error::InvalidData(
                    "superframe contains an empty coded frame",
                ));
            }
            let end = payload_cursor
                .checked_add(size)
                .ok_or(Vp9Error::IntegerOverflow)?;
            if end > index_start {
                return Err(Vp9Error::InvalidData(
                    "superframe sizes exceed packet payload",
                ));
            }
            ranges.push(payload_cursor..end);
            payload_cursor = end;
        }
        if payload_cursor != index_start {
            return Err(Vp9Error::InvalidData(
                "superframe sizes do not cover packet payload",
            ));
        }
        debug_assert_eq!(packet[cursor], marker);
        Ok(Self { ranges })
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.ranges.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }

    #[inline]
    pub fn frames<'packet>(&'packet self, packet: &'packet [u8]) -> SuperframeFrames<'packet> {
        SuperframeFrames {
            packet,
            ranges: self.ranges.iter(),
        }
    }
}

pub struct SuperframeFrames<'a> {
    packet: &'a [u8],
    ranges: std::slice::Iter<'a, Range<usize>>,
}

impl<'a> Iterator for SuperframeFrames<'a> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<Self::Item> {
        let range = self.ranges.next()?;
        self.packet.get(range.clone())
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.ranges.size_hint()
    }
}

impl ExactSizeIterator for SuperframeFrames<'_> {}

#[cfg(test)]
mod tests {
    use super::Superframe;
    use crate::Vp9Error;

    #[test]
    fn ordinary_packet_is_one_frame() {
        let packet = [0x82, 1, 2, 3];
        let index = Superframe::parse(&packet).unwrap();
        assert_eq!(index.frames(&packet).collect::<Vec<_>>(), [&packet[..]]);
    }

    #[test]
    fn splits_little_endian_superframe_sizes() {
        let mut packet = vec![1; 260];
        packet.extend_from_slice(&[2, 3, 4]);
        let marker = 0xc0 | 1 | (1 << 3);
        packet.extend_from_slice(&[marker, 4, 1, 3, 0, marker]);
        let index = Superframe::parse(&packet).unwrap();
        let frames = index.frames(&packet).collect::<Vec<_>>();
        assert_eq!(
            frames.iter().map(|frame| frame.len()).collect::<Vec<_>>(),
            [260, 3]
        );
    }

    #[test]
    fn rejects_mismatched_index_and_sizes() {
        assert_eq!(
            Superframe::parse(&[1, 2, 0xc0]),
            Err(Vp9Error::InvalidData(
                "superframe index markers do not match"
            ))
        );
    }
}
