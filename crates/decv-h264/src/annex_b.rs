//! Annex-B start-code detection and zero-copy NAL-unit splitting.

use std::iter::FusedIterator;

use memchr::memchr;

use crate::{H264Error, Result};

/// One NAL unit borrowed directly from an Annex-B byte stream.
///
/// `bytes` contains the NAL header followed by its EBSP payload. It does not
/// contain the Annex-B start code or surrounding zero bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AnnexBNalUnit<'a> {
    bytes: &'a [u8],
    stream_offset: usize,
}

impl<'a> AnnexBNalUnit<'a> {
    #[inline]
    const fn new(bytes: &'a [u8], stream_offset: usize) -> Self {
        Self {
            bytes,
            stream_offset,
        }
    }

    #[inline]
    pub const fn bytes(self) -> &'a [u8] {
        self.bytes
    }

    /// Byte offset of the NAL header in the original Annex-B stream.
    #[inline]
    pub const fn stream_offset(self) -> usize {
        self.stream_offset
    }
}

impl<'a> AsRef<[u8]> for AnnexBNalUnit<'a> {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        self.bytes
    }
}

/// Splits one contiguous Annex-B byte stream into borrowed NAL units.
///
/// This iterator does not join start codes split across separate input
/// buffers. A packet-oriented decoder should retain incomplete suffix bytes
/// before constructing the next reader.
#[derive(Debug, Clone)]
pub struct AnnexBReader<'a> {
    data: &'a [u8],
    nal_start: usize,
    started: bool,
    finished: bool,
}

impl<'a> AnnexBReader<'a> {
    #[inline]
    pub const fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            nal_start: 0,
            started: false,
            finished: false,
        }
    }

    fn start(&mut self) -> Option<Result<()>> {
        if self.data.is_empty() {
            self.finished = true;
            return None;
        }

        let Some(start_code) = find_start_code(self.data, 0) else {
            self.finished = true;
            return Some(Err(H264Error::InvalidStartCode));
        };

        // Bytes preceding the first prefix are allowed only as
        // leading_zero_8bits / zero_byte.
        if self.data[..start_code.prefix_start]
            .iter()
            .any(|&byte| byte != 0)
        {
            self.finished = true;
            return Some(Err(H264Error::InvalidStartCode));
        }

        self.nal_start = start_code.nal_start;
        self.started = true;
        Some(Ok(()))
    }
}

impl<'a> Iterator for AnnexBReader<'a> {
    type Item = Result<AnnexBNalUnit<'a>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        if !self.started {
            match self.start() {
                Some(Ok(())) => {}
                Some(Err(error)) => return Some(Err(error)),
                None => return None,
            }
        }

        let current_start = self.nal_start;
        let nal_end;

        if let Some(next_start_code) = find_start_code(self.data, current_start) {
            nal_end =
                trim_trailing_zero_bytes(self.data, current_start, next_start_code.prefix_start);
            self.nal_start = next_start_code.nal_start;
        } else {
            nal_end = trim_trailing_zero_bytes(self.data, current_start, self.data.len());
            self.finished = true;
        }

        if nal_end == current_start {
            return Some(Err(H264Error::EmptyNalUnit {
                offset: current_start,
            }));
        }

        Some(Ok(AnnexBNalUnit::new(
            &self.data[current_start..nal_end],
            current_start,
        )))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.finished {
            (0, Some(0))
        } else {
            (0, None)
        }
    }
}

impl FusedIterator for AnnexBReader<'_> {}

#[derive(Debug, Clone, Copy)]
struct StartCode {
    /// Start of the three-byte `00 00 01` prefix. Any additional preceding
    /// zero byte is kept outside the NAL and removed as boundary padding.
    prefix_start: usize,
    /// First byte after the terminating `01`.
    nal_start: usize,
}

#[inline]
fn find_start_code(data: &[u8], from: usize) -> Option<StartCode> {
    let search_start = from.checked_add(2)?;
    let mut candidates = data.get(search_start..)?;
    let mut candidate_offset = search_start;

    while let Some(relative) = memchr(1, candidates) {
        let index = candidate_offset + relative;
        if data[index] == 1 && data[index - 1] == 0 && data[index - 2] == 0 {
            return Some(StartCode {
                prefix_start: index - 2,
                nal_start: index + 1,
            });
        }
        candidate_offset = index + 1;
        candidates = &data[candidate_offset..];
    }

    None
}

#[inline]
fn trim_trailing_zero_bytes(data: &[u8], start: usize, mut end: usize) -> usize {
    while end > start && data[end - 1] == 0 {
        end -= 1;
    }
    end
}

#[cfg(test)]
mod tests {
    use super::{AnnexBNalUnit, AnnexBReader};
    use crate::H264Error;

    type AnnBNal<'a> = AnnexBNalUnit<'a>;

    fn collect(data: &[u8]) -> Vec<crate::Result<AnnBNal<'_>>> {
        AnnexBReader::new(data).collect()
    }

    #[test]
    fn splits_three_and_four_byte_start_codes() {
        let data = [
            0x00, 0x00, 0x00, 0x01, 0x67, 0xaa, 0x00, 0x00, 0x01, 0x68, 0xbb, 0xcc,
        ];
        let units = collect(&data);

        assert_eq!(units.len(), 2);
        assert_eq!(units[0], Ok(AnnBNal::new(&data[4..6], 4)));
        assert_eq!(units[1], Ok(AnnBNal::new(&data[9..12], 9)));
    }

    #[test]
    fn removes_leading_and_trailing_zero_bytes() {
        let data = [
            0x00, 0x00, 0x00, 0x00, 0x01, 0x67, 0xaa, 0x00, 0x00, 0x00, 0x00, 0x01, 0x68, 0xbb,
            0x00, 0x00,
        ];
        let units = collect(&data);

        assert_eq!(units.len(), 2);
        assert_eq!(units[0], Ok(AnnBNal::new(&data[5..7], 5)));
        assert_eq!(units[1], Ok(AnnBNal::new(&data[12..14], 12)));
    }

    #[test]
    fn does_not_split_an_emulation_prevention_sequence() {
        let data = [0x00, 0x00, 0x01, 0x65, 0x00, 0x00, 0x03, 0x01, 0x99];
        let units = collect(&data);

        assert_eq!(units, [Ok(AnnBNal::new(&data[3..], 3))]);
    }

    #[test]
    fn reports_an_empty_nal_and_continues() {
        let data = [0x00, 0x00, 0x01, 0x00, 0x00, 0x01, 0x65, 0x88];
        let units = collect(&data);

        assert_eq!(
            units,
            [
                Err(H264Error::EmptyNalUnit { offset: 3 }),
                Ok(AnnBNal::new(&data[6..], 6)),
            ]
        );
    }

    #[test]
    fn rejects_nonzero_data_before_the_first_start_code() {
        let data = [0xff, 0x00, 0x00, 0x01, 0x65];
        let mut reader = AnnexBReader::new(&data);

        assert_eq!(reader.next(), Some(Err(H264Error::InvalidStartCode)));
        assert_eq!(reader.next(), None);
        assert_eq!(reader.next(), None);
    }

    #[test]
    fn rejects_nonempty_data_without_a_start_code() {
        let mut reader = AnnexBReader::new(&[0x65, 0x88]);

        assert_eq!(reader.next(), Some(Err(H264Error::InvalidStartCode)));
        assert_eq!(reader.next(), None);
    }

    #[test]
    fn accepts_an_empty_stream() {
        let mut reader = AnnexBReader::new(&[]);

        assert_eq!(reader.next(), None);
        assert_eq!(reader.next(), None);
    }
}
