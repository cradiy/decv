use std::ops::Range;

use decv_core::MediaInput;

use crate::{FourCc, Mp4Error, Result};

const BASE_HEADER_SIZE: u64 = 8;
const EXTENDED_HEADER_SIZE: u64 = 16;
const UUID_USER_TYPE_SIZE: u64 = 16;
const UUID: FourCc = FourCc::new(*b"uuid");

/// One validated ISO BMFF box header and its absolute file range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoxHeader {
    kind: FourCc,
    offset: u64,
    size: u64,
    header_size: u64,
}

impl BoxHeader {
    #[inline]
    pub const fn kind(self) -> FourCc {
        self.kind
    }

    #[inline]
    pub const fn offset(self) -> u64 {
        self.offset
    }

    #[inline]
    pub const fn size(self) -> u64 {
        self.size
    }

    #[inline]
    pub const fn header_size(self) -> u64 {
        self.header_size
    }

    #[inline]
    pub fn end(self) -> Result<u64> {
        self.offset
            .checked_add(self.size)
            .ok_or(Mp4Error::IntegerOverflow)
    }

    #[inline]
    pub fn payload_offset(self) -> Result<u64> {
        self.offset
            .checked_add(self.header_size)
            .ok_or(Mp4Error::IntegerOverflow)
    }

    #[inline]
    pub const fn payload_size(self) -> u64 {
        self.size - self.header_size
    }

    pub fn payload_range(self) -> Result<Range<u64>> {
        Ok(self.payload_offset()?..self.end()?)
    }

    fn read(input: &dyn MediaInput, offset: u64, parent_end: u64) -> Result<Self> {
        let remaining = parent_end
            .checked_sub(offset)
            .ok_or(Mp4Error::InvalidData("box begins after its parent"))?;
        if remaining < BASE_HEADER_SIZE {
            return Err(Mp4Error::InvalidData("truncated MP4 box header"));
        }

        let mut base = [0; 8];
        read_exact_at(input, offset, &mut base)?;
        let size32 = u32::from_be_bytes(base[..4].try_into().expect("four-byte size"));
        let kind = FourCc::new(base[4..].try_into().expect("four-byte type"));
        let (mut size, mut header_size) = if size32 == 1 {
            if remaining < EXTENDED_HEADER_SIZE {
                return Err(Mp4Error::InvalidData("truncated extended MP4 box header"));
            }
            let mut extended = [0; 8];
            read_exact_at(
                input,
                offset
                    .checked_add(BASE_HEADER_SIZE)
                    .ok_or(Mp4Error::IntegerOverflow)?,
                &mut extended,
            )?;
            (u64::from_be_bytes(extended), EXTENDED_HEADER_SIZE)
        } else if size32 == 0 {
            (remaining, BASE_HEADER_SIZE)
        } else {
            (u64::from(size32), BASE_HEADER_SIZE)
        };

        if kind == UUID {
            header_size = header_size
                .checked_add(UUID_USER_TYPE_SIZE)
                .ok_or(Mp4Error::IntegerOverflow)?;
        }
        if size32 == 0 {
            size = remaining;
        }
        if size < header_size {
            return Err(Mp4Error::InvalidData(
                "MP4 box size is smaller than its header",
            ));
        }
        if size > remaining {
            return Err(Mp4Error::InvalidData("MP4 box exceeds its parent"));
        }

        Ok(Self {
            kind,
            offset,
            size,
            header_size,
        })
    }
}

/// A file whose top-level extent is known and can be traversed lazily.
#[derive(Clone, Copy)]
pub struct Mp4File<'input> {
    input: &'input dyn MediaInput,
    length: u64,
}

impl<'input> Mp4File<'input> {
    pub fn open(input: &'input dyn MediaInput) -> Result<Self> {
        let length = input.len()?.ok_or(Mp4Error::UnknownInputLength)?;
        Ok(Self { input, length })
    }

    #[inline]
    pub const fn length(self) -> u64 {
        self.length
    }

    pub fn boxes(self) -> BoxIter<'input> {
        BoxIter::new(self.input, 0, self.length)
    }

    pub fn children(self, parent: BoxHeader) -> Result<BoxIter<'input>> {
        let range = parent.payload_range()?;
        if range.end > self.length {
            return Err(Mp4Error::InvalidData("parent box exceeds the input"));
        }
        Ok(BoxIter::new(self.input, range.start, range.end))
    }
}

impl std::fmt::Debug for Mp4File<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Mp4File")
            .field("length", &self.length)
            .finish_non_exhaustive()
    }
}

/// Lazy sibling-box traversal constrained to one validated parent range.
pub struct BoxIter<'input> {
    input: &'input dyn MediaInput,
    next_offset: u64,
    end: u64,
    failed: bool,
}

impl<'input> BoxIter<'input> {
    fn new(input: &'input dyn MediaInput, start: u64, end: u64) -> Self {
        Self {
            input,
            next_offset: start,
            end,
            failed: false,
        }
    }
}

impl Iterator for BoxIter<'_> {
    type Item = Result<BoxHeader>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.failed || self.next_offset == self.end {
            return None;
        }
        match BoxHeader::read(self.input, self.next_offset, self.end) {
            Ok(header) => {
                self.next_offset = match header.end() {
                    Ok(end) => end,
                    Err(error) => {
                        self.failed = true;
                        return Some(Err(error));
                    }
                };
                Some(Ok(header))
            }
            Err(error) => {
                self.failed = true;
                Some(Err(error))
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.failed || self.next_offset == self.end {
            (0, Some(0))
        } else {
            (0, None)
        }
    }
}

fn read_exact_at(input: &dyn MediaInput, offset: u64, mut buffer: &mut [u8]) -> Result<()> {
    let mut current = offset;
    while !buffer.is_empty() {
        let read = input.read_at(current, buffer)?;
        if read == 0 {
            return Err(Mp4Error::InvalidData("unexpected end of MP4 input"));
        }
        if read > buffer.len() {
            return Err(Mp4Error::InvalidData(
                "MediaInput returned more bytes than requested",
            ));
        }
        current = current
            .checked_add(u64::try_from(read).map_err(|_| Mp4Error::IntegerOverflow)?)
            .ok_or(Mp4Error::IntegerOverflow)?;
        buffer = &mut buffer[read..];
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::io;

    use super::*;

    #[derive(Debug)]
    struct MemoryInput {
        bytes: Vec<u8>,
        max_read: usize,
        known_length: bool,
    }

    impl MemoryInput {
        fn new(bytes: Vec<u8>) -> Self {
            Self {
                bytes,
                max_read: usize::MAX,
                known_length: true,
            }
        }
    }

    impl MediaInput for MemoryInput {
        fn len(&self) -> io::Result<Option<u64>> {
            Ok(self
                .known_length
                .then(|| u64::try_from(self.bytes.len()).unwrap()))
        }

        fn read_at(&self, offset: u64, buffer: &mut [u8]) -> io::Result<usize> {
            let offset = usize::try_from(offset)
                .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "offset"))?;
            let Some(source) = self.bytes.get(offset..) else {
                return Ok(0);
            };
            let count = source.len().min(buffer.len()).min(self.max_read);
            buffer[..count].copy_from_slice(&source[..count]);
            Ok(count)
        }
    }

    fn ordinary(kind: [u8; 4], payload: &[u8]) -> Vec<u8> {
        let size = u32::try_from(8 + payload.len()).unwrap();
        let mut bytes = Vec::from(size.to_be_bytes());
        bytes.extend_from_slice(&kind);
        bytes.extend_from_slice(payload);
        bytes
    }

    fn extended(kind: [u8; 4], payload: &[u8]) -> Vec<u8> {
        let size = u64::try_from(16 + payload.len()).unwrap();
        let mut bytes = Vec::from(1u32.to_be_bytes());
        bytes.extend_from_slice(&kind);
        bytes.extend_from_slice(&size.to_be_bytes());
        bytes.extend_from_slice(payload);
        bytes
    }

    #[test]
    fn walks_ordinary_and_extended_top_level_boxes() {
        let mut bytes = ordinary(*b"ftyp", b"isom");
        bytes.extend_from_slice(&extended(*b"mdat", b"payload"));
        let input = MemoryInput::new(bytes);
        let file = Mp4File::open(&input).unwrap();
        let boxes = file.boxes().collect::<Result<Vec<_>>>().unwrap();

        assert_eq!(file.length(), 35);
        assert_eq!(
            boxes,
            [
                BoxHeader {
                    kind: FourCc::new(*b"ftyp"),
                    offset: 0,
                    size: 12,
                    header_size: 8,
                },
                BoxHeader {
                    kind: FourCc::new(*b"mdat"),
                    offset: 12,
                    size: 23,
                    header_size: 16,
                },
            ]
        );
        assert_eq!(boxes[1].payload_range().unwrap(), 28..35);
    }

    #[test]
    fn size_zero_extends_to_the_parent_end() {
        let mut bytes = Vec::from(0u32.to_be_bytes());
        bytes.extend_from_slice(b"free");
        bytes.extend_from_slice(b"rest");
        let input = MemoryInput::new(bytes);
        let boxes = Mp4File::open(&input)
            .unwrap()
            .boxes()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        assert_eq!(boxes[0].size(), 12);
        assert_eq!(boxes[0].payload_size(), 4);
    }

    #[test]
    fn traverses_children_within_the_parent_payload() {
        let child0 = ordinary(*b"trak", &[]);
        let child1 = ordinary(*b"mvhd", b"data");
        let mut payload = child0;
        payload.extend_from_slice(&child1);
        let input = MemoryInput::new(ordinary(*b"moov", &payload));
        let file = Mp4File::open(&input).unwrap();
        let parent = file.boxes().next().unwrap().unwrap();
        let children = file
            .children(parent)
            .unwrap()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        assert_eq!(
            children
                .iter()
                .map(|child| child.kind())
                .collect::<Vec<_>>(),
            [FourCc::new(*b"trak"), FourCc::new(*b"mvhd")]
        );
        assert_eq!(children[0].offset(), 8);
        assert_eq!(children[1].offset(), 16);
    }

    #[test]
    fn accounts_for_uuid_user_type_in_the_header() {
        let mut bytes = Vec::from(28u32.to_be_bytes());
        bytes.extend_from_slice(b"uuid");
        bytes.extend_from_slice(&[7; 16]);
        bytes.extend_from_slice(b"data");
        let input = MemoryInput::new(bytes);
        let header = Mp4File::open(&input)
            .unwrap()
            .boxes()
            .next()
            .unwrap()
            .unwrap();
        assert_eq!(header.header_size(), 24);
        assert_eq!(header.payload_size(), 4);
    }

    #[test]
    fn supports_inputs_that_return_short_reads() {
        let mut input = MemoryInput::new(extended(*b"mdat", b"data"));
        input.max_read = 2;
        let header = Mp4File::open(&input)
            .unwrap()
            .boxes()
            .next()
            .unwrap()
            .unwrap();
        assert_eq!(header.size(), 20);
    }

    #[test]
    fn rejects_unknown_top_level_length() {
        let mut input = MemoryInput::new(Vec::new());
        input.known_length = false;
        assert!(matches!(
            Mp4File::open(&input),
            Err(Mp4Error::UnknownInputLength)
        ));
    }

    #[test]
    fn rejects_truncated_and_out_of_parent_boxes() {
        for bytes in [
            vec![0; 7],
            {
                let mut bytes = Vec::from(4u32.to_be_bytes());
                bytes.extend_from_slice(b"free");
                bytes
            },
            {
                let mut bytes = Vec::from(20u32.to_be_bytes());
                bytes.extend_from_slice(b"free");
                bytes
            },
            {
                let mut bytes = Vec::from(1u32.to_be_bytes());
                bytes.extend_from_slice(b"free");
                bytes
            },
        ] {
            let input = MemoryInput::new(bytes);
            assert!(matches!(
                Mp4File::open(&input).unwrap().boxes().next().unwrap(),
                Err(Mp4Error::InvalidData(_))
            ));
        }
    }
}
