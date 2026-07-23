use decv_core::MediaInput;

use crate::{FourCc, Mp4Error, Result};

pub(crate) struct BoundedReader<'input> {
    input: &'input dyn MediaInput,
    position: u64,
    end: u64,
}

impl<'input> BoundedReader<'input> {
    pub(crate) fn new(input: &'input dyn MediaInput, start: u64, end: u64) -> Result<Self> {
        if start > end {
            return Err(Mp4Error::InvalidData("reader range is reversed"));
        }
        Ok(Self {
            input,
            position: start,
            end,
        })
    }

    #[inline]
    pub(crate) const fn position(&self) -> u64 {
        self.position
    }

    #[inline]
    pub(crate) fn remaining(&self) -> Result<u64> {
        self.end
            .checked_sub(self.position)
            .ok_or(Mp4Error::IntegerOverflow)
    }

    pub(crate) fn skip(&mut self, count: u64) -> Result<()> {
        let position = self
            .position
            .checked_add(count)
            .ok_or(Mp4Error::IntegerOverflow)?;
        if position > self.end {
            return Err(Mp4Error::InvalidData("MP4 field exceeds its box"));
        }
        self.position = position;
        Ok(())
    }

    pub(crate) fn read_u8(&mut self) -> Result<u8> {
        Ok(self.read_array::<1>()?[0])
    }

    pub(crate) fn read_u16(&mut self) -> Result<u16> {
        Ok(u16::from_be_bytes(self.read_array()?))
    }

    pub(crate) fn read_i16(&mut self) -> Result<i16> {
        Ok(i16::from_be_bytes(self.read_array()?))
    }

    pub(crate) fn read_u24(&mut self) -> Result<u32> {
        let bytes = self.read_array::<3>()?;
        Ok(u32::from(bytes[0]) << 16 | u32::from(bytes[1]) << 8 | u32::from(bytes[2]))
    }

    pub(crate) fn read_u32(&mut self) -> Result<u32> {
        Ok(u32::from_be_bytes(self.read_array()?))
    }

    pub(crate) fn read_i32(&mut self) -> Result<i32> {
        Ok(i32::from_be_bytes(self.read_array()?))
    }

    pub(crate) fn read_u64(&mut self) -> Result<u64> {
        Ok(u64::from_be_bytes(self.read_array()?))
    }

    pub(crate) fn read_i64(&mut self) -> Result<i64> {
        Ok(i64::from_be_bytes(self.read_array()?))
    }

    pub(crate) fn read_fourcc(&mut self) -> Result<FourCc> {
        Ok(FourCc::new(self.read_array()?))
    }

    pub(crate) fn read_vec(&mut self, length: u64, maximum: usize) -> Result<Vec<u8>> {
        let length = usize::try_from(length).map_err(|_| Mp4Error::IntegerOverflow)?;
        if length > maximum {
            return Err(Mp4Error::InvalidData("MP4 allocation exceeds its limit"));
        }
        let mut bytes = vec![0; length];
        self.read_exact(&mut bytes)?;
        Ok(bytes)
    }

    fn read_array<const N: usize>(&mut self) -> Result<[u8; N]> {
        let mut bytes = [0; N];
        self.read_exact(&mut bytes)?;
        Ok(bytes)
    }

    fn read_exact(&mut self, mut buffer: &mut [u8]) -> Result<()> {
        let requested = u64::try_from(buffer.len()).map_err(|_| Mp4Error::IntegerOverflow)?;
        if requested > self.remaining()? {
            return Err(Mp4Error::InvalidData("MP4 field exceeds its box"));
        }
        while !buffer.is_empty() {
            let read = self.input.read_at(self.position, buffer)?;
            if read == 0 {
                return Err(Mp4Error::InvalidData("unexpected end of MP4 input"));
            }
            if read > buffer.len() {
                return Err(Mp4Error::InvalidData(
                    "MediaInput returned more bytes than requested",
                ));
            }
            self.position = self
                .position
                .checked_add(u64::try_from(read).map_err(|_| Mp4Error::IntegerOverflow)?)
                .ok_or(Mp4Error::IntegerOverflow)?;
            buffer = &mut buffer[read..];
        }
        Ok(())
    }
}

pub(crate) fn read_full_box(reader: &mut BoundedReader<'_>) -> Result<(u8, u32)> {
    Ok((reader.read_u8()?, reader.read_u24()?))
}
