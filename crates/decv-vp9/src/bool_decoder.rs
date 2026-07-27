use crate::{Result, Vp9Error};

/// VP9's binary arithmetic decoder.
///
/// `probability` is the probability of a zero in units of 1/256. The decoder
/// maintains an eight-bit normalized range and sixteen bits of coded
/// lookahead; bytes are only pulled when normalization has consumed a full
/// byte.
#[derive(Debug, Clone)]
pub(crate) struct BoolDecoder<'a> {
    data: &'a [u8],
    cursor: usize,
    range: u16,
    value: u32,
    pending_shift: u8,
    padded_bytes: usize,
}

impl<'a> BoolDecoder<'a> {
    pub(crate) fn new(data: &'a [u8]) -> Result<Self> {
        if data.is_empty() {
            return Err(Vp9Error::Truncated("boolean-coded partition"));
        }
        let first = u32::from(data[0]);
        let second = u32::from(data.get(1).copied().unwrap_or(0));
        let mut decoder = Self {
            data,
            cursor: data.len().min(2),
            range: 255,
            value: first << 8 | second,
            pending_shift: 0,
            padded_bytes: usize::from(data.len() < 2),
        };
        if decoder.read_bool(128)? {
            return Err(Vp9Error::InvalidData(
                "boolean-coded partition marker must be zero",
            ));
        }
        Ok(decoder)
    }

    #[inline(always)]
    pub(crate) fn read_bool(&mut self, probability: u8) -> Result<bool> {
        let probability = u16::from(probability);
        let split = (self.range * probability + (256 - probability)) >> 8;
        debug_assert!(split != 0 && split <= self.range);
        let big_split = u32::from(split) << 8;

        let bit = self.value >= big_split;
        if bit {
            self.range -= split;
            self.value -= big_split;
        } else {
            self.range = split;
        }

        let shift = (self.range as u8).leading_zeros() as u8;
        self.range <<= shift;
        self.value = self.value.wrapping_shl(u32::from(shift)) & 0xffff;
        self.pending_shift += shift;
        while self.pending_shift >= 8 {
            self.pending_shift -= 8;
            let byte = if let Some(&byte) = self.data.get(self.cursor) {
                self.cursor += 1;
                byte
            } else {
                // VP9's normative boolean reader extends a finite partition
                // with zero bits. Syntax bounds, rather than an arbitrary
                // lookahead limit, prevent an unbounded read.
                self.padded_bytes += 1;
                0
            };
            self.value |= u32::from(byte) << self.pending_shift;
        }
        Ok(bit)
    }

    #[inline]
    pub(crate) fn padded_bytes(&self) -> usize {
        self.padded_bytes
    }

    #[inline]
    pub(crate) fn read_bit(&mut self) -> Result<bool> {
        self.read_bool(128)
    }

    pub(crate) fn read_literal(&mut self, count: u8) -> Result<u32> {
        if count > 32 {
            return Err(Vp9Error::InvalidData("boolean literal exceeds 32 bits"));
        }
        let mut value = 0;
        for _ in 0..count {
            value = value << 1 | u32::from(self.read_bit()?);
        }
        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    use super::BoolDecoder;

    #[test]
    fn half_probability_decodes_zero_fill_as_zeroes() {
        let mut decoder = BoolDecoder::new(&[0; 8]).unwrap();
        for _ in 0..32 {
            assert!(!decoder.read_bit().unwrap());
        }
    }

    #[test]
    fn half_probability_decodes_upper_half_as_ones_after_marker() {
        let mut decoder = BoolDecoder::new(&[0x7f, 0xff, 0xff, 0xff, 0xff]).unwrap();
        for _ in 0..24 {
            assert!(decoder.read_bit().unwrap());
        }
    }

    #[test]
    fn reads_literals_most_significant_bit_first() {
        let mut decoder = BoolDecoder::new(&[0x28, 0, 0]).unwrap();
        assert_eq!(decoder.read_literal(3).unwrap(), 0b010);
    }
}
