/// A forward-only, MSB-first bit reader.
///
/// The next unread bits are kept in the most significant bits of `cache`.
/// Reads of up to 32 bits therefore need only a shift on the hot path. The
/// cache is refilled 32 bits at a time, with a byte-at-a-time fallback for the
/// end of the input.
#[derive(Debug, Clone, Copy)]
pub struct BitReader<'a> {
    data: &'a [u8],
    /// Index of the next byte that has not been loaded into `cache`.
    cursor: usize,
    /// Cached bits, aligned to the most significant end.
    cache: u64,
    /// Number of valid high bits in `cache`.
    cached_bits: u32,
}

impl<'a> BitReader<'a> {
    #[inline]
    pub const fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            cursor: 0,
            cache: 0,
            cached_bits: 0,
        }
    }

    #[inline]
    pub const fn byte_len(&self) -> usize {
        self.data.len()
    }

    #[inline]
    pub fn bit_len(&self) -> usize {
        self.data.len().saturating_mul(8)
    }

    /// Number of bits consumed so far.
    #[inline]
    pub fn bit_position(&self) -> usize {
        self.cursor
            .saturating_mul(8)
            .saturating_sub(self.cached_bits as usize)
    }

    /// Number of whole bytes consumed so far.
    #[inline]
    pub fn byte_position(&self) -> usize {
        self.bit_position() / 8
    }

    /// Offset of the next bit within the current byte.
    #[inline]
    pub fn bit_offset(&self) -> u32 {
        (self.bit_position() % 8) as u32
    }

    #[inline]
    pub fn remaining_bits(&self) -> usize {
        let unloaded_bits = (self.data.len() - self.cursor).saturating_mul(8);
        unloaded_bits.saturating_add(self.cached_bits as usize)
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.remaining_bits() == 0
    }

    /// Reads one bit.
    ///
    /// Returns `None` at the end of the input without changing the reader.
    #[inline(always)]
    pub fn read_bit(&mut self) -> Option<u8> {
        if self.cached_bits == 0 {
            if self.cursor == self.data.len() {
                return None;
            }

            if self.data.len() - self.cursor >= 4 {
                self.refill_word();
            } else {
                self.refill_byte();
            }
        }

        let bit = (self.cache >> 63) as u8;
        self.cache <<= 1;
        self.cached_bits -= 1;
        Some(bit)
    }

    /// Reads between 0 and 32 bits, MSB first.
    ///
    /// Reading zero bits succeeds with zero. Invalid widths and truncated
    /// input return `None` without changing the reader.
    #[inline(always)]
    pub fn read_bits(&mut self, count: u32) -> Option<u32> {
        let value = self.peek_bits(count)?;
        self.consume_cached(count);
        Some(value)
    }

    /// Reads a compile-time constant number of bits, MSB first.
    ///
    /// Prefer this variant for fixed-width syntax fields. It allows the
    /// compiler to eliminate the width check and generate constant shifts.
    #[inline(always)]
    pub fn read_bits_const<const COUNT: u32>(&mut self) -> Option<u32> {
        const {
            assert!(COUNT <= 32, "BitReader can read at most 32 bits at once");
        }

        if COUNT == 0 {
            return Some(0);
        }
        if !self.has_bits(COUNT) {
            return None;
        }

        if self.cached_bits < COUNT {
            self.refill_for(COUNT);
        }

        let value = (self.cache >> (64 - COUNT)) as u32;
        self.consume_cached(COUNT);
        Some(value)
    }

    /// Reads an unsigned exponential-Golomb value (`ue(v)`).
    ///
    /// The common case is decoded from one 32-bit lookahead using
    /// `leading_zeros`. Truncated input and values larger than `u32::MAX`
    /// return `None` without consuming input.
    #[inline]
    pub fn read_ue(&mut self) -> Option<u32> {
        if self.cached_bits < 32 && self.data.len() - self.cursor >= 4 {
            self.refill_word();
        }
        while self.cached_bits < 32 && self.cursor < self.data.len() {
            self.refill_byte();
        }

        let available = self.cached_bits.min(32);
        if available == 0 {
            return None;
        }

        // Valid cache bits are already MSB-aligned. When fewer than 32 are
        // available, the unused low bits are zero and naturally act as
        // lookahead padding.
        let window = (self.cache >> 32) as u32;
        let leading_zeros = window.leading_zeros();

        // A code with at most 15 leading zeroes fits completely in the
        // 32-bit lookahead. This covers the overwhelmingly common case.
        if leading_zeros <= 15 {
            let code_bits = leading_zeros * 2 + 1;
            if code_bits > available {
                return None;
            }

            let suffix = if leading_zeros == 0 {
                0
            } else {
                let code = window >> (32 - code_bits);
                code & ((1 << leading_zeros) - 1)
            };

            self.consume_cached(code_bits);
            return Some(((1 << leading_zeros) - 1) + suffix);
        }

        self.read_ue_slow()
    }

    /// Reads a signed exponential-Golomb value (`se(v)`).
    ///
    /// A result outside the `i32` range returns `None` without consuming
    /// input.
    #[inline]
    pub fn read_se(&mut self) -> Option<i32> {
        let mut probe = *self;
        let code_num = probe.read_ue()?;

        let value = if code_num & 1 == 0 {
            -(code_num as i64 / 2)
        } else {
            (code_num as i64 + 1) / 2
        };
        let value = i32::try_from(value).ok()?;

        *self = probe;
        Some(value)
    }

    /// Peeks between 0 and 32 bits without consuming them.
    ///
    /// Refilling the internal cache is not considered observable consumption:
    /// the reported position and subsequent output remain unchanged.
    #[inline(always)]
    pub fn peek_bits(&mut self, count: u32) -> Option<u32> {
        if count == 0 {
            return Some(0);
        }
        if count > 32 || !self.has_bits(count) {
            return None;
        }

        if self.cached_bits < count {
            self.refill_for(count);
        }

        debug_assert!(self.cached_bits >= count);
        Some((self.cache >> (64 - count)) as u32)
    }

    /// Skips `count` bits.
    ///
    /// Unlike `read_bits`, this operation efficiently handles arbitrarily
    /// large skips by bypassing whole bytes after draining the cache.
    #[inline]
    pub fn skip_bits(&mut self, mut count: usize) -> bool {
        if count > self.remaining_bits() {
            return false;
        }

        if count <= self.cached_bits as usize {
            self.consume_cached(count as u32);
            return true;
        }

        // Draining the cache lands on a byte boundary because only complete
        // bytes are ever loaded into it.
        if self.cached_bits != 0 {
            let cached = self.cached_bits as usize;
            count -= cached;
            self.cache = 0;
            self.cached_bits = 0;
        }

        let whole_bytes = count / 8;
        self.cursor += whole_bytes;
        count %= 8;

        if count != 0 {
            self.refill_for(count as u32);
            self.consume_cached(count as u32);
        }

        true
    }

    /// Advances to the next byte boundary.
    #[inline]
    pub fn byte_align(&mut self) -> bool {
        let bits = (8 - self.bit_offset()) % 8;
        self.skip_bits(bits as usize)
    }

    #[inline(always)]
    fn consume_cached(&mut self, count: u32) {
        debug_assert!(count <= self.cached_bits);

        if count == 0 {
            return;
        }

        self.cache <<= count;
        self.cached_bits -= count;
    }

    #[inline(always)]
    fn has_bits(&self, count: u32) -> bool {
        if self.cached_bits >= count {
            return true;
        }

        let missing = count - self.cached_bits;
        let required_bytes = missing.div_ceil(8) as usize;
        self.data.len() - self.cursor >= required_bytes
    }

    #[inline]
    fn refill_for(&mut self, count: u32) {
        debug_assert!((1..=32).contains(&count));
        debug_assert!(self.has_bits(count));

        while self.cached_bits < count {
            if self.data.len() - self.cursor >= 4 {
                self.refill_word();
            } else {
                self.refill_byte();
            }
        }
    }

    #[cold]
    #[inline(never)]
    fn read_ue_slow(&mut self) -> Option<u32> {
        // Decode on a copy so truncation and overflow remain atomic.
        let mut probe = *self;
        let mut leading_zeros = 0;

        while probe.read_bit()? == 0 {
            leading_zeros += 1;
            if leading_zeros > 32 {
                return None;
            }
        }

        let suffix = probe.read_bits(leading_zeros)?;
        let value = ((1u64 << leading_zeros) - 1) + suffix as u64;
        let value = u32::try_from(value).ok()?;

        *self = probe;
        Some(value)
    }

    #[inline(always)]
    fn refill_word(&mut self) {
        debug_assert!(self.cached_bits < 32);
        debug_assert!(self.data.len() - self.cursor >= 4);

        // SAFETY: The length check above guarantees that four bytes starting
        // at `cursor` belong to `data`. `read_unaligned` imposes no alignment
        // requirement. Reading a u32 does not outlive or mutate the slice.
        let word = unsafe {
            let pointer = self.data.as_ptr().add(self.cursor).cast::<u32>();
            u32::from_be(std::ptr::read_unaligned(pointer))
        };

        self.cache |= (word as u64) << (32 - self.cached_bits);
        self.cached_bits += 32;
        self.cursor += 4;
    }

    #[cold]
    #[inline(never)]
    fn refill_byte(&mut self) {
        debug_assert!(self.cached_bits < 32);
        debug_assert!(self.cursor < self.data.len());

        // SAFETY: The bound is established above. This function is only
        // reached for the final one to three bytes of the input.
        let byte = unsafe { *self.data.get_unchecked(self.cursor) };

        self.cache |= (byte as u64) << (56 - self.cached_bits);
        self.cached_bits += 8;
        self.cursor += 1;
    }
}

impl<'a> From<&'a [u8]> for BitReader<'a> {
    #[inline]
    fn from(data: &'a [u8]) -> Self {
        Self::new(data)
    }
}

#[cfg(test)]
mod tests {
    use super::BitReader;

    #[test]
    fn reads_msb_first() {
        let mut reader = BitReader::new(&[0b1011_0010]);

        assert_eq!(reader.read_bits(3), Some(0b101));
        assert_eq!(reader.read_bits(2), Some(0b10));
        assert_eq!(reader.read_bits(3), Some(0b010));
        assert!(reader.is_empty());
    }

    #[test]
    fn reads_across_refill_boundaries() {
        let data = [0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0];
        let mut reader = BitReader::new(&data);

        assert_eq!(reader.read_bits(28), Some(0x0123_4567));
        assert_eq!(reader.read_bits(12), Some(0x0000_089a));
        assert_eq!(reader.read_bits(24), Some(0x00bc_def0));
        assert!(reader.is_empty());
    }

    #[test]
    fn reads_full_words() {
        let mut reader = BitReader::new(&[0xde, 0xad, 0xbe, 0xef]);

        assert_eq!(reader.read_bits(32), Some(0xdead_beef));
        assert_eq!(reader.bit_position(), 32);
        assert_eq!(reader.byte_position(), 4);
    }

    #[test]
    fn reads_compile_time_widths() {
        let mut reader = BitReader::new(&[0xde, 0xad, 0xbe, 0xef]);

        assert_eq!(reader.read_bits_const::<0>(), Some(0));
        assert_eq!(reader.read_bits_const::<4>(), Some(0xd));
        assert_eq!(reader.read_bits_const::<12>(), Some(0xead));
        assert_eq!(reader.read_bits_const::<16>(), Some(0xbeef));
        assert_eq!(reader.read_bits_const::<1>(), None);
        assert_eq!(reader.bit_position(), 32);
    }

    #[test]
    fn reads_unsigned_exponential_golomb_values() {
        let values = [0, 1, 2, 3, 4, 5, 6, 7, 31, 255, 65_535];
        let data = encode_ue_values(&values);
        let mut reader = BitReader::new(&data);

        for expected in values {
            assert_eq!(reader.read_ue(), Some(expected));
        }
    }

    #[test]
    fn reads_signed_exponential_golomb_values() {
        // se(v): 0, 1, -1, 2, -2, 3, -3
        let data = encode_ue_values(&[0, 1, 2, 3, 4, 5, 6]);
        let mut reader = BitReader::new(&data);

        for expected in [0, 1, -1, 2, -2, 3, -3] {
            assert_eq!(reader.read_se(), Some(expected));
        }
    }

    #[test]
    fn exponential_golomb_failures_are_atomic() {
        let mut truncated = BitReader::new(&[0]);
        assert_eq!(truncated.read_ue(), None);
        assert_eq!(truncated.bit_position(), 0);

        // u32::MAX is represented by 32 zeroes, a one, and 32 zeroes.
        let maximum = encode_ue_values(&[u32::MAX]);
        let mut reader = BitReader::new(&maximum);
        assert_eq!(reader.read_ue(), Some(u32::MAX));

        let mut signed_overflow = BitReader::new(&maximum);
        assert_eq!(signed_overflow.read_se(), None);
        assert_eq!(signed_overflow.bit_position(), 0);

        // Same prefix as u32::MAX, but a non-zero suffix overflows u32.
        let overflow = [0, 0, 0, 0, 0x80, 0, 0, 0, 0x80];
        let mut reader = BitReader::new(&overflow);
        assert_eq!(reader.read_ue(), None);
        assert_eq!(reader.bit_position(), 0);
    }

    #[test]
    fn peeking_does_not_consume_input() {
        let mut reader = BitReader::new(&[0xab, 0xcd]);

        assert_eq!(reader.peek_bits(12), Some(0xabc));
        assert_eq!(reader.bit_position(), 0);
        assert_eq!(reader.remaining_bits(), 16);
        assert_eq!(reader.peek_bits(12), Some(0xabc));
        assert_eq!(reader.read_bits(12), Some(0xabc));
        assert_eq!(reader.read_bits(4), Some(0xd));
    }

    #[test]
    fn failed_reads_are_atomic() {
        let mut reader = BitReader::new(&[0x80]);

        assert_eq!(reader.read_bits(9), None);
        assert_eq!(reader.read_bits(33), None);
        assert_eq!(reader.bit_position(), 0);
        assert_eq!(reader.read_bit(), Some(1));
        assert_eq!(reader.bit_position(), 1);

        assert!(!reader.skip_bits(8));
        assert_eq!(reader.bit_position(), 1);
        assert_eq!(reader.remaining_bits(), 7);
    }

    #[test]
    fn handles_zero_width_reads() {
        let mut reader = BitReader::new(&[]);

        assert_eq!(reader.read_bits(0), Some(0));
        assert_eq!(reader.peek_bits(0), Some(0));
        assert!(reader.skip_bits(0));
        assert_eq!(reader.bit_position(), 0);
    }

    #[test]
    fn skips_cached_and_uncached_data() {
        let data = [0xff, 0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0, 0x55];
        let mut reader = BitReader::new(&data);

        assert_eq!(reader.read_bits(4), Some(0xf));
        assert!(reader.skip_bits(68));
        assert_eq!(reader.bit_position(), 72);
        assert_eq!(reader.byte_position(), 9);
        assert_eq!(reader.bit_offset(), 0);
        assert_eq!(reader.read_bits(8), Some(0x55));
    }

    #[test]
    fn aligns_to_the_next_byte() {
        let mut reader = BitReader::new(&[0b1011_0010, 0x5a]);

        assert_eq!(reader.read_bits(3), Some(0b101));
        assert_eq!(reader.bit_offset(), 3);
        assert!(reader.byte_align());
        assert_eq!(reader.bit_position(), 8);
        assert_eq!(reader.bit_offset(), 0);
        assert_eq!(reader.read_bits(8), Some(0x5a));
    }

    #[test]
    fn handles_short_tail_refills() {
        for length in 1..=3 {
            let data = [0x12, 0x34, 0x56];
            let mut reader = BitReader::new(&data[..length]);

            for expected in &data[..length] {
                assert_eq!(reader.read_bits(8), Some(*expected as u32));
            }

            assert_eq!(reader.read_bit(), None);
        }
    }

    #[test]
    fn reports_positions_and_lengths() {
        let mut reader = BitReader::from(&[0xaa, 0xbb, 0xcc][..]);

        assert_eq!(reader.byte_len(), 3);
        assert_eq!(reader.bit_len(), 24);
        assert_eq!(reader.remaining_bits(), 24);

        assert_eq!(reader.read_bits(11), Some(0b101_0101_0101));
        assert_eq!(reader.bit_position(), 11);
        assert_eq!(reader.byte_position(), 1);
        assert_eq!(reader.bit_offset(), 3);
        assert_eq!(reader.remaining_bits(), 13);
    }

    #[test]
    fn matches_a_naive_reader_for_mixed_operations() {
        fn naive_read(data: &[u8], position: usize, count: u32) -> Option<u32> {
            if count > 32 || position + count as usize > data.len() * 8 {
                return None;
            }

            let mut value = 0;
            for index in 0..count as usize {
                let bit_position = position + index;
                let byte = data[bit_position / 8];
                let bit = (byte >> (7 - bit_position % 8)) & 1;
                value = (value << 1) | bit as u32;
            }
            Some(value)
        }

        let data = [
            0x03, 0x8f, 0x51, 0xc7, 0x9a, 0xe4, 0x2d, 0x60, 0xbb, 0x17, 0xd2, 0x48, 0xf5, 0x6c,
            0x81, 0x3e, 0xa9,
        ];

        for initial_skip in 0..=40 {
            for width in 0..=32 {
                let mut reader = BitReader::new(&data);
                assert!(reader.skip_bits(initial_skip));

                let expected = naive_read(&data, initial_skip, width);
                assert_eq!(reader.peek_bits(width), expected);
                assert_eq!(reader.bit_position(), initial_skip);
                assert_eq!(reader.read_bits(width), expected);
                assert_eq!(reader.bit_position(), initial_skip + width as usize);
            }
        }

        let mut reader = BitReader::new(&data);
        let widths = [1, 7, 3, 32, 19, 4, 28, 5, 17, 8, 12];
        let mut position = 0;

        for width in widths {
            let expected = naive_read(&data, position, width);
            assert_eq!(reader.read_bits(width), expected);

            if expected.is_none() {
                assert_eq!(reader.bit_position(), position);
                break;
            }

            position += width as usize;
            assert_eq!(reader.bit_position(), position);
        }
    }

    fn encode_ue_values(values: &[u32]) -> Vec<u8> {
        let mut bits = Vec::new();

        for &value in values {
            let code_num = value as u64 + 1;
            let width = 64 - code_num.leading_zeros();
            let leading_zeros = width - 1;

            bits.extend(std::iter::repeat_n(0, leading_zeros as usize));
            for shift in (0..width).rev() {
                bits.push(((code_num >> shift) & 1) as u8);
            }
        }

        let mut bytes = vec![0; bits.len().div_ceil(8)];
        for (position, bit) in bits.into_iter().enumerate() {
            bytes[position / 8] |= bit << (7 - position % 8);
        }
        bytes
    }
}
