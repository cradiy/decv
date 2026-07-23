//! Reusable CABAC binarization readers.

use crate::{CabacContextSet, CabacDecoder, H264Error, Result};

/// Couples the arithmetic engine with one slice's adaptive context models.
///
/// Higher H.264 syntax layers use this type to decode bin strings without
/// reaching into either object's storage representation.
#[derive(Debug)]
pub struct CabacSyntaxDecoder<'syntax, 'reader, 'data> {
    arithmetic: &'syntax mut CabacDecoder<'reader, 'data>,
    contexts: &'syntax mut CabacContextSet,
}

impl<'syntax, 'reader, 'data> CabacSyntaxDecoder<'syntax, 'reader, 'data> {
    pub const fn new(
        arithmetic: &'syntax mut CabacDecoder<'reader, 'data>,
        contexts: &'syntax mut CabacContextSet,
    ) -> Self {
        Self {
            arithmetic,
            contexts,
        }
    }

    /// Decodes one decision bin and updates the selected context in place.
    #[inline]
    pub fn decision(&mut self, context_index: usize) -> Result<u8> {
        self.arithmetic
            .decode_decision(self.contexts.get_mut(context_index)?)
    }

    /// Decodes one bypass bin.
    #[inline]
    pub fn bypass(&mut self) -> Result<u8> {
        self.arithmetic.decode_bypass()
    }

    /// Decodes one terminate bin.
    #[inline]
    pub fn terminate(&mut self) -> Result<u8> {
        self.arithmetic.decode_terminate()
    }

    /// Decodes a fixed-length bypass-coded bin string, MSB first.
    pub fn bypass_bits(&mut self, bit_count: u8) -> Result<u32> {
        decode_fixed_length(bit_count, || self.bypass())
    }

    /// Decodes truncated unary with an explicit context progression.
    ///
    /// `context_indices[0]` selects the first bin. Later entries select later
    /// bins; once the list is exhausted its final entry is repeated. A string
    /// of `maximum_value` one-bins represents the maximum without a terminating
    /// zero-bin.
    pub fn truncated_unary(
        &mut self,
        context_indices: &[usize],
        maximum_value: u32,
    ) -> Result<u32> {
        decode_truncated_unary(context_indices, maximum_value, |context_index| {
            self.decision(context_index)
        })
    }

    /// Decodes unary with an explicit upper bound for malformed streams.
    ///
    /// Unlike truncated unary, the maximum legal value still requires a
    /// terminating zero-bin. A longer run of one-bins is rejected.
    pub fn unary(&mut self, context_indices: &[usize], maximum_value: u32) -> Result<u32> {
        decode_unary(context_indices, maximum_value, |context_index| {
            self.decision(context_index)
        })
    }
}

fn decode_fixed_length(mut bit_count: u8, mut decode: impl FnMut() -> Result<u8>) -> Result<u32> {
    if bit_count > 32 {
        return Err(H264Error::InvalidSyntax(
            "CABAC fixed-length binarization exceeds 32 bits",
        ));
    }
    let mut value = 0u32;
    while bit_count != 0 {
        value = (value << 1) | u32::from(decode()?);
        bit_count -= 1;
    }
    Ok(value)
}

fn decode_truncated_unary(
    context_indices: &[usize],
    maximum_value: u32,
    mut decode: impl FnMut(usize) -> Result<u8>,
) -> Result<u32> {
    if maximum_value == 0 {
        return Ok(0);
    }
    validate_context_progression(context_indices)?;
    for value in 0..maximum_value {
        let context_index = progressing_context_index(context_indices, value);
        if decode(context_index)? == 0 {
            return Ok(value);
        }
    }
    Ok(maximum_value)
}

fn decode_unary(
    context_indices: &[usize],
    maximum_value: u32,
    mut decode: impl FnMut(usize) -> Result<u8>,
) -> Result<u32> {
    validate_context_progression(context_indices)?;
    for value in 0..=maximum_value {
        let context_index = progressing_context_index(context_indices, value);
        if decode(context_index)? == 0 {
            return Ok(value);
        }
    }
    Err(H264Error::InvalidSyntax(
        "CABAC unary binarization exceeds its syntax bound",
    ))
}

fn validate_context_progression(context_indices: &[usize]) -> Result<()> {
    if context_indices.is_empty() {
        return Err(H264Error::InvalidSyntax(
            "CABAC context progression is empty",
        ));
    }
    Ok(())
}

fn progressing_context_index(context_indices: &[usize], bin_index: u32) -> usize {
    context_indices[usize::try_from(bin_index)
        .unwrap_or(usize::MAX)
        .min(context_indices.len() - 1)]
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use super::*;

    #[test]
    fn decodes_fixed_length_bits_most_significant_first() {
        let mut bins = VecDeque::from([1, 0, 1, 1]);
        assert_eq!(
            decode_fixed_length(4, || Ok(bins.pop_front().unwrap())).unwrap(),
            0b1011
        );
        assert!(decode_fixed_length(33, || Ok(0)).is_err());
    }

    #[test]
    fn truncated_unary_progresses_then_repeats_the_last_context() {
        let mut bins = VecDeque::from([1, 1, 1, 0]);
        let mut visited = Vec::new();
        let value = decode_truncated_unary(&[10, 11, 12], 5, |context_index| {
            visited.push(context_index);
            Ok(bins.pop_front().unwrap())
        })
        .unwrap();
        assert_eq!(value, 3);
        assert_eq!(visited, [10, 11, 12, 12]);
    }

    #[test]
    fn truncated_unary_omits_the_terminal_zero_at_the_maximum() {
        let mut bins = VecDeque::from([1, 1, 1]);
        assert_eq!(
            decode_truncated_unary(&[20], 3, |_| Ok(bins.pop_front().unwrap())).unwrap(),
            3
        );
        assert!(bins.is_empty());
        assert_eq!(
            decode_truncated_unary(&[], 0, |_| unreachable!()).unwrap(),
            0
        );
    }

    #[test]
    fn unary_requires_a_terminal_zero_and_enforces_its_bound() {
        let mut valid = VecDeque::from([1, 1, 0]);
        assert_eq!(
            decode_unary(&[30, 31], 2, |_| Ok(valid.pop_front().unwrap())).unwrap(),
            2
        );

        let mut invalid = VecDeque::from([1, 1, 1]);
        assert!(matches!(
            decode_unary(&[30], 2, |_| Ok(invalid.pop_front().unwrap())),
            Err(H264Error::InvalidSyntax(_))
        ));
        assert!(decode_unary(&[], 2, |_| Ok(0)).is_err());
    }
}
