//! EBSP-to-RBSP conversion and RBSP trailing-bit validation.

use std::borrow::Cow;

use bit_readers::BitReader;
use memchr::memchr;

use crate::{H264Error, Result};

/// Removes H.264 emulation-prevention bytes from an EBSP payload.
///
/// The input is returned as a borrowed slice when it contains no escape bytes.
/// An allocation is made only when at least one `00 00 03 xx` sequence must be
/// transformed.
pub fn decode_rbsp(ebsp: &[u8]) -> Result<Cow<'_, [u8]>> {
    let mut output: Option<Vec<u8>> = None;
    let mut search_from = 0;
    let mut copy_from = 0;

    while let Some(relative) = memchr(0, &ebsp[search_from..]) {
        let first_zero = search_from + relative;
        if ebsp.get(first_zero + 1) != Some(&0) {
            search_from = first_zero + 1;
            continue;
        }

        let Some(&following) = ebsp.get(first_zero + 2) else {
            break;
        };
        match following {
            0x00..=0x02 => return Err(H264Error::InvalidRbspEscape),
            0x03 => {
                let next = ebsp
                    .get(first_zero + 3)
                    .copied()
                    .ok_or(H264Error::InvalidRbspEscape)?;
                if next > 0x03 {
                    return Err(H264Error::InvalidRbspEscape);
                }

                let decoded = output.get_or_insert_with(|| Vec::with_capacity(ebsp.len() - 1));
                decoded.extend_from_slice(&ebsp[copy_from..first_zero + 2]);
                copy_from = first_zero + 3;
                search_from = copy_from;
            }
            _ => {
                search_from = first_zero + 2;
            }
        }
    }

    match output {
        Some(mut output) => {
            output.extend_from_slice(&ebsp[copy_from..]);
            Ok(Cow::Owned(output))
        }
        None => Ok(Cow::Borrowed(ebsp)),
    }
}

/// Consumes and validates `rbsp_trailing_bits()`.
///
/// The caller invokes this after parsing the final syntax element in an RBSP.
/// A valid tail is one stop bit followed only by zero alignment bits.
pub fn consume_rbsp_trailing_bits(reader: &mut BitReader<'_>) -> Result<()> {
    match reader.read_bit() {
        Some(1) => {}
        Some(0) => return Err(H264Error::InvalidTrailingBits),
        Some(_) => unreachable!("BitReader returns only zero or one"),
        None => return Err(H264Error::UnexpectedEof),
    }

    while let Some(bit) = reader.read_bit() {
        if bit != 0 {
            return Err(H264Error::InvalidTrailingBits);
        }
    }

    Ok(())
}

/// Returns whether syntax data remains before `rbsp_trailing_bits()`.
///
/// The reader is not advanced. A malformed or missing tail is left for
/// [`consume_rbsp_trailing_bits`] to diagnose after the caller finishes
/// parsing its optional syntax.
pub(crate) fn more_rbsp_data(reader: &BitReader<'_>) -> bool {
    let mut probe = *reader;

    match probe.read_bit() {
        Some(1) => {
            while let Some(bit) = probe.read_bit() {
                if bit != 0 {
                    return true;
                }
            }
            false
        }
        Some(0) => true,
        Some(_) => unreachable!("BitReader returns only zero or one"),
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use bit_readers::BitReader;

    use super::{consume_rbsp_trailing_bits, decode_rbsp, more_rbsp_data};
    use crate::H264Error;

    #[test]
    fn borrows_an_unescaped_payload() {
        let ebsp = [0x42, 0x00, 0x1f, 0x80];
        let rbsp = decode_rbsp(&ebsp).unwrap();

        assert!(matches!(rbsp, Cow::Borrowed(_)));
        assert_eq!(rbsp.as_ptr(), ebsp.as_ptr());
        assert_eq!(rbsp.as_ref(), ebsp);
    }

    #[test]
    fn removes_one_or_more_emulation_prevention_bytes() {
        let ebsp = [
            0x11, 0x00, 0x00, 0x03, 0x01, 0x22, 0x00, 0x00, 0x03, 0x03, 0x80,
        ];
        let rbsp = decode_rbsp(&ebsp).unwrap();

        assert!(matches!(rbsp, Cow::Owned(_)));
        assert_eq!(
            rbsp.as_ref(),
            &[0x11, 0x00, 0x00, 0x01, 0x22, 0x00, 0x00, 0x03, 0x80]
        );
    }

    #[test]
    fn rejects_invalid_escape_sequences() {
        for ebsp in [&[0x00, 0x00, 0x03][..], &[0x00, 0x00, 0x03, 0x04]] {
            assert_eq!(decode_rbsp(ebsp), Err(H264Error::InvalidRbspEscape));
        }

        for ebsp in [
            &[0x00, 0x00, 0x00][..],
            &[0x00, 0x00, 0x01],
            &[0x00, 0x00, 0x02],
        ] {
            assert_eq!(decode_rbsp(ebsp), Err(H264Error::InvalidRbspEscape));
        }
    }

    #[test]
    fn consumes_valid_rbsp_trailing_bits() {
        let mut aligned = BitReader::new(&[0x80]);
        assert_eq!(consume_rbsp_trailing_bits(&mut aligned), Ok(()));

        let mut unaligned = BitReader::new(&[0b1011_0000]);
        assert_eq!(unaligned.read_bits_const::<3>(), Some(0b101));
        assert_eq!(consume_rbsp_trailing_bits(&mut unaligned), Ok(()));
    }

    #[test]
    fn rejects_invalid_or_missing_trailing_bits() {
        let mut missing_stop_bit = BitReader::new(&[0x00]);
        assert_eq!(
            consume_rbsp_trailing_bits(&mut missing_stop_bit),
            Err(H264Error::InvalidTrailingBits)
        );

        let mut nonzero_alignment = BitReader::new(&[0b1100_0000]);
        assert_eq!(
            consume_rbsp_trailing_bits(&mut nonzero_alignment),
            Err(H264Error::InvalidTrailingBits)
        );

        let mut empty = BitReader::new(&[]);
        assert_eq!(
            consume_rbsp_trailing_bits(&mut empty),
            Err(H264Error::UnexpectedEof)
        );
    }

    #[test]
    fn distinguishes_payload_from_rbsp_trailing_bits() {
        let trailing_only = BitReader::new(&[0x80]);
        assert!(!more_rbsp_data(&trailing_only));

        let mut unaligned_tail = BitReader::new(&[0b1011_0000]);
        assert_eq!(unaligned_tail.read_bits_const::<3>(), Some(0b101));
        assert!(!more_rbsp_data(&unaligned_tail));

        let payload_then_tail = BitReader::new(&[0b0100_0000]);
        assert!(more_rbsp_data(&payload_then_tail));
    }
}
