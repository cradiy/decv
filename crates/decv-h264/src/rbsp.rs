//! EBSP-to-RBSP conversion and RBSP trailing-bit validation.

use std::borrow::Cow;

use bit_readers::BitReader;

use crate::{H264Error, Result};

/// Removes H.264 emulation-prevention bytes from an EBSP payload.
///
/// The input is returned as a borrowed slice when it contains no escape bytes.
/// An allocation is made only when at least one `00 00 03 xx` sequence must be
/// transformed.
pub fn decode_rbsp(ebsp: &[u8]) -> Result<Cow<'_, [u8]>> {
    let mut output = None;
    let mut zero_count = 0u8;
    let mut index = 0;

    while index < ebsp.len() {
        let byte = ebsp[index];

        if zero_count == 2 {
            if byte == 0x03 {
                let next = ebsp
                    .get(index + 1)
                    .copied()
                    .ok_or(H264Error::InvalidRbspEscape)?;
                if next > 0x03 {
                    return Err(H264Error::InvalidRbspEscape);
                }

                output.get_or_insert_with(|| {
                    let mut decoded = Vec::with_capacity(ebsp.len() - 1);
                    decoded.extend_from_slice(&ebsp[..index]);
                    decoded
                });

                // The escape byte breaks the EBSP zero run, but is omitted
                // from RBSP output.
                zero_count = 0;
                index += 1;
                continue;
            }

            // Values 00..02 after two zero bytes must have been escaped.
            if byte <= 0x02 {
                return Err(H264Error::InvalidRbspEscape);
            }
        }

        if let Some(decoded) = &mut output {
            decoded.push(byte);
        }

        zero_count = if byte == 0 {
            zero_count.saturating_add(1).min(2)
        } else {
            0
        };
        index += 1;
    }

    match output {
        Some(output) => Ok(Cow::Owned(output)),
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

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use bit_readers::BitReader;

    use super::{consume_rbsp_trailing_bits, decode_rbsp};
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
}
