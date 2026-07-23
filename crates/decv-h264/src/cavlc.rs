//! Context-adaptive variable-length coefficient decoding.

use bit_readers::BitReader;

use crate::{H264Error, Result};

#[path = "cavlc_tables.rs"]
mod tables;

use tables::{
    COEFF_TOKEN_0_TO_1, COEFF_TOKEN_2_TO_3, COEFF_TOKEN_4_TO_7, COEFF_TOKEN_CHROMA_DC_420, VlcEntry,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoeffTokenContext {
    /// The `nC` value derived from available left and top transform blocks.
    NeighborTotal(u8),
    /// Chroma DC for 4:2:0 sampling (`nC == -1`).
    ChromaDc420,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct CoeffToken {
    pub total_coeff: u8,
    pub trailing_ones: u8,
}

/// Decodes `coeff_token` while preserving the reader on any failure.
pub fn decode_coeff_token(
    reader: &mut BitReader<'_>,
    context: CoeffTokenContext,
    max_num_coeff: u8,
) -> Result<CoeffToken> {
    if !(1..=16).contains(&max_num_coeff) {
        return Err(H264Error::InvalidSyntax(
            "maxNumCoeff for CAVLC must be in 1..=16",
        ));
    }

    let mut probe = *reader;
    let token = match context {
        CoeffTokenContext::NeighborTotal(n_c) if n_c >= 8 => decode_fixed_coeff_token(&mut probe)?,
        CoeffTokenContext::NeighborTotal(n_c) => {
            let table = match n_c {
                0..=1 => COEFF_TOKEN_0_TO_1,
                2..=3 => COEFF_TOKEN_2_TO_3,
                4..=7 => COEFF_TOKEN_4_TO_7,
                _ => unreachable!("u8 context is covered by the ranges above"),
            };
            decode_vlc(&mut probe, table)?
        }
        CoeffTokenContext::ChromaDc420 => decode_vlc(&mut probe, COEFF_TOKEN_CHROMA_DC_420)?,
    };

    if token.total_coeff > max_num_coeff || token.trailing_ones > token.total_coeff.min(3) {
        return Err(H264Error::InvalidSyntax(
            "coeff_token exceeds the transform block bounds",
        ));
    }

    *reader = probe;
    Ok(token)
}

fn decode_fixed_coeff_token(reader: &mut BitReader<'_>) -> Result<CoeffToken> {
    let code = reader
        .read_bits_const::<6>()
        .ok_or(H264Error::UnexpectedEof)? as u8;
    if code == 3 {
        return Ok(CoeffToken::default());
    }

    let total_coeff = code / 4 + 1;
    let trailing_ones = code % 4;
    if total_coeff == 1 && trailing_ones > 1 {
        return Err(H264Error::InvalidSyntax("invalid fixed-length coeff_token"));
    }
    Ok(CoeffToken {
        total_coeff,
        trailing_ones,
    })
}

fn decode_vlc(reader: &mut BitReader<'_>, table: &[VlcEntry]) -> Result<CoeffToken> {
    let mut bits = 0u16;
    for length in 1..=16 {
        bits = (bits << 1) | u16::from(reader.read_bit().ok_or(H264Error::UnexpectedEof)?);
        if let Some(entry) = table
            .iter()
            .find(|entry| entry.length == length && entry.bits == bits)
        {
            return Ok(CoeffToken {
                total_coeff: entry.total_coeff,
                trailing_ones: entry.trailing_ones,
            });
        }
    }
    Err(H264Error::InvalidSyntax("invalid coeff_token VLC"))
}

#[cfg(test)]
mod tests {
    use bit_readers::BitReader;

    use super::{CoeffToken, CoeffTokenContext, decode_coeff_token};
    use crate::H264Error;

    #[test]
    fn decodes_each_coeff_token_context_table() {
        let vectors = [
            (
                "1",
                CoeffTokenContext::NeighborTotal(0),
                CoeffToken {
                    total_coeff: 0,
                    trailing_ones: 0,
                },
            ),
            (
                "001",
                CoeffTokenContext::NeighborTotal(1),
                CoeffToken {
                    total_coeff: 2,
                    trailing_ones: 2,
                },
            ),
            (
                "0101",
                CoeffTokenContext::NeighborTotal(2),
                CoeffToken {
                    total_coeff: 3,
                    trailing_ones: 3,
                },
            ),
            (
                "1000",
                CoeffTokenContext::NeighborTotal(4),
                CoeffToken {
                    total_coeff: 7,
                    trailing_ones: 3,
                },
            ),
            (
                "111111",
                CoeffTokenContext::NeighborTotal(8),
                CoeffToken {
                    total_coeff: 16,
                    trailing_ones: 3,
                },
            ),
            (
                "0000000",
                CoeffTokenContext::ChromaDc420,
                CoeffToken {
                    total_coeff: 4,
                    trailing_ones: 3,
                },
            ),
        ];

        for (bits, context, expected) in vectors {
            let bytes = bit_string(bits);
            let mut reader = BitReader::new(&bytes);
            assert_eq!(
                decode_coeff_token(&mut reader, context, 16),
                Ok(expected),
                "{bits}"
            );
            assert_eq!(reader.bit_position(), bits.len());
        }
    }

    #[test]
    fn rejects_invalid_or_out_of_block_tokens_atomically() {
        let invalid_fixed = bit_string("000010");
        let mut reader = BitReader::new(&invalid_fixed);
        assert!(matches!(
            decode_coeff_token(&mut reader, CoeffTokenContext::NeighborTotal(8), 16),
            Err(H264Error::InvalidSyntax(_))
        ));
        assert_eq!(reader.bit_position(), 0);

        let total_sixteen = bit_string("0000000000000100");
        let mut reader = BitReader::new(&total_sixteen);
        assert!(matches!(
            decode_coeff_token(&mut reader, CoeffTokenContext::NeighborTotal(0), 15),
            Err(H264Error::InvalidSyntax(_))
        ));
        assert_eq!(reader.bit_position(), 0);

        let mut truncated = BitReader::new(&[0]);
        assert_eq!(
            decode_coeff_token(&mut truncated, CoeffTokenContext::NeighborTotal(0), 16),
            Err(H264Error::UnexpectedEof)
        );
        assert_eq!(truncated.bit_position(), 0);
    }

    fn bit_string(bits: &str) -> Vec<u8> {
        let mut bytes = vec![0; bits.len().div_ceil(8)];
        for (index, bit) in bits.bytes().enumerate() {
            if bit == b'1' {
                bytes[index / 8] |= 1 << (7 - index % 8);
            }
        }
        bytes
    }
}
