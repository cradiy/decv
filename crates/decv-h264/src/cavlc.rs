//! Context-adaptive variable-length coefficient decoding.

use bit_readers::BitReader;

use crate::{H264Error, Result};

#[path = "cavlc_tables.rs"]
mod tables;

use tables::{
    COEFF_TOKEN_0_TO_1, COEFF_TOKEN_0_TO_1_LOOKUP, COEFF_TOKEN_2_TO_3, COEFF_TOKEN_2_TO_3_LOOKUP,
    COEFF_TOKEN_4_TO_7, COEFF_TOKEN_4_TO_7_LOOKUP, COEFF_TOKEN_CHROMA_DC_420,
    COEFF_TOKEN_CHROMA_DC_420_LOOKUP, COEFF_TOKEN_LOOKUP_BITS, RUN_BEFORE_SMALL, TOTAL_ZEROS_4X4,
    TOTAL_ZEROS_CHROMA_DC_420, VlcCode, VlcEntry,
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

/// One decoded CAVLC transform block in coefficient scan order.
///
/// Only the first `max_num_coeff` entries can be populated. Mapping scan order
/// to transform-matrix coordinates is deliberately left to the reconstruction
/// stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidualBlock {
    pub coefficients: [i32; 16],
    pub total_coeff: u8,
    pub max_num_coeff: u8,
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
            let (table, lookup) = match n_c {
                0..=1 => (COEFF_TOKEN_0_TO_1, &COEFF_TOKEN_0_TO_1_LOOKUP),
                2..=3 => (COEFF_TOKEN_2_TO_3, &COEFF_TOKEN_2_TO_3_LOOKUP),
                4..=7 => (COEFF_TOKEN_4_TO_7, &COEFF_TOKEN_4_TO_7_LOOKUP),
                _ => unreachable!("u8 context is covered by the ranges above"),
            };
            decode_vlc(&mut probe, table, lookup)?
        }
        CoeffTokenContext::ChromaDc420 => decode_vlc(
            &mut probe,
            COEFF_TOKEN_CHROMA_DC_420,
            &COEFF_TOKEN_CHROMA_DC_420_LOOKUP,
        )?,
    };

    if token.total_coeff > max_num_coeff || token.trailing_ones > token.total_coeff.min(3) {
        return Err(H264Error::InvalidSyntax(
            "coeff_token exceeds the transform block bounds",
        ));
    }

    *reader = probe;
    Ok(token)
}

/// Decodes one complete CAVLC residual block.
///
/// The currently supported 4:2:0 pipeline uses block sizes 16 (4x4), 15
/// (Intra16x16 AC), and 4 (chroma DC). The reader is advanced only when the
/// complete block is valid.
pub fn decode_residual_block(
    reader: &mut BitReader<'_>,
    context: CoeffTokenContext,
    max_num_coeff: u8,
) -> Result<ResidualBlock> {
    if !matches!(max_num_coeff, 4 | 15 | 16) {
        return Err(H264Error::UnsupportedFeature(
            "CAVLC residual block sizes other than 4, 15, and 16",
        ));
    }

    let mut probe = *reader;
    let token = decode_coeff_token(&mut probe, context, max_num_coeff)?;
    let mut coefficients = [0i32; 16];
    if token.total_coeff == 0 {
        *reader = probe;
        return Ok(ResidualBlock {
            coefficients,
            total_coeff: 0,
            max_num_coeff,
        });
    }

    let levels = decode_levels(&mut probe, token)?;
    let runs = decode_runs(&mut probe, token.total_coeff, max_num_coeff)?;
    combine_levels_and_runs(
        &mut coefficients,
        &levels,
        &runs,
        token.total_coeff,
        max_num_coeff,
    )?;

    *reader = probe;
    Ok(ResidualBlock {
        coefficients,
        total_coeff: token.total_coeff,
        max_num_coeff,
    })
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

fn decode_vlc(
    reader: &mut BitReader<'_>,
    table: &[VlcEntry],
    lookup: &[u16],
) -> Result<CoeffToken> {
    if let Some(prefix) = reader.peek_bits(COEFF_TOKEN_LOOKUP_BITS) {
        let packed = lookup[prefix as usize];
        if packed != 0 {
            let length = packed & 0x1f;
            let skipped = reader.skip_bits(usize::from(length));
            debug_assert!(skipped);
            return Ok(CoeffToken {
                total_coeff: ((packed >> 5) & 0x1f) as u8,
                trailing_ones: ((packed >> 10) & 0x03) as u8,
            });
        }
    }

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

fn decode_levels(reader: &mut BitReader<'_>, token: CoeffToken) -> Result<[i32; 16]> {
    let mut levels = [0i32; 16];
    let mut index = 0usize;

    while index < usize::from(token.trailing_ones) {
        levels[index] = if reader.read_bit().ok_or(H264Error::UnexpectedEof)? == 0 {
            1
        } else {
            -1
        };
        index += 1;
    }

    let mut suffix_length = u32::from(token.total_coeff > 10 && token.trailing_ones < 3);
    while index < usize::from(token.total_coeff) {
        let level_prefix = decode_level_prefix(reader)?;
        let suffix_size = if level_prefix == 14 && suffix_length == 0 {
            4
        } else if level_prefix >= 15 {
            level_prefix - 3
        } else {
            suffix_length
        };
        let level_suffix = reader
            .read_bits(suffix_size)
            .ok_or(H264Error::UnexpectedEof)? as i64;

        let mut level_code = (i64::from(level_prefix.min(15)) << suffix_length) + level_suffix;
        if level_prefix >= 15 && suffix_length == 0 {
            level_code += 15;
        }
        if level_prefix >= 16 {
            level_code = level_code
                .checked_add((1i64 << (level_prefix - 3)) - 4096)
                .ok_or(H264Error::IntegerOverflow)?;
        }
        if index == usize::from(token.trailing_ones) && token.trailing_ones < 3 {
            level_code += 2;
        }

        let level = if level_code & 1 == 0 {
            (level_code + 2) >> 1
        } else {
            (-level_code - 1) >> 1
        };
        levels[index] = i32::try_from(level).map_err(|_| H264Error::IntegerOverflow)?;

        if suffix_length == 0 {
            suffix_length = 1;
        }
        if i64::from(levels[index].unsigned_abs()) > (3i64 << (suffix_length - 1))
            && suffix_length < 6
        {
            suffix_length += 1;
        }
        index += 1;
    }

    Ok(levels)
}

fn decode_level_prefix(reader: &mut BitReader<'_>) -> Result<u32> {
    let mut leading_zeros = 0u32;
    loop {
        if reader.read_bit().ok_or(H264Error::UnexpectedEof)? != 0 {
            return Ok(leading_zeros);
        }
        leading_zeros += 1;
        if leading_zeros > 31 {
            return Err(H264Error::IntegerOverflow);
        }
    }
}

fn decode_runs(reader: &mut BitReader<'_>, total_coeff: u8, max_num_coeff: u8) -> Result<[u8; 16]> {
    let mut runs = [0u8; 16];
    let mut zeros_left = if total_coeff == max_num_coeff {
        0
    } else {
        decode_total_zeros(reader, total_coeff, max_num_coeff)?
    };

    for run in runs
        .iter_mut()
        .take(usize::from(total_coeff.saturating_sub(1)))
    {
        if zeros_left != 0 {
            *run = decode_run_before(reader, zeros_left)?;
            zeros_left = zeros_left
                .checked_sub(*run)
                .ok_or(H264Error::InvalidSyntax("run_before exceeds zerosLeft"))?;
        }
    }
    runs[usize::from(total_coeff - 1)] = zeros_left;
    Ok(runs)
}

fn decode_total_zeros(
    reader: &mut BitReader<'_>,
    total_coeff: u8,
    max_num_coeff: u8,
) -> Result<u8> {
    let table = match max_num_coeff {
        4 => TOTAL_ZEROS_CHROMA_DC_420
            .get(usize::from(total_coeff - 1))
            .copied(),
        15 | 16 => TOTAL_ZEROS_4X4.get(usize::from(total_coeff - 1)).copied(),
        _ => None,
    }
    .ok_or(H264Error::InvalidSyntax(
        "invalid TotalCoeff for total_zeros",
    ))?;

    let total_zeros = decode_code_index(reader, table, "invalid total_zeros VLC")?;
    if total_zeros > max_num_coeff - total_coeff {
        return Err(H264Error::InvalidSyntax(
            "total_zeros exceeds the transform block bounds",
        ));
    }
    Ok(total_zeros)
}

fn decode_run_before(reader: &mut BitReader<'_>, zeros_left: u8) -> Result<u8> {
    if zeros_left <= 6 {
        let table = RUN_BEFORE_SMALL[usize::from(zeros_left - 1)];
        return decode_code_index(reader, table, "invalid run_before VLC");
    }

    let prefix = reader
        .read_bits_const::<3>()
        .ok_or(H264Error::UnexpectedEof)? as u8;
    let run = if prefix != 0 {
        7 - prefix
    } else {
        let mut run = 7u8;
        while reader.read_bit().ok_or(H264Error::UnexpectedEof)? == 0 {
            run += 1;
            if run > 14 {
                return Err(H264Error::InvalidSyntax("invalid run_before VLC"));
            }
        }
        run
    };

    if run > zeros_left {
        return Err(H264Error::InvalidSyntax("run_before exceeds zerosLeft"));
    }
    Ok(run)
}

fn decode_code_index(
    reader: &mut BitReader<'_>,
    table: &[VlcCode],
    invalid_syntax: &'static str,
) -> Result<u8> {
    let max_length = table.iter().map(|code| code.length).max().unwrap_or(0);
    if let Some(window) = reader.peek_bits(u32::from(max_length)) {
        for (value, code) in table.iter().enumerate() {
            let prefix = window >> (max_length - code.length);
            if prefix == u32::from(code.bits) {
                let skipped = reader.skip_bits(usize::from(code.length));
                debug_assert!(skipped);
                return u8::try_from(value).map_err(|_| H264Error::IntegerOverflow);
            }
        }
        return Err(H264Error::InvalidSyntax(invalid_syntax));
    }

    let mut bits = 0u16;
    for length in 1..=max_length {
        bits = (bits << 1) | u16::from(reader.read_bit().ok_or(H264Error::UnexpectedEof)?);
        if let Some(value) = table
            .iter()
            .position(|code| code.length == length && code.bits == bits)
        {
            return u8::try_from(value).map_err(|_| H264Error::IntegerOverflow);
        }
    }
    Err(H264Error::InvalidSyntax(invalid_syntax))
}

fn combine_levels_and_runs(
    coefficients: &mut [i32; 16],
    levels: &[i32; 16],
    runs: &[u8; 16],
    total_coeff: u8,
    max_num_coeff: u8,
) -> Result<()> {
    let mut coefficient_index = 0usize;
    for index in (0..usize::from(total_coeff)).rev() {
        coefficient_index = coefficient_index
            .checked_add(usize::from(runs[index]))
            .ok_or(H264Error::IntegerOverflow)?;
        if coefficient_index >= usize::from(max_num_coeff) {
            return Err(H264Error::InvalidSyntax(
                "CAVLC run places a coefficient outside the transform block",
            ));
        }
        coefficients[coefficient_index] = levels[index];
        coefficient_index += 1;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use bit_readers::BitReader;

    use super::{
        CoeffToken, CoeffTokenContext, ResidualBlock, decode_code_index, decode_coeff_token,
        decode_levels, decode_residual_block, decode_run_before, decode_total_zeros,
    };
    use crate::H264Error;
    use crate::cavlc::tables::{
        COEFF_TOKEN_0_TO_1, COEFF_TOKEN_2_TO_3, COEFF_TOKEN_4_TO_7, COEFF_TOKEN_CHROMA_DC_420,
        RUN_BEFORE_SMALL, TOTAL_ZEROS_4X4, TOTAL_ZEROS_CHROMA_DC_420,
    };

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

    #[test]
    fn every_variable_length_table_entry_round_trips() {
        for (context, table) in [
            (CoeffTokenContext::NeighborTotal(0), COEFF_TOKEN_0_TO_1),
            (CoeffTokenContext::NeighborTotal(2), COEFF_TOKEN_2_TO_3),
            (CoeffTokenContext::NeighborTotal(4), COEFF_TOKEN_4_TO_7),
            (CoeffTokenContext::ChromaDc420, COEFF_TOKEN_CHROMA_DC_420),
        ] {
            for entry in table {
                let bytes = code_bytes(entry.bits, entry.length);
                let mut reader = BitReader::new(&bytes);
                assert_eq!(
                    decode_coeff_token(&mut reader, context, 16),
                    Ok(CoeffToken {
                        total_coeff: entry.total_coeff,
                        trailing_ones: entry.trailing_ones,
                    })
                );
                assert_eq!(reader.bit_position(), usize::from(entry.length));
            }
        }

        for (table_index, table) in TOTAL_ZEROS_4X4.iter().enumerate() {
            for (value, code) in table.iter().enumerate() {
                let bytes = code_bytes(code.bits, code.length);
                let mut reader = BitReader::new(&bytes);
                assert_eq!(
                    decode_total_zeros(&mut reader, table_index as u8 + 1, 16),
                    Ok(value as u8)
                );
            }
        }
        for (table_index, table) in TOTAL_ZEROS_CHROMA_DC_420.iter().enumerate() {
            for (value, code) in table.iter().enumerate() {
                let bytes = code_bytes(code.bits, code.length);
                let mut reader = BitReader::new(&bytes);
                assert_eq!(
                    decode_total_zeros(&mut reader, table_index as u8 + 1, 4),
                    Ok(value as u8)
                );
            }
        }
        for (table_index, table) in RUN_BEFORE_SMALL.iter().enumerate() {
            for (value, code) in table.iter().enumerate() {
                let bytes = code_bytes(code.bits, code.length);
                let mut reader = BitReader::new(&bytes);
                assert_eq!(
                    decode_run_before(&mut reader, table_index as u8 + 1),
                    Ok(value as u8)
                );
            }
        }

        // Exercise all unary extensions in the zerosLeft > 6 column.
        for run in 0..=14u8 {
            let bits = if run <= 6 {
                format!("{:03b}", 7 - run)
            } else {
                format!("{}1", "0".repeat(usize::from(run - 4)))
            };
            let bytes = bit_string(&bits);
            let mut reader = BitReader::new(&bytes);
            assert_eq!(decode_run_before(&mut reader, 14), Ok(run));
        }

        // Keep the generic index decoder covered independently of its callers.
        let bytes = bit_string("01");
        let mut reader = BitReader::new(&bytes);
        assert_eq!(
            decode_code_index(&mut reader, RUN_BEFORE_SMALL[1], "invalid"),
            Ok(1)
        );
    }

    #[test]
    fn decodes_complete_residual_blocks_in_scan_order() {
        // TotalCoeff=1, TrailingOnes=0, level=+2, total_zeros=0.
        let single = bit_string("00010111");
        let mut reader = BitReader::new(&single);
        let block =
            decode_residual_block(&mut reader, CoeffTokenContext::NeighborTotal(0), 16).unwrap();
        assert_eq!(
            block,
            ResidualBlock {
                coefficients: [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                total_coeff: 1,
                max_num_coeff: 16,
            }
        );
        assert_eq!(reader.bit_position(), 8);

        // Three trailing ones, two total zeros, and runs [1, 0, 1].
        let sparse = bit_string("00011010110011");
        let mut reader = BitReader::new(&sparse);
        let block =
            decode_residual_block(&mut reader, CoeffTokenContext::NeighborTotal(0), 16).unwrap();
        assert_eq!(&block.coefficients[..5], &[0, 1, -1, 0, 1]);
        assert_eq!(block.total_coeff, 3);
        assert_eq!(reader.bit_position(), 14);

        // 4:2:0 chroma DC: two trailing ones, one zero, and runs [0, 1].
        let chroma_dc = bit_string("00101011");
        let mut reader = BitReader::new(&chroma_dc);
        let block = decode_residual_block(&mut reader, CoeffTokenContext::ChromaDc420, 4).unwrap();
        assert_eq!(&block.coefficients[..4], &[0, -1, 1, 0]);
        assert_eq!(block.total_coeff, 2);
        assert_eq!(reader.bit_position(), 8);

        let empty = bit_string("1");
        let mut reader = BitReader::new(&empty);
        assert_eq!(
            decode_residual_block(&mut reader, CoeffTokenContext::NeighborTotal(0), 16),
            Ok(ResidualBlock {
                coefficients: [0; 16],
                total_coeff: 0,
                max_num_coeff: 16,
            })
        );
        assert_eq!(reader.bit_position(), 1);
    }

    #[test]
    fn decodes_level_suffix_and_escape_branches() {
        let cases = [
            ("1", 2), // level_prefix=0 plus the first-level adjustment.
            ("01", -2),
            ("001", 3),
            ("0000000000000011111", -16), // prefix 14, four-bit suffix.
            ("0000000000000001000000000000", 17), // prefix 15, 12-bit suffix.
        ];
        for (bits, expected) in cases {
            let bytes = bit_string(bits);
            let mut reader = BitReader::new(&bytes);
            let levels = decode_levels(
                &mut reader,
                CoeffToken {
                    total_coeff: 1,
                    trailing_ones: 0,
                },
            )
            .unwrap();
            assert_eq!(levels[0], expected, "{bits}");
            assert_eq!(reader.bit_position(), bits.len(), "{bits}");
        }

        // After the first non-trailing value, suffixLength has advanced to 1.
        let bytes = bit_string("111");
        let mut reader = BitReader::new(&bytes);
        let levels = decode_levels(
            &mut reader,
            CoeffToken {
                total_coeff: 2,
                trailing_ones: 0,
            },
        )
        .unwrap();
        assert_eq!(&levels[..2], &[2, -1]);
        assert_eq!(reader.bit_position(), 3);
    }

    #[test]
    fn residual_block_failures_are_atomic() {
        // TotalCoeff=1 followed by an unterminated level_prefix.
        let data = [0b00010100];
        let mut reader = BitReader::new(&data);
        assert!(matches!(
            decode_residual_block(&mut reader, CoeffTokenContext::NeighborTotal(0), 16),
            Err(H264Error::UnexpectedEof)
        ));
        assert_eq!(reader.bit_position(), 0);

        assert!(matches!(
            decode_residual_block(&mut reader, CoeffTokenContext::NeighborTotal(0), 8),
            Err(H264Error::UnsupportedFeature(_))
        ));
        assert_eq!(reader.bit_position(), 0);

        // maxNumCoeff=15 permits at most one zero when TotalCoeff=14.
        let out_of_bounds_total_zeros = bit_string("1");
        let mut reader = BitReader::new(&out_of_bounds_total_zeros);
        assert!(matches!(
            decode_total_zeros(&mut reader, 14, 15),
            Err(H264Error::InvalidSyntax(_))
        ));
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

    fn code_bytes(bits: u16, length: u8) -> Vec<u8> {
        let mut bits = format!("{bits:0width$b}", width = usize::from(length));
        bits.push_str(&"0".repeat(16usize.saturating_sub(bits.len())));
        bit_string(&bits)
    }
}
