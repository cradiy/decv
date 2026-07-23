//! Inverse scan, inverse quantization, and inverse integer transforms.

use crate::{H264Error, Result};

pub type Block4x4 = [[i32; 4]; 4];

pub const FLAT_SCALING_LIST_4X4: [u8; 16] = [16; 16];

const FRAME_SCAN_4X4: [(usize, usize); 16] = [
    (0, 0),
    (0, 1),
    (1, 0),
    (2, 0),
    (1, 1),
    (0, 2),
    (0, 3),
    (1, 2),
    (2, 1),
    (3, 0),
    (3, 1),
    (2, 2),
    (1, 3),
    (2, 3),
    (3, 2),
    (3, 3),
];

const FIELD_SCAN_4X4: [(usize, usize); 16] = [
    (0, 0),
    (1, 0),
    (0, 1),
    (2, 0),
    (3, 0),
    (1, 1),
    (2, 1),
    (3, 1),
    (0, 2),
    (1, 2),
    (2, 2),
    (3, 2),
    (0, 3),
    (1, 3),
    (2, 3),
    (3, 3),
];

// Rows are qP % 6; columns select even/even, odd/odd, or mixed positions.
const NORM_ADJUST_4X4: [[i32; 3]; 6] = [
    [10, 16, 13],
    [11, 18, 14],
    [13, 20, 16],
    [14, 23, 18],
    [16, 25, 20],
    [18, 29, 23],
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScanMode {
    Frame,
    Field,
}

/// Maps a 16-element coefficient list into its 4x4 transform positions.
pub fn inverse_scan_4x4(values: &[i32; 16], mode: ScanMode) -> Block4x4 {
    let coordinates = match mode {
        ScanMode::Frame => &FRAME_SCAN_4X4,
        ScanMode::Field => &FIELD_SCAN_4X4,
    };
    let mut block = [[0; 4]; 4];
    for (value, &(row, column)) in values.iter().zip(coordinates) {
        block[row][column] = *value;
    }
    block
}

/// Converts a scaling list from zig-zag order into matrix order.
pub fn inverse_scan_scaling_list_4x4(values: &[u8; 16]) -> [[u8; 4]; 4] {
    let mut matrix = [[0; 4]; 4];
    for (value, &(row, column)) in values.iter().zip(&FRAME_SCAN_4X4) {
        matrix[row][column] = *value;
    }
    matrix
}

/// Applies H.264 4x4 inverse quantization.
///
/// `preserve_dc` implements the special d00 = c00 rule used after separately
/// transforming Intra16x16 luma DC or chroma DC coefficients.
pub fn inverse_scale_4x4(
    coefficients: &Block4x4,
    qp: u8,
    scaling_list: &[u8; 16],
    preserve_dc: bool,
) -> Result<Block4x4> {
    if qp > 51 {
        return Err(H264Error::InvalidSyntax("8-bit transform QP exceeds 51"));
    }
    if scaling_list.contains(&0) {
        return Err(H264Error::InvalidSyntax(
            "4x4 scaling-list entries must be non-zero",
        ));
    }

    let weights = inverse_scan_scaling_list_4x4(scaling_list);
    let qp_div_6 = u32::from(qp / 6);
    let norm_row = NORM_ADJUST_4X4[usize::from(qp % 6)];
    let mut scaled = [[0; 4]; 4];
    for row in 0..4 {
        for column in 0..4 {
            if preserve_dc && row == 0 && column == 0 {
                scaled[row][column] = coefficients[row][column];
                continue;
            }

            let norm = if row % 2 == 0 && column % 2 == 0 {
                norm_row[0]
            } else if row % 2 == 1 && column % 2 == 1 {
                norm_row[1]
            } else {
                norm_row[2]
            };
            let level_scale = i64::from(weights[row][column]) * i64::from(norm);
            let coefficient = i64::from(coefficients[row][column]);
            let value = if qp >= 24 {
                coefficient
                    .checked_mul(level_scale)
                    .and_then(|value| value.checked_shl(qp_div_6 - 4))
                    .ok_or(H264Error::IntegerOverflow)?
            } else {
                let rounding = 1i64 << (3 - qp_div_6);
                (coefficient
                    .checked_mul(level_scale)
                    .and_then(|value| value.checked_add(rounding))
                    .ok_or(H264Error::IntegerOverflow)?)
                    >> (4 - qp_div_6)
            };
            scaled[row][column] = i32::try_from(value).map_err(|_| H264Error::IntegerOverflow)?;
        }
    }
    Ok(scaled)
}

/// Applies the normative separable H.264 4x4 inverse integer transform.
pub fn inverse_transform_4x4(scaled: &Block4x4) -> Result<Block4x4> {
    let mut horizontal = [[0i64; 4]; 4];
    for row in 0..4 {
        horizontal[row] = inverse_transform_1d([
            i64::from(scaled[row][0]),
            i64::from(scaled[row][1]),
            i64::from(scaled[row][2]),
            i64::from(scaled[row][3]),
        ])?;
    }

    let mut residual = [[0; 4]; 4];
    for column in 0..4 {
        let transformed = inverse_transform_1d([
            horizontal[0][column],
            horizontal[1][column],
            horizontal[2][column],
            horizontal[3][column],
        ])?;
        for row in 0..4 {
            let value = (transformed[row] + 32) >> 6;
            residual[row][column] = i32::try_from(value).map_err(|_| H264Error::IntegerOverflow)?;
        }
    }
    Ok(residual)
}

/// Runs inverse scan, inverse scaling, and the inverse integer transform.
pub fn reconstruct_residual_4x4(
    values: &[i32; 16],
    scan_mode: ScanMode,
    qp: u8,
    scaling_list: &[u8; 16],
    preserve_dc: bool,
) -> Result<Block4x4> {
    let coefficients = inverse_scan_4x4(values, scan_mode);
    let scaled = inverse_scale_4x4(&coefficients, qp, scaling_list, preserve_dc)?;
    inverse_transform_4x4(&scaled)
}

fn inverse_transform_1d(values: [i64; 4]) -> Result<[i64; 4]> {
    let e0 = values[0]
        .checked_add(values[2])
        .ok_or(H264Error::IntegerOverflow)?;
    let e1 = values[0]
        .checked_sub(values[2])
        .ok_or(H264Error::IntegerOverflow)?;
    let e2 = (values[1] >> 1)
        .checked_sub(values[3])
        .ok_or(H264Error::IntegerOverflow)?;
    let e3 = values[1]
        .checked_add(values[3] >> 1)
        .ok_or(H264Error::IntegerOverflow)?;
    Ok([
        e0.checked_add(e3).ok_or(H264Error::IntegerOverflow)?,
        e1.checked_add(e2).ok_or(H264Error::IntegerOverflow)?,
        e1.checked_sub(e2).ok_or(H264Error::IntegerOverflow)?,
        e0.checked_sub(e3).ok_or(H264Error::IntegerOverflow)?,
    ])
}

#[cfg(test)]
mod tests {
    use super::{
        FLAT_SCALING_LIST_4X4, ScanMode, inverse_scale_4x4, inverse_scan_4x4,
        inverse_scan_scaling_list_4x4, inverse_transform_4x4, reconstruct_residual_4x4,
    };
    use crate::H264Error;

    #[test]
    fn inverse_scans_frame_and_field_coefficients() {
        let values = std::array::from_fn(|index| index as i32);
        assert_eq!(
            inverse_scan_4x4(&values, ScanMode::Frame),
            [[0, 1, 5, 6], [2, 4, 7, 12], [3, 8, 11, 13], [9, 10, 14, 15]]
        );
        assert_eq!(
            inverse_scan_4x4(&values, ScanMode::Field),
            [[0, 2, 8, 12], [1, 5, 9, 13], [3, 6, 10, 14], [4, 7, 11, 15]]
        );
    }

    #[test]
    fn scaling_lists_always_use_zig_zag_scan() {
        let values = std::array::from_fn(|index| index as u8 + 1);
        assert_eq!(
            inverse_scan_scaling_list_4x4(&values),
            [
                [1, 2, 6, 7],
                [3, 5, 8, 13],
                [4, 9, 12, 14],
                [10, 11, 15, 16]
            ]
        );
    }

    #[test]
    fn inverse_scales_flat_lists_and_preserves_special_dc() {
        let mut coefficients = [[0; 4]; 4];
        coefficients[0][0] = 1;
        coefficients[0][1] = 1;
        coefficients[1][1] = 1;
        assert_eq!(
            inverse_scale_4x4(&coefficients, 0, &FLAT_SCALING_LIST_4X4, false).unwrap()[..2],
            [[10, 13, 0, 0], [0, 16, 0, 0]]
        );
        assert_eq!(
            inverse_scale_4x4(&coefficients, 0, &FLAT_SCALING_LIST_4X4, true).unwrap()[0][0],
            1
        );

        // qP=24 switches to the left-shift branch with a zero shift.
        assert_eq!(
            inverse_scale_4x4(&coefficients, 24, &FLAT_SCALING_LIST_4X4, false).unwrap()[0][0],
            160
        );
    }

    #[test]
    fn inverse_transform_handles_dc_and_ac_impulses() {
        let mut dc = [[0; 4]; 4];
        dc[0][0] = 64;
        assert_eq!(inverse_transform_4x4(&dc), Ok([[1; 4]; 4]));

        let mut ac = [[0; 4]; 4];
        ac[0][1] = 64;
        assert_eq!(inverse_transform_4x4(&ac), Ok([[1, 1, 0, -1]; 4]));
    }

    #[test]
    fn reconstructs_a_flat_qp_zero_dc_block() {
        let mut values = [0; 16];
        values[0] = 64;
        assert_eq!(
            reconstruct_residual_4x4(&values, ScanMode::Frame, 0, &FLAT_SCALING_LIST_4X4, false,),
            Ok([[10; 4]; 4])
        );
    }

    #[test]
    fn rejects_invalid_qp_scaling_lists_and_overflow() {
        let coefficients = [[1; 4]; 4];
        assert!(matches!(
            inverse_scale_4x4(&coefficients, 52, &FLAT_SCALING_LIST_4X4, false),
            Err(H264Error::InvalidSyntax(_))
        ));
        let mut invalid = FLAT_SCALING_LIST_4X4;
        invalid[5] = 0;
        assert!(matches!(
            inverse_scale_4x4(&coefficients, 0, &invalid, false),
            Err(H264Error::InvalidSyntax(_))
        ));
        let huge = [[i32::MAX; 4]; 4];
        assert_eq!(
            inverse_scale_4x4(&huge, 51, &[u8::MAX; 16], false),
            Err(H264Error::IntegerOverflow)
        );
    }
}
