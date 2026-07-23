//! Inverse scan, inverse quantization, and inverse integer transforms.

use crate::{H264Error, Result, ScalingList, ScalingMatrices};

pub type Block4x4 = [[i32; 4]; 4];

pub const FLAT_SCALING_LIST_4X4: [u8; 16] = [16; 16];
pub const DEFAULT_SCALING_LIST_4X4_INTRA: [u8; 16] = [
    6, 13, 13, 20, 20, 20, 28, 28, 28, 28, 32, 32, 32, 37, 37, 42,
];
pub const DEFAULT_SCALING_LIST_4X4_INTER: [u8; 16] = [
    10, 14, 14, 20, 20, 20, 24, 24, 24, 24, 27, 27, 27, 30, 30, 34,
];

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredictionClass {
    Intra,
    Inter,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorComponent {
    Luma,
    Cb,
    Cr,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedScalingLists4x4 {
    lists: [[u8; 16]; 6],
}

impl ResolvedScalingLists4x4 {
    #[inline]
    pub const fn get(&self, prediction: PredictionClass, component: ColorComponent) -> &[u8; 16] {
        let prediction_offset = match prediction {
            PredictionClass::Intra => 0,
            PredictionClass::Inter => 3,
        };
        let component_offset = match component {
            ColorComponent::Luma => 0,
            ColorComponent::Cb => 1,
            ColorComponent::Cr => 2,
        };
        &self.lists[prediction_offset + component_offset]
    }
}

/// Resolves SPS/PPS 4x4 scaling-list fallbacks into six concrete lists.
pub fn resolve_scaling_lists_4x4(
    sequence: Option<&ScalingMatrices>,
    picture: Option<&ScalingMatrices>,
) -> Result<ResolvedScalingLists4x4> {
    let sequence_lists = resolve_sequence_scaling_lists_4x4(sequence)?;
    let Some(picture) = picture else {
        return Ok(ResolvedScalingLists4x4 {
            lists: sequence_lists,
        });
    };
    if picture.lists.len() < 6 {
        return Err(H264Error::InvalidSyntax(
            "picture scaling matrices contain fewer than six 4x4 lists",
        ));
    }

    let mut lists = [[0u8; 16]; 6];
    for index in 0..6 {
        lists[index] = match picture.lists[index].as_ref() {
            Some(list) => resolve_present_scaling_list_4x4(list, index)?,
            None if index == 0 || index == 3 => {
                if sequence.is_some() {
                    sequence_lists[index]
                } else {
                    default_scaling_list_4x4(index)
                }
            }
            None => lists[index - 1],
        };
    }
    Ok(ResolvedScalingLists4x4 { lists })
}

fn resolve_sequence_scaling_lists_4x4(sequence: Option<&ScalingMatrices>) -> Result<[[u8; 16]; 6]> {
    let Some(sequence) = sequence else {
        return Ok([FLAT_SCALING_LIST_4X4; 6]);
    };
    if sequence.lists.len() < 6 {
        return Err(H264Error::InvalidSyntax(
            "sequence scaling matrices contain fewer than six 4x4 lists",
        ));
    }

    let mut lists = [[0u8; 16]; 6];
    for index in 0..6 {
        lists[index] = match sequence.lists[index].as_ref() {
            Some(list) => resolve_present_scaling_list_4x4(list, index)?,
            None if index == 0 || index == 3 => default_scaling_list_4x4(index),
            None => lists[index - 1],
        };
    }
    Ok(lists)
}

fn resolve_present_scaling_list_4x4(list: &ScalingList, index: usize) -> Result<[u8; 16]> {
    if list.use_default {
        return Ok(default_scaling_list_4x4(index));
    }
    list.values
        .as_slice()
        .try_into()
        .map_err(|_| H264Error::InvalidSyntax("4x4 scaling list does not contain 16 entries"))
}

const fn default_scaling_list_4x4(index: usize) -> [u8; 16] {
    if index < 3 {
        DEFAULT_SCALING_LIST_4X4_INTRA
    } else {
        DEFAULT_SCALING_LIST_4X4_INTER
    }
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

/// Applies the Intra16x16 luma DC inverse Hadamard transform and scaling.
pub fn inverse_transform_luma_dc_4x4(
    values: &[i32; 16],
    scan_mode: ScanMode,
    qp: u8,
    scaling_list: &[u8; 16],
) -> Result<Block4x4> {
    validate_qp_and_scaling_list(qp, scaling_list)?;
    let coefficients = inverse_scan_4x4(values, scan_mode);
    let mut vertical = [[0i64; 4]; 4];
    for column in 0..4 {
        let transformed = inverse_hadamard_4([
            i64::from(coefficients[0][column]),
            i64::from(coefficients[1][column]),
            i64::from(coefficients[2][column]),
            i64::from(coefficients[3][column]),
        ])?;
        for row in 0..4 {
            vertical[row][column] = transformed[row];
        }
    }

    let qp_div_6 = u32::from(qp / 6);
    let level_scale =
        i64::from(scaling_list[0]) * i64::from(NORM_ADJUST_4X4[usize::from(qp % 6)][0]);
    let mut dc = [[0; 4]; 4];
    for row in 0..4 {
        let transformed = inverse_hadamard_4(vertical[row])?;
        for column in 0..4 {
            let value = if qp >= 36 {
                transformed[column]
                    .checked_mul(level_scale)
                    .and_then(|value| value.checked_shl(qp_div_6 - 6))
                    .ok_or(H264Error::IntegerOverflow)?
            } else {
                let rounding = 1i64 << (5 - qp_div_6);
                (transformed[column]
                    .checked_mul(level_scale)
                    .and_then(|value| value.checked_add(rounding))
                    .ok_or(H264Error::IntegerOverflow)?)
                    >> (6 - qp_div_6)
            };
            dc[row][column] = i32::try_from(value).map_err(|_| H264Error::IntegerOverflow)?;
        }
    }
    Ok(dc)
}

/// Applies the 4:2:0 chroma DC 2x2 inverse Hadamard transform and scaling.
pub fn inverse_transform_chroma_dc_420(
    values: &[i32; 4],
    qp: u8,
    scaling_list: &[u8; 16],
) -> Result<[[i32; 2]; 2]> {
    validate_qp_and_scaling_list(qp, scaling_list)?;
    let c00 = i64::from(values[0]);
    let c01 = i64::from(values[1]);
    let c10 = i64::from(values[2]);
    let c11 = i64::from(values[3]);
    let transformed = [
        [
            c00.checked_add(c01)
                .and_then(|value| value.checked_add(c10))
                .and_then(|value| value.checked_add(c11))
                .ok_or(H264Error::IntegerOverflow)?,
            c00.checked_sub(c01)
                .and_then(|value| value.checked_add(c10))
                .and_then(|value| value.checked_sub(c11))
                .ok_or(H264Error::IntegerOverflow)?,
        ],
        [
            c00.checked_add(c01)
                .and_then(|value| value.checked_sub(c10))
                .and_then(|value| value.checked_sub(c11))
                .ok_or(H264Error::IntegerOverflow)?,
            c00.checked_sub(c01)
                .and_then(|value| value.checked_sub(c10))
                .and_then(|value| value.checked_add(c11))
                .ok_or(H264Error::IntegerOverflow)?,
        ],
    ];

    let qp_div_6 = u32::from(qp / 6);
    let level_scale =
        i64::from(scaling_list[0]) * i64::from(NORM_ADJUST_4X4[usize::from(qp % 6)][0]);
    let mut dc = [[0; 2]; 2];
    for row in 0..2 {
        for column in 0..2 {
            let value = transformed[row][column]
                .checked_mul(level_scale)
                .and_then(|value| value.checked_shl(qp_div_6))
                .ok_or(H264Error::IntegerOverflow)?
                >> 5;
            dc[row][column] = i32::try_from(value).map_err(|_| H264Error::IntegerOverflow)?;
        }
    }
    Ok(dc)
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

fn inverse_hadamard_4(values: [i64; 4]) -> Result<[i64; 4]> {
    let sum_01 = values[0]
        .checked_add(values[1])
        .ok_or(H264Error::IntegerOverflow)?;
    let difference_01 = values[0]
        .checked_sub(values[1])
        .ok_or(H264Error::IntegerOverflow)?;
    let sum_23 = values[2]
        .checked_add(values[3])
        .ok_or(H264Error::IntegerOverflow)?;
    let difference_23 = values[2]
        .checked_sub(values[3])
        .ok_or(H264Error::IntegerOverflow)?;
    Ok([
        sum_01
            .checked_add(sum_23)
            .ok_or(H264Error::IntegerOverflow)?,
        sum_01
            .checked_sub(sum_23)
            .ok_or(H264Error::IntegerOverflow)?,
        difference_01
            .checked_sub(difference_23)
            .ok_or(H264Error::IntegerOverflow)?,
        difference_01
            .checked_add(difference_23)
            .ok_or(H264Error::IntegerOverflow)?,
    ])
}

fn validate_qp_and_scaling_list(qp: u8, scaling_list: &[u8; 16]) -> Result<()> {
    if qp > 51 {
        return Err(H264Error::InvalidSyntax("8-bit transform QP exceeds 51"));
    }
    if scaling_list.contains(&0) {
        return Err(H264Error::InvalidSyntax(
            "4x4 scaling-list entries must be non-zero",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        ColorComponent, DEFAULT_SCALING_LIST_4X4_INTRA, FLAT_SCALING_LIST_4X4, PredictionClass,
        ScanMode, inverse_scale_4x4, inverse_scan_4x4, inverse_scan_scaling_list_4x4,
        inverse_transform_4x4, inverse_transform_chroma_dc_420, inverse_transform_luma_dc_4x4,
        reconstruct_residual_4x4, resolve_scaling_lists_4x4,
    };
    use crate::{H264Error, ScalingList, ScalingMatrices};

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
    fn transforms_and_scales_luma_dc_coefficients() {
        let mut impulse = [0; 16];
        impulse[0] = 1;
        assert_eq!(
            inverse_transform_luma_dc_4x4(&impulse, ScanMode::Frame, 0, &FLAT_SCALING_LIST_4X4,),
            Ok([[3; 4]; 4])
        );

        let all_one = [1; 16];
        assert_eq!(
            inverse_transform_luma_dc_4x4(&all_one, ScanMode::Frame, 0, &FLAT_SCALING_LIST_4X4,),
            Ok([[40, 0, 0, 0], [0; 4], [0; 4], [0; 4]])
        );
    }

    #[test]
    fn transforms_and_scales_chroma_dc_coefficients() {
        assert_eq!(
            inverse_transform_chroma_dc_420(&[1, 0, 0, 0], 0, &FLAT_SCALING_LIST_4X4),
            Ok([[5; 2]; 2])
        );
        assert_eq!(
            inverse_transform_chroma_dc_420(&[1; 4], 0, &FLAT_SCALING_LIST_4X4),
            Ok([[20, 0], [0, 0]])
        );
    }

    #[test]
    fn resolves_sequence_and_picture_scaling_list_fallbacks() {
        let flat = resolve_scaling_lists_4x4(None, None).unwrap();
        assert_eq!(
            flat.get(PredictionClass::Intra, ColorComponent::Luma),
            &FLAT_SCALING_LIST_4X4
        );
        assert_eq!(
            flat.get(PredictionClass::Inter, ColorComponent::Cr),
            &FLAT_SCALING_LIST_4X4
        );

        let sequence = ScalingMatrices {
            lists: vec![
                Some(scaling_list(8, false)),
                None,
                None,
                Some(scaling_list(9, false)),
                None,
                None,
            ],
        };
        let resolved = resolve_scaling_lists_4x4(Some(&sequence), None).unwrap();
        assert_eq!(
            resolved.get(PredictionClass::Intra, ColorComponent::Cr),
            &[8; 16]
        );
        assert_eq!(
            resolved.get(PredictionClass::Inter, ColorComponent::Cr),
            &[9; 16]
        );

        let picture = ScalingMatrices {
            lists: vec![
                Some(scaling_list(7, false)),
                None,
                Some(scaling_list(99, true)),
                None,
                None,
                Some(scaling_list(11, false)),
            ],
        };
        let resolved = resolve_scaling_lists_4x4(Some(&sequence), Some(&picture)).unwrap();
        assert_eq!(
            resolved.get(PredictionClass::Intra, ColorComponent::Cb),
            &[7; 16]
        );
        assert_eq!(
            resolved.get(PredictionClass::Intra, ColorComponent::Cr),
            &DEFAULT_SCALING_LIST_4X4_INTRA
        );
        assert_eq!(
            resolved.get(PredictionClass::Inter, ColorComponent::Luma),
            &[9; 16]
        );
        assert_eq!(
            resolved.get(PredictionClass::Inter, ColorComponent::Cr),
            &[11; 16]
        );
    }

    #[test]
    fn rejects_malformed_scaling_matrix_shapes() {
        let too_few = ScalingMatrices {
            lists: vec![None; 5],
        };
        assert!(matches!(
            resolve_scaling_lists_4x4(Some(&too_few), None),
            Err(H264Error::InvalidSyntax(_))
        ));

        let malformed = ScalingMatrices {
            lists: vec![
                Some(ScalingList {
                    values: vec![8; 15],
                    use_default: false,
                }),
                None,
                None,
                None,
                None,
                None,
            ],
        };
        assert!(matches!(
            resolve_scaling_lists_4x4(Some(&malformed), None),
            Err(H264Error::InvalidSyntax(_))
        ));
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

    fn scaling_list(value: u8, use_default: bool) -> ScalingList {
        ScalingList {
            values: vec![value; 16],
            use_default,
        }
    }
}
