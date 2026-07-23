//! Intra- and inter-prediction primitives.

use crate::{H264Error, Result};

pub type Prediction16x16 = [[u8; 16]; 16];
pub type Prediction8x8 = [[u8; 8]; 8];
pub type Prediction4x4 = [[u8; 4]; 4];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Intra4x4References {
    /// The eight samples above the block. The caller substitutes `top[3]` for
    /// unavailable top-right samples as required by H.264 clause 8.3.1.2.
    pub top: Option<[u8; 8]>,
    pub left: Option<[u8; 4]>,
    pub top_left: Option<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Intra16x16References {
    pub top: Option<[u8; 16]>,
    pub left: Option<[u8; 16]>,
    pub top_left: Option<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IntraChroma420References {
    pub top: Option<[u8; 8]>,
    pub left: Option<[u8; 8]>,
    pub top_left: Option<u8>,
}

/// Generates one 8-bit Intra4x4 luma prediction (modes 0 through 8).
pub fn predict_intra_4x4(mode: u8, references: &Intra4x4References) -> Result<Prediction4x4> {
    match mode {
        0 => {
            let top = required_top4(references, "Intra4x4 vertical")?;
            Ok([top; 4])
        }
        1 => {
            let left = required_left4(references, "Intra4x4 horizontal")?;
            Ok(std::array::from_fn(|row| [left[row]; 4]))
        }
        2 => {
            let value = dc_prediction_4(
                references.top.as_ref().map(|samples| &samples[..4]),
                references.left.as_ref().map(|samples| &samples[..]),
            );
            Ok([[value; 4]; 4])
        }
        3 => {
            let top = required_top8(references, "Intra4x4 diagonal-down-left")?;
            Ok(std::array::from_fn(|row| {
                std::array::from_fn(|column| {
                    if row == 3 && column == 3 {
                        weighted3(top[6], top[7], top[7])
                    } else {
                        let index = row + column;
                        weighted3(top[index], top[index + 1], top[index + 2])
                    }
                })
            }))
        }
        4 => {
            let (top, left, top_left) =
                required_corner_references(references, "Intra4x4 diagonal-down-right")?;
            Ok(std::array::from_fn(|row| {
                std::array::from_fn(|column| match column.cmp(&row) {
                    std::cmp::Ordering::Greater => {
                        let delta = column - row;
                        weighted3(
                            top_or_corner(top, top_left, delta as i32 - 2),
                            top_or_corner(top, top_left, delta as i32 - 1),
                            top[delta],
                        )
                    }
                    std::cmp::Ordering::Less => {
                        let delta = row - column;
                        weighted3(
                            left_or_corner(left, top_left, delta as i32 - 2),
                            left_or_corner(left, top_left, delta as i32 - 1),
                            left[delta],
                        )
                    }
                    std::cmp::Ordering::Equal => weighted3(top[0], top_left, left[0]),
                })
            }))
        }
        5 => {
            let (top, left, top_left) =
                required_corner_references(references, "Intra4x4 vertical-right")?;
            Ok(std::array::from_fn(|row| {
                std::array::from_fn(|column| {
                    let x = column as i32;
                    let y = row as i32;
                    let z = 2 * x - y;
                    match z {
                        0 | 2 | 4 | 6 => average2(
                            top_or_corner(top, top_left, x - (y >> 1) - 1),
                            top_or_corner(top, top_left, x - (y >> 1)),
                        ),
                        1 | 3 | 5 => weighted3(
                            top_or_corner(top, top_left, x - (y >> 1) - 2),
                            top_or_corner(top, top_left, x - (y >> 1) - 1),
                            top_or_corner(top, top_left, x - (y >> 1)),
                        ),
                        -1 => weighted3(left[0], top_left, top[0]),
                        -3 | -2 => weighted3(
                            left_or_corner(left, top_left, y - 1),
                            left_or_corner(left, top_left, y - 2),
                            left_or_corner(left, top_left, y - 3),
                        ),
                        _ => unreachable!("4x4 coordinates constrain zVR to -3..6"),
                    }
                })
            }))
        }
        6 => {
            let (top, left, top_left) =
                required_corner_references(references, "Intra4x4 horizontal-down")?;
            Ok(std::array::from_fn(|row| {
                std::array::from_fn(|column| {
                    let x = column as i32;
                    let y = row as i32;
                    let z = 2 * y - x;
                    match z {
                        0 | 2 | 4 | 6 => average2(
                            left_or_corner(left, top_left, y - (x >> 1) - 1),
                            left_or_corner(left, top_left, y - (x >> 1)),
                        ),
                        1 | 3 | 5 => weighted3(
                            left_or_corner(left, top_left, y - (x >> 1) - 2),
                            left_or_corner(left, top_left, y - (x >> 1) - 1),
                            left_or_corner(left, top_left, y - (x >> 1)),
                        ),
                        -1 => weighted3(left[0], top_left, top[0]),
                        -3 | -2 => weighted3(
                            top_or_corner(top, top_left, x - 1),
                            top_or_corner(top, top_left, x - 2),
                            top_or_corner(top, top_left, x - 3),
                        ),
                        _ => unreachable!("4x4 coordinates constrain zHD to -3..6"),
                    }
                })
            }))
        }
        7 => {
            let top = required_top8(references, "Intra4x4 vertical-left")?;
            Ok(std::array::from_fn(|row| {
                std::array::from_fn(|column| {
                    let index = column + (row >> 1);
                    if row & 1 == 0 {
                        average2(top[index], top[index + 1])
                    } else {
                        weighted3(top[index], top[index + 1], top[index + 2])
                    }
                })
            }))
        }
        8 => {
            let left = required_left4(references, "Intra4x4 horizontal-up")?;
            Ok(std::array::from_fn(|row| {
                std::array::from_fn(|column| {
                    let z = column + 2 * row;
                    match z {
                        0 | 2 | 4 => {
                            let index = row + (column >> 1);
                            average2(left[index], left[index + 1])
                        }
                        1 | 3 => {
                            let index = row + (column >> 1);
                            weighted3(left[index], left[index + 1], left[index + 2])
                        }
                        5 => weighted3(left[2], left[3], left[3]),
                        _ => left[3],
                    }
                })
            }))
        }
        _ => Err(H264Error::InvalidSyntax("Intra4x4PredMode exceeds 8")),
    }
}

/// Generates one 8-bit Intra16x16 luma prediction (modes 0 through 3).
pub fn predict_intra_16x16(mode: u8, references: &Intra16x16References) -> Result<Prediction16x16> {
    match mode {
        0 => {
            let top = references.top.ok_or(H264Error::InvalidSyntax(
                "Intra16x16 vertical prediction requires top samples",
            ))?;
            Ok([top; 16])
        }
        1 => {
            let left = references.left.ok_or(H264Error::InvalidSyntax(
                "Intra16x16 horizontal prediction requires left samples",
            ))?;
            Ok(std::array::from_fn(|row| [left[row]; 16]))
        }
        2 => {
            let value = dc_prediction_16(references.top.as_ref(), references.left.as_ref());
            Ok([[value; 16]; 16])
        }
        3 => predict_intra_16x16_plane(references),
        _ => Err(H264Error::InvalidSyntax("Intra16x16PredMode exceeds 3")),
    }
}

/// Generates one 8-bit 4:2:0 chroma prediction (modes 0 through 3).
pub fn predict_intra_chroma_420(
    mode: u8,
    references: &IntraChroma420References,
) -> Result<Prediction8x8> {
    match mode {
        0 => Ok(predict_chroma_dc(references)),
        1 => {
            let left = references.left.ok_or(H264Error::InvalidSyntax(
                "chroma horizontal prediction requires left samples",
            ))?;
            Ok(std::array::from_fn(|row| [left[row]; 8]))
        }
        2 => {
            let top = references.top.ok_or(H264Error::InvalidSyntax(
                "chroma vertical prediction requires top samples",
            ))?;
            Ok([top; 8])
        }
        3 => predict_chroma_plane(references),
        _ => Err(H264Error::InvalidSyntax("intra_chroma_pred_mode exceeds 3")),
    }
}

fn required_top4(references: &Intra4x4References, mode: &'static str) -> Result<[u8; 4]> {
    references
        .top
        .map(|samples| [samples[0], samples[1], samples[2], samples[3]])
        .ok_or(H264Error::InvalidSyntax(match mode {
            "Intra4x4 vertical" => "Intra4x4 vertical prediction requires top samples",
            _ => "Intra4x4 prediction requires top samples",
        }))
}

fn required_top8(references: &Intra4x4References, _mode: &'static str) -> Result<[u8; 8]> {
    references.top.ok_or(H264Error::InvalidSyntax(
        "Intra4x4 prediction requires all eight top samples",
    ))
}

fn required_left4(references: &Intra4x4References, _mode: &'static str) -> Result<[u8; 4]> {
    references.left.ok_or(H264Error::InvalidSyntax(
        "Intra4x4 prediction requires left samples",
    ))
}

fn required_corner_references(
    references: &Intra4x4References,
    _mode: &'static str,
) -> Result<([u8; 8], [u8; 4], u8)> {
    let top = required_top8(references, "corner mode")?;
    let left = required_left4(references, "corner mode")?;
    let top_left = references.top_left.ok_or(H264Error::InvalidSyntax(
        "Intra4x4 prediction requires the top-left sample",
    ))?;
    Ok((top, left, top_left))
}

#[inline]
fn top_or_corner(top: [u8; 8], top_left: u8, index: i32) -> u8 {
    if index == -1 {
        top_left
    } else {
        top[index as usize]
    }
}

#[inline]
fn left_or_corner(left: [u8; 4], top_left: u8, index: i32) -> u8 {
    if index == -1 {
        top_left
    } else {
        left[index as usize]
    }
}

#[inline]
fn average2(first: u8, second: u8) -> u8 {
    ((u16::from(first) + u16::from(second) + 1) >> 1) as u8
}

#[inline]
fn weighted3(first: u8, middle: u8, last: u8) -> u8 {
    ((u16::from(first) + 2 * u16::from(middle) + u16::from(last) + 2) >> 2) as u8
}

fn dc_prediction_16(top: Option<&[u8; 16]>, left: Option<&[u8; 16]>) -> u8 {
    match (top, left) {
        (Some(top), Some(left)) => ((sum(top) + sum(left) + 16) >> 5) as u8,
        (Some(samples), None) | (None, Some(samples)) => ((sum(samples) + 8) >> 4) as u8,
        (None, None) => 128,
    }
}

fn predict_intra_16x16_plane(references: &Intra16x16References) -> Result<Prediction16x16> {
    let top = references.top.ok_or(H264Error::InvalidSyntax(
        "Intra16x16 plane prediction requires top samples",
    ))?;
    let left = references.left.ok_or(H264Error::InvalidSyntax(
        "Intra16x16 plane prediction requires left samples",
    ))?;
    let top_left = references.top_left.ok_or(H264Error::InvalidSyntax(
        "Intra16x16 plane prediction requires the top-left sample",
    ))?;

    let mut horizontal_gradient = 0i32;
    let mut vertical_gradient = 0i32;
    for offset in 0..8usize {
        let lower_top = if offset == 7 {
            i32::from(top_left)
        } else {
            i32::from(top[6 - offset])
        };
        let lower_left = if offset == 7 {
            i32::from(top_left)
        } else {
            i32::from(left[6 - offset])
        };
        horizontal_gradient += (offset as i32 + 1) * (i32::from(top[8 + offset]) - lower_top);
        vertical_gradient += (offset as i32 + 1) * (i32::from(left[8 + offset]) - lower_left);
    }
    let a = 16 * (i32::from(top[15]) + i32::from(left[15]));
    let b = (5 * horizontal_gradient + 32) >> 6;
    let c = (5 * vertical_gradient + 32) >> 6;
    Ok(std::array::from_fn(|row| {
        std::array::from_fn(|column| {
            clip_u8((a + b * (column as i32 - 7) + c * (row as i32 - 7) + 16) >> 5)
        })
    }))
}

fn predict_chroma_dc(references: &IntraChroma420References) -> Prediction8x8 {
    let mut prediction = [[0; 8]; 8];
    for block_y in 0..2 {
        for block_x in 0..2 {
            let top = references
                .top
                .as_ref()
                .map(|samples| &samples[block_x * 4..block_x * 4 + 4]);
            let left = references
                .left
                .as_ref()
                .map(|samples| &samples[block_y * 4..block_y * 4 + 4]);
            let value = if block_x == 0 && block_y == 0 || block_x != 0 && block_y != 0 {
                dc_prediction_4(top, left)
            } else if block_y == 0 {
                top.map_or_else(
                    || dc_prediction_4(None, left),
                    |top| dc_prediction_4(Some(top), None),
                )
            } else {
                left.map_or_else(
                    || dc_prediction_4(top, None),
                    |left| dc_prediction_4(None, Some(left)),
                )
            };
            for row in prediction.iter_mut().skip(block_y * 4).take(4) {
                for sample in row.iter_mut().skip(block_x * 4).take(4) {
                    *sample = value;
                }
            }
        }
    }
    prediction
}

fn dc_prediction_4(top: Option<&[u8]>, left: Option<&[u8]>) -> u8 {
    match (top, left) {
        (Some(top), Some(left)) => ((sum_slice(top) + sum_slice(left) + 4) >> 3) as u8,
        (Some(samples), None) | (None, Some(samples)) => ((sum_slice(samples) + 2) >> 2) as u8,
        (None, None) => 128,
    }
}

fn predict_chroma_plane(references: &IntraChroma420References) -> Result<Prediction8x8> {
    let top = references.top.ok_or(H264Error::InvalidSyntax(
        "chroma plane prediction requires top samples",
    ))?;
    let left = references.left.ok_or(H264Error::InvalidSyntax(
        "chroma plane prediction requires left samples",
    ))?;
    let top_left = references.top_left.ok_or(H264Error::InvalidSyntax(
        "chroma plane prediction requires the top-left sample",
    ))?;

    let mut horizontal_gradient = 0i32;
    let mut vertical_gradient = 0i32;
    for offset in 0..4usize {
        let lower_top = if offset == 3 {
            i32::from(top_left)
        } else {
            i32::from(top[2 - offset])
        };
        let lower_left = if offset == 3 {
            i32::from(top_left)
        } else {
            i32::from(left[2 - offset])
        };
        horizontal_gradient += (offset as i32 + 1) * (i32::from(top[4 + offset]) - lower_top);
        vertical_gradient += (offset as i32 + 1) * (i32::from(left[4 + offset]) - lower_left);
    }
    let a = 16 * (i32::from(top[7]) + i32::from(left[7]));
    let b = (34 * horizontal_gradient + 32) >> 6;
    let c = (34 * vertical_gradient + 32) >> 6;
    Ok(std::array::from_fn(|row| {
        std::array::from_fn(|column| {
            clip_u8((a + b * (column as i32 - 3) + c * (row as i32 - 3) + 16) >> 5)
        })
    }))
}

#[inline]
fn clip_u8(value: i32) -> u8 {
    value.clamp(0, 255) as u8
}

fn sum<const LENGTH: usize>(samples: &[u8; LENGTH]) -> u32 {
    samples.iter().map(|&value| u32::from(value)).sum()
}

fn sum_slice(samples: &[u8]) -> u32 {
    samples.iter().map(|&value| u32::from(value)).sum()
}

#[cfg(test)]
mod tests {
    use super::{
        Intra4x4References, Intra16x16References, IntraChroma420References, predict_intra_4x4,
        predict_intra_16x16, predict_intra_chroma_420,
    };
    use crate::H264Error;

    fn intra4_references() -> Intra4x4References {
        Intra4x4References {
            top: Some([10, 20, 30, 40, 50, 60, 70, 80]),
            left: Some([90, 100, 110, 120]),
            top_left: Some(50),
        }
    }

    #[test]
    fn predicts_intra4_vertical_horizontal_and_dc() {
        let references = intra4_references();
        assert_eq!(
            predict_intra_4x4(0, &references).unwrap(),
            [[10, 20, 30, 40]; 4]
        );
        assert_eq!(
            predict_intra_4x4(1, &references).unwrap(),
            [[90; 4], [100; 4], [110; 4], [120; 4]]
        );
        assert_eq!(predict_intra_4x4(2, &references).unwrap(), [[65; 4]; 4]);
        assert_eq!(
            predict_intra_4x4(
                2,
                &Intra4x4References {
                    top: None,
                    left: None,
                    top_left: None,
                }
            )
            .unwrap(),
            [[128; 4]; 4]
        );
    }

    #[test]
    fn predicts_intra4_down_left_and_vertical_left() {
        let references = intra4_references();
        assert_eq!(
            predict_intra_4x4(3, &references).unwrap(),
            [
                [20, 30, 40, 50],
                [30, 40, 50, 60],
                [40, 50, 60, 70],
                [50, 60, 70, 78],
            ]
        );
        assert_eq!(
            predict_intra_4x4(7, &references).unwrap(),
            [
                [15, 25, 35, 45],
                [20, 30, 40, 50],
                [25, 35, 45, 55],
                [30, 40, 50, 60],
            ]
        );
    }

    #[test]
    fn predicts_intra4_horizontal_up() {
        assert_eq!(
            predict_intra_4x4(8, &intra4_references()).unwrap(),
            [
                [95, 100, 105, 110],
                [105, 110, 115, 118],
                [115, 118, 120, 120],
                [120, 120, 120, 120],
            ]
        );
    }

    #[test]
    fn predicts_intra4_corner_modes_and_checks_references() {
        let references = intra4_references();
        assert_eq!(
            predict_intra_4x4(4, &references).unwrap(),
            [
                [50, 23, 20, 30],
                [83, 50, 23, 20],
                [100, 83, 50, 23],
                [110, 100, 83, 50],
            ]
        );
        assert_eq!(
            predict_intra_4x4(5, &references).unwrap(),
            [
                [30, 15, 25, 35],
                [50, 23, 20, 30],
                [83, 30, 15, 25],
                [100, 50, 23, 20],
            ]
        );
        assert_eq!(
            predict_intra_4x4(6, &references).unwrap(),
            [
                [70, 50, 23, 20],
                [95, 83, 70, 50],
                [105, 100, 95, 83],
                [115, 110, 105, 100],
            ]
        );

        let constant = Intra4x4References {
            top: Some([77; 8]),
            left: Some([77; 4]),
            top_left: Some(77),
        };
        for mode in 4..=6 {
            assert_eq!(predict_intra_4x4(mode, &constant).unwrap(), [[77; 4]; 4]);
        }

        let missing_corner = Intra4x4References {
            top_left: None,
            ..constant
        };
        assert!(predict_intra_4x4(4, &missing_corner).is_err());
        let missing_top = Intra4x4References {
            top: None,
            ..constant
        };
        assert!(predict_intra_4x4(0, &missing_top).is_err());
        assert!(predict_intra_4x4(3, &missing_top).is_err());
        assert!(predict_intra_4x4(9, &constant).is_err());
    }

    #[test]
    fn predicts_intra16_vertical_and_horizontal() {
        let top = std::array::from_fn(|index| index as u8);
        let left = std::array::from_fn(|index| 100 + index as u8);
        let references = Intra16x16References {
            top: Some(top),
            left: Some(left),
            top_left: Some(50),
        };
        let vertical = predict_intra_16x16(0, &references).unwrap();
        assert!(vertical.iter().all(|row| row == &top));

        let horizontal = predict_intra_16x16(1, &references).unwrap();
        for row in 0..16 {
            assert_eq!(horizontal[row], [left[row]; 16]);
        }
    }

    #[test]
    fn predicts_intra16_dc_for_each_availability_shape() {
        let both = Intra16x16References {
            top: Some([10; 16]),
            left: Some([20; 16]),
            top_left: None,
        };
        assert_eq!(predict_intra_16x16(2, &both), Ok([[15; 16]; 16]));

        let top_only = Intra16x16References {
            top: Some([11; 16]),
            left: None,
            top_left: None,
        };
        assert_eq!(predict_intra_16x16(2, &top_only), Ok([[11; 16]; 16]));
        let none = Intra16x16References {
            top: None,
            left: None,
            top_left: None,
        };
        assert_eq!(predict_intra_16x16(2, &none), Ok([[128; 16]; 16]));
    }

    #[test]
    fn predicts_constant_intra16_plane() {
        let references = Intra16x16References {
            top: Some([64; 16]),
            left: Some([64; 16]),
            top_left: Some(64),
        };
        assert_eq!(predict_intra_16x16(3, &references), Ok([[64; 16]; 16]));
    }

    #[test]
    fn predicts_chroma_dc_per_4x4_region() {
        let references = IntraChroma420References {
            top: Some([10, 10, 10, 10, 30, 30, 30, 30]),
            left: Some([20, 20, 20, 20, 40, 40, 40, 40]),
            top_left: None,
        };
        let prediction = predict_intra_chroma_420(0, &references).unwrap();
        assert_eq!(prediction[0], [15, 15, 15, 15, 30, 30, 30, 30]);
        assert_eq!(prediction[7], [40, 40, 40, 40, 35, 35, 35, 35]);
    }

    #[test]
    fn predicts_chroma_directional_and_plane_modes() {
        let top = std::array::from_fn(|index| index as u8);
        let left = std::array::from_fn(|index| 100 + index as u8);
        let references = IntraChroma420References {
            top: Some(top),
            left: Some(left),
            top_left: Some(50),
        };
        assert_eq!(predict_intra_chroma_420(2, &references), Ok([top; 8]));
        let horizontal = predict_intra_chroma_420(1, &references).unwrap();
        for row in 0..8 {
            assert_eq!(horizontal[row], [left[row]; 8]);
        }

        let constant = IntraChroma420References {
            top: Some([64; 8]),
            left: Some([64; 8]),
            top_left: Some(64),
        };
        assert_eq!(predict_intra_chroma_420(3, &constant), Ok([[64; 8]; 8]));
    }

    #[test]
    fn rejects_modes_with_missing_required_samples() {
        let missing = Intra16x16References {
            top: None,
            left: None,
            top_left: None,
        };
        assert!(matches!(
            predict_intra_16x16(0, &missing),
            Err(H264Error::InvalidSyntax(_))
        ));
        assert!(matches!(
            predict_intra_16x16(3, &missing),
            Err(H264Error::InvalidSyntax(_))
        ));
        assert!(matches!(
            predict_intra_16x16(4, &missing),
            Err(H264Error::InvalidSyntax(_))
        ));
    }
}
