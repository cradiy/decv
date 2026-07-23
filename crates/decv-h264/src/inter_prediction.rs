//! Fractional-sample inter prediction for progressive 8-bit 4:2:0 pictures.

use crate::{H264Error, ResolvedPPartition, Result, Yuv420Picture};

/// Fixed-capacity prediction storage for one P partition.
///
/// Only `height` rows and `width` columns of `luma` are valid. Cb and Cr use
/// half of those dimensions for 4:2:0 video. Fixed capacity avoids a heap
/// allocation for every 4x4 through 16x16 partition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InterPrediction420 {
    pub width: u8,
    pub height: u8,
    pub luma: [[u8; 16]; 16],
    pub cb: [[u8; 8]; 8],
    pub cr: [[u8; 8]; 8],
}

impl Yuv420Picture {
    /// Predicts one partition from this already reconstructed reference
    /// picture using normative H.264 luma and 4:2:0 chroma interpolation.
    pub fn predict_inter_420(
        &self,
        macroblock_x: usize,
        macroblock_y: usize,
        partition: ResolvedPPartition,
    ) -> Result<InterPrediction420> {
        validate_partition(partition)?;
        let current_x = macroblock_x
            .checked_mul(16)
            .and_then(|value| value.checked_add(usize::from(partition.x)))
            .ok_or(H264Error::IntegerOverflow)?;
        let current_y = macroblock_y
            .checked_mul(16)
            .and_then(|value| value.checked_add(usize::from(partition.y)))
            .ok_or(H264Error::IntegerOverflow)?;
        let current_right = current_x
            .checked_add(usize::from(partition.width))
            .ok_or(H264Error::IntegerOverflow)?;
        let current_bottom = current_y
            .checked_add(usize::from(partition.height))
            .ok_or(H264Error::IntegerOverflow)?;
        let (picture_width, picture_height) = self.dimensions();
        if current_right > picture_width || current_bottom > picture_height {
            return Err(H264Error::InvalidSyntax(
                "inter prediction partition lies outside the current picture",
            ));
        }

        let (luma, cb, cr) = self.planes();
        let motion = partition.motion_vector;
        let integer_motion_x = i32::from(motion.x).div_euclid(4);
        let integer_motion_y = i32::from(motion.y).div_euclid(4);
        let fractional_motion_x = i32::from(motion.x).rem_euclid(4) as u8;
        let fractional_motion_y = i32::from(motion.y).rem_euclid(4) as u8;
        let mut prediction = InterPrediction420 {
            width: partition.width,
            height: partition.height,
            luma: [[0; 16]; 16],
            cb: [[0; 8]; 8],
            cr: [[0; 8]; 8],
        };

        let reference_luma_x = usize_to_i32(current_x)? + integer_motion_x;
        let reference_luma_y = usize_to_i32(current_y)? + integer_motion_y;
        let luma_is_interior = interpolation_window_is_inside(
            reference_luma_x,
            reference_luma_y,
            usize::from(partition.width),
            usize::from(partition.height),
            picture_width,
            picture_height,
            2,
            3,
        );
        if luma_is_interior {
            predict_luma::<false>(
                &mut prediction,
                luma,
                picture_width,
                picture_height,
                reference_luma_x,
                reference_luma_y,
                fractional_motion_x,
                fractional_motion_y,
            );
        } else {
            predict_luma::<true>(
                &mut prediction,
                luma,
                picture_width,
                picture_height,
                reference_luma_x,
                reference_luma_y,
                fractional_motion_x,
                fractional_motion_y,
            );
        }

        let chroma_width = picture_width / 2;
        let chroma_height = picture_height / 2;
        let current_chroma_x = current_x / 2;
        let current_chroma_y = current_y / 2;
        let integer_chroma_x = i32::from(motion.x).div_euclid(8);
        let integer_chroma_y = i32::from(motion.y).div_euclid(8);
        let fractional_chroma_x = i32::from(motion.x).rem_euclid(8) as u8;
        let fractional_chroma_y = i32::from(motion.y).rem_euclid(8) as u8;
        let reference_chroma_x = usize_to_i32(current_chroma_x)? + integer_chroma_x;
        let reference_chroma_y = usize_to_i32(current_chroma_y)? + integer_chroma_y;
        let chroma_is_interior = interpolation_window_is_inside(
            reference_chroma_x,
            reference_chroma_y,
            usize::from(partition.width / 2),
            usize::from(partition.height / 2),
            chroma_width,
            chroma_height,
            0,
            1,
        );
        if chroma_is_interior {
            predict_chroma::<false>(
                &mut prediction,
                cb,
                cr,
                chroma_width,
                chroma_height,
                reference_chroma_x,
                reference_chroma_y,
                fractional_chroma_x,
                fractional_chroma_y,
            );
        } else {
            predict_chroma::<true>(
                &mut prediction,
                cb,
                cr,
                chroma_width,
                chroma_height,
                reference_chroma_x,
                reference_chroma_y,
                fractional_chroma_x,
                fractional_chroma_y,
            );
        }
        Ok(prediction)
    }
}

fn validate_partition(partition: ResolvedPPartition) -> Result<()> {
    if partition.width < 4
        || partition.height < 4
        || partition.width > 16
        || partition.height > 16
        || !partition.x.is_multiple_of(4)
        || !partition.y.is_multiple_of(4)
        || !partition.width.is_multiple_of(4)
        || !partition.height.is_multiple_of(4)
        || partition
            .x
            .checked_add(partition.width)
            .is_none_or(|x| x > 16)
        || partition
            .y
            .checked_add(partition.height)
            .is_none_or(|y| y > 16)
    {
        return Err(H264Error::InvalidSyntax(
            "inter prediction partition geometry is invalid",
        ));
    }
    Ok(())
}

#[inline]
fn usize_to_i32(value: usize) -> Result<i32> {
    i32::try_from(value).map_err(|_| H264Error::IntegerOverflow)
}

#[allow(clippy::too_many_arguments)]
fn interpolation_window_is_inside(
    x: i32,
    y: i32,
    width: usize,
    height: usize,
    plane_width: usize,
    plane_height: usize,
    margin_before: i32,
    margin_after: i32,
) -> bool {
    let Ok(width) = i32::try_from(width) else {
        return false;
    };
    let Ok(height) = i32::try_from(height) else {
        return false;
    };
    let Ok(plane_width) = i32::try_from(plane_width) else {
        return false;
    };
    let Ok(plane_height) = i32::try_from(plane_height) else {
        return false;
    };
    x >= margin_before
        && y >= margin_before
        && x.checked_add(width - 1 + margin_after)
            .is_some_and(|right| right < plane_width)
        && y.checked_add(height - 1 + margin_after)
            .is_some_and(|bottom| bottom < plane_height)
}

#[allow(clippy::too_many_arguments)]
fn predict_luma<const CLIP: bool>(
    prediction: &mut InterPrediction420,
    plane: &[u8],
    width: usize,
    height: usize,
    reference_x: i32,
    reference_y: i32,
    x_fraction: u8,
    y_fraction: u8,
) {
    for y in 0..usize::from(prediction.height) {
        for x in 0..usize::from(prediction.width) {
            prediction.luma[y][x] = interpolate_luma_inner::<CLIP>(
                plane,
                width,
                height,
                reference_x + x as i32,
                reference_y + y as i32,
                x_fraction,
                y_fraction,
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn predict_chroma<const CLIP: bool>(
    prediction: &mut InterPrediction420,
    cb: &[u8],
    cr: &[u8],
    width: usize,
    height: usize,
    reference_x: i32,
    reference_y: i32,
    x_fraction: u8,
    y_fraction: u8,
) {
    for output_y in 0..usize::from(prediction.height / 2) {
        for output_x in 0..usize::from(prediction.width / 2) {
            let x = reference_x + output_x as i32;
            let y = reference_y + output_y as i32;
            prediction.cb[output_y][output_x] =
                interpolate_chroma_inner::<CLIP>(cb, width, height, x, y, x_fraction, y_fraction);
            prediction.cr[output_y][output_x] =
                interpolate_chroma_inner::<CLIP>(cr, width, height, x, y, x_fraction, y_fraction);
        }
    }
}

#[cfg(test)]
fn interpolate_luma(
    plane: &[u8],
    width: usize,
    height: usize,
    x: i32,
    y: i32,
    x_fraction: u8,
    y_fraction: u8,
) -> u8 {
    interpolate_luma_inner::<true>(plane, width, height, x, y, x_fraction, y_fraction)
}

fn interpolate_luma_inner<const CLIP: bool>(
    plane: &[u8],
    width: usize,
    height: usize,
    x: i32,
    y: i32,
    x_fraction: u8,
    y_fraction: u8,
) -> u8 {
    debug_assert!(x_fraction < 4 && y_fraction < 4);
    let integer = sample::<CLIP>(plane, width, height, x, y);
    if x_fraction == 0 && y_fraction == 0 {
        return integer;
    }

    match (x_fraction, y_fraction) {
        (0, 1) => rounded_average(integer, half_vertical::<CLIP>(plane, width, height, x, y)),
        (0, 2) => half_vertical::<CLIP>(plane, width, height, x, y),
        (0, 3) => rounded_average(
            sample::<CLIP>(plane, width, height, x, y + 1),
            half_vertical::<CLIP>(plane, width, height, x, y),
        ),
        (1, 0) => rounded_average(integer, half_horizontal::<CLIP>(plane, width, height, x, y)),
        (1, 1) => rounded_average(
            half_horizontal::<CLIP>(plane, width, height, x, y),
            half_vertical::<CLIP>(plane, width, height, x, y),
        ),
        (1, 2) => rounded_average(
            half_vertical::<CLIP>(plane, width, height, x, y),
            half_diagonal::<CLIP>(plane, width, height, x, y),
        ),
        (1, 3) => rounded_average(
            half_vertical::<CLIP>(plane, width, height, x, y),
            half_horizontal::<CLIP>(plane, width, height, x, y + 1),
        ),
        (2, 0) => half_horizontal::<CLIP>(plane, width, height, x, y),
        (2, 1) => rounded_average(
            half_horizontal::<CLIP>(plane, width, height, x, y),
            half_diagonal::<CLIP>(plane, width, height, x, y),
        ),
        (2, 2) => half_diagonal::<CLIP>(plane, width, height, x, y),
        (2, 3) => rounded_average(
            half_diagonal::<CLIP>(plane, width, height, x, y),
            half_horizontal::<CLIP>(plane, width, height, x, y + 1),
        ),
        (3, 0) => rounded_average(
            sample::<CLIP>(plane, width, height, x + 1, y),
            half_horizontal::<CLIP>(plane, width, height, x, y),
        ),
        (3, 1) => rounded_average(
            half_horizontal::<CLIP>(plane, width, height, x, y),
            half_vertical::<CLIP>(plane, width, height, x + 1, y),
        ),
        (3, 2) => rounded_average(
            half_diagonal::<CLIP>(plane, width, height, x, y),
            half_vertical::<CLIP>(plane, width, height, x + 1, y),
        ),
        (3, 3) => rounded_average(
            half_vertical::<CLIP>(plane, width, height, x + 1, y),
            half_horizontal::<CLIP>(plane, width, height, x, y + 1),
        ),
        _ => unreachable!("fractional luma positions are in 0..=3"),
    }
}

#[inline]
fn half_horizontal<const CLIP: bool>(
    plane: &[u8],
    width: usize,
    height: usize,
    x: i32,
    y: i32,
) -> u8 {
    ((horizontal_six_tap::<CLIP>(plane, width, height, x, y) + 16) >> 5).clamp(0, 255) as u8
}

#[inline]
fn half_vertical<const CLIP: bool>(
    plane: &[u8],
    width: usize,
    height: usize,
    x: i32,
    y: i32,
) -> u8 {
    let value = sample::<CLIP>(plane, width, height, x, y - 2) as i32
        - 5 * sample::<CLIP>(plane, width, height, x, y - 1) as i32
        + 20 * sample::<CLIP>(plane, width, height, x, y) as i32
        + 20 * sample::<CLIP>(plane, width, height, x, y + 1) as i32
        - 5 * sample::<CLIP>(plane, width, height, x, y + 2) as i32
        + sample::<CLIP>(plane, width, height, x, y + 3) as i32;
    ((value + 16) >> 5).clamp(0, 255) as u8
}

#[inline]
fn half_diagonal<const CLIP: bool>(
    plane: &[u8],
    width: usize,
    height: usize,
    x: i32,
    y: i32,
) -> u8 {
    let value = horizontal_six_tap::<CLIP>(plane, width, height, x, y - 2)
        - 5 * horizontal_six_tap::<CLIP>(plane, width, height, x, y - 1)
        + 20 * horizontal_six_tap::<CLIP>(plane, width, height, x, y)
        + 20 * horizontal_six_tap::<CLIP>(plane, width, height, x, y + 1)
        - 5 * horizontal_six_tap::<CLIP>(plane, width, height, x, y + 2)
        + horizontal_six_tap::<CLIP>(plane, width, height, x, y + 3);
    ((value + 512) >> 10).clamp(0, 255) as u8
}

#[inline]
fn horizontal_six_tap<const CLIP: bool>(
    plane: &[u8],
    width: usize,
    height: usize,
    x: i32,
    y: i32,
) -> i32 {
    sample::<CLIP>(plane, width, height, x - 2, y) as i32
        - 5 * sample::<CLIP>(plane, width, height, x - 1, y) as i32
        + 20 * sample::<CLIP>(plane, width, height, x, y) as i32
        + 20 * sample::<CLIP>(plane, width, height, x + 1, y) as i32
        - 5 * sample::<CLIP>(plane, width, height, x + 2, y) as i32
        + sample::<CLIP>(plane, width, height, x + 3, y) as i32
}

#[inline]
#[cfg(test)]
fn interpolate_chroma(
    plane: &[u8],
    width: usize,
    height: usize,
    x: i32,
    y: i32,
    x_fraction: u8,
    y_fraction: u8,
) -> u8 {
    interpolate_chroma_inner::<true>(plane, width, height, x, y, x_fraction, y_fraction)
}

fn interpolate_chroma_inner<const CLIP: bool>(
    plane: &[u8],
    width: usize,
    height: usize,
    x: i32,
    y: i32,
    x_fraction: u8,
    y_fraction: u8,
) -> u8 {
    debug_assert!(x_fraction < 8 && y_fraction < 8);
    let a = u32::from(sample::<CLIP>(plane, width, height, x, y));
    if x_fraction == 0 && y_fraction == 0 {
        return a as u8;
    }
    let b = u32::from(sample::<CLIP>(plane, width, height, x + 1, y));
    let c = u32::from(sample::<CLIP>(plane, width, height, x, y + 1));
    let x_fraction = u32::from(x_fraction);
    let y_fraction = u32::from(y_fraction);
    if y_fraction == 0 {
        return (((8 - x_fraction) * a + x_fraction * b + 4) >> 3) as u8;
    }
    if x_fraction == 0 {
        return (((8 - y_fraction) * a + y_fraction * c + 4) >> 3) as u8;
    }
    let d = u32::from(sample::<CLIP>(plane, width, height, x + 1, y + 1));
    (((8 - x_fraction) * (8 - y_fraction) * a
        + x_fraction * (8 - y_fraction) * b
        + (8 - x_fraction) * y_fraction * c
        + x_fraction * y_fraction * d
        + 32)
        >> 6) as u8
}

#[inline(always)]
fn sample<const CLIP: bool>(plane: &[u8], width: usize, height: usize, x: i32, y: i32) -> u8 {
    if CLIP {
        return sample_clipped(plane, width, height, x, y);
    }

    debug_assert!(x >= 0 && x < width as i32);
    debug_assert!(y >= 0 && y < height as i32);
    let index = y as usize * width + x as usize;
    // SAFETY: the caller selects this specialization only after validating
    // the complete interpolation footprint against the plane dimensions.
    unsafe { *plane.get_unchecked(index) }
}

#[inline]
fn sample_clipped(plane: &[u8], width: usize, height: usize, x: i32, y: i32) -> u8 {
    let x = x.clamp(0, width as i32 - 1) as usize;
    let y = y.clamp(0, height as i32 - 1) as usize;
    plane[y * width + x]
}

#[inline]
fn rounded_average(a: u8, b: u8) -> u8 {
    (u16::from(a) + u16::from(b)).div_ceil(2) as u8
}

#[cfg(test)]
mod tests {
    use decv_core::Size;

    use super::*;
    use crate::MotionVector;

    fn gradient_picture() -> Yuv420Picture {
        let mut picture = Yuv420Picture::new(Size {
            width: 16,
            height: 16,
        })
        .unwrap();
        let (luma, cb, cr) = picture.planes_mut();
        for y in 0..16 {
            for x in 0..16 {
                luma[y * 16 + x] = 20 + 4 * x as u8 + 8 * y as u8;
            }
        }
        for y in 0..8 {
            for x in 0..8 {
                cb[y * 8 + x] = 10 + 8 * x as u8 + 16 * y as u8;
                cr[y * 8 + x] = 11 + 8 * x as u8 + 16 * y as u8;
            }
        }
        picture
    }

    #[test]
    fn interpolates_all_sixteen_luma_fractional_positions() {
        let picture = gradient_picture();
        let (luma, _, _) = picture.planes();
        for y_fraction in 0..4 {
            for x_fraction in 0..4 {
                assert_eq!(
                    interpolate_luma(luma, 16, 16, 4, 4, x_fraction, y_fraction),
                    68 + x_fraction + 2 * y_fraction
                );
            }
        }
    }

    #[test]
    fn interpolates_all_sixty_four_chroma_fractional_positions() {
        let picture = gradient_picture();
        let (_, cb, _) = picture.planes();
        for y_fraction in 0..8 {
            for x_fraction in 0..8 {
                assert_eq!(
                    interpolate_chroma(cb, 8, 8, 2, 2, x_fraction, y_fraction),
                    58 + x_fraction + 2 * y_fraction
                );
            }
        }
    }

    #[test]
    fn builds_luma_and_chroma_prediction_without_allocating_per_plane() {
        let picture = gradient_picture();
        let prediction = picture
            .predict_inter_420(
                0,
                0,
                ResolvedPPartition {
                    x: 0,
                    y: 0,
                    width: 8,
                    height: 4,
                    reference_index: 0,
                    motion_vector: MotionVector { x: 17, y: 19 },
                },
            )
            .unwrap();
        assert_eq!((prediction.width, prediction.height), (8, 4));
        assert_eq!(prediction.luma[0][0], 75);
        assert_eq!(prediction.luma[3][7], 127);
        assert_eq!(prediction.cb[0][0], 65);
        assert_eq!(prediction.cr[0][0], 66);
    }

    #[test]
    fn clips_every_filter_tap_to_reference_picture_edges() {
        let picture = gradient_picture();
        let prediction = picture
            .predict_inter_420(
                0,
                0,
                ResolvedPPartition {
                    x: 0,
                    y: 0,
                    width: 4,
                    height: 4,
                    reference_index: 0,
                    motion_vector: MotionVector {
                        x: i16::MIN,
                        y: i16::MIN,
                    },
                },
            )
            .unwrap();
        assert_eq!(prediction.luma[0][0], 20);
        assert_eq!(prediction.cb[0][0], 10);
        assert_eq!(prediction.cr[0][0], 11);
    }

    #[test]
    fn rejects_invalid_partition_geometry_and_current_location() {
        let picture = gradient_picture();
        let invalid = ResolvedPPartition {
            x: 14,
            y: 0,
            width: 4,
            height: 4,
            reference_index: 0,
            motion_vector: MotionVector::default(),
        };
        assert!(matches!(
            picture.predict_inter_420(0, 0, invalid),
            Err(H264Error::InvalidSyntax(_))
        ));
        let valid = ResolvedPPartition {
            x: 0,
            y: 0,
            width: 16,
            height: 16,
            reference_index: 0,
            motion_vector: MotionVector::default(),
        };
        assert!(matches!(
            picture.predict_inter_420(1, 0, valid),
            Err(H264Error::InvalidSyntax(_))
        ));
    }
}
