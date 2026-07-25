//! Fractional-sample inter prediction for progressive 8-bit 4:2:0 pictures.

use crate::{H264Error, MotionVector, ResolvedPPartition, Result, Yuv420Picture};

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

impl InterPrediction420 {
    pub(crate) const fn empty() -> Self {
        Self {
            width: 0,
            height: 0,
            luma: [[0; 16]; 16],
            cb: [[0; 8]; 8],
            cr: [[0; 8]; 8],
        }
    }
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
        let mut prediction = InterPrediction420::empty();
        self.predict_inter_420_into(macroblock_x, macroblock_y, partition, &mut prediction)?;
        Ok(prediction)
    }

    pub(crate) fn predict_inter_420_into(
        &self,
        macroblock_x: usize,
        macroblock_y: usize,
        partition: ResolvedPPartition,
        prediction: &mut InterPrediction420,
    ) -> Result<()> {
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
        prediction.width = partition.width;
        prediction.height = partition.height;

        let reference_luma_x = usize_to_i32(current_x)? + integer_motion_x;
        let reference_luma_y = usize_to_i32(current_y)? + integer_motion_y;
        let luma_is_interior = interpolation_window_is_inside(
            reference_luma_x,
            reference_luma_y,
            usize::from(partition.width),
            usize::from(partition.height),
            picture_width,
            picture_height,
            if fractional_motion_x == 0 { 0 } else { 2 },
            if fractional_motion_x == 0 { 0 } else { 3 },
            if fractional_motion_y == 0 { 0 } else { 2 },
            if fractional_motion_y == 0 { 0 } else { 3 },
        );
        if luma_is_interior {
            predict_luma::<false>(
                prediction,
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
                prediction,
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
            u8::from(fractional_chroma_x != 0),
            0,
            u8::from(fractional_chroma_y != 0),
        );
        #[cfg(feature = "internal-profiling")]
        crate::profiling::record_inter_prediction(
            partition.width,
            partition.height,
            fractional_motion_x,
            fractional_motion_y,
            !luma_is_interior,
            fractional_chroma_x,
            fractional_chroma_y,
            !chroma_is_interior,
        );
        if chroma_is_interior {
            predict_chroma::<false>(
                prediction,
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
                prediction,
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
        Ok(())
    }

    #[inline]
    pub(crate) fn try_copy_integer_macroblock_420_into(
        &self,
        macroblock_x: usize,
        macroblock_y: usize,
        motion: MotionVector,
        predicted_luma: &mut [[u8; 16]; 16],
        predicted_cb: &mut [[u8; 8]; 8],
        predicted_cr: &mut [[u8; 8]; 8],
    ) -> Result<bool> {
        let motion_x = i32::from(motion.x);
        let motion_y = i32::from(motion.y);
        if motion_x.rem_euclid(8) != 0 || motion_y.rem_euclid(8) != 0 {
            return Ok(false);
        }
        let current_x = macroblock_x
            .checked_mul(16)
            .ok_or(H264Error::IntegerOverflow)?;
        let current_y = macroblock_y
            .checked_mul(16)
            .ok_or(H264Error::IntegerOverflow)?;
        let (picture_width, picture_height) = self.dimensions();
        if current_x
            .checked_add(16)
            .is_none_or(|right| right > picture_width)
            || current_y
                .checked_add(16)
                .is_none_or(|bottom| bottom > picture_height)
        {
            return Err(H264Error::InvalidSyntax(
                "inter prediction macroblock lies outside the current picture",
            ));
        }
        let reference_luma_x = usize_to_i32(current_x)? + motion_x.div_euclid(4);
        let reference_luma_y = usize_to_i32(current_y)? + motion_y.div_euclid(4);
        if !interpolation_window_is_inside(
            reference_luma_x,
            reference_luma_y,
            16,
            16,
            picture_width,
            picture_height,
            0,
            0,
            0,
            0,
        ) {
            return Ok(false);
        }

        let chroma_width = picture_width / 2;
        let chroma_height = picture_height / 2;
        let reference_chroma_x = usize_to_i32(current_x / 2)? + motion_x.div_euclid(8);
        let reference_chroma_y = usize_to_i32(current_y / 2)? + motion_y.div_euclid(8);
        if !interpolation_window_is_inside(
            reference_chroma_x,
            reference_chroma_y,
            8,
            8,
            chroma_width,
            chroma_height,
            0,
            0,
            0,
            0,
        ) {
            return Ok(false);
        }

        let reference_luma_x =
            usize::try_from(reference_luma_x).map_err(|_| H264Error::IntegerOverflow)?;
        let reference_luma_y =
            usize::try_from(reference_luma_y).map_err(|_| H264Error::IntegerOverflow)?;
        let reference_chroma_x =
            usize::try_from(reference_chroma_x).map_err(|_| H264Error::IntegerOverflow)?;
        let reference_chroma_y =
            usize::try_from(reference_chroma_y).map_err(|_| H264Error::IntegerOverflow)?;
        let (luma, cb, cr) = self.planes();
        let luma_source = reference_luma_y * picture_width + reference_luma_x;
        let chroma_source = reference_chroma_y * chroma_width + reference_chroma_x;
        // SAFETY: the complete source rectangles were validated above, the
        // fixed staging planes contain all destination rows, and reference
        // pictures cannot overlap the staging macroblock.
        unsafe {
            copy_fixed_rows::<16, 16>(
                predicted_luma.as_mut_ptr().cast(),
                luma.as_ptr().add(luma_source),
                picture_width,
                16,
            );
            copy_fixed_rows::<8, 8>(
                predicted_cb.as_mut_ptr().cast(),
                cb.as_ptr().add(chroma_source),
                chroma_width,
                8,
            );
            copy_fixed_rows::<8, 8>(
                predicted_cr.as_mut_ptr().cast(),
                cr.as_ptr().add(chroma_source),
                chroma_width,
                8,
            );
        }
        Ok(true)
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
    margin_left: u8,
    margin_right: u8,
    margin_top: u8,
    margin_bottom: u8,
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
    let margin_left = i32::from(margin_left);
    let margin_right = i32::from(margin_right);
    let margin_top = i32::from(margin_top);
    let margin_bottom = i32::from(margin_bottom);
    x >= margin_left
        && y >= margin_top
        && x.checked_add(width - 1 + margin_right)
            .is_some_and(|right| right < plane_width)
        && y.checked_add(height - 1 + margin_bottom)
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
    if !CLIP && x_fraction == 0 && y_fraction == 0 {
        let output_width = usize::from(prediction.width);
        let output_height = usize::from(prediction.height);
        let reference_x = reference_x as usize;
        let reference_y = reference_y as usize;
        let start = reference_y * width + reference_x;
        // SAFETY: The complete source rectangle was validated as interior,
        // every destination row has sixteen bytes, and the selected constant
        // width is the validated partition width.
        unsafe {
            match output_width {
                4 => copy_fixed_rows::<4, 16>(
                    prediction.luma.as_mut_ptr().cast(),
                    plane.as_ptr().add(start),
                    width,
                    output_height,
                ),
                8 => copy_fixed_rows::<8, 16>(
                    prediction.luma.as_mut_ptr().cast(),
                    plane.as_ptr().add(start),
                    width,
                    output_height,
                ),
                16 => copy_fixed_rows::<16, 16>(
                    prediction.luma.as_mut_ptr().cast(),
                    plane.as_ptr().add(start),
                    width,
                    output_height,
                ),
                _ => unreachable!("validated luma partition widths are 4, 8, or 16"),
            }
        }
        return;
    }

    #[cfg(target_arch = "x86_64")]
    if !CLIP
        && prediction.width == 16
        && ((x_fraction == 0) != (y_fraction == 0))
        && std::is_x86_feature_detected!("avx2")
    {
        // SAFETY: Runtime detection proves AVX2 support. The caller selected
        // the non-clipping specialization only after checking the complete
        // six-tap interpolation window.
        unsafe {
            predict_luma_axis_avx2(
                prediction,
                plane,
                width,
                reference_x as usize,
                reference_y as usize,
                x_fraction,
                y_fraction,
            );
        }
        return;
    }

    #[cfg(target_arch = "x86_64")]
    if !CLIP && matches!(prediction.width, 8 | 16) && ((x_fraction == 0) != (y_fraction == 0)) {
        // SAFETY: SSE2 is part of the x86_64 baseline. The caller selected
        // the non-clipping specialization only after checking the complete
        // six-tap interpolation window.
        unsafe {
            predict_luma_axis_sse2(
                prediction,
                plane,
                width,
                reference_x as usize,
                reference_y as usize,
                x_fraction,
                y_fraction,
            );
        }
        return;
    }

    #[cfg(target_arch = "x86_64")]
    if !CLIP && matches!(prediction.width, 8 | 16) && x_fraction != 0 && y_fraction != 0 {
        // SAFETY: SSE2 is part of the x86_64 baseline. The non-clipping
        // specialization is selected only after validating the complete
        // horizontal and vertical six-tap footprint.
        unsafe {
            predict_luma_two_dimensional_sse2(
                prediction,
                plane,
                width,
                reference_x as usize,
                reference_y as usize,
                x_fraction,
                y_fraction,
            );
        }
        return;
    }

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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn predict_luma_axis_avx2(
    prediction: &mut InterPrediction420,
    plane: &[u8],
    stride: usize,
    reference_x: usize,
    reference_y: usize,
    x_fraction: u8,
    y_fraction: u8,
) {
    use std::arch::x86_64::{
        __m128i, __m256i, _mm_avg_epu8, _mm_loadu_si128, _mm_storeu_si128, _mm_unpacklo_epi64,
        _mm256_add_epi16, _mm256_castsi256_si128, _mm256_cvtepu8_epi16, _mm256_extracti128_si256,
        _mm256_mullo_epi16, _mm256_packus_epi16, _mm256_set1_epi16, _mm256_setzero_si256,
        _mm256_srai_epi16, _mm256_sub_epi16,
    };

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn load_sixteen(ptr: *const u8) -> __m256i {
        // SAFETY: The caller validated the complete sixteen-byte source row.
        let bytes = unsafe { _mm_loadu_si128(ptr.cast::<__m128i>()) };
        _mm256_cvtepu8_epi16(bytes)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn six_tap(
        s0: *const u8,
        s1: *const u8,
        s2: *const u8,
        s3: *const u8,
        s4: *const u8,
        s5: *const u8,
    ) -> __m128i {
        // SAFETY: Every pointer addresses sixteen validated source samples.
        let s0 = unsafe { load_sixteen(s0) };
        // SAFETY: See above.
        let s1 = unsafe { load_sixteen(s1) };
        // SAFETY: See above.
        let s2 = unsafe { load_sixteen(s2) };
        // SAFETY: See above.
        let s3 = unsafe { load_sixteen(s3) };
        // SAFETY: See above.
        let s4 = unsafe { load_sixteen(s4) };
        // SAFETY: See above.
        let s5 = unsafe { load_sixteen(s5) };
        let positive = _mm256_add_epi16(
            _mm256_add_epi16(s0, s5),
            _mm256_mullo_epi16(_mm256_add_epi16(s2, s3), _mm256_set1_epi16(20)),
        );
        let negative = _mm256_mullo_epi16(_mm256_add_epi16(s1, s4), _mm256_set1_epi16(5));
        let filtered = _mm256_srai_epi16::<5>(_mm256_add_epi16(
            _mm256_sub_epi16(positive, negative),
            _mm256_set1_epi16(16),
        ));
        let packed = _mm256_packus_epi16(filtered, _mm256_setzero_si256());
        _mm_unpacklo_epi64(
            _mm256_castsi256_si128(packed),
            _mm256_extracti128_si256::<1>(packed),
        )
    }

    debug_assert_eq!(prediction.width, 16);
    let output_height = usize::from(prediction.height);
    for output_y in 0..output_height {
        let base = plane
            .as_ptr()
            .wrapping_add((reference_y + output_y) * stride + reference_x);
        let half = if y_fraction == 0 {
            // SAFETY: The validated window includes x - 2 through x + 18.
            unsafe {
                six_tap(
                    base.wrapping_sub(2),
                    base.wrapping_sub(1),
                    base,
                    base.wrapping_add(1),
                    base.wrapping_add(2),
                    base.wrapping_add(3),
                )
            }
        } else {
            // SAFETY: The validated window includes all six source rows.
            unsafe {
                six_tap(
                    base.wrapping_sub(2 * stride),
                    base.wrapping_sub(stride),
                    base,
                    base.wrapping_add(stride),
                    base.wrapping_add(2 * stride),
                    base.wrapping_add(3 * stride),
                )
            }
        };
        let fraction = if y_fraction == 0 {
            x_fraction
        } else {
            y_fraction
        };
        let output = if fraction == 2 {
            half
        } else {
            let integer = if fraction == 1 {
                base
            } else if y_fraction == 0 {
                base.wrapping_add(1)
            } else {
                base.wrapping_add(stride)
            };
            // SAFETY: The integer row is part of the validated window.
            let integer = unsafe { _mm_loadu_si128(integer.cast::<__m128i>()) };
            _mm_avg_epu8(integer, half)
        };
        // SAFETY: Every luma row contains sixteen writable bytes.
        unsafe {
            _mm_storeu_si128(
                prediction.luma[output_y].as_mut_ptr().cast::<__m128i>(),
                output,
            );
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn predict_luma_axis_sse2(
    prediction: &mut InterPrediction420,
    plane: &[u8],
    stride: usize,
    reference_x: usize,
    reference_y: usize,
    x_fraction: u8,
    y_fraction: u8,
) {
    use std::arch::x86_64::{
        __m128i, _mm_add_epi16, _mm_avg_epu8, _mm_loadl_epi64, _mm_mullo_epi16, _mm_packus_epi16,
        _mm_set1_epi16, _mm_setzero_si128, _mm_srai_epi16, _mm_storel_epi64, _mm_sub_epi16,
        _mm_unpacklo_epi8,
    };

    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn load_eight(ptr: *const u8, zero: __m128i) -> __m128i {
        // SAFETY: The caller validated the complete eight-byte source range.
        let bytes = unsafe { _mm_loadl_epi64(ptr.cast::<__m128i>()) };
        _mm_unpacklo_epi8(bytes, zero)
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn six_tap(
        s0: *const u8,
        s1: *const u8,
        s2: *const u8,
        s3: *const u8,
        s4: *const u8,
        s5: *const u8,
        zero: __m128i,
    ) -> __m128i {
        // SAFETY: Every pointer addresses an eight-byte interpolation row.
        let s0 = unsafe { load_eight(s0, zero) };
        // SAFETY: See above.
        let s1 = unsafe { load_eight(s1, zero) };
        // SAFETY: See above.
        let s2 = unsafe { load_eight(s2, zero) };
        // SAFETY: See above.
        let s3 = unsafe { load_eight(s3, zero) };
        // SAFETY: See above.
        let s4 = unsafe { load_eight(s4, zero) };
        // SAFETY: See above.
        let s5 = unsafe { load_eight(s5, zero) };
        let positive = _mm_add_epi16(
            _mm_add_epi16(s0, s5),
            _mm_mullo_epi16(_mm_add_epi16(s2, s3), _mm_set1_epi16(20)),
        );
        let negative = _mm_mullo_epi16(_mm_add_epi16(s1, s4), _mm_set1_epi16(5));
        let filtered = _mm_srai_epi16::<5>(_mm_add_epi16(
            _mm_sub_epi16(positive, negative),
            _mm_set1_epi16(16),
        ));
        _mm_packus_epi16(filtered, zero)
    }

    let zero = _mm_setzero_si128();
    let output_width = usize::from(prediction.width);
    let output_height = usize::from(prediction.height);
    for output_y in 0..output_height {
        for output_x in (0..output_width).step_by(8) {
            let base = plane
                .as_ptr()
                .wrapping_add((reference_y + output_y) * stride + reference_x + output_x);
            let half = if y_fraction == 0 {
                // SAFETY: The non-clipping window includes x - 2 through
                // the final output sample plus 3.
                unsafe {
                    six_tap(
                        base.wrapping_sub(2),
                        base.wrapping_sub(1),
                        base,
                        base.wrapping_add(1),
                        base.wrapping_add(2),
                        base.wrapping_add(3),
                        zero,
                    )
                }
            } else {
                // SAFETY: The non-clipping window includes the six source
                // rows from y - 2 through y + 3.
                unsafe {
                    six_tap(
                        base.wrapping_sub(2 * stride),
                        base.wrapping_sub(stride),
                        base,
                        base.wrapping_add(stride),
                        base.wrapping_add(2 * stride),
                        base.wrapping_add(3 * stride),
                        zero,
                    )
                }
            };
            let fraction = if y_fraction == 0 {
                x_fraction
            } else {
                y_fraction
            };
            let output = if fraction == 2 {
                half
            } else {
                let integer = if fraction == 1 {
                    base
                } else if y_fraction == 0 {
                    base.wrapping_add(1)
                } else {
                    base.wrapping_add(stride)
                };
                // SAFETY: The integer row is part of the validated window.
                let integer = unsafe { _mm_loadl_epi64(integer.cast::<__m128i>()) };
                _mm_avg_epu8(integer, half)
            };
            let destination = prediction.luma[output_y]
                .as_mut_ptr()
                .wrapping_add(output_x);
            // SAFETY: Width is 8 or 16 and the loop writes one full chunk.
            unsafe {
                _mm_storel_epi64(destination.cast::<__m128i>(), output);
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn predict_luma_two_dimensional_sse2(
    prediction: &mut InterPrediction420,
    plane: &[u8],
    stride: usize,
    reference_x: usize,
    reference_y: usize,
    x_fraction: u8,
    y_fraction: u8,
) {
    use std::arch::x86_64::{
        __m128i, _mm_add_epi16, _mm_add_epi32, _mm_avg_epu8, _mm_loadl_epi64, _mm_madd_epi16,
        _mm_mullo_epi16, _mm_packs_epi32, _mm_packus_epi16, _mm_set_epi16, _mm_set1_epi16,
        _mm_set1_epi32, _mm_setzero_si128, _mm_srai_epi16, _mm_srai_epi32, _mm_storel_epi64,
        _mm_sub_epi16, _mm_unpackhi_epi16, _mm_unpacklo_epi8, _mm_unpacklo_epi16,
    };

    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn load_eight(ptr: *const u8, zero: __m128i) -> __m128i {
        // SAFETY: The caller validated the complete eight-byte source range.
        let bytes = unsafe { _mm_loadl_epi64(ptr.cast::<__m128i>()) };
        _mm_unpacklo_epi8(bytes, zero)
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn six_tap_raw(
        s0: *const u8,
        s1: *const u8,
        s2: *const u8,
        s3: *const u8,
        s4: *const u8,
        s5: *const u8,
        zero: __m128i,
    ) -> __m128i {
        // SAFETY: Every pointer addresses eight validated source samples.
        let s0 = unsafe { load_eight(s0, zero) };
        // SAFETY: See above.
        let s1 = unsafe { load_eight(s1, zero) };
        // SAFETY: See above.
        let s2 = unsafe { load_eight(s2, zero) };
        // SAFETY: See above.
        let s3 = unsafe { load_eight(s3, zero) };
        // SAFETY: See above.
        let s4 = unsafe { load_eight(s4, zero) };
        // SAFETY: See above.
        let s5 = unsafe { load_eight(s5, zero) };
        let positive = _mm_add_epi16(
            _mm_add_epi16(s0, s5),
            _mm_mullo_epi16(_mm_add_epi16(s2, s3), _mm_set1_epi16(20)),
        );
        let negative = _mm_mullo_epi16(_mm_add_epi16(s1, s4), _mm_set1_epi16(5));
        _mm_sub_epi16(positive, negative)
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    fn rounded_half(raw: __m128i, zero: __m128i) -> __m128i {
        let rounded = _mm_srai_epi16::<5>(_mm_add_epi16(raw, _mm_set1_epi16(16)));
        _mm_packus_epi16(rounded, zero)
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn horizontal_raw(base: *const u8, zero: __m128i) -> __m128i {
        // SAFETY: The interior footprint includes x - 2 through x + 10 for
        // this eight-sample output chunk.
        unsafe {
            six_tap_raw(
                base.wrapping_sub(2),
                base.wrapping_sub(1),
                base,
                base.wrapping_add(1),
                base.wrapping_add(2),
                base.wrapping_add(3),
                zero,
            )
        }
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn vertical_half(base: *const u8, stride: usize, zero: __m128i) -> __m128i {
        // SAFETY: The interior footprint includes the six required rows.
        let raw = unsafe {
            six_tap_raw(
                base.wrapping_sub(2 * stride),
                base.wrapping_sub(stride),
                base,
                base.wrapping_add(stride),
                base.wrapping_add(2 * stride),
                base.wrapping_add(3 * stride),
                zero,
            )
        };
        rounded_half(raw, zero)
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn diagonal_half(base: *const u8, stride: usize, zero: __m128i) -> __m128i {
        // SAFETY: The interior footprint includes all six horizontal taps on
        // all six vertical tap rows.
        let h0 = unsafe { horizontal_raw(base.wrapping_sub(2 * stride), zero) };
        // SAFETY: See above.
        let h1 = unsafe { horizontal_raw(base.wrapping_sub(stride), zero) };
        // SAFETY: See above.
        let h2 = unsafe { horizontal_raw(base, zero) };
        // SAFETY: See above.
        let h3 = unsafe { horizontal_raw(base.wrapping_add(stride), zero) };
        // SAFETY: See above.
        let h4 = unsafe { horizontal_raw(base.wrapping_add(2 * stride), zero) };
        // SAFETY: See above.
        let h5 = unsafe { horizontal_raw(base.wrapping_add(3 * stride), zero) };

        let coefficients_01 = _mm_set_epi16(-5, 1, -5, 1, -5, 1, -5, 1);
        let coefficients_23 = _mm_set1_epi16(20);
        let coefficients_45 = _mm_set_epi16(1, -5, 1, -5, 1, -5, 1, -5);
        let combine = |first: __m128i, second: __m128i, high: bool| {
            let pair_01 = if high {
                _mm_unpackhi_epi16(h0, h1)
            } else {
                _mm_unpacklo_epi16(h0, h1)
            };
            let pair_23 = if high {
                _mm_unpackhi_epi16(h2, h3)
            } else {
                _mm_unpacklo_epi16(h2, h3)
            };
            let pair_45 = if high {
                _mm_unpackhi_epi16(h4, h5)
            } else {
                _mm_unpacklo_epi16(h4, h5)
            };
            _mm_add_epi32(
                _mm_add_epi32(
                    _mm_madd_epi16(pair_01, first),
                    _mm_madd_epi16(pair_23, second),
                ),
                _mm_madd_epi16(pair_45, coefficients_45),
            )
        };
        let low = combine(coefficients_01, coefficients_23, false);
        let high = combine(coefficients_01, coefficients_23, true);
        let low = _mm_srai_epi32::<10>(_mm_add_epi32(low, _mm_set1_epi32(512)));
        let high = _mm_srai_epi32::<10>(_mm_add_epi32(high, _mm_set1_epi32(512)));
        _mm_packus_epi16(_mm_packs_epi32(low, high), zero)
    }

    debug_assert!((1..=3).contains(&x_fraction));
    debug_assert!((1..=3).contains(&y_fraction));
    let zero = _mm_setzero_si128();
    let output_width = usize::from(prediction.width);
    let output_height = usize::from(prediction.height);
    for output_y in 0..output_height {
        for output_x in (0..output_width).step_by(8) {
            let base = plane
                .as_ptr()
                .wrapping_add((reference_y + output_y) * stride + reference_x + output_x);
            let output = match (x_fraction, y_fraction) {
                (1, 1) => {
                    // SAFETY: The validated footprint covers both filters.
                    let horizontal = rounded_half(unsafe { horizontal_raw(base, zero) }, zero);
                    // SAFETY: See above.
                    let vertical = unsafe { vertical_half(base, stride, zero) };
                    _mm_avg_epu8(horizontal, vertical)
                }
                (1, 2) => {
                    // SAFETY: The validated footprint covers both filters.
                    let vertical = unsafe { vertical_half(base, stride, zero) };
                    // SAFETY: See above.
                    let diagonal = unsafe { diagonal_half(base, stride, zero) };
                    _mm_avg_epu8(vertical, diagonal)
                }
                (1, 3) => {
                    // SAFETY: The validated footprint covers both filters.
                    let vertical = unsafe { vertical_half(base, stride, zero) };
                    // SAFETY: See above.
                    let horizontal_next = rounded_half(
                        unsafe { horizontal_raw(base.wrapping_add(stride), zero) },
                        zero,
                    );
                    _mm_avg_epu8(vertical, horizontal_next)
                }
                (2, 1) => {
                    // SAFETY: The validated footprint covers both filters.
                    let horizontal = rounded_half(unsafe { horizontal_raw(base, zero) }, zero);
                    // SAFETY: See above.
                    let diagonal = unsafe { diagonal_half(base, stride, zero) };
                    _mm_avg_epu8(horizontal, diagonal)
                }
                (2, 2) => {
                    // SAFETY: The validated footprint covers the diagonal filter.
                    unsafe { diagonal_half(base, stride, zero) }
                }
                (2, 3) => {
                    // SAFETY: The validated footprint covers both filters.
                    let diagonal = unsafe { diagonal_half(base, stride, zero) };
                    // SAFETY: See above.
                    let horizontal_next = rounded_half(
                        unsafe { horizontal_raw(base.wrapping_add(stride), zero) },
                        zero,
                    );
                    _mm_avg_epu8(diagonal, horizontal_next)
                }
                (3, 1) => {
                    // SAFETY: The validated footprint covers both filters.
                    let horizontal = rounded_half(unsafe { horizontal_raw(base, zero) }, zero);
                    // SAFETY: See above.
                    let vertical_next =
                        unsafe { vertical_half(base.wrapping_add(1), stride, zero) };
                    _mm_avg_epu8(horizontal, vertical_next)
                }
                (3, 2) => {
                    // SAFETY: The validated footprint covers both filters.
                    let diagonal = unsafe { diagonal_half(base, stride, zero) };
                    // SAFETY: See above.
                    let vertical_next =
                        unsafe { vertical_half(base.wrapping_add(1), stride, zero) };
                    _mm_avg_epu8(diagonal, vertical_next)
                }
                (3, 3) => {
                    // SAFETY: The validated footprint covers both filters.
                    let vertical_next =
                        unsafe { vertical_half(base.wrapping_add(1), stride, zero) };
                    // SAFETY: See above.
                    let horizontal_next = rounded_half(
                        unsafe { horizontal_raw(base.wrapping_add(stride), zero) },
                        zero,
                    );
                    _mm_avg_epu8(vertical_next, horizontal_next)
                }
                _ => unreachable!("two-dimensional luma fractions are in 1..=3"),
            };
            let destination = prediction.luma[output_y]
                .as_mut_ptr()
                .wrapping_add(output_x);
            // SAFETY: Width is 8 or 16 and the loop writes one full chunk.
            unsafe {
                _mm_storel_epi64(destination.cast::<__m128i>(), output);
            }
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
    if !CLIP && x_fraction == 0 && y_fraction == 0 {
        let output_width = usize::from(prediction.width / 2);
        let output_height = usize::from(prediction.height / 2);
        let reference_x = reference_x as usize;
        let reference_y = reference_y as usize;
        let start = reference_y * width + reference_x;
        // SAFETY: Both complete source rectangles were validated as interior,
        // each destination row has eight bytes, and the selected constant
        // width is the validated chroma partition width.
        unsafe {
            match output_width {
                2 => {
                    copy_fixed_rows::<2, 8>(
                        prediction.cb.as_mut_ptr().cast(),
                        cb.as_ptr().add(start),
                        width,
                        output_height,
                    );
                    copy_fixed_rows::<2, 8>(
                        prediction.cr.as_mut_ptr().cast(),
                        cr.as_ptr().add(start),
                        width,
                        output_height,
                    );
                }
                4 => {
                    copy_fixed_rows::<4, 8>(
                        prediction.cb.as_mut_ptr().cast(),
                        cb.as_ptr().add(start),
                        width,
                        output_height,
                    );
                    copy_fixed_rows::<4, 8>(
                        prediction.cr.as_mut_ptr().cast(),
                        cr.as_ptr().add(start),
                        width,
                        output_height,
                    );
                }
                8 => {
                    copy_fixed_rows::<8, 8>(
                        prediction.cb.as_mut_ptr().cast(),
                        cb.as_ptr().add(start),
                        width,
                        output_height,
                    );
                    copy_fixed_rows::<8, 8>(
                        prediction.cr.as_mut_ptr().cast(),
                        cr.as_ptr().add(start),
                        width,
                        output_height,
                    );
                }
                _ => unreachable!("validated chroma partition widths are 2, 4, or 8"),
            }
        }
        return;
    }

    #[cfg(target_arch = "x86_64")]
    if !CLIP && prediction.width == 8 {
        // SAFETY: SSE2 is part of the x86_64 baseline. The non-clipping path
        // is selected only after validating four samples plus the right
        // neighbour from both chroma planes.
        unsafe {
            predict_chroma_pair_sse2(
                prediction,
                cb,
                cr,
                width,
                reference_x as usize,
                reference_y as usize,
                x_fraction,
                y_fraction,
            );
        }
        return;
    }

    #[cfg(target_arch = "x86_64")]
    if !CLIP && prediction.width == 16 && std::is_x86_feature_detected!("avx2") {
        // SAFETY: Runtime detection proves AVX2 support. The non-clipping
        // path is selected only after validating both chroma source planes.
        unsafe {
            predict_chroma_bilinear_avx2(
                prediction,
                cb,
                cr,
                width,
                reference_x as usize,
                reference_y as usize,
                x_fraction,
                y_fraction,
            );
        }
        return;
    }

    #[cfg(target_arch = "x86_64")]
    if !CLIP && matches!(prediction.width, 8 | 16) {
        // SAFETY: SSE2 is part of the x86_64 baseline. The non-clipping path
        // is selected only after validating the current and next chroma rows,
        // and luma widths 8/16 map to complete chroma vectors of 4/8 samples.
        unsafe {
            predict_chroma_bilinear_sse2(
                prediction,
                cb,
                cr,
                width,
                reference_x as usize,
                reference_y as usize,
                x_fraction,
                y_fraction,
            );
        }
        return;
    }

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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[allow(clippy::too_many_arguments)]
unsafe fn predict_chroma_pair_sse2(
    prediction: &mut InterPrediction420,
    cb: &[u8],
    cr: &[u8],
    stride: usize,
    reference_x: usize,
    reference_y: usize,
    x_fraction: u8,
    y_fraction: u8,
) {
    use std::arch::x86_64::{
        __m128i, _mm_add_epi16, _mm_cvtsi32_si128, _mm_cvtsi128_si32, _mm_mullo_epi16,
        _mm_packus_epi16, _mm_set1_epi16, _mm_setzero_si128, _mm_srli_epi16, _mm_srli_si128,
        _mm_unpacklo_epi8, _mm_unpacklo_epi32,
    };

    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn load_pair(cb: *const u8, cr: *const u8) -> __m128i {
        // SAFETY: The caller validated four samples from each plane. Unaligned
        // integer loads have no alignment requirement.
        let cb = unsafe { std::ptr::read_unaligned(cb.cast::<i32>()) };
        // SAFETY: See above.
        let cr = unsafe { std::ptr::read_unaligned(cr.cast::<i32>()) };
        _mm_unpacklo_epi8(
            _mm_unpacklo_epi32(_mm_cvtsi32_si128(cb), _mm_cvtsi32_si128(cr)),
            _mm_setzero_si128(),
        )
    }

    debug_assert_eq!(prediction.width, 8);
    let x = i16::from(x_fraction);
    let y = i16::from(y_fraction);
    let w00 = _mm_set1_epi16((8 - x) * (8 - y));
    let w10 = _mm_set1_epi16(x * (8 - y));
    let w01 = _mm_set1_epi16((8 - x) * y);
    let w11 = _mm_set1_epi16(x * y);
    let rounding = _mm_set1_epi16(32);
    let zero = _mm_setzero_si128();
    for output_y in 0..usize::from(prediction.height / 2) {
        let offset = (reference_y + output_y) * stride + reference_x;
        let cb_base = cb.as_ptr().wrapping_add(offset);
        let cr_base = cr.as_ptr().wrapping_add(offset);
        // SAFETY: Interior validation includes the current and next rows plus
        // one sample to the right in both planes.
        let a = unsafe { load_pair(cb_base, cr_base) };
        let weighted = if x_fraction == 0 {
            // SAFETY: See above.
            let c =
                unsafe { load_pair(cb_base.wrapping_add(stride), cr_base.wrapping_add(stride)) };
            _mm_add_epi16(_mm_mullo_epi16(a, w00), _mm_mullo_epi16(c, w01))
        } else if y_fraction == 0 {
            // SAFETY: See above.
            let b = unsafe { load_pair(cb_base.wrapping_add(1), cr_base.wrapping_add(1)) };
            _mm_add_epi16(_mm_mullo_epi16(a, w00), _mm_mullo_epi16(b, w10))
        } else {
            // SAFETY: See above.
            let b = unsafe { load_pair(cb_base.wrapping_add(1), cr_base.wrapping_add(1)) };
            // SAFETY: See above.
            let c =
                unsafe { load_pair(cb_base.wrapping_add(stride), cr_base.wrapping_add(stride)) };
            // SAFETY: See above.
            let d = unsafe {
                load_pair(
                    cb_base.wrapping_add(stride + 1),
                    cr_base.wrapping_add(stride + 1),
                )
            };
            _mm_add_epi16(
                _mm_add_epi16(_mm_mullo_epi16(a, w00), _mm_mullo_epi16(b, w10)),
                _mm_add_epi16(_mm_mullo_epi16(c, w01), _mm_mullo_epi16(d, w11)),
            )
        };
        let packed = _mm_packus_epi16(_mm_srli_epi16::<6>(_mm_add_epi16(weighted, rounding)), zero);
        // SAFETY: Each destination row contains four writable bytes, and
        // unaligned integer stores have no alignment requirement.
        unsafe {
            std::ptr::write_unaligned(
                prediction.cb[output_y].as_mut_ptr().cast::<i32>(),
                _mm_cvtsi128_si32(packed),
            );
            std::ptr::write_unaligned(
                prediction.cr[output_y].as_mut_ptr().cast::<i32>(),
                _mm_cvtsi128_si32(_mm_srli_si128::<4>(packed)),
            );
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn predict_chroma_bilinear_avx2(
    prediction: &mut InterPrediction420,
    cb: &[u8],
    cr: &[u8],
    stride: usize,
    reference_x: usize,
    reference_y: usize,
    x_fraction: u8,
    y_fraction: u8,
) {
    use std::arch::x86_64::{
        __m128i, __m256i, _mm_loadl_epi64, _mm_storel_epi64, _mm_unpacklo_epi64, _mm256_add_epi16,
        _mm256_castsi256_si128, _mm256_cvtepu8_epi16, _mm256_extracti128_si256, _mm256_mullo_epi16,
        _mm256_packus_epi16, _mm256_set1_epi16, _mm256_setzero_si256, _mm256_srli_epi16,
    };

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn load_pair(cb: *const u8, cr: *const u8) -> __m256i {
        // SAFETY: The caller validated eight samples from both planes.
        let cb = unsafe { _mm_loadl_epi64(cb.cast::<__m128i>()) };
        // SAFETY: See above.
        let cr = unsafe { _mm_loadl_epi64(cr.cast::<__m128i>()) };
        _mm256_cvtepu8_epi16(_mm_unpacklo_epi64(cb, cr))
    }

    debug_assert_eq!(prediction.width, 16);
    let x = i16::from(x_fraction);
    let y = i16::from(y_fraction);
    let w00 = _mm256_set1_epi16((8 - x) * (8 - y));
    let w10 = _mm256_set1_epi16(x * (8 - y));
    let w01 = _mm256_set1_epi16((8 - x) * y);
    let w11 = _mm256_set1_epi16(x * y);
    let rounding = _mm256_set1_epi16(32);
    let zero = _mm256_setzero_si256();
    for output_y in 0..usize::from(prediction.height / 2) {
        let offset = (reference_y + output_y) * stride + reference_x;
        let cb_base = cb.as_ptr().wrapping_add(offset);
        let cr_base = cr.as_ptr().wrapping_add(offset);
        // SAFETY: Interior validation includes the current and next rows plus
        // any right/bottom neighbour required by a non-zero fraction.
        let a = unsafe { load_pair(cb_base, cr_base) };
        let weighted = if x_fraction == 0 {
            // SAFETY: A non-zero vertical fraction makes the next row part of
            // the validated interpolation window.
            let c =
                unsafe { load_pair(cb_base.wrapping_add(stride), cr_base.wrapping_add(stride)) };
            _mm256_add_epi16(_mm256_mullo_epi16(a, w00), _mm256_mullo_epi16(c, w01))
        } else if y_fraction == 0 {
            // SAFETY: A non-zero horizontal fraction makes the right
            // neighbour part of the validated interpolation window.
            let b = unsafe { load_pair(cb_base.wrapping_add(1), cr_base.wrapping_add(1)) };
            _mm256_add_epi16(_mm256_mullo_epi16(a, w00), _mm256_mullo_epi16(b, w10))
        } else {
            // SAFETY: Both non-zero fractions make the right, bottom, and
            // bottom-right neighbours part of the validated window.
            let b = unsafe { load_pair(cb_base.wrapping_add(1), cr_base.wrapping_add(1)) };
            // SAFETY: See above.
            let c =
                unsafe { load_pair(cb_base.wrapping_add(stride), cr_base.wrapping_add(stride)) };
            // SAFETY: See above.
            let d = unsafe {
                load_pair(
                    cb_base.wrapping_add(stride + 1),
                    cr_base.wrapping_add(stride + 1),
                )
            };
            _mm256_add_epi16(
                _mm256_add_epi16(_mm256_mullo_epi16(a, w00), _mm256_mullo_epi16(b, w10)),
                _mm256_add_epi16(_mm256_mullo_epi16(c, w01), _mm256_mullo_epi16(d, w11)),
            )
        };
        let packed = _mm256_packus_epi16(
            _mm256_srli_epi16::<6>(_mm256_add_epi16(weighted, rounding)),
            zero,
        );
        // SAFETY: Each destination row contains eight writable bytes.
        unsafe {
            _mm_storel_epi64(
                prediction.cb[output_y].as_mut_ptr().cast::<__m128i>(),
                _mm256_castsi256_si128(packed),
            );
            _mm_storel_epi64(
                prediction.cr[output_y].as_mut_ptr().cast::<__m128i>(),
                _mm256_extracti128_si256::<1>(packed),
            );
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[allow(clippy::too_many_arguments)]
unsafe fn predict_chroma_bilinear_sse2(
    prediction: &mut InterPrediction420,
    cb: &[u8],
    cr: &[u8],
    stride: usize,
    reference_x: usize,
    reference_y: usize,
    x_fraction: u8,
    y_fraction: u8,
) {
    use std::arch::x86_64::{
        __m128i, _mm_add_epi16, _mm_cvtsi32_si128, _mm_cvtsi128_si32, _mm_loadl_epi64,
        _mm_mullo_epi16, _mm_packus_epi16, _mm_set1_epi16, _mm_setzero_si128, _mm_srli_epi16,
        _mm_storel_epi64, _mm_unpacklo_epi8,
    };

    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn load(ptr: *const u8, width: usize) -> __m128i {
        if width == 8 {
            // SAFETY: The caller validated eight source samples.
            unsafe { _mm_loadl_epi64(ptr.cast::<__m128i>()) }
        } else {
            debug_assert_eq!(width, 4);
            // SAFETY: The caller validated four source samples. The unaligned
            // integer load has no alignment requirement.
            let value = unsafe { std::ptr::read_unaligned(ptr.cast::<i32>()) };
            _mm_cvtsi32_si128(value)
        }
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    #[allow(clippy::too_many_arguments)]
    unsafe fn predict_plane(
        output: &mut [[u8; 8]; 8],
        plane: &[u8],
        stride: usize,
        reference_x: usize,
        reference_y: usize,
        output_width: usize,
        output_height: usize,
        weights: [i16; 4],
    ) {
        let zero = _mm_setzero_si128();
        let w00 = _mm_set1_epi16(weights[0]);
        let w10 = _mm_set1_epi16(weights[1]);
        let w01 = _mm_set1_epi16(weights[2]);
        let w11 = _mm_set1_epi16(weights[3]);
        let rounding = _mm_set1_epi16(32);
        for (output_y, output_row) in output.iter_mut().enumerate().take(output_height) {
            let base = plane
                .as_ptr()
                .wrapping_add((reference_y + output_y) * stride + reference_x);
            // SAFETY: Interior validation always includes the current row.
            let a = _mm_unpacklo_epi8(unsafe { load(base, output_width) }, zero);
            let weighted = if weights[1] == 0 && weights[3] == 0 {
                // SAFETY: A non-zero vertical fraction makes the next row
                // part of the validated interpolation window.
                let c = _mm_unpacklo_epi8(
                    unsafe { load(base.wrapping_add(stride), output_width) },
                    zero,
                );
                _mm_add_epi16(_mm_mullo_epi16(a, w00), _mm_mullo_epi16(c, w01))
            } else if weights[2] == 0 && weights[3] == 0 {
                // SAFETY: A non-zero horizontal fraction makes the right
                // neighbour part of the validated interpolation window.
                let b =
                    _mm_unpacklo_epi8(unsafe { load(base.wrapping_add(1), output_width) }, zero);
                _mm_add_epi16(_mm_mullo_epi16(a, w00), _mm_mullo_epi16(b, w10))
            } else {
                // SAFETY: Both fractions are non-zero, so the validated
                // window includes right, bottom, and bottom-right neighbours.
                let b =
                    _mm_unpacklo_epi8(unsafe { load(base.wrapping_add(1), output_width) }, zero);
                // SAFETY: See above.
                let c = _mm_unpacklo_epi8(
                    unsafe { load(base.wrapping_add(stride), output_width) },
                    zero,
                );
                // SAFETY: See above.
                let d = _mm_unpacklo_epi8(
                    unsafe { load(base.wrapping_add(stride + 1), output_width) },
                    zero,
                );
                _mm_add_epi16(
                    _mm_add_epi16(_mm_mullo_epi16(a, w00), _mm_mullo_epi16(b, w10)),
                    _mm_add_epi16(_mm_mullo_epi16(c, w01), _mm_mullo_epi16(d, w11)),
                )
            };
            let packed =
                _mm_packus_epi16(_mm_srli_epi16::<6>(_mm_add_epi16(weighted, rounding)), zero);
            if output_width == 8 {
                // SAFETY: The output row contains eight writable bytes.
                unsafe {
                    _mm_storel_epi64(output_row.as_mut_ptr().cast::<__m128i>(), packed);
                }
            } else {
                // SAFETY: The output row contains four writable bytes, and
                // unaligned integer stores have no alignment requirement.
                unsafe {
                    std::ptr::write_unaligned(
                        output_row.as_mut_ptr().cast::<i32>(),
                        _mm_cvtsi128_si32(packed),
                    );
                }
            }
        }
    }

    let x = i16::from(x_fraction);
    let y = i16::from(y_fraction);
    let weights = [(8 - x) * (8 - y), x * (8 - y), (8 - x) * y, x * y];
    let output_width = usize::from(prediction.width / 2);
    let output_height = usize::from(prediction.height / 2);
    // SAFETY: The caller validated the two source rectangles.
    unsafe {
        predict_plane(
            &mut prediction.cb,
            cb,
            stride,
            reference_x,
            reference_y,
            output_width,
            output_height,
            weights,
        );
        predict_plane(
            &mut prediction.cr,
            cr,
            stride,
            reference_x,
            reference_y,
            output_width,
            output_height,
            weights,
        );
    }
}

#[inline(always)]
pub(crate) unsafe fn copy_fixed_row(destination: *mut u8, source: *const u8, width: usize) {
    use std::ptr::{read_unaligned, write_unaligned};

    match width {
        2 => {
            // SAFETY: the caller guarantees two readable and writable bytes.
            unsafe { write_unaligned(destination.cast::<u16>(), read_unaligned(source.cast())) };
        }
        4 => {
            // SAFETY: the caller guarantees four readable and writable bytes.
            unsafe { write_unaligned(destination.cast::<u32>(), read_unaligned(source.cast())) };
        }
        8 => {
            // SAFETY: the caller guarantees eight readable and writable bytes.
            unsafe { write_unaligned(destination.cast::<u64>(), read_unaligned(source.cast())) };
        }
        16 => {
            // SAFETY: the caller guarantees sixteen readable and writable bytes.
            unsafe { write_unaligned(destination.cast::<u128>(), read_unaligned(source.cast())) };
        }
        _ => unreachable!("validated 4:2:0 partition rows have fixed power-of-two widths"),
    }
}

#[inline(always)]
unsafe fn copy_fixed_rows<const ROW_WIDTH: usize, const DESTINATION_STRIDE: usize>(
    mut destination: *mut u8,
    mut source: *const u8,
    source_stride: usize,
    mut rows: usize,
) {
    debug_assert!(matches!(ROW_WIDTH, 2 | 4 | 8 | 16));
    debug_assert!(ROW_WIDTH <= DESTINATION_STRIDE);
    debug_assert!(matches!(rows, 2 | 4 | 8 | 16));
    while rows >= 4 {
        // SAFETY: The caller validates every source and destination row.
        // Four independent fixed-size copies expose row-level memory
        // parallelism without repeating the runtime width dispatch.
        unsafe {
            copy_fixed_row(destination, source, ROW_WIDTH);
            copy_fixed_row(
                destination.add(DESTINATION_STRIDE),
                source.add(source_stride),
                ROW_WIDTH,
            );
            copy_fixed_row(
                destination.add(2 * DESTINATION_STRIDE),
                source.add(2 * source_stride),
                ROW_WIDTH,
            );
            copy_fixed_row(
                destination.add(3 * DESTINATION_STRIDE),
                source.add(3 * source_stride),
                ROW_WIDTH,
            );
            destination = destination.add(4 * DESTINATION_STRIDE);
            source = source.add(4 * source_stride);
        }
        rows -= 4;
    }
    if rows == 2 {
        // SAFETY: Chroma partitions may leave exactly two validated rows.
        unsafe {
            copy_fixed_row(destination, source, ROW_WIDTH);
            copy_fixed_row(
                destination.add(DESTINATION_STRIDE),
                source.add(source_stride),
                ROW_WIDTH,
            );
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
    fn integer_macroblock_copy_matches_normative_prediction_and_falls_back() {
        let mut picture = Yuv420Picture::new(Size {
            width: 48,
            height: 48,
        })
        .unwrap();
        let (luma, cb, cr) = picture.planes_mut();
        for (index, sample) in luma.iter_mut().enumerate() {
            *sample = (index * 73 + index / 17 * 29) as u8;
        }
        for (index, sample) in cb.iter_mut().enumerate() {
            *sample = (index * 31 + 7) as u8;
        }
        for (index, sample) in cr.iter_mut().enumerate() {
            *sample = (index * 43 + 19) as u8;
        }
        let partition = ResolvedPPartition {
            x: 0,
            y: 0,
            width: 16,
            height: 16,
            reference_index: 0,
            motion_vector: MotionVector { x: 8, y: -8 },
        };
        let expected = picture.predict_inter_420(1, 1, partition).unwrap();
        let mut luma = [[0xa5; 16]; 16];
        let mut cb = [[0xa5; 8]; 8];
        let mut cr = [[0xa5; 8]; 8];
        assert!(
            picture
                .try_copy_integer_macroblock_420_into(
                    1,
                    1,
                    partition.motion_vector,
                    &mut luma,
                    &mut cb,
                    &mut cr,
                )
                .unwrap()
        );
        assert_eq!(luma, expected.luma);
        assert_eq!(cb, expected.cb);
        assert_eq!(cr, expected.cr);

        assert!(
            !picture
                .try_copy_integer_macroblock_420_into(
                    1,
                    1,
                    MotionVector { x: 4, y: 0 },
                    &mut luma,
                    &mut cb,
                    &mut cr,
                )
                .unwrap()
        );
        assert!(
            !picture
                .try_copy_integer_macroblock_420_into(
                    0,
                    0,
                    MotionVector { x: -8, y: 0 },
                    &mut luma,
                    &mut cb,
                    &mut cr,
                )
                .unwrap()
        );
    }

    #[test]
    fn fixed_row_copy_matches_slice_copy_at_unaligned_addresses() {
        let source: Vec<u8> = (0..32).map(|index| (index * 37 + 11) as u8).collect();
        for width in [2usize, 4, 8, 16] {
            let mut actual = [0xa5; 24];
            let mut expected = actual;
            expected[3..3 + width].copy_from_slice(&source[1..1 + width]);
            // SAFETY: both explicitly selected ranges contain `width` bytes;
            // offsets one and three exercise unaligned loads and stores.
            unsafe {
                copy_fixed_row(actual.as_mut_ptr().add(3), source.as_ptr().add(1), width);
            }
            assert_eq!(actual, expected, "width={width}");
        }
    }

    #[test]
    fn fixed_rectangle_copy_matches_row_slices_for_all_partition_shapes() {
        fn check<const WIDTH: usize>(rows: usize) {
            const SOURCE_STRIDE: usize = 23;
            const DESTINATION_STRIDE: usize = 20;
            let source: Vec<u8> = (0..1 + 16 * SOURCE_STRIDE)
                .map(|index| (index * 37 + 11) as u8)
                .collect();
            let mut actual = [0xa5; 3 + 16 * DESTINATION_STRIDE];
            let mut expected = actual;
            for row in 0..rows {
                let source_start = 1 + row * SOURCE_STRIDE;
                let destination_start = 3 + row * DESTINATION_STRIDE;
                expected[destination_start..destination_start + WIDTH]
                    .copy_from_slice(&source[source_start..source_start + WIDTH]);
            }
            // SAFETY: Every selected source and destination row contains
            // `WIDTH` bytes. Offsets one and three exercise unaligned access.
            unsafe {
                copy_fixed_rows::<WIDTH, DESTINATION_STRIDE>(
                    actual.as_mut_ptr().add(3),
                    source.as_ptr().add(1),
                    SOURCE_STRIDE,
                    rows,
                );
            }
            assert_eq!(actual, expected, "width={WIDTH} rows={rows}");
        }

        for rows in [2, 4, 8, 16] {
            check::<2>(rows);
            check::<4>(rows);
            check::<8>(rows);
            check::<16>(rows);
        }
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

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn axis_sse2_matches_per_sample_luma_interpolation() {
        let plane: Vec<u8> = (0..40 * 40)
            .map(|index| ((index * 73 + index / 11 * 29) & 0xff) as u8)
            .collect();
        for size in [8u8, 16] {
            for (x_fraction, y_fraction) in [(1, 0), (2, 0), (3, 0), (0, 1), (0, 2), (0, 3)] {
                let mut prediction = InterPrediction420::empty();
                prediction.width = size;
                prediction.height = size;
                // SAFETY: SSE2 is part of the x86_64 baseline, and the
                // reference position leaves the full six-tap window inside
                // this 40x40 plane.
                unsafe {
                    predict_luma_axis_sse2(
                        &mut prediction,
                        &plane,
                        40,
                        8,
                        8,
                        x_fraction,
                        y_fraction,
                    );
                }
                for y in 0..usize::from(size) {
                    for x in 0..usize::from(size) {
                        assert_eq!(
                            prediction.luma[y][x],
                            interpolate_luma_inner::<false>(
                                &plane,
                                40,
                                40,
                                8 + x as i32,
                                8 + y as i32,
                                x_fraction,
                                y_fraction,
                            ),
                            "size={size} fraction=({x_fraction},{y_fraction}) x={x} y={y}"
                        );
                    }
                }
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn axis_avx2_matches_per_sample_luma_interpolation() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let plane: Vec<u8> = (0..40 * 40)
            .map(|index| ((index * 73 + index / 11 * 29) & 0xff) as u8)
            .collect();
        for (x_fraction, y_fraction) in [(1, 0), (2, 0), (3, 0), (0, 1), (0, 2), (0, 3)] {
            let mut prediction = InterPrediction420::empty();
            prediction.width = 16;
            prediction.height = 16;
            // SAFETY: Runtime detection proves AVX2 support, and the
            // reference position leaves the complete six-tap window inside
            // this 40x40 plane.
            unsafe {
                predict_luma_axis_avx2(&mut prediction, &plane, 40, 8, 8, x_fraction, y_fraction);
            }
            for y in 0..16 {
                for x in 0..16 {
                    assert_eq!(
                        prediction.luma[y][x],
                        interpolate_luma_inner::<false>(
                            &plane,
                            40,
                            40,
                            8 + x as i32,
                            8 + y as i32,
                            x_fraction,
                            y_fraction,
                        ),
                        "fraction=({x_fraction},{y_fraction}) x={x} y={y}"
                    );
                }
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn two_dimensional_sse2_matches_per_sample_luma_interpolation() {
        let plane: Vec<u8> = (0..40 * 40)
            .map(|index| ((index * 73 + index / 11 * 29) & 0xff) as u8)
            .collect();
        for width in [8u8, 16] {
            for height in [4u8, 8, 16] {
                for y_fraction in 1..=3 {
                    for x_fraction in 1..=3 {
                        let mut prediction = InterPrediction420::empty();
                        prediction.width = width;
                        prediction.height = height;
                        // SAFETY: SSE2 is part of the x86_64 baseline, and
                        // the reference position leaves every six-tap row and
                        // column inside this 40x40 plane.
                        unsafe {
                            predict_luma_two_dimensional_sse2(
                                &mut prediction,
                                &plane,
                                40,
                                8,
                                8,
                                x_fraction,
                                y_fraction,
                            );
                        }
                        for y in 0..usize::from(height) {
                            for x in 0..usize::from(width) {
                                assert_eq!(
                                    prediction.luma[y][x],
                                    interpolate_luma_inner::<false>(
                                        &plane,
                                        40,
                                        40,
                                        8 + x as i32,
                                        8 + y as i32,
                                        x_fraction,
                                        y_fraction,
                                    ),
                                    "size={width}x{height} fraction=({x_fraction},{y_fraction}) x={x} y={y}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn chroma_simd_matches_per_sample_bilinear_interpolation() {
        let cb: Vec<u8> = (0..40 * 40)
            .map(|index| ((index * 37 + index / 13 * 19) & 0xff) as u8)
            .collect();
        let cr: Vec<u8> = (0..40 * 40)
            .map(|index| ((index * 61 + index / 7 * 23) & 0xff) as u8)
            .collect();
        for width in [8u8, 16] {
            for height in [4u8, 8, 16] {
                for y_fraction in 0..8 {
                    for x_fraction in 0..8 {
                        if x_fraction == 0 && y_fraction == 0 {
                            continue;
                        }
                        let mut prediction = InterPrediction420::empty();
                        prediction.width = width;
                        prediction.height = height;
                        // SAFETY: SSE2 is part of the x86_64 baseline, and
                        // the reference rectangle plus right/bottom samples
                        // lies inside both 40x40 planes.
                        unsafe {
                            predict_chroma_bilinear_sse2(
                                &mut prediction,
                                &cb,
                                &cr,
                                40,
                                8,
                                8,
                                x_fraction,
                                y_fraction,
                            );
                        }
                        if width == 8 {
                            let mut paired = InterPrediction420::empty();
                            paired.width = width;
                            paired.height = height;
                            // SAFETY: SSE2 is part of the x86_64 baseline,
                            // and the same interior source rectangle was used
                            // by the per-plane SSE2 oracle above.
                            unsafe {
                                predict_chroma_pair_sse2(
                                    &mut paired,
                                    &cb,
                                    &cr,
                                    40,
                                    8,
                                    8,
                                    x_fraction,
                                    y_fraction,
                                );
                            }
                            assert_eq!(paired, prediction);
                        }
                        if width == 16 && std::is_x86_feature_detected!("avx2") {
                            let mut avx2 = InterPrediction420::empty();
                            avx2.width = width;
                            avx2.height = height;
                            // SAFETY: Runtime detection proves AVX2 support,
                            // and the same interior source rectangle was used
                            // by the SSE2 oracle above.
                            unsafe {
                                predict_chroma_bilinear_avx2(
                                    &mut avx2, &cb, &cr, 40, 8, 8, x_fraction, y_fraction,
                                );
                            }
                            assert_eq!(avx2, prediction);
                        }
                        for y in 0..usize::from(height / 2) {
                            for x in 0..usize::from(width / 2) {
                                let expected_cb = interpolate_chroma_inner::<false>(
                                    &cb,
                                    40,
                                    40,
                                    8 + x as i32,
                                    8 + y as i32,
                                    x_fraction,
                                    y_fraction,
                                );
                                let expected_cr = interpolate_chroma_inner::<false>(
                                    &cr,
                                    40,
                                    40,
                                    8 + x as i32,
                                    8 + y as i32,
                                    x_fraction,
                                    y_fraction,
                                );
                                assert_eq!(
                                    prediction.cb[y][x], expected_cb,
                                    "Cb size={width}x{height} fraction=({x_fraction},{y_fraction}) x={x} y={y}"
                                );
                                assert_eq!(
                                    prediction.cr[y][x], expected_cr,
                                    "Cr size={width}x{height} fraction=({x_fraction},{y_fraction}) x={x} y={y}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_chroma_does_not_read_unused_neighbours_at_plane_edges() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }

        // The horizontal-only case ends on the final plane row; the
        // vertical-only case ends on the final plane column. Reading the
        // unused next row or right neighbour is outside these exact buffers.
        for (stride, rows, x_fraction, y_fraction) in [(9, 4, 1, 0), (8, 5, 0, 1)] {
            let cb = (0..stride * rows)
                .map(|index| ((index * 37 + 11) & 0xff) as u8)
                .collect::<Vec<_>>();
            let cr = (0..stride * rows)
                .map(|index| ((index * 61 + 23) & 0xff) as u8)
                .collect::<Vec<_>>();
            let mut expected = InterPrediction420::empty();
            expected.width = 16;
            expected.height = 8;
            let mut actual = expected.clone();

            // SAFETY: SSE2 is the x86_64 baseline. Each case provides exactly
            // the source neighbour required by its one non-zero fraction.
            unsafe {
                predict_chroma_bilinear_sse2(
                    &mut expected,
                    &cb,
                    &cr,
                    stride,
                    0,
                    0,
                    x_fraction,
                    y_fraction,
                );
            }
            // SAFETY: Runtime detection proves AVX2 support, and the same
            // exact source rectangles validated for the SSE2 oracle apply.
            unsafe {
                predict_chroma_bilinear_avx2(
                    &mut actual,
                    &cb,
                    &cr,
                    stride,
                    0,
                    0,
                    x_fraction,
                    y_fraction,
                );
            }
            assert_eq!(actual, expected);
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
    fn single_axis_filters_remain_fast_at_the_other_picture_edge() {
        let picture = gradient_picture();
        let (luma, cb, cr) = picture.planes();
        for partition in [
            ResolvedPPartition {
                x: 4,
                y: 0,
                width: 8,
                height: 4,
                reference_index: 0,
                motion_vector: MotionVector { x: 1, y: 0 },
            },
            ResolvedPPartition {
                x: 0,
                y: 4,
                width: 8,
                height: 8,
                reference_index: 0,
                motion_vector: MotionVector { x: 0, y: 1 },
            },
        ] {
            let prediction = picture.predict_inter_420(0, 0, partition).unwrap();
            let current_x = i32::from(partition.x);
            let current_y = i32::from(partition.y);
            let reference_luma_x = current_x + i32::from(partition.motion_vector.x).div_euclid(4);
            let reference_luma_y = current_y + i32::from(partition.motion_vector.y).div_euclid(4);
            let luma_fraction_x = i32::from(partition.motion_vector.x).rem_euclid(4) as u8;
            let luma_fraction_y = i32::from(partition.motion_vector.y).rem_euclid(4) as u8;
            assert!(interpolation_window_is_inside(
                reference_luma_x,
                reference_luma_y,
                usize::from(partition.width),
                usize::from(partition.height),
                16,
                16,
                if luma_fraction_x == 0 { 0 } else { 2 },
                if luma_fraction_x == 0 { 0 } else { 3 },
                if luma_fraction_y == 0 { 0 } else { 2 },
                if luma_fraction_y == 0 { 0 } else { 3 },
            ));
            for y in 0..usize::from(partition.height) {
                for x in 0..usize::from(partition.width) {
                    assert_eq!(
                        prediction.luma[y][x],
                        interpolate_luma_inner::<true>(
                            luma,
                            16,
                            16,
                            reference_luma_x + x as i32,
                            reference_luma_y + y as i32,
                            luma_fraction_x,
                            luma_fraction_y,
                        )
                    );
                }
            }

            let reference_chroma_x =
                current_x / 2 + i32::from(partition.motion_vector.x).div_euclid(8);
            let reference_chroma_y =
                current_y / 2 + i32::from(partition.motion_vector.y).div_euclid(8);
            let chroma_fraction_x = i32::from(partition.motion_vector.x).rem_euclid(8) as u8;
            let chroma_fraction_y = i32::from(partition.motion_vector.y).rem_euclid(8) as u8;
            assert!(interpolation_window_is_inside(
                reference_chroma_x,
                reference_chroma_y,
                usize::from(partition.width / 2),
                usize::from(partition.height / 2),
                8,
                8,
                0,
                u8::from(chroma_fraction_x != 0),
                0,
                u8::from(chroma_fraction_y != 0),
            ));
            for y in 0..usize::from(partition.height / 2) {
                for x in 0..usize::from(partition.width / 2) {
                    let sample_x = reference_chroma_x + x as i32;
                    let sample_y = reference_chroma_y + y as i32;
                    assert_eq!(
                        prediction.cb[y][x],
                        interpolate_chroma_inner::<true>(
                            cb,
                            8,
                            8,
                            sample_x,
                            sample_y,
                            chroma_fraction_x,
                            chroma_fraction_y,
                        )
                    );
                    assert_eq!(
                        prediction.cr[y][x],
                        interpolate_chroma_inner::<true>(
                            cr,
                            8,
                            8,
                            sample_x,
                            sample_y,
                            chroma_fraction_x,
                            chroma_fraction_y,
                        )
                    );
                }
            }
        }
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
