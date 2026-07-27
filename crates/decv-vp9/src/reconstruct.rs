use std::sync::Arc;

use crate::{
    block::{IntraMode, TransformSize, TransformType},
    inverse_transform::{inverse_adst, inverse_dct},
};

/// Reconstructed 8-bit 4:2:0 picture.
#[derive(Debug, Clone)]
pub struct IntraPicture {
    width: usize,
    height: usize,
    origin_x: usize,
    storage_width: usize,
    strides: [usize; 3],
    planes: [Arc<[u8]>; 3],
}

impl IntraPicture {
    pub(crate) fn new(width: usize, height: usize) -> Self {
        let chroma_width = width.div_ceil(2);
        let chroma_height = height.div_ceil(2);
        Self {
            width,
            height,
            origin_x: 0,
            storage_width: width,
            strides: [width, chroma_width, chroma_width],
            planes: [
                vec![0; width * height].into(),
                vec![0; chroma_width * chroma_height].into(),
                vec![0; chroma_width * chroma_height].into(),
            ],
        }
    }

    pub(crate) fn new_strip(
        width: usize,
        height: usize,
        origin_x: usize,
        storage_width: usize,
    ) -> Self {
        debug_assert!(origin_x <= width && storage_width <= width - origin_x);
        debug_assert!(origin_x.is_multiple_of(2));
        let chroma_width = storage_width.div_ceil(2);
        let chroma_height = height.div_ceil(2);
        Self {
            width,
            height,
            origin_x,
            storage_width,
            strides: [storage_width, chroma_width, chroma_width],
            planes: [
                vec![0; storage_width * height].into(),
                vec![0; chroma_width * chroma_height].into(),
                vec![0; chroma_width * chroma_height].into(),
            ],
        }
    }

    pub(crate) fn copy_strip_from(&mut self, strip: &Self) {
        debug_assert_eq!(self.width, strip.width);
        debug_assert_eq!(self.height, strip.height);
        debug_assert_eq!(self.origin_x, 0);
        for plane in 0..3 {
            let subsampling = usize::from(plane != 0);
            let origin_x = strip.origin_x >> subsampling;
            let width = strip.storage_width.div_ceil(1 << subsampling);
            let height = self.height.div_ceil(1 << subsampling);
            let source_stride = strip.strides[plane];
            let target_stride = self.strides[plane];
            let source = &strip.planes[plane];
            let target = Arc::make_mut(&mut self.planes[plane]);
            for row in 0..height {
                let source_start = row * source_stride;
                let target_start = row * target_stride + origin_x;
                target[target_start..target_start + width]
                    .copy_from_slice(&source[source_start..source_start + width]);
            }
        }
    }

    #[inline]
    pub fn width(&self) -> usize {
        self.width
    }

    #[inline]
    pub fn height(&self) -> usize {
        self.height
    }

    #[inline]
    pub fn stride(&self, plane: usize) -> usize {
        self.strides[plane]
    }

    #[inline]
    pub fn plane(&self, plane: usize) -> &[u8] {
        &self.planes[plane]
    }

    #[inline]
    pub(crate) fn shared_plane(&self, plane: usize) -> Arc<[u8]> {
        Arc::clone(&self.planes[plane])
    }

    pub(crate) fn planes_mut(&mut self) -> [&mut [u8]; 3] {
        let [y, u, v] = &mut self.planes;
        [Arc::make_mut(y), Arc::make_mut(u), Arc::make_mut(v)]
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn predict(
        &mut self,
        plane: usize,
        x: usize,
        y: usize,
        size: usize,
        mode: IntraMode,
        tile_left: usize,
        tile_top: usize,
        right_available: bool,
    ) {
        let width = if plane == 0 {
            self.storage_width
        } else {
            self.storage_width.div_ceil(2)
        };
        let height = if plane == 0 {
            self.height
        } else {
            self.height.div_ceil(2)
        };
        let subsampling = usize::from(plane != 0);
        let plane_origin_x = self.origin_x >> subsampling;
        let x = x
            .checked_sub(plane_origin_x)
            .expect("prediction belongs to this picture strip");
        let tile_left = tile_left
            .checked_sub(plane_origin_x)
            .expect("tile begins inside this picture strip");
        let have_top = y > tile_top;
        let have_left = x > tile_left;
        debug_assert!(size <= 32);
        let mut above = [127u8; 64];
        let mut left = [129u8; 32];
        let stride = self.strides[plane];
        let pixels = &self.planes[plane];

        if have_top {
            let available = width.saturating_sub(x).min(size * 2);
            for index in 0..available {
                above[index] = pixels[(y - 1) * stride + x + index];
            }
            let extension = available.saturating_sub(1);
            for index in available..size * 2 {
                above[index] = above[extension];
            }
            if !right_available || size != 4 {
                let edge = above[size.saturating_sub(1)];
                above[size..].fill(edge);
            }
        }
        if have_left {
            let available = height.saturating_sub(y).min(size);
            for index in 0..available {
                left[index] = pixels[(y + index) * stride + x - 1];
            }
            let extension = available.saturating_sub(1);
            for index in available..size {
                left[index] = left[extension];
            }
        }
        let top_left = match (have_top, have_left) {
            (true, true) => pixels[(y - 1) * stride + x - 1],
            (true, false) => 129,
            (false, _) => 127,
        };

        let mut prediction = [0u8; 32 * 32];
        intra_predict(
            &mut prediction[..size * size],
            size,
            mode,
            &above[..size * 2],
            &left[..size],
            top_left,
            have_top,
            have_left,
        );
        let pixels = Arc::make_mut(&mut self.planes[plane]);
        let visible_width = size.min(width.saturating_sub(x));
        let visible_height = size.min(height.saturating_sub(y));
        for row in 0..visible_height {
            let target = (y + row) * stride + x;
            pixels[target..target + visible_width]
                .copy_from_slice(&prediction[row * size..row * size + visible_width]);
        }
    }

    pub(crate) fn add_residual(
        &mut self,
        plane: usize,
        x: usize,
        y: usize,
        transform_size: TransformSize,
        transform_type: TransformType,
        coefficients: &[i32],
    ) {
        let size = 4usize << transform_size as usize;
        let width = if plane == 0 {
            self.storage_width
        } else {
            self.storage_width.div_ceil(2)
        };
        let height = if plane == 0 {
            self.height
        } else {
            self.height.div_ceil(2)
        };
        let plane_origin_x = self.origin_x >> usize::from(plane != 0);
        let x = x
            .checked_sub(plane_origin_x)
            .expect("residual belongs to this picture strip");
        if transform_type == TransformType::DctDct
            && coefficients[1..]
                .iter()
                .all(|&coefficient| coefficient == 0)
        {
            if coefficients[0] == 0 {
                return;
            }
            const DCT_DC_BASIS: i64 = 11_585;
            let first = (i64::from(coefficients[0]) * DCT_DC_BASIS + (1 << 13)) >> 14;
            let second = (first * DCT_DC_BASIS + (1 << 13)) >> 14;
            let final_shift = match transform_size {
                TransformSize::Tx4x4 => 4,
                TransformSize::Tx8x8 => 5,
                TransformSize::Tx16x16 | TransformSize::Tx32x32 => 6,
            };
            let residual = round_power_of_two(second as i32, final_shift);
            let visible_width = size.min(width.saturating_sub(x));
            let visible_height = size.min(height.saturating_sub(y));
            let stride = self.strides[plane];
            let pixels = Arc::make_mut(&mut self.planes[plane]);
            for row in 0..visible_height {
                let start = (y + row) * stride + x;
                for pixel in &mut pixels[start..start + visible_width] {
                    *pixel = (i32::from(*pixel) + residual).clamp(0, 255) as u8;
                }
            }
            return;
        }
        let mut intermediate = [0i32; 32 * 32];
        let mut input = [0i32; 32];
        let mut output = [0i32; 32];
        let (rows, columns) = transform_axes(transform_type);

        for row in 0..size {
            input[..size].copy_from_slice(&coefficients[row * size..(row + 1) * size]);
            inverse_1d_sparse(&input, &mut output, size, rows);
            intermediate[row * size..(row + 1) * size].copy_from_slice(&output[..size]);
        }

        let final_shift = match transform_size {
            TransformSize::Tx4x4 => 4,
            TransformSize::Tx8x8 => 5,
            TransformSize::Tx16x16 | TransformSize::Tx32x32 => 6,
        };
        let stride = self.strides[plane];
        let pixels = Arc::make_mut(&mut self.planes[plane]);
        for column in 0..size.min(width.saturating_sub(x)) {
            for row in 0..size {
                input[row] = intermediate[row * size + column];
            }
            inverse_1d_sparse(&input, &mut output, size, columns);
            for (row, &value) in output
                .iter()
                .enumerate()
                .take(size.min(height.saturating_sub(y)))
            {
                let residual = round_power_of_two(value, final_shift);
                let index = (y + row) * stride + x + column;
                pixels[index] = (i32::from(pixels[index]) + residual).clamp(0, 255) as u8;
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn predict_inter(
        &mut self,
        reference: &Self,
        plane: usize,
        x: usize,
        y: usize,
        width: usize,
        height: usize,
        motion_row_q4: i32,
        motion_column_q4: i32,
        kernel: &[i16; 128],
        average: bool,
    ) {
        let reference_plane_width = if plane == 0 {
            reference.width
        } else {
            reference.width.div_ceil(2)
        };
        let reference_plane_height = if plane == 0 {
            reference.height
        } else {
            reference.height.div_ceil(2)
        };
        let plane_width = if plane == 0 {
            self.width
        } else {
            self.width.div_ceil(2)
        };
        let plane_height = if plane == 0 {
            self.height
        } else {
            self.height.div_ceil(2)
        };
        let plane_origin_x = self.origin_x >> usize::from(plane != 0);
        let target_x = x
            .checked_sub(plane_origin_x)
            .expect("inter prediction belongs to this picture strip");
        let target_width = if plane == 0 {
            self.storage_width
        } else {
            self.storage_width.div_ceil(2)
        };
        let output_width = width
            .min(plane_width.saturating_sub(x))
            .min(target_width.saturating_sub(target_x));
        let output_height = height.min(plane_height.saturating_sub(y));
        if output_width == 0 || output_height == 0 {
            return;
        }
        if reference.width != self.width || reference.height != self.height {
            self.predict_inter_scaled(
                reference,
                plane,
                x,
                y,
                target_x,
                output_width,
                output_height,
                motion_row_q4,
                motion_column_q4,
                kernel,
                average,
            );
            return;
        }

        let integer_x = motion_column_q4 >> 4;
        let integer_y = motion_row_q4 >> 4;
        let phase_x = (motion_column_q4 & 15) as usize;
        let phase_y = (motion_row_q4 & 15) as usize;
        let filter_x = &kernel[phase_x * 8..phase_x * 8 + 8];
        let filter_y = &kernel[phase_y * 8..phase_y * 8 + 8];
        let source_stride = reference.strides[plane];
        let source = &reference.planes[plane];
        let sample = |source_x: i32, source_y: i32| -> u8 {
            let source_x =
                source_x.clamp(0, reference_plane_width.saturating_sub(1) as i32) as usize;
            let source_y =
                source_y.clamp(0, reference_plane_height.saturating_sub(1) as i32) as usize;
            source[source_y * source_stride + source_x]
        };
        let origin_x = x as i32 + integer_x;
        let origin_y = y as i32 + integer_y;
        let target_stride = self.strides[plane];
        let target = Arc::make_mut(&mut self.planes[plane]);

        match (phase_x == 0, phase_y == 0) {
            (true, true) => {
                let source_in_bounds = origin_x >= 0
                    && origin_y >= 0
                    && origin_x as usize + output_width <= reference_plane_width
                    && origin_y as usize + output_height <= reference_plane_height;
                if source_in_bounds {
                    let source_x = origin_x as usize;
                    let source_y = origin_y as usize;
                    for row in 0..output_height {
                        let source_start = (source_y + row) * source_stride + source_x;
                        let target_start = (y + row) * target_stride + target_x;
                        let source_row = &source[source_start..source_start + output_width];
                        let target_row = &mut target[target_start..target_start + output_width];
                        if average {
                            for (target, &prediction) in target_row.iter_mut().zip(source_row) {
                                *target = avg2(*target, prediction);
                            }
                        } else {
                            target_row.copy_from_slice(source_row);
                        }
                    }
                } else {
                    for row in 0..output_height {
                        for column in 0..output_width {
                            let prediction =
                                sample(origin_x + column as i32, origin_y + row as i32);
                            let index = (y + row) * target_stride + target_x + column;
                            write_prediction(&mut target[index], prediction, average);
                        }
                    }
                }
            }
            (false, true) => {
                let source_in_bounds = origin_x >= 3
                    && origin_y >= 0
                    && origin_x as usize + output_width + 4 <= reference_plane_width
                    && origin_y as usize + output_height <= reference_plane_height;
                if source_in_bounds {
                    let source_x = origin_x as usize - 3;
                    let source_y = origin_y as usize;
                    let mut predictions = [0u8; 64];
                    for row in 0..output_height {
                        let source_start = (source_y + row) * source_stride + source_x;
                        convolve_8_horizontal_row(
                            &source[source_start..source_start + output_width + 7],
                            &mut predictions[..output_width],
                            filter_x,
                        );
                        let target_start = (y + row) * target_stride + target_x;
                        write_prediction_row(
                            &mut target[target_start..target_start + output_width],
                            &predictions[..output_width],
                            average,
                        );
                    }
                } else {
                    for row in 0..output_height {
                        for column in 0..output_width {
                            let mut sum = 0i32;
                            for (tap, &coefficient) in filter_x.iter().enumerate() {
                                sum += i32::from(coefficient)
                                    * i32::from(sample(
                                        origin_x + column as i32 + tap as i32 - 3,
                                        origin_y + row as i32,
                                    ));
                            }
                            let prediction = ((sum + 64) >> 7).clamp(0, 255) as u8;
                            let index = (y + row) * target_stride + target_x + column;
                            write_prediction(&mut target[index], prediction, average);
                        }
                    }
                }
            }
            (true, false) => {
                let source_in_bounds = origin_x >= 0
                    && origin_y >= 3
                    && origin_x as usize + output_width <= reference_plane_width
                    && origin_y as usize + output_height + 4 <= reference_plane_height;
                if source_in_bounds {
                    let source_x = origin_x as usize;
                    let source_y = origin_y as usize - 3;
                    let mut predictions = [0u8; 64];
                    for row in 0..output_height {
                        let source_start = (source_y + row) * source_stride + source_x;
                        convolve_8_vertical_row(
                            source,
                            source_start,
                            source_stride,
                            &mut predictions[..output_width],
                            filter_y,
                        );
                        let target_start = (y + row) * target_stride + target_x;
                        write_prediction_row(
                            &mut target[target_start..target_start + output_width],
                            &predictions[..output_width],
                            average,
                        );
                    }
                } else {
                    for row in 0..output_height {
                        for column in 0..output_width {
                            let mut sum = 0i32;
                            for (tap, &coefficient) in filter_y.iter().enumerate() {
                                sum += i32::from(coefficient)
                                    * i32::from(sample(
                                        origin_x + column as i32,
                                        origin_y + row as i32 + tap as i32 - 3,
                                    ));
                            }
                            let prediction = ((sum + 64) >> 7).clamp(0, 255) as u8;
                            let index = (y + row) * target_stride + target_x + column;
                            write_prediction(&mut target[index], prediction, average);
                        }
                    }
                }
            }
            (false, false) => {
                const MAXIMUM_INTERMEDIATE_SAMPLES: usize = 64 * (64 + 7);
                debug_assert!(output_width <= 64 && output_height <= 64);
                let temporary_height = output_height + 7;
                let mut temporary = [0u8; MAXIMUM_INTERMEDIATE_SAMPLES];
                let source_in_bounds = origin_x >= 3
                    && origin_y >= 3
                    && origin_x as usize + output_width + 4 <= reference_plane_width
                    && origin_y as usize + output_height + 4 <= reference_plane_height;
                if source_in_bounds {
                    let source_x = origin_x as usize - 3;
                    let source_y = origin_y as usize - 3;
                    for row in 0..temporary_height {
                        let source_start = (source_y + row) * source_stride + source_x;
                        let target_start = row * output_width;
                        convolve_8_horizontal_row(
                            &source[source_start..source_start + output_width + 7],
                            &mut temporary[target_start..target_start + output_width],
                            filter_x,
                        );
                    }
                } else {
                    for row in 0..temporary_height {
                        for column in 0..output_width {
                            let mut sum = 0i32;
                            for (tap, &coefficient) in filter_x.iter().enumerate() {
                                sum += i32::from(coefficient)
                                    * i32::from(sample(
                                        origin_x + column as i32 + tap as i32 - 3,
                                        origin_y + row as i32 - 3,
                                    ));
                            }
                            temporary[row * output_width + column] =
                                ((sum + 64) >> 7).clamp(0, 255) as u8;
                        }
                    }
                }
                let mut predictions = [0u8; 64];
                for row in 0..output_height {
                    convolve_8_vertical_row(
                        &temporary,
                        row * output_width,
                        output_width,
                        &mut predictions[..output_width],
                        filter_y,
                    );
                    let target_start = (y + row) * target_stride + target_x;
                    write_prediction_row(
                        &mut target[target_start..target_start + output_width],
                        &predictions[..output_width],
                        average,
                    );
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn predict_inter_scaled(
        &mut self,
        reference: &Self,
        plane: usize,
        x: usize,
        y: usize,
        target_x: usize,
        output_width: usize,
        output_height: usize,
        motion_row_q4: i32,
        motion_column_q4: i32,
        kernel: &[i16; 128],
        average: bool,
    ) {
        const REF_SCALE_SHIFT: u32 = 14;
        let x_scale = ((reference.width as i64) << REF_SCALE_SHIFT) / self.width as i64;
        let y_scale = ((reference.height as i64) << REF_SCALE_SHIFT) / self.height as i64;
        let scale = |value: i64, factor: i64| (value * factor) >> REF_SCALE_SHIFT;
        let start_x_q4 =
            scale(x as i64 * 16, x_scale) + scale(i64::from(motion_column_q4), x_scale);
        let start_y_q4 = scale(y as i64 * 16, y_scale) + scale(i64::from(motion_row_q4), y_scale);
        let x_step_q4 = scale(16, x_scale);
        let y_step_q4 = scale(16, y_scale);

        let source_width = if plane == 0 {
            reference.width
        } else {
            reference.width.div_ceil(2)
        };
        let source_height = if plane == 0 {
            reference.height
        } else {
            reference.height.div_ceil(2)
        };
        let source_stride = reference.strides[plane];
        let source = &reference.planes[plane];
        let sample = |source_x: i64, source_y: i64| -> u8 {
            let source_x = source_x.clamp(0, source_width.saturating_sub(1) as i64) as usize;
            let source_y = source_y.clamp(0, source_height.saturating_sub(1) as i64) as usize;
            source[source_y * source_stride + source_x]
        };

        let target_stride = self.strides[plane];
        let target = Arc::make_mut(&mut self.planes[plane]);
        for row in 0..output_height {
            let source_y_q4 = start_y_q4 + row as i64 * y_step_q4;
            let integer_y = source_y_q4 >> 4;
            let filter_y = &kernel[(source_y_q4 & 15) as usize * 8..][..8];
            for column in 0..output_width {
                let source_x_q4 = start_x_q4 + column as i64 * x_step_q4;
                let integer_x = source_x_q4 >> 4;
                let filter_x = &kernel[(source_x_q4 & 15) as usize * 8..][..8];
                let mut vertical_sum = 0i32;
                for (vertical_tap, &vertical_coefficient) in filter_y.iter().enumerate() {
                    let source_y = integer_y + vertical_tap as i64 - 3;
                    let mut horizontal_sum = 0i32;
                    for (horizontal_tap, &horizontal_coefficient) in filter_x.iter().enumerate() {
                        let source_x = integer_x + horizontal_tap as i64 - 3;
                        horizontal_sum += i32::from(horizontal_coefficient)
                            * i32::from(sample(source_x, source_y));
                    }
                    let horizontal = ((horizontal_sum + 64) >> 7).clamp(0, 255);
                    vertical_sum += i32::from(vertical_coefficient) * horizontal;
                }
                let prediction = ((vertical_sum + 64) >> 7).clamp(0, 255) as u8;
                let index = (y + row) * target_stride + target_x + column;
                write_prediction(&mut target[index], prediction, average);
            }
        }
    }
}

#[inline(always)]
fn convolve_8_scalar(samples: &[u8], coefficients: &[i16]) -> u8 {
    let mut sum = 0i32;
    for index in 0..8 {
        sum += i32::from(coefficients[index]) * i32::from(samples[index]);
    }
    ((sum + 64) >> 7).clamp(0, 255) as u8
}

fn convolve_8_horizontal_row(source: &[u8], target: &mut [u8], coefficients: &[i16]) {
    debug_assert!(source.len() >= target.len() + 7 && coefficients.len() >= 8);
    let mut offset = 0;
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        let vectorized = target.len() / 16 * 16;
        // SAFETY: runtime feature detection guarantees AVX2 and the slice
        // lengths prove every 16-byte load and store is within its allocation.
        unsafe {
            x86::convolve_8_horizontal_avx2(
                source.as_ptr(),
                target.as_mut_ptr(),
                vectorized,
                coefficients.as_ptr(),
            );
        }
        offset = vectorized;
    }
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("sse4.1") && target.len() - offset >= 8 {
        let vectorized = (target.len() - offset) / 8 * 8;
        // SAFETY: runtime feature detection guarantees SSE4.1. The slice
        // lengths prove every 8-byte load and store is within its allocation.
        unsafe {
            x86::convolve_8_horizontal_sse41(
                source.as_ptr().add(offset),
                target.as_mut_ptr().add(offset),
                vectorized,
                coefficients.as_ptr(),
            );
        }
        offset += vectorized;
    }
    for column in offset..target.len() {
        target[column] = convolve_8_scalar(&source[column..column + 8], coefficients);
    }
}

fn convolve_8_vertical_row(
    samples: &[u8],
    start: usize,
    stride: usize,
    target: &mut [u8],
    coefficients: &[i16],
) {
    debug_assert!(
        coefficients.len() >= 8
            && (target.is_empty() || start + 7 * stride + target.len() <= samples.len())
    );
    let mut offset = 0;
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        let vectorized = target.len() / 16 * 16;
        // SAFETY: runtime feature detection guarantees AVX2. The assertion
        // above proves all eight rows and every target chunk are in bounds.
        unsafe {
            x86::convolve_8_vertical_avx2(
                samples.as_ptr().add(start),
                stride,
                target.as_mut_ptr(),
                vectorized,
                coefficients.as_ptr(),
            );
        }
        offset = vectorized;
    }
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("sse4.1") && target.len() - offset >= 8 {
        let vectorized = (target.len() - offset) / 8 * 8;
        // SAFETY: runtime feature detection guarantees SSE4.1. The assertion
        // above proves all eight rows and every target chunk are in bounds.
        unsafe {
            x86::convolve_8_vertical_sse41(
                samples.as_ptr().add(start + offset),
                stride,
                target.as_mut_ptr().add(offset),
                vectorized,
                coefficients.as_ptr(),
            );
        }
        offset += vectorized;
    }
    for column in offset..target.len() {
        let mut sum = 0i32;
        for index in 0..8 {
            sum += i32::from(coefficients[index])
                * i32::from(samples[start + index * stride + column]);
        }
        target[column] = ((sum + 64) >> 7).clamp(0, 255) as u8;
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn convolve_8_horizontal_avx2(
        source: *const u8,
        target: *mut u8,
        length: usize,
        coefficients: *const i16,
    ) {
        let rounding = _mm256_set1_epi32(64);
        for column in (0..length).step_by(16) {
            let mut low = _mm256_setzero_si256();
            let mut high = _mm256_setzero_si256();
            for tap in 0..8 {
                let samples =
                    unsafe { _mm_loadu_si128(source.add(column + tap).cast::<__m128i>()) };
                let samples = _mm256_cvtepu8_epi16(samples);
                let coefficient = _mm256_set1_epi16(unsafe { *coefficients.add(tap) });
                let products = _mm256_mullo_epi16(samples, coefficient);
                low =
                    _mm256_add_epi32(low, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(products)));
                high = _mm256_add_epi32(
                    high,
                    _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(products)),
                );
            }
            low = _mm256_srai_epi32::<7>(_mm256_add_epi32(low, rounding));
            high = _mm256_srai_epi32::<7>(_mm256_add_epi32(high, rounding));
            let packed = _mm256_packs_epi32(low, high);
            let packed = _mm256_permute4x64_epi64::<0xd8>(packed);
            let bytes = _mm_packus_epi16(
                _mm256_castsi256_si128(packed),
                _mm256_extracti128_si256::<1>(packed),
            );
            unsafe {
                _mm_storeu_si128(target.add(column).cast::<__m128i>(), bytes);
            }
        }
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn convolve_8_vertical_avx2(
        source: *const u8,
        stride: usize,
        target: *mut u8,
        length: usize,
        coefficients: *const i16,
    ) {
        let rounding = _mm256_set1_epi32(64);
        for column in (0..length).step_by(16) {
            let mut low = _mm256_setzero_si256();
            let mut high = _mm256_setzero_si256();
            for tap in 0..8 {
                let samples =
                    unsafe { _mm_loadu_si128(source.add(tap * stride + column).cast::<__m128i>()) };
                let samples = _mm256_cvtepu8_epi16(samples);
                let coefficient = _mm256_set1_epi16(unsafe { *coefficients.add(tap) });
                let products = _mm256_mullo_epi16(samples, coefficient);
                low =
                    _mm256_add_epi32(low, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(products)));
                high = _mm256_add_epi32(
                    high,
                    _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(products)),
                );
            }
            low = _mm256_srai_epi32::<7>(_mm256_add_epi32(low, rounding));
            high = _mm256_srai_epi32::<7>(_mm256_add_epi32(high, rounding));
            let packed = _mm256_packs_epi32(low, high);
            let packed = _mm256_permute4x64_epi64::<0xd8>(packed);
            let bytes = _mm_packus_epi16(
                _mm256_castsi256_si128(packed),
                _mm256_extracti128_si256::<1>(packed),
            );
            unsafe {
                _mm_storeu_si128(target.add(column).cast::<__m128i>(), bytes);
            }
        }
    }

    #[target_feature(enable = "sse4.1")]
    pub(super) unsafe fn convolve_8_horizontal_sse41(
        source: *const u8,
        target: *mut u8,
        length: usize,
        coefficients: *const i16,
    ) {
        let rounding = _mm_set1_epi32(64);
        for column in (0..length).step_by(8) {
            let mut low = _mm_setzero_si128();
            let mut high = _mm_setzero_si128();
            for tap in 0..8 {
                // SAFETY: the caller proves the full source, target, and
                // coefficient ranges used by these fixed-size accesses.
                let samples =
                    unsafe { _mm_loadl_epi64(source.add(column + tap).cast::<__m128i>()) };
                let samples = _mm_cvtepu8_epi16(samples);
                let coefficient = _mm_set1_epi16(unsafe { *coefficients.add(tap) });
                let products = _mm_mullo_epi16(samples, coefficient);
                low = _mm_add_epi32(low, _mm_cvtepi16_epi32(products));
                high = _mm_add_epi32(high, _mm_cvtepi16_epi32(_mm_srli_si128::<8>(products)));
            }
            low = _mm_srai_epi32::<7>(_mm_add_epi32(low, rounding));
            high = _mm_srai_epi32::<7>(_mm_add_epi32(high, rounding));
            let packed = _mm_packs_epi32(low, high);
            let packed = _mm_packus_epi16(packed, _mm_setzero_si128());
            unsafe {
                _mm_storel_epi64(target.add(column).cast::<__m128i>(), packed);
            }
        }
    }

    #[target_feature(enable = "sse4.1")]
    pub(super) unsafe fn convolve_8_vertical_sse41(
        source: *const u8,
        stride: usize,
        target: *mut u8,
        length: usize,
        coefficients: *const i16,
    ) {
        let rounding = _mm_set1_epi32(64);
        for column in (0..length).step_by(8) {
            let mut low = _mm_setzero_si128();
            let mut high = _mm_setzero_si128();
            for tap in 0..8 {
                let samples =
                    unsafe { _mm_loadl_epi64(source.add(tap * stride + column).cast::<__m128i>()) };
                let samples = _mm_cvtepu8_epi16(samples);
                let coefficient = _mm_set1_epi16(unsafe { *coefficients.add(tap) });
                let products = _mm_mullo_epi16(samples, coefficient);
                low = _mm_add_epi32(low, _mm_cvtepi16_epi32(products));
                high = _mm_add_epi32(high, _mm_cvtepi16_epi32(_mm_srli_si128::<8>(products)));
            }
            low = _mm_srai_epi32::<7>(_mm_add_epi32(low, rounding));
            high = _mm_srai_epi32::<7>(_mm_add_epi32(high, rounding));
            let packed = _mm_packs_epi32(low, high);
            let packed = _mm_packus_epi16(packed, _mm_setzero_si128());
            unsafe {
                _mm_storel_epi64(target.add(column).cast::<__m128i>(), packed);
            }
        }
    }
}

#[derive(Clone, Copy)]
enum AxisTransform {
    Dct,
    Adst,
}

fn transform_axes(transform: TransformType) -> (AxisTransform, AxisTransform) {
    match transform {
        TransformType::DctDct => (AxisTransform::Dct, AxisTransform::Dct),
        TransformType::AdstDct => (AxisTransform::Dct, AxisTransform::Adst),
        TransformType::DctAdst => (AxisTransform::Adst, AxisTransform::Dct),
        TransformType::AdstAdst => (AxisTransform::Adst, AxisTransform::Adst),
    }
}

fn inverse_1d_sparse(
    input: &[i32; 32],
    output: &mut [i32; 32],
    size: usize,
    transform: AxisTransform,
) {
    if input[..size].iter().all(|&value| value == 0) {
        output[..size].fill(0);
        return;
    }
    match transform {
        AxisTransform::Dct => inverse_dct(input, output, size),
        AxisTransform::Adst => inverse_adst(input, output, size),
    }
}

fn round_power_of_two(value: i32, shift: u32) -> i32 {
    (value + (1 << (shift - 1))) >> shift
}

fn avg2(first: u8, second: u8) -> u8 {
    ((u16::from(first) + u16::from(second) + 1) >> 1) as u8
}

#[inline(always)]
fn write_prediction(target: &mut u8, prediction: u8, average: bool) {
    if average {
        *target = avg2(*target, prediction);
    } else {
        *target = prediction;
    }
}

fn write_prediction_row(target: &mut [u8], prediction: &[u8], average: bool) {
    debug_assert_eq!(target.len(), prediction.len());
    if average {
        for (target, &prediction) in target.iter_mut().zip(prediction) {
            *target = avg2(*target, prediction);
        }
    } else {
        target.copy_from_slice(prediction);
    }
}

fn avg3(first: u8, middle: u8, last: u8) -> u8 {
    ((u16::from(first) + 2 * u16::from(middle) + u16::from(last) + 2) >> 2) as u8
}

#[allow(clippy::too_many_arguments)]
fn intra_predict(
    target: &mut [u8],
    size: usize,
    mode: IntraMode,
    above: &[u8],
    left: &[u8],
    top_left: u8,
    have_top: bool,
    have_left: bool,
) {
    let pixel = |x: usize, y: usize| y * size + x;
    match mode {
        IntraMode::Dc => {
            let value = match (have_top, have_left) {
                (false, false) => 128,
                (true, false) => {
                    (above[..size].iter().map(|&x| usize::from(x)).sum::<usize>() + size / 2) / size
                }
                (false, true) => {
                    (left.iter().map(|&x| usize::from(x)).sum::<usize>() + size / 2) / size
                }
                (true, true) => {
                    (above[..size].iter().map(|&x| usize::from(x)).sum::<usize>()
                        + left.iter().map(|&x| usize::from(x)).sum::<usize>()
                        + size)
                        / (size * 2)
                }
            } as u8;
            target.fill(value);
        }
        IntraMode::Vertical => {
            for row in target.chunks_exact_mut(size) {
                row.copy_from_slice(&above[..size]);
            }
        }
        IntraMode::Horizontal => {
            for (row, target) in target.chunks_exact_mut(size).enumerate() {
                target.fill(left[row]);
            }
        }
        IntraMode::TrueMotion => {
            for row in 0..size {
                for column in 0..size {
                    target[pixel(column, row)] = (i16::from(left[row]) + i16::from(above[column])
                        - i16::from(top_left))
                    .clamp(0, 255) as u8;
                }
            }
        }
        IntraMode::D45 => {
            if size == 4 {
                for row in 0..size {
                    for column in 0..size {
                        target[pixel(column, row)] = if row == 3 && column == 3 {
                            above[7]
                        } else {
                            let offset = row + column;
                            avg3(above[offset], above[offset + 1], above[offset + 2])
                        };
                    }
                }
            } else {
                let edge = above[size - 1];
                for column in 0..size - 1 {
                    target[column] = avg3(above[column], above[column + 1], above[column + 2]);
                }
                target[size - 1] = edge;
                for row in 1..size {
                    for column in 0..size {
                        target[pixel(column, row)] = if column + row < size - 1 {
                            target[column + row]
                        } else {
                            edge
                        };
                    }
                }
            }
        }
        IntraMode::D63 => {
            if size == 4 {
                for row in 0..size {
                    for column in 0..size {
                        let offset = column + row / 2;
                        target[pixel(column, row)] = if row.is_multiple_of(2) {
                            avg2(above[offset], above[offset + 1])
                        } else {
                            avg3(above[offset], above[offset + 1], above[offset + 2])
                        };
                    }
                }
            } else {
                let edge = above[size - 1];
                for column in 0..size {
                    target[column] = avg2(above[column], above[column + 1]);
                    target[size + column] =
                        avg3(above[column], above[column + 1], above[column + 2]);
                }
                for row in 2..size {
                    let shift = row / 2;
                    let copy_count = size - shift - 1;
                    let source_row = row & 1;
                    for column in 0..size {
                        target[pixel(column, row)] = if column < copy_count {
                            target[pixel(column + shift, source_row)]
                        } else {
                            edge
                        };
                    }
                }
            }
        }
        IntraMode::D207 => {
            for row in 0..size {
                for column in 0..size {
                    let offset = (row + column / 2).min(size - 1);
                    target[pixel(column, row)] = if column.is_multiple_of(2) {
                        avg2(left[offset], left[(offset + 1).min(size - 1)])
                    } else {
                        avg3(
                            left[offset],
                            left[(offset + 1).min(size - 1)],
                            left[(offset + 2).min(size - 1)],
                        )
                    };
                }
            }
        }
        IntraMode::D135 => {
            let mut border = vec![0u8; size * 2 - 1];
            for index in 0..size - 2 {
                border[index] = avg3(
                    left[size - 3 - index],
                    left[size - 2 - index],
                    left[size - 1 - index],
                );
            }
            border[size - 2] = avg3(top_left, left[0], left[1]);
            border[size - 1] = avg3(left[0], top_left, above[0]);
            border[size] = avg3(top_left, above[0], above[1]);
            for index in 0..size - 2 {
                border[size + 1 + index] = avg3(above[index], above[index + 1], above[index + 2]);
            }
            for row in 0..size {
                target[row * size..(row + 1) * size]
                    .copy_from_slice(&border[size - 1 - row..size * 2 - 1 - row]);
            }
        }
        IntraMode::D117 => {
            for row in 0..size {
                for column in 0..size {
                    target[pixel(column, row)] = if row == 0 {
                        avg2(
                            if column == 0 {
                                top_left
                            } else {
                                above[column - 1]
                            },
                            above[column],
                        )
                    } else if row == 1 {
                        avg3(
                            if column == 0 {
                                left[0]
                            } else if column == 1 {
                                top_left
                            } else {
                                above[column - 2]
                            },
                            if column == 0 {
                                top_left
                            } else {
                                above[column - 1]
                            },
                            above[column],
                        )
                    } else if column == 0 {
                        avg3(
                            if row == 2 { top_left } else { left[row - 3] },
                            left[row - 2],
                            left[row - 1],
                        )
                    } else {
                        target[pixel(column - 1, row - 2)]
                    };
                }
            }
        }
        IntraMode::D153 => {
            for row in 0..size {
                for column in 0..size {
                    target[pixel(column, row)] = if column == 0 {
                        avg2(if row == 0 { top_left } else { left[row - 1] }, left[row])
                    } else if column == 1 {
                        avg3(
                            if row == 0 {
                                left[0]
                            } else if row == 1 {
                                top_left
                            } else {
                                left[row - 2]
                            },
                            if row == 0 { top_left } else { left[row - 1] },
                            if row == 0 { above[0] } else { left[row] },
                        )
                    } else if row == 0 {
                        avg3(
                            if column == 2 {
                                top_left
                            } else {
                                above[column - 3]
                            },
                            above[column - 2],
                            above[column - 1],
                        )
                    } else {
                        target[pixel(column - 2, row - 1)]
                    };
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{IntraMode, IntraPicture, intra_predict};

    #[test]
    fn vertical_prediction_repeats_top_row() {
        let mut target = [0; 16];
        intra_predict(
            &mut target,
            4,
            IntraMode::Vertical,
            &[1, 2, 3, 4, 4, 4, 4, 4],
            &[9; 4],
            8,
            true,
            true,
        );
        assert_eq!(target, [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]);
    }

    #[test]
    fn scaled_inter_prediction_maps_reference_coordinates() {
        let mut reference = IntraPicture::new(4, 4);
        reference.planes_mut()[0]
            .copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
        let mut target = IntraPicture::new(8, 8);
        let mut nearest_kernel = [0i16; 128];
        for phase in 0..16 {
            nearest_kernel[phase * 8 + 3] = 128;
        }
        target.predict_inter(&reference, 0, 0, 0, 8, 8, 0, 0, &nearest_kernel, false);
        assert_eq!(
            target.plane(0),
            &[
                1, 1, 2, 2, 3, 3, 4, 4, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 5, 5, 6, 6,
                7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14,
                14, 15, 15, 16, 16, 13, 13, 14, 14, 15, 15, 16, 16,
            ]
        );
    }

    #[test]
    fn directional_predictors_match_scalar_reference_hashes() {
        let modes = [
            IntraMode::Vertical,
            IntraMode::Horizontal,
            IntraMode::TrueMotion,
            IntraMode::D45,
            IntraMode::D63,
            IntraMode::D117,
            IntraMode::D135,
            IntraMode::D153,
            IntraMode::D207,
        ];
        let expected = [
            [
                12_385_777_618_468_341_763,
                17_481_041_929_905_823_107,
                9_162_175_879_373_427_723,
                2_951_750_245_695_463_603,
                178_186_053_736_457_363,
                9_968_858_480_488_691_406,
                16_916_242_600_893_846_268,
                14_100_189_506_615_379_203,
                7_765_643_777_602_061_011,
            ],
            [
                7_617_513_244_772_354_435,
                7_722_650_720_878_138_131,
                8_900_944_676_041_817_988,
                11_865_569_275_046_706_791,
                4_563_822_202_903_225_025,
                642_264_254_256_673_494,
                3_802_063_769_585_604_452,
                11_899_294_097_599_334_869,
                15_510_928_977_528_816_603,
            ],
            [
                13_983_866_196_833_557_891,
                2_247_242_578_578_510_243,
                16_953_500_656_830_245_178,
                7_535_794_851_790_941_115,
                3_868_825_529_465_454_833,
                6_926_231_847_567_959_486,
                7_377_452_461_317_928_316,
                9_707_382_032_725_213_221,
                5_018_596_726_807_400_035,
            ],
            [
                13_884_625_836_434_813_827,
                10_717_571_983_384_104_515,
                15_589_949_899_131_379_792,
                6_778_394_571_013_274_035,
                15_203_466_764_615_908_553,
                16_066_928_015_638_142_678,
                1_763_118_948_851_886_444,
                4_912_625_095_993_828_469,
                10_569_393_731_538_595_331,
            ],
        ];
        let above: Vec<_> = (0..64).map(|index| (17 + index * 29) as u8).collect();
        let left: Vec<_> = (0..32).map(|index| (211i32 - index * 23) as u8).collect();
        for (size_index, size) in [4, 8, 16, 32].into_iter().enumerate() {
            for (mode_index, &mode) in modes.iter().enumerate() {
                let mut target = vec![0; size * size];
                intra_predict(
                    &mut target,
                    size,
                    mode,
                    &above,
                    &left[..size],
                    93,
                    true,
                    true,
                );
                let hash = target
                    .into_iter()
                    .fold(1_469_598_103_934_665_603u64, |hash, value| {
                        (hash ^ u64::from(value)).wrapping_mul(1_099_511_628_211)
                    });
                assert_eq!(
                    hash, expected[size_index][mode_index],
                    "{mode:?} {size}x{size}"
                );
            }
        }
    }
}
