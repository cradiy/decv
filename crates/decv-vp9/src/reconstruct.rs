use std::sync::Arc;

use crate::{
    BitDepth, ChromaSubsampling,
    block::{IntraMode, TransformSize, TransformType},
    inverse_transform::{inverse_adst, inverse_dct},
};

#[derive(Debug, Clone)]
enum PicturePlanes {
    Eight([Arc<[u8]>; 3]),
    High([Arc<[u16]>; 3]),
}

trait Sample: Copy + Default {
    fn from_u32(value: u32) -> Self;
    fn to_u32(self) -> u32;
}

impl Sample for u8 {
    #[inline]
    fn from_u32(value: u32) -> Self {
        value as Self
    }

    #[inline]
    fn to_u32(self) -> u32 {
        u32::from(self)
    }
}

impl Sample for u16 {
    #[inline]
    fn from_u32(value: u32) -> Self {
        value as Self
    }

    #[inline]
    fn to_u32(self) -> u32 {
        u32::from(self)
    }
}

fn copy_strip_plane<T: Copy>(
    source: &[T],
    target: &mut [T],
    source_stride: usize,
    target_stride: usize,
    origin_x: usize,
    width: usize,
    height: usize,
) {
    for row in 0..height {
        let source_start = row * source_stride;
        let target_start = row * target_stride + origin_x;
        target[target_start..target_start + width]
            .copy_from_slice(&source[source_start..source_start + width]);
    }
}

/// Reconstructed planar YUV picture in its native VP9 sample depth.
#[derive(Debug, Clone)]
pub struct IntraPicture {
    width: usize,
    height: usize,
    origin_x: usize,
    storage_width: usize,
    bit_depth: BitDepth,
    subsampling: ChromaSubsampling,
    /// Plane strides in samples, not bytes.
    strides: [usize; 3],
    planes: PicturePlanes,
}

impl IntraPicture {
    pub(crate) fn new(
        width: usize,
        height: usize,
        subsampling: ChromaSubsampling,
        bit_depth: BitDepth,
    ) -> Self {
        let chroma_width = width.div_ceil(1 << subsampling.x_shift());
        let chroma_height = height.div_ceil(1 << subsampling.y_shift());
        let plane_lengths = [
            width * height,
            chroma_width * chroma_height,
            chroma_width * chroma_height,
        ];
        Self {
            width,
            height,
            origin_x: 0,
            storage_width: width,
            bit_depth,
            subsampling,
            strides: [width, chroma_width, chroma_width],
            planes: Self::allocate_planes(bit_depth, plane_lengths),
        }
    }

    pub(crate) fn new_strip(
        width: usize,
        height: usize,
        origin_x: usize,
        storage_width: usize,
        subsampling: ChromaSubsampling,
        bit_depth: BitDepth,
    ) -> Self {
        debug_assert!(origin_x <= width && storage_width <= width - origin_x);
        debug_assert!(origin_x.is_multiple_of(1 << subsampling.x_shift()));
        let chroma_width = storage_width.div_ceil(1 << subsampling.x_shift());
        let chroma_height = height.div_ceil(1 << subsampling.y_shift());
        let plane_lengths = [
            storage_width * height,
            chroma_width * chroma_height,
            chroma_width * chroma_height,
        ];
        Self {
            width,
            height,
            origin_x,
            storage_width,
            bit_depth,
            subsampling,
            strides: [storage_width, chroma_width, chroma_width],
            planes: Self::allocate_planes(bit_depth, plane_lengths),
        }
    }

    fn allocate_planes(bit_depth: BitDepth, lengths: [usize; 3]) -> PicturePlanes {
        match bit_depth {
            BitDepth::Eight => PicturePlanes::Eight(lengths.map(|length| vec![0; length].into())),
            BitDepth::Ten | BitDepth::Twelve => {
                PicturePlanes::High(lengths.map(|length| vec![0; length].into()))
            }
        }
    }

    pub(crate) fn copy_strip_from(&mut self, strip: &Self) {
        debug_assert_eq!(self.width, strip.width);
        debug_assert_eq!(self.height, strip.height);
        debug_assert_eq!(self.bit_depth, strip.bit_depth);
        debug_assert_eq!(self.subsampling, strip.subsampling);
        debug_assert_eq!(self.origin_x, 0);
        for plane in 0..3 {
            let subsampling_x = self.subsampling_x(plane);
            let subsampling_y = self.subsampling_y(plane);
            let origin_x = strip.origin_x >> subsampling_x;
            let width = strip.storage_width.div_ceil(1 << subsampling_x);
            let height = self.height.div_ceil(1 << subsampling_y);
            let source_stride = strip.strides[plane];
            let target_stride = self.strides[plane];
            match (&strip.planes, &mut self.planes) {
                (PicturePlanes::Eight(source), PicturePlanes::Eight(target)) => {
                    copy_strip_plane(
                        &source[plane],
                        Arc::make_mut(&mut target[plane]),
                        source_stride,
                        target_stride,
                        origin_x,
                        width,
                        height,
                    );
                }
                (PicturePlanes::High(source), PicturePlanes::High(target)) => {
                    copy_strip_plane(
                        &source[plane],
                        Arc::make_mut(&mut target[plane]),
                        source_stride,
                        target_stride,
                        origin_x,
                        width,
                        height,
                    );
                }
                _ => unreachable!("picture strips must have matching sample depths"),
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
    pub fn bit_depth(&self) -> BitDepth {
        self.bit_depth
    }

    #[inline]
    pub fn subsampling(&self) -> ChromaSubsampling {
        self.subsampling
    }

    #[inline]
    pub(crate) fn subsampling_x(&self, plane: usize) -> usize {
        usize::from(plane != 0) * self.subsampling.x_shift()
    }

    #[inline]
    pub(crate) fn subsampling_y(&self, plane: usize) -> usize {
        usize::from(plane != 0) * self.subsampling.y_shift()
    }

    #[inline]
    pub(crate) fn plane_width(&self, plane: usize) -> usize {
        self.width.div_ceil(1 << self.subsampling_x(plane))
    }

    #[inline]
    pub(crate) fn plane_height(&self, plane: usize) -> usize {
        self.height.div_ceil(1 << self.subsampling_y(plane))
    }

    #[inline]
    pub fn stride(&self, plane: usize) -> usize {
        self.strides[plane] * self.bytes_per_sample()
    }

    #[inline]
    pub(crate) fn sample_stride(&self, plane: usize) -> usize {
        self.strides[plane]
    }

    /// Returns an 8-bit plane.
    ///
    /// This compatibility accessor is only valid for Profile 0/1 pictures.
    /// Use [`Self::plane_u16`] for Profile 2/3 pictures.
    #[inline]
    pub fn plane(&self, plane: usize) -> &[u8] {
        match &self.planes {
            PicturePlanes::Eight(planes) => &planes[plane],
            PicturePlanes::High(_) => panic!("16-bit picture accessed through plane()"),
        }
    }

    #[inline]
    pub fn plane_u16(&self, plane: usize) -> Option<&[u16]> {
        match &self.planes {
            PicturePlanes::Eight(_) => None,
            PicturePlanes::High(planes) => Some(&planes[plane]),
        }
    }

    #[inline]
    pub(crate) fn shared_plane_u8(&self, plane: usize) -> Option<Arc<[u8]>> {
        match &self.planes {
            PicturePlanes::Eight(planes) => Some(Arc::clone(&planes[plane])),
            PicturePlanes::High(_) => None,
        }
    }

    #[inline]
    pub(crate) fn shared_plane_u16(&self, plane: usize) -> Option<Arc<[u16]>> {
        match &self.planes {
            PicturePlanes::Eight(_) => None,
            PicturePlanes::High(planes) => Some(Arc::clone(&planes[plane])),
        }
    }

    pub(crate) fn planes_mut(&mut self) -> [&mut [u8]; 3] {
        let PicturePlanes::Eight(planes) = &mut self.planes else {
            panic!("16-bit picture accessed through planes_mut()");
        };
        let [y, u, v] = planes;
        [Arc::make_mut(y), Arc::make_mut(u), Arc::make_mut(v)]
    }

    #[inline]
    fn plane_u8(&self, plane: usize) -> &[u8] {
        self.plane(plane)
    }

    #[inline]
    fn plane_u8_mut(&mut self, plane: usize) -> &mut [u8] {
        let PicturePlanes::Eight(planes) = &mut self.planes else {
            panic!("16-bit picture reached the 8-bit reconstruction path");
        };
        Arc::make_mut(&mut planes[plane])
    }

    pub(crate) fn planes_u16_mut(&mut self) -> [&mut [u16]; 3] {
        let PicturePlanes::High(planes) = &mut self.planes else {
            panic!("8-bit picture accessed through planes_u16_mut()");
        };
        let [y, u, v] = planes;
        [Arc::make_mut(y), Arc::make_mut(u), Arc::make_mut(v)]
    }

    #[inline]
    const fn bytes_per_sample(&self) -> usize {
        match self.bit_depth {
            BitDepth::Eight => 1,
            BitDepth::Ten | BitDepth::Twelve => 2,
        }
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
        let subsampling_x = self.subsampling_x(plane);
        let width = self.storage_width.div_ceil(1 << subsampling_x);
        let height = self.plane_height(plane);
        let plane_origin_x = self.origin_x >> subsampling_x;
        let x = x
            .checked_sub(plane_origin_x)
            .expect("prediction belongs to this picture strip");
        let tile_left = tile_left
            .checked_sub(plane_origin_x)
            .expect("tile begins inside this picture strip");
        let have_top = y > tile_top;
        let have_left = x > tile_left;
        debug_assert!(size <= 32);
        let stride = self.strides[plane];
        let bit_depth = self.bit_depth;
        match &mut self.planes {
            PicturePlanes::Eight(planes) => predict_plane(
                &mut planes[plane],
                stride,
                width,
                height,
                x,
                y,
                size,
                mode,
                have_top,
                have_left,
                right_available,
                bit_depth,
            ),
            PicturePlanes::High(planes) => predict_plane(
                &mut planes[plane],
                stride,
                width,
                height,
                x,
                y,
                size,
                mode,
                have_top,
                have_left,
                right_available,
                bit_depth,
            ),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn add_residual(
        &mut self,
        plane: usize,
        x: usize,
        y: usize,
        transform_size: TransformSize,
        transform_type: TransformType,
        lossless: bool,
        coefficients: &[i32],
    ) {
        let subsampling_x = self.subsampling_x(plane);
        let width = self.storage_width.div_ceil(1 << subsampling_x);
        let height = self.plane_height(plane);
        let plane_origin_x = self.origin_x >> subsampling_x;
        let x = x
            .checked_sub(plane_origin_x)
            .expect("residual belongs to this picture strip");
        let stride = self.strides[plane];
        let max_sample = self.bit_depth.max_sample();
        match &mut self.planes {
            PicturePlanes::Eight(planes) => add_residual_to_plane(
                &mut planes[plane],
                stride,
                x,
                y,
                width,
                height,
                transform_size,
                transform_type,
                lossless,
                coefficients,
                max_sample,
            ),
            PicturePlanes::High(planes) => add_residual_to_plane(
                &mut planes[plane],
                stride,
                x,
                y,
                width,
                height,
                transform_size,
                transform_type,
                lossless,
                coefficients,
                max_sample,
            ),
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
        debug_assert_eq!(self.subsampling, reference.subsampling);
        debug_assert_eq!(self.bit_depth, reference.bit_depth);
        let subsampling_x = self.subsampling_x(plane);
        let reference_plane_width = reference.plane_width(plane);
        let reference_plane_height = reference.plane_height(plane);
        let plane_width = self.plane_width(plane);
        let plane_height = self.plane_height(plane);
        let plane_origin_x = self.origin_x >> subsampling_x;
        let target_x = x
            .checked_sub(plane_origin_x)
            .expect("inter prediction belongs to this picture strip");
        let target_width = self.storage_width.div_ceil(1 << subsampling_x);
        let output_width = width
            .min(plane_width.saturating_sub(x))
            .min(target_width.saturating_sub(target_x));
        let output_height = height.min(plane_height.saturating_sub(y));
        if output_width == 0 || output_height == 0 {
            return;
        }
        if !matches!(self.bit_depth, BitDepth::Eight) {
            let PicturePlanes::High(reference_planes) = &reference.planes else {
                unreachable!("high-bit-depth picture requires matching references");
            };
            let source = &reference_planes[plane];
            let source_stride = reference.strides[plane];
            let source_width = reference.plane_width(plane);
            let source_height = reference.plane_height(plane);
            let reference_width = reference.width;
            let reference_height = reference.height;
            let current_width = self.width;
            let current_height = self.height;
            let max_sample = self.bit_depth.max_sample();
            let target_stride = self.strides[plane];
            let PicturePlanes::High(target_planes) = &mut self.planes else {
                unreachable!("high-bit-depth picture requires word storage");
            };
            predict_inter_high(
                source,
                source_stride,
                source_width,
                source_height,
                reference_width,
                reference_height,
                Arc::make_mut(&mut target_planes[plane]),
                target_stride,
                current_width,
                current_height,
                x,
                y,
                target_x,
                output_width,
                output_height,
                motion_row_q4,
                motion_column_q4,
                kernel,
                average,
                max_sample,
            );
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
        let source = reference.plane_u8(plane);
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
        let target = self.plane_u8_mut(plane);

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

        let source_width = reference.plane_width(plane);
        let source_height = reference.plane_height(plane);
        let source_stride = reference.strides[plane];
        let source = reference.plane_u8(plane);
        let sample = |source_x: i64, source_y: i64| -> u8 {
            let source_x = source_x.clamp(0, source_width.saturating_sub(1) as i64) as usize;
            let source_y = source_y.clamp(0, source_height.saturating_sub(1) as i64) as usize;
            source[source_y * source_stride + source_x]
        };

        let target_stride = self.strides[plane];
        let target = self.plane_u8_mut(plane);
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

#[allow(clippy::too_many_arguments)]
fn predict_inter_high(
    source: &[u16],
    source_stride: usize,
    source_width: usize,
    source_height: usize,
    reference_width: usize,
    reference_height: usize,
    target: &mut [u16],
    target_stride: usize,
    current_width: usize,
    current_height: usize,
    x: usize,
    y: usize,
    target_x: usize,
    output_width: usize,
    output_height: usize,
    motion_row_q4: i32,
    motion_column_q4: i32,
    kernel: &[i16; 128],
    average: bool,
    max_sample: u16,
) {
    const REF_SCALE_SHIFT: u32 = 14;
    let scaled = reference_width != current_width || reference_height != current_height;
    let x_scale = if scaled {
        ((reference_width as i64) << REF_SCALE_SHIFT) / current_width as i64
    } else {
        1 << REF_SCALE_SHIFT
    };
    let y_scale = if scaled {
        ((reference_height as i64) << REF_SCALE_SHIFT) / current_height as i64
    } else {
        1 << REF_SCALE_SHIFT
    };
    let scale = |value: i64, factor: i64| (value * factor) >> REF_SCALE_SHIFT;
    let start_x_q4 = scale(x as i64 * 16, x_scale) + scale(i64::from(motion_column_q4), x_scale);
    let start_y_q4 = scale(y as i64 * 16, y_scale) + scale(i64::from(motion_row_q4), y_scale);
    let x_step_q4 = scale(16, x_scale);
    let y_step_q4 = scale(16, y_scale);
    let sample = |source_x: i64, source_y: i64| -> u16 {
        let source_x = source_x.clamp(0, source_width.saturating_sub(1) as i64) as usize;
        let source_y = source_y.clamp(0, source_height.saturating_sub(1) as i64) as usize;
        source[source_y * source_stride + source_x]
    };
    let clip = |value: i32| value.clamp(0, i32::from(max_sample)) as u16;

    if !scaled && motion_column_q4 & 15 == 0 && motion_row_q4 & 15 == 0 {
        let origin_x = x as i64 + i64::from(motion_column_q4 >> 4);
        let origin_y = y as i64 + i64::from(motion_row_q4 >> 4);
        let source_in_bounds = origin_x >= 0
            && origin_y >= 0
            && origin_x as usize + output_width <= source_width
            && origin_y as usize + output_height <= source_height;
        if source_in_bounds {
            let source_x = origin_x as usize;
            let source_y = origin_y as usize;
            for row in 0..output_height {
                let source_start = (source_y + row) * source_stride + source_x;
                let target_start = (y + row) * target_stride + target_x;
                write_prediction_row_high(
                    &mut target[target_start..target_start + output_width],
                    &source[source_start..source_start + output_width],
                    average,
                );
            }
        } else {
            for row in 0..output_height {
                for column in 0..output_width {
                    let prediction = sample(origin_x + column as i64, origin_y + row as i64);
                    let index = (y + row) * target_stride + target_x + column;
                    write_prediction(&mut target[index], prediction, average);
                }
            }
        }
        return;
    }

    if !scaled {
        let integer_x = motion_column_q4 >> 4;
        let integer_y = motion_row_q4 >> 4;
        let phase_x = (motion_column_q4 & 15) as usize;
        let phase_y = (motion_row_q4 & 15) as usize;
        let filter_x = &kernel[phase_x * 8..phase_x * 8 + 8];
        let filter_y = &kernel[phase_y * 8..phase_y * 8 + 8];
        let origin_x = x as i32 + integer_x;
        let origin_y = y as i32 + integer_y;

        match (phase_x == 0, phase_y == 0) {
            (true, true) => unreachable!("integer-pel prediction returned above"),
            (false, true)
                if origin_x >= 3
                    && origin_y >= 0
                    && origin_x as usize + output_width + 4 <= source_width
                    && origin_y as usize + output_height <= source_height =>
            {
                let source_x = origin_x as usize - 3;
                let source_y = origin_y as usize;
                let mut predictions = [0u16; 64];
                for row in 0..output_height {
                    let source_start = (source_y + row) * source_stride + source_x;
                    convolve_8_horizontal_row_high(
                        &source[source_start..source_start + output_width + 7],
                        &mut predictions[..output_width],
                        filter_x,
                        max_sample,
                    );
                    let target_start = (y + row) * target_stride + target_x;
                    write_prediction_row_high(
                        &mut target[target_start..target_start + output_width],
                        &predictions[..output_width],
                        average,
                    );
                }
                return;
            }
            (true, false)
                if origin_x >= 0
                    && origin_y >= 3
                    && origin_x as usize + output_width <= source_width
                    && origin_y as usize + output_height + 4 <= source_height =>
            {
                let source_x = origin_x as usize;
                let source_y = origin_y as usize - 3;
                let mut predictions = [0u16; 64];
                for row in 0..output_height {
                    let source_start = (source_y + row) * source_stride + source_x;
                    convolve_8_vertical_row_high(
                        source,
                        source_start,
                        source_stride,
                        &mut predictions[..output_width],
                        filter_y,
                        max_sample,
                    );
                    let target_start = (y + row) * target_stride + target_x;
                    write_prediction_row_high(
                        &mut target[target_start..target_start + output_width],
                        &predictions[..output_width],
                        average,
                    );
                }
                return;
            }
            (false, false)
                if origin_x >= 3
                    && origin_y >= 3
                    && origin_x as usize + output_width + 4 <= source_width
                    && origin_y as usize + output_height + 4 <= source_height =>
            {
                const MAXIMUM_INTERMEDIATE_SAMPLES: usize = 64 * (64 + 7);
                debug_assert!(output_width <= 64 && output_height <= 64);
                let temporary_height = output_height + 7;
                let source_x = origin_x as usize - 3;
                let source_y = origin_y as usize - 3;
                let mut temporary = [0u16; MAXIMUM_INTERMEDIATE_SAMPLES];
                for row in 0..temporary_height {
                    let source_start = (source_y + row) * source_stride + source_x;
                    let target_start = row * output_width;
                    convolve_8_horizontal_row_high(
                        &source[source_start..source_start + output_width + 7],
                        &mut temporary[target_start..target_start + output_width],
                        filter_x,
                        max_sample,
                    );
                }
                let mut predictions = [0u16; 64];
                for row in 0..output_height {
                    convolve_8_vertical_row_high(
                        &temporary,
                        row * output_width,
                        output_width,
                        &mut predictions[..output_width],
                        filter_y,
                        max_sample,
                    );
                    let target_start = (y + row) * target_stride + target_x;
                    write_prediction_row_high(
                        &mut target[target_start..target_start + output_width],
                        &predictions[..output_width],
                        average,
                    );
                }
                return;
            }
            _ => {}
        }
    }

    for row in 0..output_height {
        let source_y_q4 = start_y_q4 + row as i64 * y_step_q4;
        let integer_y = source_y_q4 >> 4;
        let phase_y = (source_y_q4 & 15) as usize;
        let filter_y = &kernel[phase_y * 8..phase_y * 8 + 8];
        for column in 0..output_width {
            let source_x_q4 = start_x_q4 + column as i64 * x_step_q4;
            let integer_x = source_x_q4 >> 4;
            let phase_x = (source_x_q4 & 15) as usize;
            let filter_x = &kernel[phase_x * 8..phase_x * 8 + 8];
            let prediction = match (phase_x == 0, phase_y == 0) {
                (true, true) => sample(integer_x, integer_y),
                (false, true) => {
                    let mut sum = 0i32;
                    for (tap, &coefficient) in filter_x.iter().enumerate() {
                        sum += i32::from(coefficient)
                            * i32::from(sample(integer_x + tap as i64 - 3, integer_y));
                    }
                    clip((sum + 64) >> 7)
                }
                (true, false) => {
                    let mut sum = 0i32;
                    for (tap, &coefficient) in filter_y.iter().enumerate() {
                        sum += i32::from(coefficient)
                            * i32::from(sample(integer_x, integer_y + tap as i64 - 3));
                    }
                    clip((sum + 64) >> 7)
                }
                (false, false) => {
                    let mut vertical_sum = 0i32;
                    for (vertical_tap, &vertical_coefficient) in filter_y.iter().enumerate() {
                        let source_y = integer_y + vertical_tap as i64 - 3;
                        let mut horizontal_sum = 0i32;
                        for (horizontal_tap, &horizontal_coefficient) in filter_x.iter().enumerate()
                        {
                            horizontal_sum += i32::from(horizontal_coefficient)
                                * i32::from(sample(
                                    integer_x + horizontal_tap as i64 - 3,
                                    source_y,
                                ));
                        }
                        let horizontal =
                            ((horizontal_sum + 64) >> 7).clamp(0, i32::from(max_sample));
                        vertical_sum += i32::from(vertical_coefficient) * horizontal;
                    }
                    clip((vertical_sum + 64) >> 7)
                }
            };
            let index = (y + row) * target_stride + target_x + column;
            write_prediction(&mut target[index], prediction, average);
        }
    }
}

#[inline(always)]
fn convolve_8_high_scalar(samples: &[u16], coefficients: &[i16], max_sample: u16) -> u16 {
    let mut sum = 0i32;
    for index in 0..8 {
        sum += i32::from(coefficients[index]) * i32::from(samples[index]);
    }
    ((sum + 64) >> 7).clamp(0, i32::from(max_sample)) as u16
}

fn convolve_8_horizontal_row_high(
    source: &[u16],
    target: &mut [u16],
    coefficients: &[i16],
    max_sample: u16,
) {
    debug_assert!(source.len() >= target.len() + 7 && coefficients.len() >= 8);
    let mut offset = 0;
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        let vectorized = target.len() / 8 * 8;
        // SAFETY: runtime feature detection guarantees AVX2 and the slice
        // lengths prove every eight-word load and store is in bounds.
        unsafe {
            x86::convolve_8_horizontal_high_avx2(
                source.as_ptr(),
                target.as_mut_ptr(),
                vectorized,
                coefficients.as_ptr(),
                max_sample,
            );
        }
        offset = vectorized;
    }
    for column in offset..target.len() {
        target[column] =
            convolve_8_high_scalar(&source[column..column + 8], coefficients, max_sample);
    }
}

fn convolve_8_vertical_row_high(
    samples: &[u16],
    start: usize,
    stride: usize,
    target: &mut [u16],
    coefficients: &[i16],
    max_sample: u16,
) {
    debug_assert!(
        coefficients.len() >= 8
            && (target.is_empty() || start + 7 * stride + target.len() <= samples.len())
    );
    let mut offset = 0;
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        let vectorized = target.len() / 8 * 8;
        // SAFETY: runtime feature detection guarantees AVX2. The assertion
        // above proves all eight source rows and the target range are valid.
        unsafe {
            x86::convolve_8_vertical_high_avx2(
                samples.as_ptr().add(start),
                stride,
                target.as_mut_ptr(),
                vectorized,
                coefficients.as_ptr(),
                max_sample,
            );
        }
        offset = vectorized;
    }
    for column in offset..target.len() {
        let mut sum = 0i32;
        for index in 0..8 {
            sum += i32::from(coefficients[index])
                * i32::from(samples[start + index * stride + column]);
        }
        target[column] = ((sum + 64) >> 7).clamp(0, i32::from(max_sample)) as u16;
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
    pub(super) unsafe fn average_high_avx2(
        target: *mut u16,
        prediction: *const u16,
        length: usize,
    ) {
        for offset in (0..length).step_by(16) {
            let current = unsafe { _mm256_loadu_si256(target.add(offset).cast::<__m256i>()) };
            let prediction =
                unsafe { _mm256_loadu_si256(prediction.add(offset).cast::<__m256i>()) };
            let average = _mm256_avg_epu16(current, prediction);
            unsafe {
                _mm256_storeu_si256(target.add(offset).cast::<__m256i>(), average);
            }
        }
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn convolve_8_horizontal_high_avx2(
        source: *const u16,
        target: *mut u16,
        length: usize,
        coefficients: *const i16,
        max_sample: u16,
    ) {
        let rounding = _mm256_set1_epi32(64);
        let zero = _mm256_setzero_si256();
        let maximum = _mm256_set1_epi32(i32::from(max_sample));
        for column in (0..length).step_by(8) {
            let mut sum = _mm256_setzero_si256();
            for tap in 0..8 {
                let samples =
                    unsafe { _mm_loadu_si128(source.add(column + tap).cast::<__m128i>()) };
                let samples = _mm256_cvtepu16_epi32(samples);
                let coefficient = _mm256_set1_epi32(i32::from(unsafe { *coefficients.add(tap) }));
                sum = _mm256_add_epi32(sum, _mm256_mullo_epi32(samples, coefficient));
            }
            sum = _mm256_srai_epi32::<7>(_mm256_add_epi32(sum, rounding));
            sum = _mm256_min_epi32(_mm256_max_epi32(sum, zero), maximum);
            let packed = _mm256_packus_epi32(sum, zero);
            let packed = _mm256_permute4x64_epi64::<0xd8>(packed);
            unsafe {
                _mm_storeu_si128(
                    target.add(column).cast::<__m128i>(),
                    _mm256_castsi256_si128(packed),
                );
            }
        }
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn convolve_8_vertical_high_avx2(
        source: *const u16,
        stride: usize,
        target: *mut u16,
        length: usize,
        coefficients: *const i16,
        max_sample: u16,
    ) {
        let rounding = _mm256_set1_epi32(64);
        let zero = _mm256_setzero_si256();
        let maximum = _mm256_set1_epi32(i32::from(max_sample));
        for column in (0..length).step_by(8) {
            let mut sum = _mm256_setzero_si256();
            for tap in 0..8 {
                let samples =
                    unsafe { _mm_loadu_si128(source.add(tap * stride + column).cast::<__m128i>()) };
                let samples = _mm256_cvtepu16_epi32(samples);
                let coefficient = _mm256_set1_epi32(i32::from(unsafe { *coefficients.add(tap) }));
                sum = _mm256_add_epi32(sum, _mm256_mullo_epi32(samples, coefficient));
            }
            sum = _mm256_srai_epi32::<7>(_mm256_add_epi32(sum, rounding));
            sum = _mm256_min_epi32(_mm256_max_epi32(sum, zero), maximum);
            let packed = _mm256_packus_epi32(sum, zero);
            let packed = _mm256_permute4x64_epi64::<0xd8>(packed);
            unsafe {
                _mm_storeu_si128(
                    target.add(column).cast::<__m128i>(),
                    _mm256_castsi256_si128(packed),
                );
            }
        }
    }

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

#[inline]
fn add_clipped<T: Sample>(sample: T, residual: i32, max_sample: u16) -> T {
    let value = sample.to_u32() as i32 + residual;
    T::from_u32(value.clamp(0, i32::from(max_sample)) as u32)
}

#[allow(clippy::too_many_arguments)]
fn add_residual_to_plane<T: Sample>(
    plane: &mut Arc<[T]>,
    stride: usize,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    transform_size: TransformSize,
    transform_type: TransformType,
    lossless: bool,
    coefficients: &[i32],
    max_sample: u16,
) {
    let size = 4usize << transform_size as usize;
    if lossless {
        debug_assert_eq!(transform_size, TransformSize::Tx4x4);
        debug_assert_eq!(transform_type, TransformType::DctDct);
        add_lossless_residual_to_plane(
            plane,
            stride,
            x,
            y,
            width,
            height,
            coefficients,
            max_sample,
        );
        return;
    }
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
        let pixels = Arc::make_mut(plane);
        for row in 0..visible_height {
            let start = (y + row) * stride + x;
            for pixel in &mut pixels[start..start + visible_width] {
                *pixel = add_clipped(*pixel, residual, max_sample);
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
    let pixels = Arc::make_mut(plane);
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
            pixels[index] = add_clipped(pixels[index], residual, max_sample);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn add_lossless_residual_to_plane<T: Sample>(
    plane: &mut Arc<[T]>,
    stride: usize,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    coefficients: &[i32],
    max_sample: u16,
) {
    debug_assert!(coefficients.len() >= 16);
    let mut intermediate = [0i32; 16];
    for row in 0..4 {
        let input = &coefficients[row * 4..row * 4 + 4];
        let mut a = input[0] >> 2;
        let mut c = input[1] >> 2;
        let mut d = input[2] >> 2;
        let mut b = input[3] >> 2;
        a += c;
        d -= b;
        let e = (a - d) >> 1;
        b = e - b;
        c = e - c;
        a -= b;
        d += c;
        intermediate[row * 4..row * 4 + 4].copy_from_slice(&[a, b, c, d]);
    }

    let visible_width = 4.min(width.saturating_sub(x));
    let visible_height = 4.min(height.saturating_sub(y));
    let pixels = Arc::make_mut(plane);
    for column in 0..visible_width {
        let mut a = intermediate[column];
        let mut c = intermediate[4 + column];
        let mut d = intermediate[8 + column];
        let mut b = intermediate[12 + column];
        a += c;
        d -= b;
        let e = (a - d) >> 1;
        b = e - b;
        c = e - c;
        a -= b;
        d += c;
        for (row, residual) in [a, b, c, d].into_iter().enumerate().take(visible_height) {
            let index = (y + row) * stride + x + column;
            pixels[index] = add_clipped(pixels[index], residual, max_sample);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn predict_plane<T: Sample>(
    plane: &mut Arc<[T]>,
    stride: usize,
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    size: usize,
    mode: IntraMode,
    have_top: bool,
    have_left: bool,
    right_available: bool,
    bit_depth: BitDepth,
) {
    let shift = u32::from(bit_depth.bits() - 8);
    let prediction_base = 128 << shift;
    let unavailable_above = T::from_u32(prediction_base - 1);
    let unavailable_left = T::from_u32(prediction_base + 1);
    let mut above = [unavailable_above; 64];
    let mut left = [unavailable_left; 32];
    let pixels = plane.as_ref();

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
        (true, false) => unavailable_left,
        (false, _) => unavailable_above,
    };

    let mut prediction = [T::default(); 32 * 32];
    intra_predict(
        &mut prediction[..size * size],
        size,
        mode,
        &above[..size * 2],
        &left[..size],
        top_left,
        have_top,
        have_left,
        bit_depth.max_sample(),
    );
    let pixels = Arc::make_mut(plane);
    let visible_width = size.min(width.saturating_sub(x));
    let visible_height = size.min(height.saturating_sub(y));
    for row in 0..visible_height {
        let target = (y + row) * stride + x;
        pixels[target..target + visible_width]
            .copy_from_slice(&prediction[row * size..row * size + visible_width]);
    }
}

fn avg2<T: Sample>(first: T, second: T) -> T {
    T::from_u32((first.to_u32() + second.to_u32() + 1) >> 1)
}

#[inline(always)]
fn write_prediction<T: Sample>(target: &mut T, prediction: T, average: bool) {
    if average {
        *target = avg2(*target, prediction);
    } else {
        *target = prediction;
    }
}

fn write_prediction_row<T: Sample>(target: &mut [T], prediction: &[T], average: bool) {
    debug_assert_eq!(target.len(), prediction.len());
    if average {
        for (target, &prediction) in target.iter_mut().zip(prediction) {
            *target = avg2(*target, prediction);
        }
    } else {
        target.copy_from_slice(prediction);
    }
}

fn write_prediction_row_high(target: &mut [u16], prediction: &[u16], average: bool) {
    debug_assert_eq!(target.len(), prediction.len());
    if !average {
        target.copy_from_slice(prediction);
        return;
    }
    let mut offset = 0;
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        let vectorized = target.len() / 16 * 16;
        // SAFETY: runtime feature detection guarantees AVX2 and both slices
        // contain the complete vectorized range.
        unsafe {
            x86::average_high_avx2(target.as_mut_ptr(), prediction.as_ptr(), vectorized);
        }
        offset = vectorized;
    }
    for (target, &prediction) in target[offset..].iter_mut().zip(&prediction[offset..]) {
        *target = avg2(*target, prediction);
    }
}

fn avg3<T: Sample>(first: T, middle: T, last: T) -> T {
    T::from_u32((first.to_u32() + 2 * middle.to_u32() + last.to_u32() + 2) >> 2)
}

#[allow(clippy::too_many_arguments)]
fn intra_predict<T: Sample>(
    target: &mut [T],
    size: usize,
    mode: IntraMode,
    above: &[T],
    left: &[T],
    top_left: T,
    have_top: bool,
    have_left: bool,
    max_sample: u16,
) {
    let pixel = |x: usize, y: usize| y * size + x;
    match mode {
        IntraMode::Dc => {
            let value = match (have_top, have_left) {
                (false, false) => (u32::from(max_sample) + 1) >> 1,
                (true, false) => {
                    (above[..size]
                        .iter()
                        .map(|&sample| sample.to_u32())
                        .sum::<u32>()
                        + size as u32 / 2)
                        / size as u32
                }
                (false, true) => {
                    (left.iter().map(|&sample| sample.to_u32()).sum::<u32>() + size as u32 / 2)
                        / size as u32
                }
                (true, true) => {
                    (above[..size]
                        .iter()
                        .map(|&sample| sample.to_u32())
                        .sum::<u32>()
                        + left.iter().map(|&sample| sample.to_u32()).sum::<u32>()
                        + size as u32)
                        / (size * 2) as u32
                }
            };
            target.fill(T::from_u32(value));
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
                    let prediction = left[row].to_u32() as i32 + above[column].to_u32() as i32
                        - top_left.to_u32() as i32;
                    target[pixel(column, row)] =
                        T::from_u32(prediction.clamp(0, i32::from(max_sample)) as u32);
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
            let mut border = vec![T::default(); size * 2 - 1];
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
    use super::{
        IntraMode, IntraPicture, convolve_8_high_scalar, convolve_8_horizontal_row_high,
        convolve_8_vertical_row_high, intra_predict,
    };
    use crate::{
        BitDepth, ChromaSubsampling,
        block::{TransformSize, TransformType},
    };

    #[test]
    fn vertical_prediction_repeats_top_row() {
        let mut target = [0u8; 16];
        intra_predict(
            &mut target,
            4,
            IntraMode::Vertical,
            &[1, 2, 3, 4, 4, 4, 4, 4],
            &[9; 4],
            8,
            true,
            true,
            255,
        );
        assert_eq!(target, [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]);
    }

    #[test]
    fn planar_layout_tracks_independent_chroma_subsampling() {
        let layouts = [
            (ChromaSubsampling::Cs444, 7, 5, 35),
            (ChromaSubsampling::Cs422, 4, 5, 20),
            (ChromaSubsampling::Cs440, 7, 3, 21),
            (ChromaSubsampling::Cs420, 4, 3, 12),
        ];
        for (subsampling, chroma_width, chroma_height, chroma_len) in layouts {
            let picture = IntraPicture::new(7, 5, subsampling, BitDepth::Eight);
            assert_eq!(picture.stride(1), chroma_width);
            assert_eq!(picture.plane_height(1), chroma_height);
            assert_eq!(picture.plane(1).len(), chroma_len);
            assert_eq!(picture.plane(2).len(), chroma_len);
        }
    }

    #[test]
    fn high_bit_depth_layout_uses_word_planes_and_byte_strides() {
        let picture = IntraPicture::new(7, 5, ChromaSubsampling::Cs420, BitDepth::Ten);

        assert_eq!(picture.bit_depth(), BitDepth::Ten);
        assert_eq!(picture.stride(0), 14);
        assert_eq!(picture.stride(1), 8);
        assert_eq!(picture.plane_u16(0).unwrap().len(), 35);
        assert_eq!(picture.plane_u16(1).unwrap().len(), 12);
        assert!(
            picture
                .plane_u16(0)
                .unwrap()
                .iter()
                .all(|&sample| sample == 0)
        );
    }

    #[test]
    fn high_bit_depth_tile_strips_merge_in_sample_coordinates() {
        let mut picture = IntraPicture::new(8, 2, ChromaSubsampling::Cs420, BitDepth::Twelve);
        let mut strip =
            IntraPicture::new_strip(8, 2, 2, 2, ChromaSubsampling::Cs420, BitDepth::Twelve);
        let [y, u, v] = strip.planes_u16_mut();
        y.fill(100);
        u.fill(200);
        v.fill(300);

        picture.copy_strip_from(&strip);

        assert_eq!(
            picture.plane_u16(0).unwrap(),
            &[0, 0, 100, 100, 0, 0, 0, 0, 0, 0, 100, 100, 0, 0, 0, 0]
        );
        assert_eq!(picture.plane_u16(1).unwrap(), &[0, 200, 0, 0]);
        assert_eq!(picture.plane_u16(2).unwrap(), &[0, 300, 0, 0]);
    }

    #[test]
    fn high_bit_depth_intra_prediction_and_lossless_residual_use_full_range() {
        let mut picture = IntraPicture::new(4, 4, ChromaSubsampling::Cs444, BitDepth::Ten);
        picture.predict(0, 0, 0, 4, IntraMode::Dc, 0, 0, false);
        assert_eq!(picture.plane_u16(0).unwrap(), &[512; 16]);

        let mut coefficients = [0; 16];
        coefficients[0] = 16;
        picture.add_residual(
            0,
            0,
            0,
            TransformSize::Tx4x4,
            TransformType::DctDct,
            true,
            &coefficients,
        );
        assert_eq!(picture.plane_u16(0).unwrap(), &[513; 16]);
    }

    #[test]
    fn high_bit_depth_directional_prediction_uses_centered_missing_edges() {
        let mut picture = IntraPicture::new(4, 4, ChromaSubsampling::Cs444, BitDepth::Ten);
        picture.predict(0, 0, 0, 4, IntraMode::Vertical, 0, 0, false);
        assert_eq!(picture.plane_u16(0).unwrap(), &[511; 16]);

        let mut picture = IntraPicture::new(4, 4, ChromaSubsampling::Cs444, BitDepth::Twelve);
        picture.predict(0, 0, 0, 4, IntraMode::Horizontal, 0, 0, false);
        assert_eq!(picture.plane_u16(0).unwrap(), &[2049; 16]);
    }

    #[test]
    fn lossless_walsh_hadamard_reconstructs_dc_block() {
        let mut picture = IntraPicture::new(4, 4, ChromaSubsampling::Cs444, BitDepth::Eight);
        let mut coefficients = [0; 16];
        coefficients[0] = 16;
        picture.add_residual(
            0,
            0,
            0,
            TransformSize::Tx4x4,
            TransformType::DctDct,
            true,
            &coefficients,
        );
        assert_eq!(picture.plane(0), &[1; 16]);
    }

    #[test]
    fn scaled_inter_prediction_maps_reference_coordinates() {
        let mut reference = IntraPicture::new(4, 4, ChromaSubsampling::Cs420, BitDepth::Eight);
        reference.planes_mut()[0]
            .copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
        let mut target = IntraPicture::new(8, 8, ChromaSubsampling::Cs420, BitDepth::Eight);
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
    fn high_bit_depth_scaled_inter_prediction_preserves_word_samples() {
        let mut reference = IntraPicture::new(4, 4, ChromaSubsampling::Cs420, BitDepth::Ten);
        reference.planes_u16_mut()[0]
            .copy_from_slice(&[4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64]);
        let mut target = IntraPicture::new(8, 8, ChromaSubsampling::Cs420, BitDepth::Ten);
        let mut nearest_kernel = [0i16; 128];
        for phase in 0..16 {
            nearest_kernel[phase * 8 + 3] = 128;
        }

        target.predict_inter(&reference, 0, 0, 0, 8, 8, 0, 0, &nearest_kernel, false);

        assert_eq!(
            &target.plane_u16(0).unwrap()[..16],
            &[4, 4, 8, 8, 12, 12, 16, 16, 4, 4, 8, 8, 12, 12, 16, 16]
        );
        assert_eq!(
            &target.plane_u16(0).unwrap()[48..],
            &[
                52, 52, 56, 56, 60, 60, 64, 64, 52, 52, 56, 56, 60, 60, 64, 64
            ]
        );
    }

    #[test]
    fn high_bit_depth_integer_inter_prediction_uses_exact_reference_samples() {
        let mut reference = IntraPicture::new(4, 4, ChromaSubsampling::Cs420, BitDepth::Ten);
        reference.planes_u16_mut()[0].copy_from_slice(&[
            100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115,
        ]);
        let mut target = IntraPicture::new(4, 4, ChromaSubsampling::Cs420, BitDepth::Ten);
        let mut nearest_kernel = [0i16; 128];
        nearest_kernel[3] = 128;

        target.predict_inter(&reference, 0, 0, 0, 4, 4, 0, 0, &nearest_kernel, false);

        assert_eq!(target.plane_u16(0), reference.plane_u16(0));

        target.planes_u16_mut()[0].fill(200);
        target.predict_inter(&reference, 0, 0, 0, 4, 4, 0, 0, &nearest_kernel, true);
        let expected: Vec<_> = reference
            .plane_u16(0)
            .unwrap()
            .iter()
            .map(|&sample| (200 + sample + 1) >> 1)
            .collect();
        assert_eq!(target.plane_u16(0).unwrap(), expected);
    }

    #[test]
    fn high_bit_depth_row_convolution_matches_scalar_reference() {
        let coefficients = [-1, 3, -7, 127, 8, -3, 1, 0];
        let mut state = 0x7a31_49c2u32;
        let mut source = [0u16; 71];
        for sample in &mut source {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *sample = ((state >> 16) & 0x0fff) as u16;
        }
        let expected: Vec<_> = (0..64)
            .map(|column| convolve_8_high_scalar(&source[column..column + 8], &coefficients, 4095))
            .collect();
        let mut actual = [0u16; 64];
        convolve_8_horizontal_row_high(&source, &mut actual, &coefficients, 4095);
        assert_eq!(actual.as_slice(), expected);

        let mut source = [0u16; 8 * 64];
        for sample in &mut source {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *sample = ((state >> 16) & 0x0fff) as u16;
        }
        let expected: Vec<_> = (0..64)
            .map(|column| {
                let samples = std::array::from_fn::<_, 8, _>(|row| source[row * 64 + column]);
                convolve_8_high_scalar(&samples, &coefficients, 4095)
            })
            .collect();
        convolve_8_vertical_row_high(&source, 0, 64, &mut actual, &coefficients, 4095);
        assert_eq!(actual.as_slice(), expected);
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
                    255,
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
