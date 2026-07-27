use crate::{
    BitDepth, FrameHeader, Result, Vp9Error,
    block::{BlockSize, TransformSize},
    reconstruct::IntraPicture,
    tile::floor_transform,
};

#[derive(Debug, Clone, Copy)]
pub(crate) struct FilterMode {
    pub(crate) block_size: BlockSize,
    pub(crate) transform_size: TransformSize,
    pub(crate) skip: bool,
    pub(crate) segment_id: u8,
    /// INTRA=0, LAST=1, GOLDEN=2, ALTREF=3.
    pub(crate) reference: u8,
    /// The loop-filter mode class: zero motion/intra=0, other inter modes=1.
    pub(crate) mode_class: u8,
}

#[derive(Debug, Clone)]
pub(crate) struct FilterModeMap {
    mi_columns: usize,
    mi_rows: usize,
    modes: Vec<FilterMode>,
}

impl FilterModeMap {
    pub(crate) fn new(mi_columns: usize, mi_rows: usize, modes: Vec<FilterMode>) -> Result<Self> {
        if modes.len() != mi_columns.saturating_mul(mi_rows) {
            return Err(Vp9Error::InvalidData(
                "loop-filter mode map dimensions are inconsistent",
            ));
        }
        Ok(Self {
            mi_columns,
            mi_rows,
            modes,
        })
    }

    #[inline]
    fn get(&self, row: usize, column: usize) -> FilterMode {
        self.modes[row * self.mi_columns + column]
    }

    pub(crate) fn segment_ids(&self) -> Vec<u8> {
        self.modes.iter().map(|mode| mode.segment_id).collect()
    }
}

#[derive(Debug, Clone, Copy)]
enum FilterWidth {
    Four,
    Eight,
    Sixteen,
}

#[derive(Debug, Clone, Copy)]
struct Thresholds {
    limit: u8,
    blimit: u8,
    hev: u8,
}

#[derive(Debug, Clone, Copy)]
struct PreparedFilter {
    mode: FilterMode,
    thresholds: Thresholds,
    transform: TransformSize,
}

trait LoopSample: Copy + Default + Send {
    fn to_i32(self) -> i32;
    fn from_i32(value: i32) -> Self;

    #[allow(clippy::too_many_arguments)]
    fn filter_vertical(
        pixels: &mut [Self],
        stride: usize,
        plane_width: usize,
        x: usize,
        y: usize,
        count: usize,
        width: FilterWidth,
        thresholds: Thresholds,
        bit_depth: BitDepth,
    );

    #[allow(clippy::too_many_arguments)]
    fn filter_horizontal(
        pixels: &mut [Self],
        stride: usize,
        plane_height: usize,
        x: usize,
        y: usize,
        count: usize,
        width: FilterWidth,
        thresholds: Thresholds,
        bit_depth: BitDepth,
    );
}

pub(crate) fn apply_loop_filter(
    picture: &mut IntraPicture,
    header: &FrameHeader,
    modes: &FilterModeMap,
) -> Result<()> {
    let Some(configuration) = &header.loop_filter else {
        return Ok(());
    };
    if configuration.level == 0 {
        return Ok(());
    }
    let expected_columns = picture.width().div_ceil(8);
    let expected_rows = picture.height().div_ceil(8);
    if modes.mi_columns != expected_columns || modes.mi_rows != expected_rows {
        return Err(Vp9Error::InvalidData(
            "loop-filter mode map does not match the picture",
        ));
    }

    let plane_widths: [usize; 3] = std::array::from_fn(|plane| picture.plane_width(plane));
    let plane_heights: [usize; 3] = std::array::from_fn(|plane| picture.plane_height(plane));
    let plane_strides: [usize; 3] = std::array::from_fn(|plane| picture.sample_stride(plane));
    let subsampling_x: [usize; 3] = std::array::from_fn(|plane| picture.subsampling_x(plane));
    let subsampling_y: [usize; 3] = std::array::from_fn(|plane| picture.subsampling_y(plane));
    let bit_depth = picture.bit_depth();
    match bit_depth {
        BitDepth::Eight => apply_all_planes(
            picture.planes_mut(),
            plane_widths,
            plane_heights,
            plane_strides,
            subsampling_x,
            subsampling_y,
            header,
            modes,
            bit_depth,
        ),
        BitDepth::Ten | BitDepth::Twelve => apply_all_planes(
            picture.planes_u16_mut(),
            plane_widths,
            plane_heights,
            plane_strides,
            subsampling_x,
            subsampling_y,
            header,
            modes,
            bit_depth,
        ),
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn apply_all_planes<T: LoopSample>(
    [luma, chroma_u, chroma_v]: [&mut [T]; 3],
    plane_widths: [usize; 3],
    plane_heights: [usize; 3],
    plane_strides: [usize; 3],
    subsampling_x: [usize; 3],
    subsampling_y: [usize; 3],
    header: &FrameHeader,
    modes: &FilterModeMap,
    bit_depth: BitDepth,
) {
    std::thread::scope(|scope| {
        let u_worker = scope.spawn(|| {
            apply_plane_loop_filter(
                chroma_u,
                plane_widths[1],
                plane_heights[1],
                plane_strides[1],
                subsampling_x[1],
                subsampling_y[1],
                header,
                modes,
                bit_depth,
            );
        });
        let v_worker = scope.spawn(|| {
            apply_plane_loop_filter(
                chroma_v,
                plane_widths[2],
                plane_heights[2],
                plane_strides[2],
                subsampling_x[2],
                subsampling_y[2],
                header,
                modes,
                bit_depth,
            );
        });
        apply_plane_loop_filter(
            luma,
            plane_widths[0],
            plane_heights[0],
            plane_strides[0],
            subsampling_x[0],
            subsampling_y[0],
            header,
            modes,
            bit_depth,
        );
        u_worker.join().expect("VP9 U-plane loop filter panicked");
        v_worker.join().expect("VP9 V-plane loop filter panicked");
    });
}

#[allow(clippy::too_many_arguments)]
fn apply_plane_loop_filter<T: LoopSample>(
    pixels: &mut [T],
    width: usize,
    height: usize,
    stride: usize,
    subsampling_x: usize,
    subsampling_y: usize,
    header: &FrameHeader,
    modes: &FilterModeMap,
    bit_depth: BitDepth,
) {
    let configuration = header.loop_filter.as_ref().expect("caller checked");
    let row_step = 1usize << subsampling_y;
    let column_step = 1usize << subsampling_x;

    for superblock_row in (0..modes.mi_rows).step_by(8) {
        let row_end = (superblock_row + 8).min(modes.mi_rows);
        for superblock_column in (0..modes.mi_columns).step_by(8) {
            let column_end = (superblock_column + 8).min(modes.mi_columns);
            let mut prepared = [None; 64];

            // VP9 completes the vertical and then horizontal pass for each
            // 64x64 superblock before moving to the next one.
            for mi_row in (superblock_row..row_end).step_by(row_step) {
                let y = (mi_row * 8) >> subsampling_y;
                let line_count = 8.min(height.saturating_sub(y));
                for mi_column in (superblock_column..column_end).step_by(column_step) {
                    let x = (mi_column * 8) >> subsampling_x;
                    let mode = modes.get(mi_row, mi_column);
                    let level = filter_level(header, mode);
                    if level == 0 {
                        continue;
                    }
                    let thresholds = thresholds(level, configuration.sharpness);
                    let transform = plane_transform(mode, subsampling_x, subsampling_y);
                    prepared[(mi_row - superblock_row) * 8 + mi_column - superblock_column] =
                        Some(PreparedFilter {
                            mode,
                            thresholds,
                            transform,
                        });
                    let block_edge = is_left_block_edge(mode.block_size, mi_column);
                    let skip_edge = mode.skip && mode.reference != 0 && !block_edge;
                    if x != 0
                        && !skip_edge
                        && let Some(width_kind) =
                            edge_filter_width(transform, x / 8, width.saturating_sub(x))
                    {
                        T::filter_vertical(
                            pixels, stride, width, x, y, line_count, width_kind, thresholds,
                            bit_depth,
                        );
                    }
                    if transform == TransformSize::Tx4x4
                        && !(mode.skip && mode.reference != 0)
                        && x + 4 < width
                    {
                        T::filter_vertical(
                            pixels,
                            stride,
                            width,
                            x + 4,
                            y,
                            line_count,
                            FilterWidth::Four,
                            thresholds,
                            bit_depth,
                        );
                    }
                }
            }

            for mi_row in (superblock_row..row_end).step_by(row_step) {
                let y = (mi_row * 8) >> subsampling_y;
                for mi_column in (superblock_column..column_end).step_by(column_step) {
                    let x = (mi_column * 8) >> subsampling_x;
                    let column_count = 8.min(width.saturating_sub(x));
                    let Some(prepared) =
                        prepared[(mi_row - superblock_row) * 8 + mi_column - superblock_column]
                    else {
                        continue;
                    };
                    let mode = prepared.mode;
                    let thresholds = prepared.thresholds;
                    let transform = prepared.transform;
                    let block_edge = is_top_block_edge(mode.block_size, mi_row);
                    let skip_edge = mode.skip && mode.reference != 0 && !block_edge;
                    if y != 0
                        && !skip_edge
                        && let Some(width_kind) =
                            edge_filter_width(transform, y / 8, height.saturating_sub(y))
                    {
                        T::filter_horizontal(
                            pixels,
                            stride,
                            height,
                            x,
                            y,
                            column_count,
                            width_kind,
                            thresholds,
                            bit_depth,
                        );
                    }
                    if transform == TransformSize::Tx4x4
                        && !(mode.skip && mode.reference != 0)
                        && y + 4 < height
                    {
                        T::filter_horizontal(
                            pixels,
                            stride,
                            height,
                            x,
                            y + 4,
                            column_count,
                            FilterWidth::Four,
                            thresholds,
                            bit_depth,
                        );
                    }
                }
            }
        }
    }
}

fn plane_transform(mode: FilterMode, subsampling_x: usize, subsampling_y: usize) -> TransformSize {
    if subsampling_x == 0 && subsampling_y == 0 {
        return mode.transform_size;
    }
    let (width, height) = if mode.block_size < BlockSize::B8x8 {
        (1, 1)
    } else {
        (
            mode.block_size.width_4x4().div_ceil(1 << subsampling_x),
            mode.block_size.height_4x4().div_ceil(1 << subsampling_y),
        )
    };
    mode.transform_size.min(floor_transform(width.min(height)))
}

fn is_left_block_edge(block: BlockSize, mi_column: usize) -> bool {
    let width = block.width_mi();
    width == 1 || mi_column.is_multiple_of(width)
}

fn is_top_block_edge(block: BlockSize, mi_row: usize) -> bool {
    let height = block.height_mi();
    height == 1 || mi_row.is_multiple_of(height)
}

fn edge_filter_width(
    transform: TransformSize,
    unit_index: usize,
    following_pixels: usize,
) -> Option<FilterWidth> {
    match transform {
        TransformSize::Tx32x32 if unit_index.is_multiple_of(4) => Some(if following_pixels >= 8 {
            FilterWidth::Sixteen
        } else {
            FilterWidth::Eight
        }),
        TransformSize::Tx16x16 if unit_index.is_multiple_of(2) => Some(if following_pixels >= 8 {
            FilterWidth::Sixteen
        } else {
            FilterWidth::Eight
        }),
        TransformSize::Tx8x8 => Some(FilterWidth::Eight),
        TransformSize::Tx4x4 if unit_index.is_multiple_of(4) => Some(FilterWidth::Eight),
        TransformSize::Tx4x4 => Some(FilterWidth::Four),
        TransformSize::Tx16x16 | TransformSize::Tx32x32 => None,
    }
}

fn filter_level(header: &FrameHeader, mode: FilterMode) -> u8 {
    let configuration = header.loop_filter.as_ref().expect("caller checked");
    let mut level = i16::from(configuration.level);
    if let Some(segmentation) = &header.segmentation {
        let alternate = segmentation.features[usize::from(mode.segment_id)][1];
        if alternate.enabled {
            level = if segmentation.absolute_values {
                alternate.value
            } else {
                level + alternate.value
            };
            level = level.clamp(0, 63);
        }
    }
    if configuration.mode_ref_delta_enabled {
        let scale = 1i16 << (configuration.level >> 5);
        level += i16::from(configuration.reference_deltas[usize::from(mode.reference)]) * scale;
        if mode.reference != 0 {
            level += i16::from(configuration.mode_deltas[usize::from(mode.mode_class)]) * scale;
        }
    }
    level.clamp(0, 63) as u8
}

fn thresholds(level: u8, sharpness: u8) -> Thresholds {
    let mut limit = level >> (u8::from(sharpness > 0) + u8::from(sharpness > 4));
    if sharpness > 0 {
        limit = limit.min(9 - sharpness);
    }
    limit = limit.max(1);
    Thresholds {
        limit,
        blimit: 2 * (level + 2) + limit,
        hev: level >> 4,
    }
}

#[allow(clippy::too_many_arguments)]
fn filter_vertical(
    pixels: &mut [u8],
    stride: usize,
    plane_width: usize,
    x: usize,
    y: usize,
    count: usize,
    width: FilterWidth,
    thresholds: Thresholds,
) {
    let reach = match width {
        FilterWidth::Sixteen => 8,
        FilterWidth::Four | FilterWidth::Eight => 4,
    };
    if x < reach || x + reach > plane_width {
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if matches!(width, FilterWidth::Four)
        && count == 8
        && std::arch::is_x86_feature_detected!("sse2")
    {
        // SAFETY: the reach checks above prove four pixels exist on each
        // side of the edge for all eight rows.
        unsafe {
            x86::filter_vertical_4_sse2(
                pixels.as_mut_ptr().add(y * stride + x),
                stride,
                thresholds,
            );
        }
        return;
    }
    filter_vertical_scalar(
        pixels,
        stride,
        plane_width,
        x,
        y,
        count,
        width,
        thresholds,
        BitDepth::Eight,
    );
}

#[allow(clippy::too_many_arguments)]
fn filter_vertical_scalar<T: LoopSample>(
    pixels: &mut [T],
    stride: usize,
    plane_width: usize,
    x: usize,
    y: usize,
    count: usize,
    width: FilterWidth,
    thresholds: Thresholds,
    bit_depth: BitDepth,
) {
    let reach = match width {
        FilterWidth::Sixteen => 8,
        FilterWidth::Four | FilterWidth::Eight => 4,
    };
    if x < reach || x + reach > plane_width {
        return;
    }
    for row in 0..count {
        let base = (y + row) * stride + x;
        let mut samples = [T::default(); 16];
        for offset in 0..reach * 2 {
            samples[8 - reach + offset] = pixels[base - reach + offset];
        }
        filter_samples(&mut samples, width, thresholds, bit_depth);
        for offset in 0..reach * 2 {
            pixels[base - reach + offset] = samples[8 - reach + offset];
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn filter_horizontal(
    pixels: &mut [u8],
    stride: usize,
    plane_height: usize,
    x: usize,
    y: usize,
    count: usize,
    width: FilterWidth,
    thresholds: Thresholds,
) {
    let reach = match width {
        FilterWidth::Sixteen => 8,
        FilterWidth::Four | FilterWidth::Eight => 4,
    };
    if y < reach || y + reach > plane_height {
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if matches!(width, FilterWidth::Four)
        && count == 8
        && std::arch::is_x86_feature_detected!("sse2")
    {
        // SAFETY: the reach checks above prove four rows exist on each side
        // and count proves every 8-byte row load/store is in bounds.
        unsafe {
            x86::filter_horizontal_4_sse2(
                pixels.as_mut_ptr().add(y * stride + x),
                stride,
                thresholds,
            );
        }
        return;
    }
    filter_horizontal_scalar(
        pixels,
        stride,
        plane_height,
        x,
        y,
        count,
        width,
        thresholds,
        BitDepth::Eight,
    );
}

#[allow(clippy::too_many_arguments)]
fn filter_horizontal_scalar<T: LoopSample>(
    pixels: &mut [T],
    stride: usize,
    plane_height: usize,
    x: usize,
    y: usize,
    count: usize,
    width: FilterWidth,
    thresholds: Thresholds,
    bit_depth: BitDepth,
) {
    let reach = match width {
        FilterWidth::Sixteen => 8,
        FilterWidth::Four | FilterWidth::Eight => 4,
    };
    if y < reach || y + reach > plane_height {
        return;
    }
    for column in 0..count {
        let mut samples = [T::default(); 16];
        for offset in 0..reach * 2 {
            samples[8 - reach + offset] = pixels[(y + offset - reach) * stride + x + column];
        }
        filter_samples(&mut samples, width, thresholds, bit_depth);
        for offset in 0..reach * 2 {
            pixels[(y + offset - reach) * stride + x + column] = samples[8 - reach + offset];
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn filter_vertical_high(
    pixels: &mut [u16],
    stride: usize,
    plane_width: usize,
    x: usize,
    y: usize,
    count: usize,
    width: FilterWidth,
    thresholds: Thresholds,
    bit_depth: BitDepth,
) {
    let reach = match width {
        FilterWidth::Sixteen => 8,
        FilterWidth::Four | FilterWidth::Eight => 4,
    };
    if x < reach || x + reach > plane_width {
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if count == 8 && std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: the reach checks above prove four samples exist on each
        // side of the edge for all eight rows, and AVX2 was detected.
        unsafe {
            x86::filter_vertical_high_avx2(
                pixels.as_mut_ptr().add(y * stride + x),
                stride,
                width,
                thresholds,
                bit_depth,
            );
        }
        return;
    }
    filter_vertical_scalar(
        pixels,
        stride,
        plane_width,
        x,
        y,
        count,
        width,
        thresholds,
        bit_depth,
    );
}

#[allow(clippy::too_many_arguments)]
fn filter_horizontal_high(
    pixels: &mut [u16],
    stride: usize,
    plane_height: usize,
    x: usize,
    y: usize,
    count: usize,
    width: FilterWidth,
    thresholds: Thresholds,
    bit_depth: BitDepth,
) {
    let reach = match width {
        FilterWidth::Sixteen => 8,
        FilterWidth::Four | FilterWidth::Eight => 4,
    };
    if y < reach || y + reach > plane_height {
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if count == 8 && std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: the reach checks above prove four rows exist on each side
        // and count proves every 8-sample row load/store is in bounds.
        unsafe {
            x86::filter_horizontal_high_avx2(
                pixels.as_mut_ptr().add(y * stride + x),
                stride,
                width,
                thresholds,
                bit_depth,
            );
        }
        return;
    }
    filter_horizontal_scalar(
        pixels,
        stride,
        plane_height,
        x,
        y,
        count,
        width,
        thresholds,
        bit_depth,
    );
}

fn filter_samples<T: LoopSample>(
    samples: &mut [T; 16],
    width: FilterWidth,
    thresholds: Thresholds,
    bit_depth: BitDepth,
) {
    let shift = u32::from(bit_depth.bits() - 8);
    let mask = filter_mask(
        u32::from(thresholds.limit) << shift,
        u32::from(thresholds.blimit) << shift,
        &samples[4..12],
    );
    if !mask {
        return;
    }
    let flat = flat_mask(&samples[4..12], shift);
    match width {
        FilterWidth::Four => filter_four(samples, thresholds.hev, bit_depth),
        FilterWidth::Eight if flat => filter_eight(samples),
        FilterWidth::Eight => filter_four(samples, thresholds.hev, bit_depth),
        FilterWidth::Sixteen if flat && flat2_mask(samples, shift) => filter_sixteen(samples),
        FilterWidth::Sixteen if flat => filter_eight(samples),
        FilterWidth::Sixteen => filter_four(samples, thresholds.hev, bit_depth),
    }
}

impl LoopSample for u8 {
    #[inline]
    fn to_i32(self) -> i32 {
        i32::from(self)
    }

    #[inline]
    fn from_i32(value: i32) -> Self {
        value as Self
    }

    fn filter_vertical(
        pixels: &mut [Self],
        stride: usize,
        plane_width: usize,
        x: usize,
        y: usize,
        count: usize,
        width: FilterWidth,
        thresholds: Thresholds,
        bit_depth: BitDepth,
    ) {
        debug_assert_eq!(bit_depth, BitDepth::Eight);
        filter_vertical(pixels, stride, plane_width, x, y, count, width, thresholds);
    }

    fn filter_horizontal(
        pixels: &mut [Self],
        stride: usize,
        plane_height: usize,
        x: usize,
        y: usize,
        count: usize,
        width: FilterWidth,
        thresholds: Thresholds,
        bit_depth: BitDepth,
    ) {
        debug_assert_eq!(bit_depth, BitDepth::Eight);
        filter_horizontal(pixels, stride, plane_height, x, y, count, width, thresholds);
    }
}

impl LoopSample for u16 {
    #[inline]
    fn to_i32(self) -> i32 {
        i32::from(self)
    }

    #[inline]
    fn from_i32(value: i32) -> Self {
        value as Self
    }

    fn filter_vertical(
        pixels: &mut [Self],
        stride: usize,
        plane_width: usize,
        x: usize,
        y: usize,
        count: usize,
        width: FilterWidth,
        thresholds: Thresholds,
        bit_depth: BitDepth,
    ) {
        filter_vertical_high(
            pixels,
            stride,
            plane_width,
            x,
            y,
            count,
            width,
            thresholds,
            bit_depth,
        );
    }

    fn filter_horizontal(
        pixels: &mut [Self],
        stride: usize,
        plane_height: usize,
        x: usize,
        y: usize,
        count: usize,
        width: FilterWidth,
        thresholds: Thresholds,
        bit_depth: BitDepth,
    ) {
        filter_horizontal_high(
            pixels,
            stride,
            plane_height,
            x,
            y,
            count,
            width,
            thresholds,
            bit_depth,
        );
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;

    use super::{BitDepth, FilterWidth, Thresholds};

    #[target_feature(enable = "sse2")]
    unsafe fn abs_diff(first: __m128i, second: __m128i) -> __m128i {
        _mm_or_si128(_mm_subs_epu8(first, second), _mm_subs_epu8(second, first))
    }

    #[target_feature(enable = "sse2")]
    #[allow(clippy::too_many_arguments)]
    unsafe fn filter4(
        p3p2: __m128i,
        p2p1: __m128i,
        p1p0: __m128i,
        q3q2: __m128i,
        q2q1: __m128i,
        q1q0: __m128i,
        q1p1: __m128i,
        q0p0: __m128i,
        thresholds: Thresholds,
    ) -> (__m128i, __m128i) {
        let zero = _mm_setzero_si128();
        let all = _mm_cmpeq_epi8(zero, zero);
        let limit_v = _mm_unpacklo_epi64(
            _mm_set1_epi8(thresholds.blimit as i8),
            _mm_set1_epi8(thresholds.limit as i8),
        );
        let threshold_v = _mm_unpacklo_epi8(_mm_set1_epi8(thresholds.hev as i8), zero);

        let mut flat = unsafe { abs_diff(q1p1, q0p0) };
        let abs_p1q1p0q0 = unsafe { abs_diff(p1p0, q1q0) };
        let mut hev = _mm_unpacklo_epi8(_mm_max_epu8(flat, _mm_srli_si128::<8>(flat)), zero);
        hev = _mm_cmpgt_epi16(hev, threshold_v);
        hev = _mm_packs_epi16(hev, hev);

        let abs_p0q0 = _mm_adds_epu8(abs_p1q1p0q0, abs_p1q1p0q0);
        let mut abs_p1q1 = _mm_unpackhi_epi8(abs_p1q1p0q0, abs_p1q1p0q0);
        abs_p1q1 = _mm_srli_epi16::<9>(abs_p1q1);
        abs_p1q1 = _mm_packs_epi16(abs_p1q1, abs_p1q1);
        let mut mask = _mm_adds_epu8(abs_p0q0, abs_p1q1);
        flat = _mm_max_epu8(unsafe { abs_diff(p3p2, p2p1) }, flat);
        flat = _mm_max_epu8(unsafe { abs_diff(q3q2, q2q1) }, flat);
        flat = _mm_max_epu8(flat, _mm_srli_si128::<8>(flat));
        mask = _mm_unpacklo_epi64(mask, flat);
        mask = _mm_subs_epu8(mask, limit_v);
        mask = _mm_cmpeq_epi8(mask, zero);
        mask = _mm_and_si128(mask, _mm_srli_si128::<8>(mask));

        let sign = _mm_set1_epi8(0x80u8 as i8);
        let mut ps1ps0 = _mm_xor_si128(p1p0, sign);
        let mut qs1qs0 = _mm_xor_si128(q1q0, sign);
        let work = _mm_subs_epi8(ps1ps0, qs1qs0);
        let mut filter = _mm_and_si128(_mm_srli_si128::<8>(work), hev);
        filter = _mm_subs_epi8(filter, work);
        filter = _mm_subs_epi8(filter, work);
        filter = _mm_subs_epi8(filter, work);
        filter = _mm_and_si128(filter, mask);
        filter = _mm_unpacklo_epi64(filter, filter);

        let additions = _mm_set_epi8(3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4);
        let mut filter2filter1 = _mm_adds_epi8(filter, additions);
        filter = _mm_unpackhi_epi8(filter2filter1, filter2filter1);
        filter2filter1 = _mm_unpacklo_epi8(filter2filter1, filter2filter1);
        filter2filter1 = _mm_srai_epi16::<11>(filter2filter1);
        filter = _mm_srai_epi16::<11>(filter);
        filter2filter1 = _mm_packs_epi16(filter2filter1, filter);

        filter = _mm_subs_epi8(filter2filter1, all);
        filter = _mm_unpacklo_epi8(filter, filter);
        filter = _mm_srai_epi16::<9>(filter);
        filter = _mm_packs_epi16(filter, filter);
        filter = _mm_andnot_si128(hev, filter);

        hev = _mm_unpackhi_epi64(filter2filter1, filter);
        filter2filter1 = _mm_unpacklo_epi64(filter2filter1, filter);
        qs1qs0 = _mm_subs_epi8(qs1qs0, filter2filter1);
        ps1ps0 = _mm_adds_epi8(ps1ps0, hev);
        (_mm_xor_si128(ps1ps0, sign), _mm_xor_si128(qs1qs0, sign))
    }

    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn filter_horizontal_4_sse2(
        edge: *mut u8,
        stride: usize,
        thresholds: Thresholds,
    ) {
        let load = |row: isize| unsafe {
            _mm_loadl_epi64(edge.offset(row * stride as isize).cast::<__m128i>())
        };
        let p3p2 = _mm_unpacklo_epi64(load(-3), load(-4));
        let q1p1 = _mm_unpacklo_epi64(load(-2), load(1));
        let q0p0 = _mm_unpacklo_epi64(load(-1), load(0));
        let q3q2 = _mm_unpacklo_epi64(load(2), load(3));
        let p1p0 = _mm_unpacklo_epi64(q0p0, q1p1);
        let p2p1 = _mm_unpacklo_epi64(q1p1, p3p2);
        let q1q0 = _mm_unpackhi_epi64(q0p0, q1p1);
        let q2q1 = _mm_unpacklo_epi64(_mm_srli_si128::<8>(q1p1), q3q2);
        let (ps1ps0, qs1qs0) =
            unsafe { filter4(p3p2, p2p1, p1p0, q3q2, q2q1, q1q0, q1p1, q0p0, thresholds) };
        unsafe {
            _mm_storel_epi64(
                edge.offset(-2 * stride as isize).cast::<__m128i>(),
                _mm_srli_si128::<8>(ps1ps0),
            );
            _mm_storel_epi64(edge.offset(-(stride as isize)).cast::<__m128i>(), ps1ps0);
            _mm_storel_epi64(edge.cast::<__m128i>(), qs1qs0);
            _mm_storel_epi64(
                edge.add(stride).cast::<__m128i>(),
                _mm_srli_si128::<8>(qs1qs0),
            );
        }
    }

    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn filter_vertical_4_sse2(
        edge: *mut u8,
        stride: usize,
        thresholds: Thresholds,
    ) {
        let load_pair = |first: usize, second: usize| unsafe {
            _mm_unpacklo_epi8(
                _mm_loadl_epi64(edge.add(first * stride).sub(4).cast::<__m128i>()),
                _mm_loadl_epi64(edge.add(second * stride).sub(4).cast::<__m128i>()),
            )
        };
        let mut q1q0 = load_pair(0, 1);
        let x1 = load_pair(2, 3);
        let mut x2 = load_pair(4, 5);
        let x3 = load_pair(6, 7);

        let mut p1p0 = _mm_unpacklo_epi16(q1q0, x1);
        let x0 = _mm_unpacklo_epi16(x2, x3);
        let mut p3p2 = _mm_unpacklo_epi32(p1p0, x0);
        p1p0 = _mm_unpackhi_epi32(p1p0, x0);
        p3p2 = _mm_unpackhi_epi64(p3p2, _mm_slli_si128::<8>(p3p2));
        p1p0 = _mm_unpackhi_epi64(p1p0, _mm_slli_si128::<8>(p1p0));

        q1q0 = _mm_unpackhi_epi16(q1q0, x1);
        x2 = _mm_unpackhi_epi16(x2, x3);
        let q3q2 = _mm_unpackhi_epi32(q1q0, x2);
        q1q0 = _mm_unpacklo_epi32(q1q0, x2);
        let q0p0 = _mm_unpacklo_epi64(p1p0, q1q0);
        let q1p1 = _mm_unpackhi_epi64(p1p0, q1q0);
        p1p0 = _mm_unpacklo_epi64(q0p0, q1p1);
        let p2p1 = _mm_unpacklo_epi64(q1p1, p3p2);
        let q2q1 = _mm_unpacklo_epi64(_mm_srli_si128::<8>(q1p1), q3q2);

        let (mut ps1ps0, mut qs1qs0) =
            unsafe { filter4(p3p2, p2p1, p1p0, q3q2, q2q1, q1q0, q1p1, q0p0, thresholds) };
        ps1ps0 = _mm_unpackhi_epi64(ps1ps0, _mm_slli_si128::<8>(ps1ps0));
        let x0 = _mm_unpackhi_epi8(ps1ps0, qs1qs0);
        ps1ps0 = _mm_unpacklo_epi8(ps1ps0, qs1qs0);
        qs1qs0 = _mm_unpackhi_epi8(ps1ps0, x0);
        ps1ps0 = _mm_unpacklo_epi8(ps1ps0, x0);

        for row in 0..4 {
            unsafe {
                std::ptr::write_unaligned(
                    edge.add(row * stride).sub(2).cast::<i32>(),
                    _mm_cvtsi128_si32(ps1ps0),
                );
            }
            ps1ps0 = _mm_srli_si128::<4>(ps1ps0);
        }
        for row in 4..8 {
            unsafe {
                std::ptr::write_unaligned(
                    edge.add(row * stride).sub(2).cast::<i32>(),
                    _mm_cvtsi128_si32(qs1qs0),
                );
            }
            qs1qs0 = _mm_srli_si128::<4>(qs1qs0);
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn load_high_row(edge: *mut u16, stride: usize, row: isize) -> __m256i {
        let samples =
            unsafe { _mm_loadu_si128(edge.offset(row * stride as isize).cast::<__m128i>()) };
        _mm256_cvtepu16_epi32(samples)
    }

    #[target_feature(enable = "avx2")]
    unsafe fn pack_high(samples: __m256i) -> __m128i {
        _mm_packus_epi32(
            _mm256_castsi256_si128(samples),
            _mm256_extracti128_si256::<1>(samples),
        )
    }

    #[target_feature(enable = "avx2")]
    unsafe fn store_high_row(edge: *mut u16, stride: usize, row: isize, samples: __m256i) {
        let packed = unsafe { pack_high(samples) };
        unsafe {
            _mm_storeu_si128(edge.offset(row * stride as isize).cast::<__m128i>(), packed);
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn transpose_8x8_u16(rows: [__m128i; 8]) -> [__m128i; 8] {
        let pair01_low = _mm_unpacklo_epi16(rows[0], rows[1]);
        let pair01_high = _mm_unpackhi_epi16(rows[0], rows[1]);
        let pair23_low = _mm_unpacklo_epi16(rows[2], rows[3]);
        let pair23_high = _mm_unpackhi_epi16(rows[2], rows[3]);
        let pair45_low = _mm_unpacklo_epi16(rows[4], rows[5]);
        let pair45_high = _mm_unpackhi_epi16(rows[4], rows[5]);
        let pair67_low = _mm_unpacklo_epi16(rows[6], rows[7]);
        let pair67_high = _mm_unpackhi_epi16(rows[6], rows[7]);

        let group03_0 = _mm_unpacklo_epi32(pair01_low, pair23_low);
        let group03_1 = _mm_unpackhi_epi32(pair01_low, pair23_low);
        let group03_2 = _mm_unpacklo_epi32(pair01_high, pair23_high);
        let group03_3 = _mm_unpackhi_epi32(pair01_high, pair23_high);
        let group47_0 = _mm_unpacklo_epi32(pair45_low, pair67_low);
        let group47_1 = _mm_unpackhi_epi32(pair45_low, pair67_low);
        let group47_2 = _mm_unpacklo_epi32(pair45_high, pair67_high);
        let group47_3 = _mm_unpackhi_epi32(pair45_high, pair67_high);

        [
            _mm_unpacklo_epi64(group03_0, group47_0),
            _mm_unpackhi_epi64(group03_0, group47_0),
            _mm_unpacklo_epi64(group03_1, group47_1),
            _mm_unpackhi_epi64(group03_1, group47_1),
            _mm_unpacklo_epi64(group03_2, group47_2),
            _mm_unpackhi_epi64(group03_2, group47_2),
            _mm_unpacklo_epi64(group03_3, group47_3),
            _mm_unpackhi_epi64(group03_3, group47_3),
        ]
    }

    #[target_feature(enable = "avx2")]
    unsafe fn abs_diff_i32(first: __m256i, second: __m256i) -> __m256i {
        _mm256_abs_epi32(_mm256_sub_epi32(first, second))
    }

    #[target_feature(enable = "avx2")]
    unsafe fn less_or_equal_i32(value: __m256i, limit: __m256i) -> __m256i {
        _mm256_xor_si256(_mm256_cmpgt_epi32(value, limit), _mm256_set1_epi32(-1))
    }

    #[target_feature(enable = "avx2")]
    unsafe fn clamp_i32(value: __m256i, minimum: __m256i, maximum: __m256i) -> __m256i {
        _mm256_max_epi32(minimum, _mm256_min_epi32(value, maximum))
    }

    #[target_feature(enable = "avx2")]
    #[allow(clippy::too_many_arguments)]
    unsafe fn filter4_high(
        p3: __m256i,
        p2: __m256i,
        p1: __m256i,
        p0: __m256i,
        q0: __m256i,
        q1: __m256i,
        q2: __m256i,
        q3: __m256i,
        thresholds: Thresholds,
        bit_depth: BitDepth,
    ) -> (__m256i, __m256i, __m256i, __m256i) {
        let shift = i32::from(bit_depth.bits() - 8);
        let limit = _mm256_set1_epi32(i32::from(thresholds.limit) << shift);
        let blimit = _mm256_set1_epi32(i32::from(thresholds.blimit) << shift);
        let hev_limit = _mm256_set1_epi32(i32::from(thresholds.hev) << shift);

        let mut mask = unsafe { less_or_equal_i32(abs_diff_i32(p3, p2), limit) };
        for difference in [
            unsafe { abs_diff_i32(p2, p1) },
            unsafe { abs_diff_i32(p1, p0) },
            unsafe { abs_diff_i32(q1, q0) },
            unsafe { abs_diff_i32(q2, q1) },
            unsafe { abs_diff_i32(q3, q2) },
        ] {
            mask = _mm256_and_si256(mask, unsafe { less_or_equal_i32(difference, limit) });
        }
        let edge_difference = _mm256_add_epi32(
            _mm256_slli_epi32::<1>(unsafe { abs_diff_i32(p0, q0) }),
            _mm256_srli_epi32::<1>(unsafe { abs_diff_i32(p1, q1) }),
        );
        mask = _mm256_and_si256(mask, unsafe { less_or_equal_i32(edge_difference, blimit) });

        let hev = _mm256_or_si256(
            _mm256_cmpgt_epi32(unsafe { abs_diff_i32(p1, p0) }, hev_limit),
            _mm256_cmpgt_epi32(unsafe { abs_diff_i32(q1, q0) }, hev_limit),
        );
        let signed_min = _mm256_set1_epi32(-(128 << shift));
        let signed_max = _mm256_set1_epi32((128 << shift) - 1);
        let pixel_min = _mm256_setzero_si256();
        let pixel_max = _mm256_set1_epi32(i32::from(bit_depth.max_sample()));

        let outer = _mm256_and_si256(hev, unsafe {
            clamp_i32(_mm256_sub_epi32(p1, q1), signed_min, signed_max)
        });
        let filter = unsafe {
            clamp_i32(
                _mm256_add_epi32(
                    outer,
                    _mm256_mullo_epi32(_mm256_sub_epi32(q0, p0), _mm256_set1_epi32(3)),
                ),
                signed_min,
                signed_max,
            )
        };
        let filter1 = _mm256_srai_epi32::<3>(unsafe {
            clamp_i32(
                _mm256_add_epi32(filter, _mm256_set1_epi32(4)),
                signed_min,
                signed_max,
            )
        });
        let filter2 = _mm256_srai_epi32::<3>(unsafe {
            clamp_i32(
                _mm256_add_epi32(filter, _mm256_set1_epi32(3)),
                signed_min,
                signed_max,
            )
        });

        let filtered_q0 = unsafe { clamp_i32(_mm256_sub_epi32(q0, filter1), pixel_min, pixel_max) };
        let filtered_p0 = unsafe { clamp_i32(_mm256_add_epi32(p0, filter2), pixel_min, pixel_max) };
        let q0 = _mm256_blendv_epi8(q0, filtered_q0, mask);
        let p0 = _mm256_blendv_epi8(p0, filtered_p0, mask);

        let adjustment = _mm256_srai_epi32::<1>(_mm256_add_epi32(filter1, _mm256_set1_epi32(1)));
        let inner_mask = _mm256_andnot_si256(hev, mask);
        let filtered_q1 =
            unsafe { clamp_i32(_mm256_sub_epi32(q1, adjustment), pixel_min, pixel_max) };
        let filtered_p1 =
            unsafe { clamp_i32(_mm256_add_epi32(p1, adjustment), pixel_min, pixel_max) };
        let q1 = _mm256_blendv_epi8(q1, filtered_q1, inner_mask);
        let p1 = _mm256_blendv_epi8(p1, filtered_p1, inner_mask);

        (p1, p0, q0, q1)
    }

    #[target_feature(enable = "avx2")]
    unsafe fn sum_i32<const N: usize>(values: [__m256i; N]) -> __m256i {
        values
            .into_iter()
            .fold(_mm256_setzero_si256(), |sum, value| {
                _mm256_add_epi32(sum, value)
            })
    }

    #[target_feature(enable = "avx2")]
    unsafe fn filter8_high(samples: &[__m256i; 16]) -> [__m256i; 6] {
        let twice = |value| _mm256_slli_epi32::<1>(value);
        let thrice = |value| _mm256_add_epi32(twice(value), value);
        let round = |value| _mm256_srai_epi32::<3>(_mm256_add_epi32(value, _mm256_set1_epi32(4)));
        [
            round(unsafe {
                sum_i32([
                    thrice(samples[4]),
                    twice(samples[5]),
                    samples[6],
                    samples[7],
                    samples[8],
                ])
            }),
            round(unsafe {
                sum_i32([
                    twice(samples[4]),
                    samples[5],
                    twice(samples[6]),
                    samples[7],
                    samples[8],
                    samples[9],
                ])
            }),
            round(unsafe {
                sum_i32([
                    samples[4],
                    samples[5],
                    samples[6],
                    twice(samples[7]),
                    samples[8],
                    samples[9],
                    samples[10],
                ])
            }),
            round(unsafe {
                sum_i32([
                    samples[5],
                    samples[6],
                    samples[7],
                    twice(samples[8]),
                    samples[9],
                    samples[10],
                    samples[11],
                ])
            }),
            round(unsafe {
                sum_i32([
                    samples[6],
                    samples[7],
                    samples[8],
                    twice(samples[9]),
                    samples[10],
                    twice(samples[11]),
                ])
            }),
            round(unsafe {
                sum_i32([
                    samples[7],
                    samples[8],
                    samples[9],
                    twice(samples[10]),
                    thrice(samples[11]),
                ])
            }),
        ]
    }

    #[target_feature(enable = "avx2")]
    unsafe fn filter16_high(samples: &[__m256i; 16]) -> [__m256i; 14] {
        let zero = _mm256_setzero_si256();
        let mut prefix = [zero; 17];
        for index in 0..16 {
            prefix[index + 1] = _mm256_add_epi32(prefix[index], samples[index]);
        }
        let mut filtered = [zero; 14];
        for index in 1..=7 {
            let endpoint = _mm256_mullo_epi32(samples[0], _mm256_set1_epi32((8 - index) as i32));
            let before = _mm256_sub_epi32(prefix[index], prefix[1]);
            let center = _mm256_slli_epi32::<1>(samples[index]);
            let after = _mm256_sub_epi32(prefix[index + 8], prefix[index + 1]);
            filtered[index - 1] = _mm256_srai_epi32::<4>(_mm256_add_epi32(
                unsafe { sum_i32([endpoint, before, center, after]) },
                _mm256_set1_epi32(8),
            ));
        }
        for index in 8..=14 {
            let before = _mm256_sub_epi32(prefix[index], prefix[index - 7]);
            let center = _mm256_slli_epi32::<1>(samples[index]);
            let after = _mm256_sub_epi32(prefix[15], prefix[index + 1]);
            let endpoint = _mm256_mullo_epi32(samples[15], _mm256_set1_epi32((index - 7) as i32));
            filtered[index - 1] = _mm256_srai_epi32::<4>(_mm256_add_epi32(
                unsafe { sum_i32([before, center, after, endpoint]) },
                _mm256_set1_epi32(8),
            ));
        }
        filtered
    }

    #[target_feature(enable = "avx2")]
    unsafe fn filter_wide_high(
        samples: &mut [__m256i; 16],
        width: FilterWidth,
        thresholds: Thresholds,
        bit_depth: BitDepth,
    ) {
        debug_assert!(!matches!(width, FilterWidth::Four));
        let shift = i32::from(bit_depth.bits() - 8);
        let limit = _mm256_set1_epi32(i32::from(thresholds.limit) << shift);
        let blimit = _mm256_set1_epi32(i32::from(thresholds.blimit) << shift);
        let mut mask = unsafe { less_or_equal_i32(abs_diff_i32(samples[4], samples[5]), limit) };
        for difference in [
            unsafe { abs_diff_i32(samples[5], samples[6]) },
            unsafe { abs_diff_i32(samples[6], samples[7]) },
            unsafe { abs_diff_i32(samples[9], samples[8]) },
            unsafe { abs_diff_i32(samples[10], samples[9]) },
            unsafe { abs_diff_i32(samples[11], samples[10]) },
        ] {
            mask = _mm256_and_si256(mask, unsafe { less_or_equal_i32(difference, limit) });
        }
        let edge_difference = _mm256_add_epi32(
            _mm256_slli_epi32::<1>(unsafe { abs_diff_i32(samples[7], samples[8]) }),
            _mm256_srli_epi32::<1>(unsafe { abs_diff_i32(samples[6], samples[9]) }),
        );
        mask = _mm256_and_si256(mask, unsafe { less_or_equal_i32(edge_difference, blimit) });

        let flat_limit = _mm256_set1_epi32(1 << shift);
        let mut flat =
            unsafe { less_or_equal_i32(abs_diff_i32(samples[6], samples[7]), flat_limit) };
        for difference in [
            unsafe { abs_diff_i32(samples[9], samples[8]) },
            unsafe { abs_diff_i32(samples[5], samples[7]) },
            unsafe { abs_diff_i32(samples[10], samples[8]) },
            unsafe { abs_diff_i32(samples[4], samples[7]) },
            unsafe { abs_diff_i32(samples[11], samples[8]) },
        ] {
            flat = _mm256_and_si256(flat, unsafe { less_or_equal_i32(difference, flat_limit) });
        }

        let (p1, p0, q0, q1) = unsafe {
            filter4_high(
                samples[4],
                samples[5],
                samples[6],
                samples[7],
                samples[8],
                samples[9],
                samples[10],
                samples[11],
                thresholds,
                bit_depth,
            )
        };
        let all = _mm256_set1_epi32(-1);
        let four_mask = _mm256_andnot_si256(flat, all);
        for (index, filtered) in [(6, p1), (7, p0), (8, q0), (9, q1)] {
            samples[index] = _mm256_blendv_epi8(samples[index], filtered, four_mask);
        }

        let mut flat2 = all;
        if matches!(width, FilterWidth::Sixteen) {
            for index in 0..=3 {
                flat2 = _mm256_and_si256(flat2, unsafe {
                    less_or_equal_i32(abs_diff_i32(samples[index], samples[7]), flat_limit)
                });
            }
            for index in 12..=15 {
                flat2 = _mm256_and_si256(flat2, unsafe {
                    less_or_equal_i32(abs_diff_i32(samples[index], samples[8]), flat_limit)
                });
            }
            let wide_mask = _mm256_and_si256(mask, _mm256_and_si256(flat, flat2));
            let filtered = unsafe { filter16_high(samples) };
            for (index, filtered) in (1..=14).zip(filtered) {
                samples[index] = _mm256_blendv_epi8(samples[index], filtered, wide_mask);
            }
        }

        let eight_mask = if matches!(width, FilterWidth::Eight) {
            _mm256_and_si256(mask, flat)
        } else {
            _mm256_and_si256(mask, _mm256_andnot_si256(flat2, flat))
        };
        let filtered = unsafe { filter8_high(samples) };
        for (index, filtered) in (5..=10).zip(filtered) {
            samples[index] = _mm256_blendv_epi8(samples[index], filtered, eight_mask);
        }
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn filter_horizontal_high_avx2(
        edge: *mut u16,
        stride: usize,
        width: FilterWidth,
        thresholds: Thresholds,
        bit_depth: BitDepth,
    ) {
        if matches!(width, FilterWidth::Four) {
            unsafe {
                filter_horizontal_4_high_avx2(edge, stride, thresholds, bit_depth);
            }
            return;
        }
        let reach = if matches!(width, FilterWidth::Sixteen) {
            8
        } else {
            4
        };
        let zero = _mm256_setzero_si256();
        let mut samples = [zero; 16];
        for offset in 0..reach * 2 {
            samples[8 - reach + offset] =
                unsafe { load_high_row(edge, stride, offset as isize - reach as isize) };
        }
        unsafe {
            filter_wide_high(&mut samples, width, thresholds, bit_depth);
        }
        for offset in 0..reach * 2 {
            unsafe {
                store_high_row(
                    edge,
                    stride,
                    offset as isize - reach as isize,
                    samples[8 - reach + offset],
                );
            }
        }
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn filter_vertical_high_avx2(
        edge: *mut u16,
        stride: usize,
        width: FilterWidth,
        thresholds: Thresholds,
        bit_depth: BitDepth,
    ) {
        if matches!(width, FilterWidth::Four) {
            unsafe {
                filter_vertical_4_high_avx2(edge, stride, thresholds, bit_depth);
            }
            return;
        }
        let reach = if matches!(width, FilterWidth::Sixteen) {
            8
        } else {
            4
        };
        let mut low_rows = [_mm_setzero_si128(); 8];
        let mut high_rows = [_mm_setzero_si128(); 8];
        for row in 0..8 {
            let row_edge = unsafe { edge.add(row * stride) };
            low_rows[row] = unsafe { _mm_loadu_si128(row_edge.sub(reach).cast::<__m128i>()) };
            if reach == 8 {
                high_rows[row] = unsafe { _mm_loadu_si128(row_edge.cast::<__m128i>()) };
            }
        }
        let low_columns = unsafe { transpose_8x8_u16(low_rows) };
        let zero = _mm256_setzero_si256();
        let mut samples = [zero; 16];
        for (index, column) in low_columns.into_iter().enumerate() {
            samples[8 - reach + index] = _mm256_cvtepu16_epi32(column);
        }
        if reach == 8 {
            let high_columns = unsafe { transpose_8x8_u16(high_rows) };
            for (index, column) in high_columns.into_iter().enumerate() {
                samples[8 + index] = _mm256_cvtepu16_epi32(column);
            }
        }
        unsafe {
            filter_wide_high(&mut samples, width, thresholds, bit_depth);
        }
        let mut low_columns = [_mm_setzero_si128(); 8];
        for (column, samples) in low_columns.iter_mut().zip(&samples[8 - reach..16 - reach]) {
            *column = unsafe { pack_high(*samples) };
        }
        let low_rows = unsafe { transpose_8x8_u16(low_columns) };
        let high_rows = if reach == 8 {
            let mut high_columns = [_mm_setzero_si128(); 8];
            for (column, samples) in high_columns.iter_mut().zip(&samples[8..16]) {
                *column = unsafe { pack_high(*samples) };
            }
            Some(unsafe { transpose_8x8_u16(high_columns) })
        } else {
            None
        };
        for row in 0..8 {
            let row_edge = unsafe { edge.add(row * stride) };
            unsafe {
                _mm_storeu_si128(row_edge.sub(reach).cast::<__m128i>(), low_rows[row]);
            }
            if let Some(high_rows) = high_rows {
                unsafe {
                    _mm_storeu_si128(row_edge.cast::<__m128i>(), high_rows[row]);
                }
            }
        }
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn filter_horizontal_4_high_avx2(
        edge: *mut u16,
        stride: usize,
        thresholds: Thresholds,
        bit_depth: BitDepth,
    ) {
        let p3 = unsafe { load_high_row(edge, stride, -4) };
        let p2 = unsafe { load_high_row(edge, stride, -3) };
        let p1 = unsafe { load_high_row(edge, stride, -2) };
        let p0 = unsafe { load_high_row(edge, stride, -1) };
        let q0 = unsafe { load_high_row(edge, stride, 0) };
        let q1 = unsafe { load_high_row(edge, stride, 1) };
        let q2 = unsafe { load_high_row(edge, stride, 2) };
        let q3 = unsafe { load_high_row(edge, stride, 3) };
        let (p1, p0, q0, q1) =
            unsafe { filter4_high(p3, p2, p1, p0, q0, q1, q2, q3, thresholds, bit_depth) };
        unsafe {
            store_high_row(edge, stride, -2, p1);
            store_high_row(edge, stride, -1, p0);
            store_high_row(edge, stride, 0, q0);
            store_high_row(edge, stride, 1, q1);
        }
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn filter_vertical_4_high_avx2(
        edge: *mut u16,
        stride: usize,
        thresholds: Thresholds,
        bit_depth: BitDepth,
    ) {
        let mut columns = [[0i32; 8]; 8];
        for row in 0..8 {
            let row_edge = unsafe { edge.add(row * stride) };
            for (column, values) in columns.iter_mut().enumerate() {
                values[row] = i32::from(unsafe { *row_edge.offset(column as isize - 4) });
            }
        }
        let load_column =
            |column: &[i32; 8]| unsafe { _mm256_loadu_si256(column.as_ptr().cast::<__m256i>()) };
        let (p1, p0, q0, q1) = unsafe {
            filter4_high(
                load_column(&columns[0]),
                load_column(&columns[1]),
                load_column(&columns[2]),
                load_column(&columns[3]),
                load_column(&columns[4]),
                load_column(&columns[5]),
                load_column(&columns[6]),
                load_column(&columns[7]),
                thresholds,
                bit_depth,
            )
        };
        let mut filtered = [[0i32; 8]; 4];
        for (values, vector) in filtered.iter_mut().zip([p1, p0, q0, q1]) {
            unsafe {
                _mm256_storeu_si256(values.as_mut_ptr().cast::<__m256i>(), vector);
            }
        }
        for row in 0..8 {
            let row_edge = unsafe { edge.add(row * stride) };
            for (column, values) in filtered.iter().enumerate() {
                unsafe {
                    *row_edge.offset(column as isize - 2) = values[row] as u16;
                }
            }
        }
    }
}

fn filter_mask<T: LoopSample>(limit: u32, blimit: u32, edge: &[T]) -> bool {
    let p3 = edge[0];
    let p2 = edge[1];
    let p1 = edge[2];
    let p0 = edge[3];
    let q0 = edge[4];
    let q1 = edge[5];
    let q2 = edge[6];
    let q3 = edge[7];
    abs_diff(p3, p2) <= limit
        && abs_diff(p2, p1) <= limit
        && abs_diff(p1, p0) <= limit
        && abs_diff(q1, q0) <= limit
        && abs_diff(q2, q1) <= limit
        && abs_diff(q3, q2) <= limit
        && abs_diff(p0, q0) * 2 + abs_diff(p1, q1) / 2 <= blimit
}

fn flat_mask<T: LoopSample>(edge: &[T], shift: u32) -> bool {
    let p3 = edge[0];
    let p2 = edge[1];
    let p1 = edge[2];
    let p0 = edge[3];
    let q0 = edge[4];
    let q1 = edge[5];
    let q2 = edge[6];
    let q3 = edge[7];
    let threshold = 1 << shift;
    abs_diff(p1, p0) <= threshold
        && abs_diff(q1, q0) <= threshold
        && abs_diff(p2, p0) <= threshold
        && abs_diff(q2, q0) <= threshold
        && abs_diff(p3, p0) <= threshold
        && abs_diff(q3, q0) <= threshold
}

fn flat2_mask<T: LoopSample>(samples: &[T; 16], shift: u32) -> bool {
    let p0 = samples[7];
    let q0 = samples[8];
    let threshold = 1 << shift;
    (0..=3).all(|index| abs_diff(samples[index], p0) <= threshold)
        && (12..=15).all(|index| abs_diff(samples[index], q0) <= threshold)
}

fn filter_four<T: LoopSample>(samples: &mut [T; 16], threshold: u8, bit_depth: BitDepth) {
    let shift = u32::from(bit_depth.bits() - 8);
    let ps1 = samples[6].to_i32();
    let ps0 = samples[7].to_i32();
    let qs0 = samples[8].to_i32();
    let qs1 = samples[9].to_i32();
    let signed_min = -(128i32 << shift);
    let signed_max = (128i32 << shift) - 1;
    let signed_clamp = |value: i32| value.clamp(signed_min, signed_max);
    let pixel_clamp = |value: i32| T::from_i32(value.clamp(0, i32::from(bit_depth.max_sample())));
    let threshold = u32::from(threshold) << shift;
    let hev = abs_diff(samples[6], samples[7]) > threshold
        || abs_diff(samples[9], samples[8]) > threshold;
    let outer = if hev { signed_clamp(ps1 - qs1) } else { 0 };
    let filter = signed_clamp(outer + 3 * (qs0 - ps0));
    let filter1 = signed_clamp(filter + 4) >> 3;
    let filter2 = signed_clamp(filter + 3) >> 3;
    samples[8] = pixel_clamp(qs0 - filter1);
    samples[7] = pixel_clamp(ps0 + filter2);
    if !hev {
        let adjustment = (filter1 + 1) >> 1;
        samples[9] = pixel_clamp(qs1 - adjustment);
        samples[6] = pixel_clamp(ps1 + adjustment);
    }
}

fn filter_eight<T: LoopSample>(samples: &mut [T; 16]) {
    let [p3, p2, p1, p0, q0, q1, q2, q3] = samples[4..12].try_into().unwrap();
    samples[5] =
        round_shift3(3 * p3.to_i32() + 2 * p2.to_i32() + p1.to_i32() + p0.to_i32() + q0.to_i32());
    samples[6] = round_shift3(
        2 * p3.to_i32() + p2.to_i32() + 2 * p1.to_i32() + p0.to_i32() + q0.to_i32() + q1.to_i32(),
    );
    samples[7] = round_shift3(
        p3.to_i32()
            + p2.to_i32()
            + p1.to_i32()
            + 2 * p0.to_i32()
            + q0.to_i32()
            + q1.to_i32()
            + q2.to_i32(),
    );
    samples[8] = round_shift3(
        p2.to_i32()
            + p1.to_i32()
            + p0.to_i32()
            + 2 * q0.to_i32()
            + q1.to_i32()
            + q2.to_i32()
            + q3.to_i32(),
    );
    samples[9] = round_shift3(
        p1.to_i32() + p0.to_i32() + q0.to_i32() + 2 * q1.to_i32() + q2.to_i32() + 2 * q3.to_i32(),
    );
    samples[10] =
        round_shift3(p0.to_i32() + q0.to_i32() + q1.to_i32() + 2 * q2.to_i32() + 3 * q3.to_i32());
}

fn filter_sixteen<T: LoopSample>(samples: &mut [T; 16]) {
    let source = *samples;
    let p = &source[..8];
    let q = &source[8..];
    let sum = |terms: &[(i32, T)]| -> T {
        let total = terms
            .iter()
            .map(|&(weight, value)| weight * value.to_i32())
            .sum::<i32>();
        T::from_i32((total + 8) >> 4)
    };
    samples[1] = sum(&[
        (7, p[0]),
        (2, p[1]),
        (1, p[2]),
        (1, p[3]),
        (1, p[4]),
        (1, p[5]),
        (1, p[6]),
        (1, p[7]),
        (1, q[0]),
    ]);
    samples[2] = sum(&[
        (6, p[0]),
        (1, p[1]),
        (2, p[2]),
        (1, p[3]),
        (1, p[4]),
        (1, p[5]),
        (1, p[6]),
        (1, p[7]),
        (1, q[0]),
        (1, q[1]),
    ]);
    samples[3] = sum(&[
        (5, p[0]),
        (1, p[1]),
        (1, p[2]),
        (2, p[3]),
        (1, p[4]),
        (1, p[5]),
        (1, p[6]),
        (1, p[7]),
        (1, q[0]),
        (1, q[1]),
        (1, q[2]),
    ]);
    samples[4] = sum(&[
        (4, p[0]),
        (1, p[1]),
        (1, p[2]),
        (1, p[3]),
        (2, p[4]),
        (1, p[5]),
        (1, p[6]),
        (1, p[7]),
        (1, q[0]),
        (1, q[1]),
        (1, q[2]),
        (1, q[3]),
    ]);
    samples[5] = sum(&[
        (3, p[0]),
        (1, p[1]),
        (1, p[2]),
        (1, p[3]),
        (1, p[4]),
        (2, p[5]),
        (1, p[6]),
        (1, p[7]),
        (1, q[0]),
        (1, q[1]),
        (1, q[2]),
        (1, q[3]),
        (1, q[4]),
    ]);
    samples[6] = sum(&[
        (2, p[0]),
        (1, p[1]),
        (1, p[2]),
        (1, p[3]),
        (1, p[4]),
        (1, p[5]),
        (2, p[6]),
        (1, p[7]),
        (1, q[0]),
        (1, q[1]),
        (1, q[2]),
        (1, q[3]),
        (1, q[4]),
        (1, q[5]),
    ]);
    samples[7] = sum(&[
        (1, p[0]),
        (1, p[1]),
        (1, p[2]),
        (1, p[3]),
        (1, p[4]),
        (1, p[5]),
        (1, p[6]),
        (2, p[7]),
        (1, q[0]),
        (1, q[1]),
        (1, q[2]),
        (1, q[3]),
        (1, q[4]),
        (1, q[5]),
        (1, q[6]),
    ]);
    samples[8] = sum(&[
        (1, p[1]),
        (1, p[2]),
        (1, p[3]),
        (1, p[4]),
        (1, p[5]),
        (1, p[6]),
        (1, p[7]),
        (2, q[0]),
        (1, q[1]),
        (1, q[2]),
        (1, q[3]),
        (1, q[4]),
        (1, q[5]),
        (1, q[6]),
        (1, q[7]),
    ]);
    samples[9] = sum(&[
        (1, p[2]),
        (1, p[3]),
        (1, p[4]),
        (1, p[5]),
        (1, p[6]),
        (1, p[7]),
        (1, q[0]),
        (2, q[1]),
        (1, q[2]),
        (1, q[3]),
        (1, q[4]),
        (1, q[5]),
        (1, q[6]),
        (2, q[7]),
    ]);
    samples[10] = sum(&[
        (1, p[3]),
        (1, p[4]),
        (1, p[5]),
        (1, p[6]),
        (1, p[7]),
        (1, q[0]),
        (1, q[1]),
        (2, q[2]),
        (1, q[3]),
        (1, q[4]),
        (1, q[5]),
        (1, q[6]),
        (3, q[7]),
    ]);
    samples[11] = sum(&[
        (1, p[4]),
        (1, p[5]),
        (1, p[6]),
        (1, p[7]),
        (1, q[0]),
        (1, q[1]),
        (1, q[2]),
        (2, q[3]),
        (1, q[4]),
        (1, q[5]),
        (1, q[6]),
        (4, q[7]),
    ]);
    samples[12] = sum(&[
        (1, p[5]),
        (1, p[6]),
        (1, p[7]),
        (1, q[0]),
        (1, q[1]),
        (1, q[2]),
        (1, q[3]),
        (2, q[4]),
        (1, q[5]),
        (1, q[6]),
        (5, q[7]),
    ]);
    samples[13] = sum(&[
        (1, p[6]),
        (1, p[7]),
        (1, q[0]),
        (1, q[1]),
        (1, q[2]),
        (1, q[3]),
        (1, q[4]),
        (2, q[5]),
        (1, q[6]),
        (6, q[7]),
    ]);
    samples[14] = sum(&[
        (1, p[7]),
        (1, q[0]),
        (1, q[1]),
        (1, q[2]),
        (1, q[3]),
        (1, q[4]),
        (1, q[5]),
        (2, q[6]),
        (7, q[7]),
    ]);
}

#[inline]
fn abs_diff<T: LoopSample>(first: T, second: T) -> u32 {
    first.to_i32().abs_diff(second.to_i32())
}

#[inline]
fn round_shift3<T: LoopSample>(value: i32) -> T {
    T::from_i32((value + 4) >> 3)
}

#[cfg(test)]
mod tests {
    use super::{
        FilterWidth, Thresholds, filter_horizontal_high, filter_horizontal_scalar, filter_samples,
        filter_vertical, filter_vertical_high, filter_vertical_scalar, thresholds,
    };
    use crate::BitDepth;

    #[test]
    fn flat_eight_tap_edge_is_smoothed_symmetrically() {
        let mut samples = [0u8; 16];
        samples[..8].fill(100);
        samples[8..].fill(104);
        filter_samples(
            &mut samples,
            FilterWidth::Eight,
            Thresholds {
                limit: 4,
                blimit: 20,
                hev: 0,
            },
            BitDepth::Eight,
        );
        assert_eq!(&samples[5..11], &[101, 101, 102, 103, 103, 104]);
    }

    #[test]
    fn high_bit_depth_filter_scales_normative_thresholds() {
        let mut samples = [0u16; 16];
        samples[..8].fill(400);
        samples[8..].fill(416);
        filter_samples(
            &mut samples,
            FilterWidth::Eight,
            Thresholds {
                limit: 4,
                blimit: 20,
                hev: 0,
            },
            BitDepth::Ten,
        );
        assert_eq!(&samples[5..11], &[402, 404, 406, 410, 412, 414]);
    }

    #[test]
    fn high_bit_depth_fast_paths_match_scalar() {
        const STRIDE: usize = 24;
        const HEIGHT: usize = 16;
        let mut state = 0x7a31_9d2bu32;
        for bit_depth in [BitDepth::Ten, BitDepth::Twelve] {
            let shift = u32::from(bit_depth.bits() - 8);
            let max_sample = u32::from(bit_depth.max_sample());
            for level in [9, 19, 63] {
                for width in [FilterWidth::Four, FilterWidth::Eight, FilterWidth::Sixteen] {
                    for iteration in 0..200 {
                        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                        let mut source = [0u16; STRIDE * HEIGHT];
                        let base = 32 + state % (max_sample - 64);
                        for (index, sample) in source.iter_mut().enumerate() {
                            let variation = match iteration % 3 {
                                0 => i32::from(index % STRIDE >= 8) << (shift + 1),
                                1 => i32::from(index / STRIDE >= 8) << (shift + 1),
                                _ => {
                                    state =
                                        state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                                    ((state >> 28) as i32 - 8) << shift
                                }
                            };
                            *sample = (base as i32 + variation).clamp(0, max_sample as i32) as u16;
                        }

                        let mut scalar = source;
                        let mut fast = source;
                        filter_vertical_scalar(
                            &mut scalar,
                            STRIDE,
                            STRIDE,
                            8,
                            4,
                            8,
                            width,
                            thresholds(level, 0),
                            bit_depth,
                        );
                        filter_vertical_high(
                            &mut fast,
                            STRIDE,
                            STRIDE,
                            8,
                            4,
                            8,
                            width,
                            thresholds(level, 0),
                            bit_depth,
                        );
                        assert_eq!(
                            fast, scalar,
                            "vertical {bit_depth:?}, {width:?}, level {level}"
                        );

                        let mut scalar = source;
                        let mut fast = source;
                        filter_horizontal_scalar(
                            &mut scalar,
                            STRIDE,
                            HEIGHT,
                            4,
                            8,
                            8,
                            width,
                            thresholds(level, 0),
                            bit_depth,
                        );
                        filter_horizontal_high(
                            &mut fast,
                            STRIDE,
                            HEIGHT,
                            4,
                            8,
                            8,
                            width,
                            thresholds(level, 0),
                            bit_depth,
                        );
                        assert_eq!(
                            fast, scalar,
                            "horizontal {bit_depth:?}, {width:?}, level {level}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn scalar_filter_kernels_match_reference_hashes() {
        fn hash(width: FilterWidth, level: u8) -> u64 {
            let mut hash = 1_469_598_103_934_665_603u64;
            let mut state = 0x7182_93a4u32;
            for _ in 0..1000 {
                let mut pixels = [0u8; 8 * 16];
                for row in 0..8 {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    let base = 8 + (state >> 25) as u8;
                    for column in 0..16 {
                        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                        pixels[row * 16 + column] = base + (state >> 31) as u8;
                    }
                }
                filter_vertical(&mut pixels, 16, 16, 8, 0, 8, width, thresholds(level, 0));
                for value in pixels {
                    hash = (hash ^ u64::from(value)).wrapping_mul(1_099_511_628_211);
                }
            }
            hash
        }

        let expected = [
            (
                FilterWidth::Four,
                [
                    16_905_686_190_371_205_798,
                    8_241_948_348_504_931_646,
                    8_241_948_348_504_931_646,
                ],
            ),
            (
                FilterWidth::Eight,
                [
                    3_969_889_870_118_261_703,
                    3_969_889_870_118_261_703,
                    3_969_889_870_118_261_703,
                ],
            ),
            (
                FilterWidth::Sixteen,
                [
                    12_322_266_613_754_353_444,
                    12_322_266_613_754_353_444,
                    12_322_266_613_754_353_444,
                ],
            ),
        ];
        for (width, expected) in expected {
            for (index, level) in [9, 19, 63].into_iter().enumerate() {
                assert_eq!(
                    hash(width, level),
                    expected[index],
                    "{width:?} level {level}"
                );
            }
        }
    }
}
