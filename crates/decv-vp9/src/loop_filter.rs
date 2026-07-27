use crate::{
    FrameHeader, Result, Vp9Error,
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

    let picture_width = picture.width();
    let picture_height = picture.height();
    let [luma, chroma_u, chroma_v] = picture.planes_mut();
    std::thread::scope(|scope| {
        let u_worker = scope.spawn(|| {
            apply_plane_loop_filter(chroma_u, 1, picture_width, picture_height, header, modes);
        });
        let v_worker = scope.spawn(|| {
            apply_plane_loop_filter(chroma_v, 2, picture_width, picture_height, header, modes);
        });
        apply_plane_loop_filter(luma, 0, picture_width, picture_height, header, modes);
        u_worker.join().expect("VP9 U-plane loop filter panicked");
        v_worker.join().expect("VP9 V-plane loop filter panicked");
    });
    Ok(())
}

fn apply_plane_loop_filter(
    pixels: &mut [u8],
    plane: usize,
    picture_width: usize,
    picture_height: usize,
    header: &FrameHeader,
    modes: &FilterModeMap,
) {
    let configuration = header.loop_filter.as_ref().expect("caller checked");
    let subsampling = usize::from(plane != 0);
    let row_step = 1usize << subsampling;
    let column_step = 1usize << subsampling;
    let width = if plane == 0 {
        picture_width
    } else {
        picture_width.div_ceil(2)
    };
    let height = if plane == 0 {
        picture_height
    } else {
        picture_height.div_ceil(2)
    };
    let stride = width;

    for superblock_row in (0..modes.mi_rows).step_by(8) {
        let row_end = (superblock_row + 8).min(modes.mi_rows);
        for superblock_column in (0..modes.mi_columns).step_by(8) {
            let column_end = (superblock_column + 8).min(modes.mi_columns);

            // VP9 completes the vertical and then horizontal pass for each
            // 64x64 superblock before moving to the next one.
            for mi_row in (superblock_row..row_end).step_by(row_step) {
                let y = (mi_row * 8) >> subsampling;
                let line_count = 8.min(height.saturating_sub(y));
                for mi_column in (superblock_column..column_end).step_by(column_step) {
                    let x = (mi_column * 8) >> subsampling;
                    let mode = modes.get(mi_row, mi_column);
                    let level = filter_level(header, mode);
                    if level == 0 {
                        continue;
                    }
                    let thresholds = thresholds(level, configuration.sharpness);
                    let transform = plane_transform(mode, subsampling);
                    let block_edge = is_left_block_edge(mode.block_size, mi_column);
                    let skip_edge = mode.skip && mode.reference != 0 && !block_edge;
                    if x != 0
                        && !skip_edge
                        && let Some(width_kind) =
                            edge_filter_width(transform, x / 8, width.saturating_sub(x))
                    {
                        filter_vertical(
                            pixels, stride, width, x, y, line_count, width_kind, thresholds,
                        );
                    }
                    if transform == TransformSize::Tx4x4
                        && !(mode.skip && mode.reference != 0)
                        && x + 4 < width
                    {
                        filter_vertical(
                            pixels,
                            stride,
                            width,
                            x + 4,
                            y,
                            line_count,
                            FilterWidth::Four,
                            thresholds,
                        );
                    }
                }
            }

            for mi_row in (superblock_row..row_end).step_by(row_step) {
                let y = (mi_row * 8) >> subsampling;
                for mi_column in (superblock_column..column_end).step_by(column_step) {
                    let x = (mi_column * 8) >> subsampling;
                    let column_count = 8.min(width.saturating_sub(x));
                    let mode = modes.get(mi_row, mi_column);
                    let level = filter_level(header, mode);
                    if level == 0 {
                        continue;
                    }
                    let thresholds = thresholds(level, configuration.sharpness);
                    let transform = plane_transform(mode, subsampling);
                    let block_edge = is_top_block_edge(mode.block_size, mi_row);
                    let skip_edge = mode.skip && mode.reference != 0 && !block_edge;
                    if y != 0
                        && !skip_edge
                        && let Some(width_kind) =
                            edge_filter_width(transform, y / 8, height.saturating_sub(y))
                    {
                        filter_horizontal(
                            pixels,
                            stride,
                            height,
                            x,
                            y,
                            column_count,
                            width_kind,
                            thresholds,
                        );
                    }
                    if transform == TransformSize::Tx4x4
                        && !(mode.skip && mode.reference != 0)
                        && y + 4 < height
                    {
                        filter_horizontal(
                            pixels,
                            stride,
                            height,
                            x,
                            y + 4,
                            column_count,
                            FilterWidth::Four,
                            thresholds,
                        );
                    }
                }
            }
        }
    }
}

fn plane_transform(mode: FilterMode, subsampling: usize) -> TransformSize {
    if subsampling == 0 {
        return mode.transform_size;
    }
    let (width, height) = if mode.block_size < BlockSize::B8x8 {
        (1, 1)
    } else {
        (
            mode.block_size.width_4x4().div_ceil(2),
            mode.block_size.height_4x4().div_ceil(2),
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
    for row in 0..count {
        let base = (y + row) * stride + x;
        let mut samples = [0u8; 16];
        for offset in 0..reach * 2 {
            samples[8 - reach + offset] = pixels[base - reach + offset];
        }
        filter_samples(&mut samples, width, thresholds);
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
    for column in 0..count {
        let mut samples = [0u8; 16];
        for offset in 0..reach * 2 {
            samples[8 - reach + offset] = pixels[(y + offset - reach) * stride + x + column];
        }
        filter_samples(&mut samples, width, thresholds);
        for offset in 0..reach * 2 {
            pixels[(y + offset - reach) * stride + x + column] = samples[8 - reach + offset];
        }
    }
}

fn filter_samples(samples: &mut [u8; 16], width: FilterWidth, thresholds: Thresholds) {
    let mask = filter_mask(thresholds.limit, thresholds.blimit, &samples[4..12]);
    if !mask {
        return;
    }
    let flat = flat_mask(&samples[4..12]);
    match width {
        FilterWidth::Four => filter_four(samples, thresholds.hev),
        FilterWidth::Eight if flat => filter_eight(samples),
        FilterWidth::Eight => filter_four(samples, thresholds.hev),
        FilterWidth::Sixteen if flat && flat2_mask(samples) => filter_sixteen(samples),
        FilterWidth::Sixteen if flat => filter_eight(samples),
        FilterWidth::Sixteen => filter_four(samples, thresholds.hev),
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;

    use super::Thresholds;

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
}

fn filter_mask(limit: u8, blimit: u8, edge: &[u8]) -> bool {
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
        && u16::from(abs_diff(p0, q0)) * 2 + u16::from(abs_diff(p1, q1)) / 2 <= u16::from(blimit)
}

fn flat_mask(edge: &[u8]) -> bool {
    let p3 = edge[0];
    let p2 = edge[1];
    let p1 = edge[2];
    let p0 = edge[3];
    let q0 = edge[4];
    let q1 = edge[5];
    let q2 = edge[6];
    let q3 = edge[7];
    abs_diff(p1, p0) <= 1
        && abs_diff(q1, q0) <= 1
        && abs_diff(p2, p0) <= 1
        && abs_diff(q2, q0) <= 1
        && abs_diff(p3, p0) <= 1
        && abs_diff(q3, q0) <= 1
}

fn flat2_mask(samples: &[u8; 16]) -> bool {
    let p0 = samples[7];
    let q0 = samples[8];
    (0..=3).all(|index| abs_diff(samples[index], p0) <= 1)
        && (12..=15).all(|index| abs_diff(samples[index], q0) <= 1)
}

fn filter_four(samples: &mut [u8; 16], threshold: u8) {
    let ps1 = i32::from((samples[6] ^ 0x80) as i8);
    let ps0 = i32::from((samples[7] ^ 0x80) as i8);
    let qs0 = i32::from((samples[8] ^ 0x80) as i8);
    let qs1 = i32::from((samples[9] ^ 0x80) as i8);
    let hev = abs_diff(samples[6], samples[7]) > threshold
        || abs_diff(samples[9], samples[8]) > threshold;
    let outer = if hev { signed_char_clamp(ps1 - qs1) } else { 0 };
    let filter = signed_char_clamp(outer + 3 * (qs0 - ps0));
    let filter1 = signed_char_clamp(filter + 4) >> 3;
    let filter2 = signed_char_clamp(filter + 3) >> 3;
    samples[8] = signed_to_pixel(signed_char_clamp(qs0 - filter1));
    samples[7] = signed_to_pixel(signed_char_clamp(ps0 + filter2));
    if !hev {
        let adjustment = (filter1 + 1) >> 1;
        samples[9] = signed_to_pixel(signed_char_clamp(qs1 - adjustment));
        samples[6] = signed_to_pixel(signed_char_clamp(ps1 + adjustment));
    }
}

fn filter_eight(samples: &mut [u8; 16]) {
    let [p3, p2, p1, p0, q0, q1, q2, q3] = samples[4..12].try_into().unwrap();
    samples[5] = round_shift3(
        3 * u16::from(p3) + 2 * u16::from(p2) + u16::from(p1) + u16::from(p0) + u16::from(q0),
    );
    samples[6] = round_shift3(
        2 * u16::from(p3)
            + u16::from(p2)
            + 2 * u16::from(p1)
            + u16::from(p0)
            + u16::from(q0)
            + u16::from(q1),
    );
    samples[7] = round_shift3(
        u16::from(p3)
            + u16::from(p2)
            + u16::from(p1)
            + 2 * u16::from(p0)
            + u16::from(q0)
            + u16::from(q1)
            + u16::from(q2),
    );
    samples[8] = round_shift3(
        u16::from(p2)
            + u16::from(p1)
            + u16::from(p0)
            + 2 * u16::from(q0)
            + u16::from(q1)
            + u16::from(q2)
            + u16::from(q3),
    );
    samples[9] = round_shift3(
        u16::from(p1)
            + u16::from(p0)
            + u16::from(q0)
            + 2 * u16::from(q1)
            + u16::from(q2)
            + 2 * u16::from(q3),
    );
    samples[10] = round_shift3(
        u16::from(p0) + u16::from(q0) + u16::from(q1) + 2 * u16::from(q2) + 3 * u16::from(q3),
    );
}

fn filter_sixteen(samples: &mut [u8; 16]) {
    let source = *samples;
    let p = &source[..8];
    let q = &source[8..];
    let sum = |terms: &[(u16, u8)]| -> u8 {
        let total = terms
            .iter()
            .map(|&(weight, value)| weight * u16::from(value))
            .sum::<u16>();
        ((total + 8) >> 4) as u8
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
fn abs_diff(first: u8, second: u8) -> u8 {
    first.abs_diff(second)
}

#[inline]
fn signed_char_clamp(value: i32) -> i32 {
    value.clamp(-128, 127)
}

#[inline]
fn signed_to_pixel(value: i32) -> u8 {
    (value as i8 as u8) ^ 0x80
}

#[inline]
fn round_shift3(value: u16) -> u8 {
    ((value + 4) >> 3) as u8
}

#[cfg(test)]
mod tests {
    use super::{FilterWidth, Thresholds, filter_samples, filter_vertical, thresholds};

    #[test]
    fn flat_eight_tap_edge_is_smoothed_symmetrically() {
        let mut samples = [0; 16];
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
        );
        assert_eq!(&samples[5..11], &[101, 101, 102, 103, 103, 104]);
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
