//! Normative H.264 in-loop deblocking.
//!
//! This module starts with the sample-level filtering processes from clauses
//! 8.7.2.2 through 8.7.2.4. Picture traversal and boundary-strength derivation
//! are deliberately kept separate: callers provide the already-derived
//! boundary strength and the QPs on both sides of one edge.

use crate::{DeblockingFilter, H264Error, MotionVector, Result, Yuv420Picture};

const ALPHA: [u8; 52] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15, 17, 20,
    22, 25, 28, 32, 36, 40, 45, 50, 56, 63, 71, 80, 90, 101, 113, 127, 144, 162, 182, 203, 226,
    255, 255,
];

const BETA: [u8; 52] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 6, 6, 7, 7, 8, 8,
    9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18,
];

const TC0: [[u8; 52]; 3] = [
    [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 6, 6, 7, 8, 9, 10, 11, 13,
    ],
    [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 7, 8, 8, 10, 11, 12, 13, 15, 17,
    ],
    [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2,
        2, 3, 3, 3, 4, 4, 4, 5, 6, 6, 7, 8, 9, 10, 11, 13, 14, 16, 18, 20, 23, 25,
    ],
];

/// Eight input samples around one horizontal or vertical 4x4 block edge.
///
/// `p[0]` and `q[0]` are immediately adjacent to the edge. Increasing indices
/// move farther away from the edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeblockEdgeSamples {
    pub p: [u8; 4],
    pub q: [u8; 4],
}

/// The samples that H.264 permits one edge operation to replace.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FilteredDeblockEdge {
    pub p: [u8; 3],
    pub q: [u8; 3],
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MacroblockDeblockInfo {
    pub slice_id: u32,
    pub is_intra: bool,
    pub luma_qp: u8,
    pub cb_qp: u8,
    pub cr_qp: u8,
    pub transform_8x8: bool,
    /// Non-zero luma coefficients at raster-ordered 4x4 granularity.
    pub luma_nonzero: [bool; 16],
    /// List-0 reference identity and motion at raster-ordered 4x4 granularity.
    pub motion: [DeblockMotion; 16],
    pub filter: DeblockingFilter,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DeblockMotion {
    reference_ids: [u8; 2],
    vectors: [MotionVector; 2],
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DeblockListMotion {
    /// Stable for the lifetime of the reconstructed picture. Zero means absent.
    pub reference_id: u8,
    pub vector: MotionVector,
}

impl DeblockMotion {
    #[inline]
    pub const fn new(list0: DeblockListMotion, list1: DeblockListMotion) -> Self {
        Self {
            reference_ids: [list0.reference_id, list1.reference_id],
            vectors: [list0.vector, list1.vector],
        }
    }

    #[inline]
    const fn list0(self) -> DeblockListMotion {
        DeblockListMotion {
            reference_id: self.reference_ids[0],
            vector: self.vectors[0],
        }
    }

    #[inline]
    const fn list1(self) -> DeblockListMotion {
        DeblockListMotion {
            reference_id: self.reference_ids[1],
            vector: self.vectors[1],
        }
    }
}

impl DeblockEdgeSamples {
    #[inline]
    const fn unchanged(self) -> FilteredDeblockEdge {
        FilteredDeblockEdge {
            p: [self.p[0], self.p[1], self.p[2]],
            q: [self.q[0], self.q[1], self.q[2]],
        }
    }
}

/// Filters one 8-bit sample set across a decoded block edge.
///
/// `alpha_offset_div2` and `beta_offset_div2` are the signed values carried in
/// the slice header, not the already-doubled `FilterOffsetA`/`FilterOffsetB`
/// values used by the equations. `chroma_style` must be true for 4:2:0 and
/// 4:2:2 chroma edges and false for luma edges.
pub fn filter_deblock_edge(
    samples: DeblockEdgeSamples,
    boundary_strength: u8,
    qp_p: u8,
    qp_q: u8,
    alpha_offset_div2: i8,
    beta_offset_div2: i8,
    chroma_style: bool,
) -> Result<FilteredDeblockEdge> {
    let parameters = prepare_edge_parameters(EdgeParameters {
        boundary_strength,
        qp_p,
        qp_q,
        alpha_offset_div2,
        beta_offset_div2,
        chroma_style,
    })?;
    Ok(parameters.map_or_else(
        || samples.unchanged(),
        |parameters| filter_deblock_edge_prepared(samples, parameters),
    ))
}

fn filter_deblock_edge_prepared(
    samples: DeblockEdgeSamples,
    parameters: PreparedEdgeParameters,
) -> FilteredDeblockEdge {
    let unchanged = samples.unchanged();

    let [p0, p1, p2, p3] = samples.p.map(i16::from);
    let [q0, q1, q2, q3] = samples.q.map(i16::from);
    if (p0 - q0).abs() >= parameters.alpha
        || (p1 - p0).abs() >= parameters.beta
        || (q1 - q0).abs() >= parameters.beta
    {
        return unchanged;
    }

    if parameters.boundary_strength < 4 {
        let ap = (p2 - p0).abs();
        let aq = (q2 - q0).abs();
        let tc = if parameters.chroma_style {
            parameters.tc0 + 1
        } else {
            parameters.tc0 + i16::from(ap < parameters.beta) + i16::from(aq < parameters.beta)
        };
        let delta = ((((q0 - p0) << 2) + (p1 - q1) + 4) >> 3).clamp(-tc, tc);

        let filtered_p1 = if !parameters.chroma_style && ap < parameters.beta {
            p1 + ((p2 + ((p0 + q0 + 1) >> 1) - (p1 << 1)) >> 1)
                .clamp(-parameters.tc0, parameters.tc0)
        } else {
            p1
        };
        let filtered_q1 = if !parameters.chroma_style && aq < parameters.beta {
            q1 + ((q2 + ((p0 + q0 + 1) >> 1) - (q1 << 1)) >> 1)
                .clamp(-parameters.tc0, parameters.tc0)
        } else {
            q1
        };

        return FilteredDeblockEdge {
            p: [
                clip_sample(p0 + delta),
                clip_sample(filtered_p1),
                samples.p[2],
            ],
            q: [
                clip_sample(q0 - delta),
                clip_sample(filtered_q1),
                samples.q[2],
            ],
        };
    }

    let ap = (p2 - p0).abs();
    let aq = (q2 - q0).abs();
    let p = if !parameters.chroma_style
        && ap < parameters.beta
        && (p0 - q0).abs() < parameters.strong_threshold
    {
        [
            clip_sample((p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3),
            clip_sample((p2 + p1 + p0 + q0 + 2) >> 2),
            clip_sample((2 * p3 + 3 * p2 + p1 + p0 + q0 + 4) >> 3),
        ]
    } else {
        [
            clip_sample((2 * p1 + p0 + q1 + 2) >> 2),
            samples.p[1],
            samples.p[2],
        ]
    };
    let q = if !parameters.chroma_style
        && aq < parameters.beta
        && (p0 - q0).abs() < parameters.strong_threshold
    {
        [
            clip_sample((p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) >> 3),
            clip_sample((p0 + q0 + q1 + q2 + 2) >> 2),
            clip_sample((2 * q3 + 3 * q2 + q1 + q0 + p0 + 4) >> 3),
        ]
    } else {
        [
            clip_sample((2 * q1 + q0 + p1 + 2) >> 2),
            samples.q[1],
            samples.q[2],
        ]
    };
    FilteredDeblockEdge { p, q }
}

#[allow(clippy::needless_range_loop)]
pub(crate) fn filter_420_picture(
    picture: &mut Yuv420Picture,
    macroblocks: &[MacroblockDeblockInfo],
    width_in_macroblocks: usize,
) -> Result<()> {
    let (width, height) = picture.dimensions();
    if width_in_macroblocks == 0
        || !width.is_multiple_of(16)
        || !height.is_multiple_of(16)
        || width / 16 != width_in_macroblocks
        || macroblocks.len() != (width / 16) * (height / 16)
    {
        return Err(H264Error::InvalidSyntax(
            "deblocking metadata does not match picture dimensions",
        ));
    }

    let chroma_stride = width / 2;
    let (luma, cb, cr) = picture.planes_mut();
    for (address, current) in macroblocks.iter().enumerate() {
        let macroblock_x = address % width_in_macroblocks;
        let macroblock_y = address / width_in_macroblocks;
        if current.filter.idc == 1 {
            continue;
        }

        let left = (macroblock_x > 0).then(|| &macroblocks[address - 1]);
        let top = (macroblock_y > 0).then(|| &macroblocks[address - width_in_macroblocks]);
        let filter_left = left.is_some_and(|neighbor| {
            current.filter.idc != 2 || neighbor.slice_id == current.slice_id
        });
        let filter_top = top.is_some_and(|neighbor| {
            current.filter.idc != 2 || neighbor.slice_id == current.slice_id
        });
        let mut vertical_strengths = [[0u8; 4]; 4];
        let mut horizontal_strengths = [[0u8; 4]; 4];
        if filter_left {
            let previous = left.expect("filter_left requires a neighbor");
            for block_row in 0..4 {
                vertical_strengths[0][block_row] =
                    boundary_strength(previous, block_row * 4 + 3, current, block_row * 4, true);
            }
        }
        let internal_edges_zero = !current.is_intra
            && !current.luma_nonzero.iter().any(|&nonzero| nonzero)
            && current.motion[1..]
                .iter()
                .all(|&motion| motion == current.motion[0]);
        if !internal_edges_zero {
            for block_column in 1..4 {
                if block_column == 2 || !current.transform_8x8 {
                    for block_row in 0..4 {
                        let q = block_row * 4 + block_column;
                        vertical_strengths[block_column][block_row] =
                            boundary_strength(current, q - 1, current, q, false);
                    }
                }
            }
            for block_row in 1..4 {
                if block_row == 2 || !current.transform_8x8 {
                    for block_column in 0..4 {
                        let q = block_row * 4 + block_column;
                        horizontal_strengths[block_row][block_column] =
                            boundary_strength(current, q - 4, current, q, false);
                    }
                }
            }
        }
        if filter_top {
            let previous = top.expect("filter_top requires a neighbor");
            for block_column in 0..4 {
                horizontal_strengths[0][block_column] =
                    boundary_strength(previous, 12 + block_column, current, block_column, true);
            }
        }

        let internal_thresholds = if internal_edges_zero {
            [None; 3]
        } else {
            [
                edge_thresholds(current, current, 0)?,
                edge_thresholds(current, current, 1)?,
                edge_thresholds(current, current, 2)?,
            ]
        };
        let left_thresholds = if filter_left {
            let previous = left.expect("filter_left requires a neighbor");
            [
                edge_thresholds(previous, current, 0)?,
                edge_thresholds(previous, current, 1)?,
                edge_thresholds(previous, current, 2)?,
            ]
        } else {
            [None; 3]
        };
        let top_thresholds = if filter_top {
            let previous = top.expect("filter_top requires a neighbor");
            [
                edge_thresholds(previous, current, 0)?,
                edge_thresholds(previous, current, 1)?,
                edge_thresholds(previous, current, 2)?,
            ]
        } else {
            [None; 3]
        };

        let luma_x = macroblock_x * 16;
        let luma_y = macroblock_y * 16;
        if filter_left && vertical_strengths[0] != [0; 4] {
            for block_row in 0..4 {
                filter_vertical_edge(
                    luma,
                    width,
                    luma_x,
                    luma_y + block_row * 4,
                    4,
                    vertical_strengths[0][block_row],
                    left_thresholds[0],
                );
            }
        }
        for block_column in 1..4 {
            if (block_column == 2 || !current.transform_8x8)
                && vertical_strengths[block_column] != [0; 4]
            {
                for block_row in 0..4 {
                    filter_vertical_edge(
                        luma,
                        width,
                        luma_x + block_column * 4,
                        luma_y + block_row * 4,
                        4,
                        vertical_strengths[block_column][block_row],
                        internal_thresholds[0],
                    );
                }
            }
        }
        if filter_top && horizontal_strengths[0] != [0; 4] {
            for block_column in 0..4 {
                filter_horizontal_edge(
                    luma,
                    width,
                    luma_x + block_column * 4,
                    luma_y,
                    4,
                    horizontal_strengths[0][block_column],
                    top_thresholds[0],
                );
            }
        }
        for block_row in 1..4 {
            if (block_row == 2 || !current.transform_8x8)
                && horizontal_strengths[block_row] != [0; 4]
            {
                for block_column in 0..4 {
                    filter_horizontal_edge(
                        luma,
                        width,
                        luma_x + block_column * 4,
                        luma_y + block_row * 4,
                        4,
                        horizontal_strengths[block_row][block_column],
                        internal_thresholds[0],
                    );
                }
            }
        }

        let chroma_x = macroblock_x * 8;
        let chroma_y = macroblock_y * 8;
        for (plane, component) in [(&mut *cb, 1usize), (&mut *cr, 2usize)] {
            if filter_left && vertical_strengths[0] != [0; 4] {
                for block_row in 0..4 {
                    filter_vertical_edge(
                        plane,
                        chroma_stride,
                        chroma_x,
                        chroma_y + block_row * 2,
                        2,
                        vertical_strengths[0][block_row],
                        left_thresholds[component],
                    );
                }
            }
            if vertical_strengths[2] != [0; 4] {
                for block_row in 0..4 {
                    filter_vertical_edge(
                        plane,
                        chroma_stride,
                        chroma_x + 4,
                        chroma_y + block_row * 2,
                        2,
                        vertical_strengths[2][block_row],
                        internal_thresholds[component],
                    );
                }
            }
            if filter_top && horizontal_strengths[0] != [0; 4] {
                filter_horizontal_chroma_edge(
                    plane,
                    chroma_stride,
                    chroma_x,
                    chroma_y,
                    horizontal_strengths[0],
                    top_thresholds[component],
                );
            }
            if horizontal_strengths[2] != [0; 4] {
                filter_horizontal_chroma_edge(
                    plane,
                    chroma_stride,
                    chroma_x,
                    chroma_y + 4,
                    horizontal_strengths[2],
                    internal_thresholds[component],
                );
            }
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct EdgeParameters {
    boundary_strength: u8,
    qp_p: u8,
    qp_q: u8,
    alpha_offset_div2: i8,
    beta_offset_div2: i8,
    chroma_style: bool,
}

#[derive(Debug, Clone, Copy)]
struct EdgeThresholds {
    alpha: i16,
    beta: i16,
    index_a: usize,
    strong_threshold: i16,
    chroma_style: bool,
}

#[derive(Debug, Clone, Copy)]
struct PreparedEdgeParameters {
    boundary_strength: u8,
    alpha: i16,
    beta: i16,
    tc0: i16,
    strong_threshold: i16,
    chroma_style: bool,
}

fn prepare_edge_parameters(parameters: EdgeParameters) -> Result<Option<PreparedEdgeParameters>> {
    validate_edge_parameters(parameters)?;
    let thresholds = prepare_edge_thresholds_unchecked(
        parameters.qp_p,
        parameters.qp_q,
        parameters.alpha_offset_div2,
        parameters.beta_offset_div2,
        parameters.chroma_style,
    );
    Ok(prepare_edge_strength(
        parameters.boundary_strength,
        thresholds,
    ))
}

fn prepare_edge_thresholds_unchecked(
    qp_p: u8,
    qp_q: u8,
    alpha_offset_div2: i8,
    beta_offset_div2: i8,
    chroma_style: bool,
) -> Option<EdgeThresholds> {
    let qp_average = (i16::from(qp_p) + i16::from(qp_q) + 1) >> 1;
    let index_a = (qp_average + i16::from(alpha_offset_div2) * 2).clamp(0, 51) as usize;
    let index_b = (qp_average + i16::from(beta_offset_div2) * 2).clamp(0, 51) as usize;
    let alpha = i16::from(ALPHA[index_a]);
    let beta = i16::from(BETA[index_b]);
    if alpha == 0 || beta == 0 {
        return None;
    }

    Some(EdgeThresholds {
        alpha,
        beta,
        index_a,
        strong_threshold: (alpha >> 2) + 2,
        chroma_style,
    })
}

#[inline]
fn prepare_edge_strength(
    boundary_strength: u8,
    thresholds: Option<EdgeThresholds>,
) -> Option<PreparedEdgeParameters> {
    if boundary_strength == 0 {
        return None;
    }
    let thresholds = thresholds?;
    Some(PreparedEdgeParameters {
        boundary_strength,
        alpha: thresholds.alpha,
        beta: thresholds.beta,
        tc0: if boundary_strength < 4 {
            i16::from(TC0[usize::from(boundary_strength - 1)][thresholds.index_a])
        } else {
            0
        },
        strong_threshold: thresholds.strong_threshold,
        chroma_style: thresholds.chroma_style,
    })
}

fn edge_thresholds(
    previous: &MacroblockDeblockInfo,
    current: &MacroblockDeblockInfo,
    component: u8,
) -> Result<Option<EdgeThresholds>> {
    let qp = |macroblock: &MacroblockDeblockInfo| match component {
        0 => macroblock.luma_qp,
        1 => macroblock.cb_qp,
        _ => macroblock.cr_qp,
    };
    let qp_p = qp(previous);
    let qp_q = qp(current);
    let alpha_offset_div2 = current.filter.alpha_c0_offset_div2;
    let beta_offset_div2 = current.filter.beta_offset_div2;
    validate_threshold_inputs(qp_p, qp_q, alpha_offset_div2, beta_offset_div2)?;
    Ok(prepare_edge_thresholds_unchecked(
        qp_p,
        qp_q,
        alpha_offset_div2,
        beta_offset_div2,
        component != 0,
    ))
}

fn boundary_strength(
    previous: &MacroblockDeblockInfo,
    previous_cell: usize,
    current: &MacroblockDeblockInfo,
    current_cell: usize,
    external: bool,
) -> u8 {
    if previous.is_intra || current.is_intra {
        return if external { 4 } else { 3 };
    }
    if previous.luma_nonzero[previous_cell] || current.luma_nonzero[current_cell] {
        return 2;
    }

    u8::from(motion_differs(
        previous.motion[previous_cell],
        current.motion[current_cell],
    ))
}

fn motion_differs(previous: DeblockMotion, current: DeblockMotion) -> bool {
    let previous_l0 = previous.list0();
    let previous_l1 = previous.list1();
    let current_l0 = current.list0();
    let current_l1 = current.list1();
    let same_order_differs = list_motion_differs(previous_l0, current_l0)
        || list_motion_differs(previous_l1, current_l1);
    if !same_order_differs {
        return false;
    }

    let references_are_swapped = previous_l0.reference_id == current_l1.reference_id
        && previous_l1.reference_id == current_l0.reference_id;
    if !references_are_swapped {
        return true;
    }

    list_motion_differs(previous_l0, current_l1) || list_motion_differs(previous_l1, current_l0)
}

#[inline]
fn list_motion_differs(previous: DeblockListMotion, current: DeblockListMotion) -> bool {
    previous.reference_id != current.reference_id
        || (i32::from(previous.vector.x) - i32::from(current.vector.x)).abs() >= 4
        || (i32::from(previous.vector.y) - i32::from(current.vector.y)).abs() >= 4
}

fn filter_vertical_edge(
    plane: &mut [u8],
    stride: usize,
    x: usize,
    y: usize,
    length: usize,
    boundary_strength: u8,
    thresholds: Option<EdgeThresholds>,
) {
    let Some(parameters) = prepare_edge_strength(boundary_strength, thresholds) else {
        return;
    };
    #[cfg(target_arch = "x86_64")]
    if length == 4 && parameters.boundary_strength < 4 && !parameters.chroma_style {
        // SAFETY: SSE2 is part of the x86_64 baseline. Picture traversal
        // guarantees four complete rows and three samples on either side.
        unsafe {
            filter_vertical_weak_luma_sse2(plane, stride, x, y, parameters);
        }
        return;
    }
    for offset in 0..length {
        let q0 = (y + offset) * stride + x;
        let samples = DeblockEdgeSamples {
            p: std::array::from_fn(|index| plane[q0 - index - 1]),
            q: std::array::from_fn(|index| plane[q0 + index]),
        };
        let filtered = apply_parameters(samples, parameters);
        for index in 0..3 {
            plane[q0 - index - 1] = filtered.p[index];
            plane[q0 + index] = filtered.q[index];
        }
    }
}

fn filter_horizontal_edge(
    plane: &mut [u8],
    stride: usize,
    x: usize,
    y: usize,
    length: usize,
    boundary_strength: u8,
    thresholds: Option<EdgeThresholds>,
) {
    let Some(parameters) = prepare_edge_strength(boundary_strength, thresholds) else {
        return;
    };
    #[cfg(target_arch = "x86_64")]
    if length == 4 && parameters.boundary_strength < 4 && !parameters.chroma_style {
        // SAFETY: SSE2 is part of the x86_64 baseline. Picture traversal
        // guarantees four horizontally adjacent samples and three complete
        // rows on both sides of this luma edge.
        unsafe {
            filter_horizontal_weak_luma_sse2(plane, stride, x, y, parameters);
        }
        return;
    }
    filter_horizontal_edge_scalar(plane, stride, x, y, length, parameters);
}

fn filter_horizontal_edge_scalar(
    plane: &mut [u8],
    stride: usize,
    x: usize,
    y: usize,
    length: usize,
    parameters: PreparedEdgeParameters,
) {
    for offset in 0..length {
        let q0 = y * stride + x + offset;
        let samples = DeblockEdgeSamples {
            p: std::array::from_fn(|index| plane[q0 - (index + 1) * stride]),
            q: std::array::from_fn(|index| plane[q0 + index * stride]),
        };
        let filtered = apply_parameters(samples, parameters);
        for index in 0..3 {
            plane[q0 - (index + 1) * stride] = filtered.p[index];
            plane[q0 + index * stride] = filtered.q[index];
        }
    }
}

fn filter_horizontal_chroma_edge(
    plane: &mut [u8],
    stride: usize,
    x: usize,
    y: usize,
    boundary_strengths: [u8; 4],
    thresholds: Option<EdgeThresholds>,
) {
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: SSE2 is part of the x86_64 baseline. Picture traversal
        // guarantees eight horizontally adjacent chroma samples and two
        // complete rows on either side of the edge.
        unsafe {
            filter_horizontal_chroma_edge_sse2(plane, stride, x, y, boundary_strengths, thresholds);
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        for (block_column, boundary_strength) in boundary_strengths.into_iter().enumerate() {
            filter_horizontal_edge(
                plane,
                stride,
                x + block_column * 2,
                y,
                2,
                boundary_strength,
                thresholds,
            );
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn filter_horizontal_chroma_edge_sse2(
    plane: &mut [u8],
    stride: usize,
    x: usize,
    y: usize,
    boundary_strengths: [u8; 4],
    thresholds: Option<EdgeThresholds>,
) {
    use std::arch::x86_64::{
        __m128i, _mm_add_epi16, _mm_and_si128, _mm_andnot_si128, _mm_cmplt_epi16,
        _mm_cvtsi64_si128, _mm_cvtsi128_si64, _mm_loadu_si128, _mm_max_epi16, _mm_min_epi16,
        _mm_or_si128, _mm_packus_epi16, _mm_set1_epi16, _mm_setzero_si128, _mm_slli_epi16,
        _mm_srai_epi16, _mm_sub_epi16, _mm_unpacklo_epi8,
    };

    macro_rules! absolute {
        ($value:expr, $zero:expr) => {{
            let value = $value;
            _mm_max_epi16(value, _mm_sub_epi16($zero, value))
        }};
    }
    macro_rules! select {
        ($mask:expr, $selected:expr, $fallback:expr) => {
            _mm_or_si128(
                _mm_and_si128($mask, $selected),
                _mm_andnot_si128($mask, $fallback),
            )
        };
    }
    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn load_eight(ptr: *const u8, zero: __m128i) -> __m128i {
        // SAFETY: The caller proves that the eight-byte row is in-bounds.
        let packed = unsafe { ptr.cast::<u64>().read_unaligned() };
        _mm_unpacklo_epi8(_mm_cvtsi64_si128(packed as i64), zero)
    }
    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn store_eight(ptr: *mut u8, values: __m128i, zero: __m128i) {
        let packed = _mm_cvtsi128_si64(_mm_packus_epi16(values, zero)) as u64;
        // SAFETY: The caller proves that the eight-byte row is in-bounds.
        unsafe {
            ptr.cast::<u64>().write_unaligned(packed);
        }
    }

    let Some(thresholds) = thresholds else {
        return;
    };
    debug_assert!(thresholds.chroma_style);
    let mut active = [0i16; 8];
    let mut strong = [0i16; 8];
    let mut tc = [0i16; 8];
    for (segment, boundary_strength) in boundary_strengths.into_iter().enumerate() {
        debug_assert!(boundary_strength <= 4);
        if boundary_strength == 0 {
            continue;
        }
        active[segment * 2..segment * 2 + 2].fill(-1);
        if boundary_strength == 4 {
            strong[segment * 2..segment * 2 + 2].fill(-1);
        } else {
            tc[segment * 2..segment * 2 + 2]
                .fill(i16::from(TC0[usize::from(boundary_strength - 1)][thresholds.index_a]) + 1);
        }
    }

    let zero = _mm_setzero_si128();
    // SAFETY: Local arrays contain exactly eight i16 lanes.
    let active = unsafe { _mm_loadu_si128(active.as_ptr().cast::<__m128i>()) };
    // SAFETY: Local arrays contain exactly eight i16 lanes.
    let strong = unsafe { _mm_loadu_si128(strong.as_ptr().cast::<__m128i>()) };
    // SAFETY: Local arrays contain exactly eight i16 lanes.
    let tc = unsafe { _mm_loadu_si128(tc.as_ptr().cast::<__m128i>()) };
    let base = plane.as_mut_ptr().wrapping_add(y * stride + x);
    // SAFETY: Picture traversal supplies two complete rows on either side.
    let p0 = unsafe { load_eight(base.wrapping_sub(stride), zero) };
    // SAFETY: See above.
    let p1 = unsafe { load_eight(base.wrapping_sub(2 * stride), zero) };
    // SAFETY: See above.
    let q0 = unsafe { load_eight(base, zero) };
    // SAFETY: See above.
    let q1 = unsafe { load_eight(base.wrapping_add(stride), zero) };

    let alpha = _mm_set1_epi16(thresholds.alpha);
    let beta = _mm_set1_epi16(thresholds.beta);
    let valid = _mm_and_si128(
        active,
        _mm_and_si128(
            _mm_cmplt_epi16(absolute!(_mm_sub_epi16(p0, q0), zero), alpha),
            _mm_and_si128(
                _mm_cmplt_epi16(absolute!(_mm_sub_epi16(p1, p0), zero), beta),
                _mm_cmplt_epi16(absolute!(_mm_sub_epi16(q1, q0), zero), beta),
            ),
        ),
    );

    let negative_tc = _mm_sub_epi16(zero, tc);
    let delta = _mm_srai_epi16::<3>(_mm_add_epi16(
        _mm_add_epi16(
            _mm_slli_epi16::<2>(_mm_sub_epi16(q0, p0)),
            _mm_sub_epi16(p1, q1),
        ),
        _mm_set1_epi16(4),
    ));
    let delta = _mm_min_epi16(_mm_max_epi16(delta, negative_tc), tc);
    let weak_p0 = _mm_add_epi16(p0, delta);
    let weak_q0 = _mm_sub_epi16(q0, delta);
    let strong_p0 = _mm_srai_epi16::<2>(_mm_add_epi16(
        _mm_add_epi16(_mm_add_epi16(_mm_slli_epi16::<1>(p1), p0), q1),
        _mm_set1_epi16(2),
    ));
    let strong_q0 = _mm_srai_epi16::<2>(_mm_add_epi16(
        _mm_add_epi16(_mm_add_epi16(_mm_slli_epi16::<1>(q1), q0), p1),
        _mm_set1_epi16(2),
    ));
    let filtered_p0 = select!(valid, select!(strong, strong_p0, weak_p0), p0);
    let filtered_q0 = select!(valid, select!(strong, strong_q0, weak_q0), q0);

    // SAFETY: The same validated rows used for the loads are writable.
    unsafe {
        store_eight(base.wrapping_sub(stride), filtered_p0, zero);
        store_eight(base, filtered_q0, zero);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn filter_horizontal_weak_luma_sse2(
    plane: &mut [u8],
    stride: usize,
    x: usize,
    y: usize,
    parameters: PreparedEdgeParameters,
) {
    use std::arch::x86_64::{
        __m128i, _mm_add_epi16, _mm_and_si128, _mm_andnot_si128, _mm_cmplt_epi16,
        _mm_cvtsi32_si128, _mm_cvtsi128_si32, _mm_max_epi16, _mm_min_epi16, _mm_or_si128,
        _mm_packus_epi16, _mm_set1_epi16, _mm_setzero_si128, _mm_slli_epi16, _mm_srai_epi16,
        _mm_sub_epi16, _mm_unpacklo_epi8,
    };

    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn load_four(ptr: *const u8, zero: __m128i) -> __m128i {
        // SAFETY: The caller proves the four-byte row range is in-bounds.
        let packed = unsafe { ptr.cast::<i32>().read_unaligned() };
        _mm_unpacklo_epi8(_mm_cvtsi32_si128(packed), zero)
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn store_four(ptr: *mut u8, values: __m128i, zero: __m128i) {
        let packed = _mm_cvtsi128_si32(_mm_packus_epi16(values, zero));
        // SAFETY: The caller proves the four-byte row range is in-bounds.
        unsafe {
            ptr.cast::<i32>().write_unaligned(packed);
        }
    }

    macro_rules! absolute {
        ($value:expr, $zero:expr) => {{
            let value = $value;
            _mm_max_epi16(value, _mm_sub_epi16($zero, value))
        }};
    }
    macro_rules! select {
        ($mask:expr, $selected:expr, $fallback:expr) => {
            _mm_or_si128(
                _mm_and_si128($mask, $selected),
                _mm_andnot_si128($mask, $fallback),
            )
        };
    }

    let zero = _mm_setzero_si128();
    let base = plane.as_mut_ptr().wrapping_add(y * stride + x);
    // SAFETY: Picture traversal supplies three complete rows on either side.
    let p0 = unsafe { load_four(base.wrapping_sub(stride), zero) };
    // SAFETY: See above.
    let p1 = unsafe { load_four(base.wrapping_sub(2 * stride), zero) };
    // SAFETY: See above.
    let p2 = unsafe { load_four(base.wrapping_sub(3 * stride), zero) };
    // SAFETY: See above.
    let q0 = unsafe { load_four(base, zero) };
    // SAFETY: See above.
    let q1 = unsafe { load_four(base.wrapping_add(stride), zero) };
    // SAFETY: See above.
    let q2 = unsafe { load_four(base.wrapping_add(2 * stride), zero) };

    let alpha = _mm_set1_epi16(parameters.alpha);
    let beta = _mm_set1_epi16(parameters.beta);
    let valid = _mm_and_si128(
        _mm_cmplt_epi16(absolute!(_mm_sub_epi16(p0, q0), zero), alpha),
        _mm_and_si128(
            _mm_cmplt_epi16(absolute!(_mm_sub_epi16(p1, p0), zero), beta),
            _mm_cmplt_epi16(absolute!(_mm_sub_epi16(q1, q0), zero), beta),
        ),
    );
    let ap = _mm_cmplt_epi16(absolute!(_mm_sub_epi16(p2, p0), zero), beta);
    let aq = _mm_cmplt_epi16(absolute!(_mm_sub_epi16(q2, q0), zero), beta);
    let one = _mm_set1_epi16(1);
    let tc0 = _mm_set1_epi16(parameters.tc0);
    let tc = _mm_add_epi16(
        tc0,
        _mm_add_epi16(_mm_and_si128(ap, one), _mm_and_si128(aq, one)),
    );
    let negative_tc = _mm_sub_epi16(zero, tc);
    let delta = _mm_srai_epi16::<3>(_mm_add_epi16(
        _mm_add_epi16(
            _mm_slli_epi16::<2>(_mm_sub_epi16(q0, p0)),
            _mm_sub_epi16(p1, q1),
        ),
        _mm_set1_epi16(4),
    ));
    let delta = _mm_min_epi16(_mm_max_epi16(delta, negative_tc), tc);

    let average = _mm_srai_epi16::<1>(_mm_add_epi16(_mm_add_epi16(p0, q0), one));
    let negative_tc0 = _mm_sub_epi16(zero, tc0);
    let p1_delta = _mm_srai_epi16::<1>(_mm_sub_epi16(
        _mm_add_epi16(p2, average),
        _mm_slli_epi16::<1>(p1),
    ));
    let p1_delta = _mm_min_epi16(_mm_max_epi16(p1_delta, negative_tc0), tc0);
    let q1_delta = _mm_srai_epi16::<1>(_mm_sub_epi16(
        _mm_add_epi16(q2, average),
        _mm_slli_epi16::<1>(q1),
    ));
    let q1_delta = _mm_min_epi16(_mm_max_epi16(q1_delta, negative_tc0), tc0);

    let filtered_p0 = select!(valid, _mm_add_epi16(p0, delta), p0);
    let filtered_q0 = select!(valid, _mm_sub_epi16(q0, delta), q0);
    let filtered_p1 = select!(valid, select!(ap, _mm_add_epi16(p1, p1_delta), p1), p1);
    let filtered_q1 = select!(valid, select!(aq, _mm_add_epi16(q1, q1_delta), q1), q1);

    // SAFETY: The same validated row ranges used for the loads are writable.
    unsafe {
        store_four(base.wrapping_sub(stride), filtered_p0, zero);
        store_four(base.wrapping_sub(2 * stride), filtered_p1, zero);
        store_four(base, filtered_q0, zero);
        store_four(base.wrapping_add(stride), filtered_q1, zero);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn filter_vertical_weak_luma_sse2(
    plane: &mut [u8],
    stride: usize,
    x: usize,
    y: usize,
    parameters: PreparedEdgeParameters,
) {
    use std::arch::x86_64::{
        _mm_add_epi16, _mm_and_si128, _mm_andnot_si128, _mm_cmplt_epi16, _mm_cvtsi64_si128,
        _mm_cvtsi128_si32, _mm_max_epi16, _mm_min_epi16, _mm_or_si128, _mm_packus_epi16,
        _mm_set1_epi16, _mm_setzero_si128, _mm_slli_epi16, _mm_srai_epi16, _mm_srli_si128,
        _mm_sub_epi16, _mm_unpacklo_epi8, _mm_unpacklo_epi16,
    };

    macro_rules! absolute {
        ($value:expr, $zero:expr) => {{
            let value = $value;
            _mm_max_epi16(value, _mm_sub_epi16($zero, value))
        }};
    }
    macro_rules! select {
        ($mask:expr, $selected:expr, $fallback:expr) => {
            _mm_or_si128(
                _mm_and_si128($mask, $selected),
                _mm_andnot_si128($mask, $fallback),
            )
        };
    }
    macro_rules! column {
        ($columns:expr, $shift:literal, $zero:expr) => {
            _mm_unpacklo_epi8(_mm_srli_si128::<$shift>($columns), $zero)
        };
    }

    let base = plane
        .as_mut_ptr()
        .wrapping_add(y * stride + x)
        .wrapping_sub(4);
    let mut rows = [0u64; 4];
    for (row, value) in rows.iter_mut().enumerate() {
        // SAFETY: The edge has at least three samples on the left and the
        // eight-byte load covers p3 through q3 within the luma row.
        *value = unsafe {
            base.wrapping_add(row * stride)
                .cast::<u64>()
                .read_unaligned()
        };
    }
    let row0 = _mm_cvtsi64_si128(rows[0] as i64);
    let row1 = _mm_cvtsi64_si128(rows[1] as i64);
    let row2 = _mm_cvtsi64_si128(rows[2] as i64);
    let row3 = _mm_cvtsi64_si128(rows[3] as i64);
    let rows01 = _mm_unpacklo_epi8(row0, row1);
    let rows23 = _mm_unpacklo_epi8(row2, row3);
    let columns0 = _mm_unpacklo_epi16(rows01, rows23);
    let columns4 = _mm_unpacklo_epi16(_mm_srli_si128::<8>(rows01), _mm_srli_si128::<8>(rows23));
    let zero = _mm_setzero_si128();
    let p2 = column!(columns0, 4, zero);
    let p1 = column!(columns0, 8, zero);
    let p0 = column!(columns0, 12, zero);
    let q0 = column!(columns4, 0, zero);
    let q1 = column!(columns4, 4, zero);
    let q2 = column!(columns4, 8, zero);

    let alpha = _mm_set1_epi16(parameters.alpha);
    let beta = _mm_set1_epi16(parameters.beta);
    let valid = _mm_and_si128(
        _mm_cmplt_epi16(absolute!(_mm_sub_epi16(p0, q0), zero), alpha),
        _mm_and_si128(
            _mm_cmplt_epi16(absolute!(_mm_sub_epi16(p1, p0), zero), beta),
            _mm_cmplt_epi16(absolute!(_mm_sub_epi16(q1, q0), zero), beta),
        ),
    );
    let ap = _mm_cmplt_epi16(absolute!(_mm_sub_epi16(p2, p0), zero), beta);
    let aq = _mm_cmplt_epi16(absolute!(_mm_sub_epi16(q2, q0), zero), beta);
    let one = _mm_set1_epi16(1);
    let tc0 = _mm_set1_epi16(parameters.tc0);
    let tc = _mm_add_epi16(
        tc0,
        _mm_add_epi16(_mm_and_si128(ap, one), _mm_and_si128(aq, one)),
    );
    let negative_tc = _mm_sub_epi16(zero, tc);
    let delta = _mm_srai_epi16::<3>(_mm_add_epi16(
        _mm_add_epi16(
            _mm_slli_epi16::<2>(_mm_sub_epi16(q0, p0)),
            _mm_sub_epi16(p1, q1),
        ),
        _mm_set1_epi16(4),
    ));
    let delta = _mm_min_epi16(_mm_max_epi16(delta, negative_tc), tc);
    let average = _mm_srai_epi16::<1>(_mm_add_epi16(_mm_add_epi16(p0, q0), one));
    let negative_tc0 = _mm_sub_epi16(zero, tc0);
    let p1_delta = _mm_srai_epi16::<1>(_mm_sub_epi16(
        _mm_add_epi16(p2, average),
        _mm_slli_epi16::<1>(p1),
    ));
    let p1_delta = _mm_min_epi16(_mm_max_epi16(p1_delta, negative_tc0), tc0);
    let q1_delta = _mm_srai_epi16::<1>(_mm_sub_epi16(
        _mm_add_epi16(q2, average),
        _mm_slli_epi16::<1>(q1),
    ));
    let q1_delta = _mm_min_epi16(_mm_max_epi16(q1_delta, negative_tc0), tc0);
    let filtered = [
        select!(valid, select!(ap, _mm_add_epi16(p1, p1_delta), p1), p1),
        select!(valid, _mm_add_epi16(p0, delta), p0),
        select!(valid, _mm_sub_epi16(q0, delta), q0),
        select!(valid, select!(aq, _mm_add_epi16(q1, q1_delta), q1), q1),
    ]
    .map(|values| _mm_cvtsi128_si32(_mm_packus_epi16(values, zero)) as u32);

    const REPLACED_BYTES: u64 = 0x0000_ffff_ffff_0000;
    for (row, row_samples) in rows.iter_mut().enumerate() {
        let replacement = u64::from(filtered[0].to_le_bytes()[row]) << 16
            | u64::from(filtered[1].to_le_bytes()[row]) << 24
            | u64::from(filtered[2].to_le_bytes()[row]) << 32
            | u64::from(filtered[3].to_le_bytes()[row]) << 40;
        *row_samples = (*row_samples & !REPLACED_BYTES) | replacement;
        // SAFETY: This is the same validated eight-byte row loaded above.
        unsafe {
            base.wrapping_add(row * stride)
                .cast::<u64>()
                .write_unaligned(*row_samples);
        }
    }
}

fn apply_parameters(
    samples: DeblockEdgeSamples,
    parameters: PreparedEdgeParameters,
) -> FilteredDeblockEdge {
    filter_deblock_edge_prepared(samples, parameters)
}

fn validate_edge_parameters(parameters: EdgeParameters) -> Result<()> {
    validate_inputs(
        parameters.boundary_strength,
        parameters.qp_p,
        parameters.qp_q,
        parameters.alpha_offset_div2,
        parameters.beta_offset_div2,
    )
}

fn validate_inputs(
    boundary_strength: u8,
    qp_p: u8,
    qp_q: u8,
    alpha_offset_div2: i8,
    beta_offset_div2: i8,
) -> Result<()> {
    if boundary_strength > 4 {
        return Err(H264Error::InvalidSyntax(
            "deblocking boundary strength exceeds 4",
        ));
    }
    validate_threshold_inputs(qp_p, qp_q, alpha_offset_div2, beta_offset_div2)
}

fn validate_threshold_inputs(
    qp_p: u8,
    qp_q: u8,
    alpha_offset_div2: i8,
    beta_offset_div2: i8,
) -> Result<()> {
    if qp_p > 51 || qp_q > 51 {
        return Err(H264Error::InvalidSyntax("deblocking QP exceeds 51"));
    }
    if !(-6..=6).contains(&alpha_offset_div2) || !(-6..=6).contains(&beta_offset_div2) {
        return Err(H264Error::InvalidSyntax(
            "deblocking offset_div2 is outside -6..=6",
        ));
    }
    Ok(())
}

#[inline]
fn clip_sample(value: i16) -> u8 {
    value.clamp(0, 255) as u8
}

#[cfg(test)]
mod tests {
    use decv_core::Size;

    use super::{
        ALPHA, BETA, DeblockEdgeSamples, DeblockListMotion, DeblockMotion, FilteredDeblockEdge,
        MacroblockDeblockInfo, TC0, boundary_strength, filter_420_picture, filter_deblock_edge,
    };
    #[cfg(target_arch = "x86_64")]
    use super::{
        EdgeParameters, filter_horizontal_chroma_edge, filter_horizontal_edge,
        filter_horizontal_edge_scalar, filter_horizontal_weak_luma_sse2,
        filter_vertical_weak_luma_sse2, prepare_edge_parameters, prepare_edge_thresholds_unchecked,
    };
    use crate::{DeblockingFilter, H264Error, MotionVector, Yuv420Picture};

    const SMOOTH_EDGE: DeblockEdgeSamples = DeblockEdgeSamples {
        p: [100, 99, 98, 97],
        q: [110, 111, 112, 113],
    };

    fn macroblock(slice_id: u32, idc: u8) -> MacroblockDeblockInfo {
        MacroblockDeblockInfo {
            slice_id,
            is_intra: true,
            luma_qp: 40,
            cb_qp: 40,
            cr_qp: 40,
            transform_8x8: false,
            luma_nonzero: [false; 16],
            motion: [DeblockMotion::default(); 16],
            filter: DeblockingFilter {
                idc,
                alpha_c0_offset_div2: 0,
                beta_offset_div2: 0,
            },
        }
    }

    fn inter_macroblock(reference_id: u8, vector: MotionVector) -> MacroblockDeblockInfo {
        MacroblockDeblockInfo {
            is_intra: false,
            motion: [DeblockMotion::new(
                DeblockListMotion {
                    reference_id,
                    vector,
                },
                DeblockListMotion::default(),
            ); 16],
            ..macroblock(1, 0)
        }
    }

    #[test]
    fn normative_threshold_tables_have_expected_boundaries() {
        assert_eq!(
            (ALPHA[15], ALPHA[16], ALPHA[50], ALPHA[51]),
            (0, 4, 255, 255)
        );
        assert_eq!((BETA[15], BETA[16], BETA[50], BETA[51]), (0, 2, 18, 18));
        assert_eq!((TC0[0][22], TC0[0][23], TC0[0][51]), (0, 1, 13));
        assert_eq!((TC0[1][20], TC0[1][21], TC0[1][51]), (0, 1, 17));
        assert_eq!((TC0[2][16], TC0[2][17], TC0[2][51]), (0, 1, 25));
    }

    #[test]
    fn motion_cells_do_not_pad_each_reference_list() {
        assert_eq!(
            std::mem::size_of::<DeblockMotion>(),
            2 * std::mem::size_of::<u8>() + 2 * std::mem::size_of::<MotionVector>()
        );
    }

    #[test]
    fn zero_strength_and_failed_threshold_gate_are_no_ops() {
        let unchanged = FilteredDeblockEdge {
            p: [100, 99, 98],
            q: [110, 111, 112],
        };
        assert_eq!(
            filter_deblock_edge(SMOOTH_EDGE, 0, 40, 40, 0, 0, false).unwrap(),
            unchanged
        );
        assert_eq!(
            filter_deblock_edge(SMOOTH_EDGE, 3, 15, 15, 0, 0, false).unwrap(),
            unchanged
        );
    }

    #[test]
    fn weak_filter_updates_luma_side_samples() {
        assert_eq!(
            filter_deblock_edge(SMOOTH_EDGE, 2, 40, 40, 0, 0, false).unwrap(),
            FilteredDeblockEdge {
                p: [104, 101, 98],
                q: [106, 108, 112],
            }
        );
    }

    #[test]
    fn weak_chroma_style_filter_leaves_p1_and_q1_unchanged() {
        assert_eq!(
            filter_deblock_edge(SMOOTH_EDGE, 2, 40, 40, 0, 0, true).unwrap(),
            FilteredDeblockEdge {
                p: [104, 99, 98],
                q: [106, 111, 112],
            }
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn horizontal_weak_luma_sse2_matches_scalar_filtering() {
        for qp in [18, 24, 32, 40, 51] {
            for boundary_strength in 1..=3 {
                let parameters = prepare_edge_parameters(EdgeParameters {
                    boundary_strength,
                    qp_p: qp,
                    qp_q: qp.saturating_sub(2),
                    alpha_offset_div2: 0,
                    beta_offset_div2: 0,
                    chroma_style: false,
                })
                .unwrap()
                .unwrap();
                for seed in 0..16usize {
                    let original: Vec<u8> = (0..32 * 32)
                        .map(|index| {
                            let x = index % 32;
                            let y = index / 32;
                            if seed.is_multiple_of(2) {
                                96 + ((x * 3 + y * 2 + seed) % 13) as u8
                            } else {
                                ((index * 73 + index / 11 * 29 + seed * 41) & 0xff) as u8
                            }
                        })
                        .collect();
                    let mut scalar = original.clone();
                    filter_horizontal_edge_scalar(&mut scalar, 32, 8, 16, 4, parameters);
                    let mut simd = original;
                    // SAFETY: SSE2 is part of the x86_64 baseline, and this
                    // test supplies the same in-bounds four-sample edge as
                    // the scalar implementation.
                    unsafe {
                        filter_horizontal_weak_luma_sse2(&mut simd, 32, 8, 16, parameters);
                    }
                    assert_eq!(
                        simd, scalar,
                        "qp={qp} boundary_strength={boundary_strength} seed={seed}"
                    );

                    let mut scalar = simd.clone();
                    for row in 0..4 {
                        let q0 = (8 + row) * 32 + 16;
                        let samples = DeblockEdgeSamples {
                            p: std::array::from_fn(|index| scalar[q0 - index - 1]),
                            q: std::array::from_fn(|index| scalar[q0 + index]),
                        };
                        let filtered = filter_deblock_edge(
                            samples,
                            boundary_strength,
                            qp,
                            qp.saturating_sub(2),
                            0,
                            0,
                            false,
                        )
                        .unwrap();
                        for index in 0..3 {
                            scalar[q0 - index - 1] = filtered.p[index];
                            scalar[q0 + index] = filtered.q[index];
                        }
                    }
                    // SAFETY: The test supplies an in-bounds four-row edge.
                    unsafe {
                        filter_vertical_weak_luma_sse2(&mut simd, 32, 16, 8, parameters);
                    }
                    assert_eq!(
                        simd, scalar,
                        "vertical qp={qp} boundary_strength={boundary_strength} seed={seed}"
                    );
                }
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn horizontal_chroma_sse2_matches_segmented_scalar_filtering() {
        const STRENGTHS: [[u8; 4]; 6] = [
            [0, 0, 0, 0],
            [1, 2, 3, 4],
            [4, 3, 2, 1],
            [1, 1, 2, 2],
            [4, 4, 4, 4],
            [0, 3, 0, 4],
        ];
        for qp in [18, 24, 32, 40, 51] {
            let thresholds =
                prepare_edge_thresholds_unchecked(qp, qp.saturating_sub(2), 0, 0, true);
            for strengths in STRENGTHS {
                for seed in 0..16usize {
                    let original: Vec<u8> = (0..32 * 32)
                        .map(|index| {
                            let x = index % 32;
                            let y = index / 32;
                            if seed.is_multiple_of(2) {
                                96 + ((x * 3 + y * 2 + seed) % 13) as u8
                            } else {
                                ((index * 73 + index / 11 * 29 + seed * 41) & 0xff) as u8
                            }
                        })
                        .collect();
                    let mut scalar = original.clone();
                    for (segment, strength) in strengths.into_iter().enumerate() {
                        filter_horizontal_edge(
                            &mut scalar,
                            32,
                            8 + segment * 2,
                            16,
                            2,
                            strength,
                            thresholds,
                        );
                    }
                    let mut simd = original;
                    filter_horizontal_chroma_edge(&mut simd, 32, 8, 16, strengths, thresholds);
                    assert_eq!(simd, scalar, "qp={qp} strengths={strengths:?} seed={seed}");
                }
            }
        }
    }

    #[test]
    fn strong_filter_uses_wide_luma_tap_set() {
        assert_eq!(
            filter_deblock_edge(SMOOTH_EDGE, 4, 40, 40, 0, 0, false).unwrap(),
            FilteredDeblockEdge {
                p: [103, 102, 100],
                q: [107, 108, 110],
            }
        );
    }

    #[test]
    fn strong_chroma_style_filter_uses_narrow_tap_set() {
        assert_eq!(
            filter_deblock_edge(SMOOTH_EDGE, 4, 40, 40, 0, 0, true).unwrap(),
            FilteredDeblockEdge {
                p: [102, 99, 98],
                q: [108, 111, 112],
            }
        );
    }

    #[test]
    fn rejects_out_of_range_derived_inputs() {
        assert_eq!(
            filter_deblock_edge(SMOOTH_EDGE, 5, 40, 40, 0, 0, false),
            Err(H264Error::InvalidSyntax(
                "deblocking boundary strength exceeds 4"
            ))
        );
        assert!(filter_deblock_edge(SMOOTH_EDGE, 1, 52, 40, 0, 0, false).is_err());
        assert!(filter_deblock_edge(SMOOTH_EDGE, 1, 40, 40, 7, 0, false).is_err());
    }

    #[test]
    fn filters_intra_macroblock_boundaries_in_place() {
        let mut picture = Yuv420Picture::new(Size::new(32, 16)).unwrap();
        let (luma, cb, cr) = picture.planes_mut();
        for row in luma.chunks_exact_mut(32) {
            row[..16].fill(100);
            row[16..].fill(110);
        }
        for plane in [cb, cr] {
            for row in plane.chunks_exact_mut(16) {
                row[..8].fill(100);
                row[8..].fill(110);
            }
        }

        filter_420_picture(&mut picture, &[macroblock(1, 0), macroblock(1, 0)], 2).unwrap();

        let (luma, cb, cr) = picture.planes_mut();
        assert_eq!(&luma[12..20], &[100, 101, 103, 104, 106, 108, 109, 110]);
        assert_eq!(&cb[4..12], &[100, 100, 100, 103, 108, 110, 110, 110]);
        assert_eq!(&cr[4..12], &[100, 100, 100, 103, 108, 110, 110, 110]);
    }

    #[test]
    fn idc_two_preserves_edges_between_slices() {
        let mut picture = Yuv420Picture::new(Size::new(32, 16)).unwrap();
        let (luma, _, _) = picture.planes_mut();
        for row in luma.chunks_exact_mut(32) {
            row[..16].fill(100);
            row[16..].fill(110);
        }

        filter_420_picture(&mut picture, &[macroblock(1, 0), macroblock(2, 2)], 2).unwrap();

        let (luma, _, _) = picture.planes_mut();
        assert_eq!(&luma[12..20], &[100, 100, 100, 100, 110, 110, 110, 110]);
    }

    #[test]
    fn derives_progressive_p_boundary_strengths() {
        let same = inter_macroblock(7, MotionVector { x: 3, y: -2 });
        assert_eq!(boundary_strength(&same, 3, &same, 0, true), 0);

        let different_reference = inter_macroblock(8, same.motion[0].list0().vector);
        assert_eq!(
            boundary_strength(&same, 3, &different_reference, 0, true),
            1
        );

        let different_vector = inter_macroblock(7, MotionVector { x: 7, y: -2 });
        assert_eq!(boundary_strength(&same, 3, &different_vector, 0, true), 1);

        let mut residual = same;
        residual.luma_nonzero[3] = true;
        assert_eq!(boundary_strength(&residual, 3, &same, 0, true), 2);
        assert_eq!(boundary_strength(&macroblock(1, 0), 3, &same, 0, true), 4);
        assert_eq!(boundary_strength(&macroblock(1, 0), 3, &same, 0, false), 3);
    }

    #[test]
    fn derives_bidirectional_boundary_strength_with_swapped_lists() {
        let list = |reference_id, x| DeblockListMotion {
            reference_id,
            vector: MotionVector { x, y: 0 },
        };
        let bidirectional = |list0, list1| MacroblockDeblockInfo {
            is_intra: false,
            motion: [DeblockMotion::new(list0, list1); 16],
            ..macroblock(1, 0)
        };

        let previous = bidirectional(list(7, 2), list(8, 9));
        let swapped = bidirectional(list(8, 9), list(7, 2));
        assert_eq!(boundary_strength(&previous, 3, &swapped, 0, true), 0);

        let crossed_difference = bidirectional(list(8, 13), list(7, 2));
        assert_eq!(
            boundary_strength(&previous, 3, &crossed_difference, 0, true),
            1
        );

        let mismatched_pair = bidirectional(list(7, 2), list(9, 9));
        assert_eq!(
            boundary_strength(&previous, 3, &mismatched_pair, 0, true),
            1
        );
    }
}
