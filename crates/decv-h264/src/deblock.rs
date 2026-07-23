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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    pub list0: DeblockListMotion,
    pub list1: DeblockListMotion,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DeblockListMotion {
    /// Stable for the lifetime of the active reference list. Zero means absent.
    pub reference_id: usize,
    pub vector: MotionVector,
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
    validate_inputs(
        boundary_strength,
        qp_p,
        qp_q,
        alpha_offset_div2,
        beta_offset_div2,
    )?;
    Ok(filter_deblock_edge_validated(
        samples,
        boundary_strength,
        qp_p,
        qp_q,
        alpha_offset_div2,
        beta_offset_div2,
        chroma_style,
    ))
}

#[allow(clippy::too_many_arguments)]
fn filter_deblock_edge_validated(
    samples: DeblockEdgeSamples,
    boundary_strength: u8,
    qp_p: u8,
    qp_q: u8,
    alpha_offset_div2: i8,
    beta_offset_div2: i8,
    chroma_style: bool,
) -> FilteredDeblockEdge {
    let unchanged = samples.unchanged();
    if boundary_strength == 0 {
        return unchanged;
    }

    let qp_average = (i16::from(qp_p) + i16::from(qp_q) + 1) >> 1;
    let index_a = (qp_average + i16::from(alpha_offset_div2) * 2).clamp(0, 51) as usize;
    let index_b = (qp_average + i16::from(beta_offset_div2) * 2).clamp(0, 51) as usize;
    let alpha = i16::from(ALPHA[index_a]);
    let beta = i16::from(BETA[index_b]);

    let [p0, p1, p2, p3] = samples.p.map(i16::from);
    let [q0, q1, q2, q3] = samples.q.map(i16::from);
    if (p0 - q0).abs() >= alpha || (p1 - p0).abs() >= beta || (q1 - q0).abs() >= beta {
        return unchanged;
    }

    if boundary_strength < 4 {
        let tc0 = i16::from(TC0[usize::from(boundary_strength - 1)][index_a]);
        let ap = (p2 - p0).abs();
        let aq = (q2 - q0).abs();
        let tc = if chroma_style {
            tc0 + 1
        } else {
            tc0 + i16::from(ap < beta) + i16::from(aq < beta)
        };
        let delta = ((((q0 - p0) << 2) + (p1 - q1) + 4) >> 3).clamp(-tc, tc);

        let filtered_p1 = if !chroma_style && ap < beta {
            p1 + ((p2 + ((p0 + q0 + 1) >> 1) - (p1 << 1)) >> 1).clamp(-tc0, tc0)
        } else {
            p1
        };
        let filtered_q1 = if !chroma_style && aq < beta {
            q1 + ((q2 + ((p0 + q0 + 1) >> 1) - (q1 << 1)) >> 1).clamp(-tc0, tc0)
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
    let strong_threshold = (alpha >> 2) + 2;
    let p = if !chroma_style && ap < beta && (p0 - q0).abs() < strong_threshold {
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
    let q = if !chroma_style && aq < beta && (p0 - q0).abs() < strong_threshold {
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
    for (address, &current) in macroblocks.iter().enumerate() {
        let macroblock_x = address % width_in_macroblocks;
        let macroblock_y = address / width_in_macroblocks;
        if current.filter.idc == 1 {
            continue;
        }

        let left = (macroblock_x > 0).then(|| macroblocks[address - 1]);
        let top = (macroblock_y > 0).then(|| macroblocks[address - width_in_macroblocks]);
        let filter_left = left.is_some_and(|neighbor| {
            current.filter.idc != 2 || neighbor.slice_id == current.slice_id
        });
        let filter_top = top.is_some_and(|neighbor| {
            current.filter.idc != 2 || neighbor.slice_id == current.slice_id
        });

        let luma_x = macroblock_x * 16;
        let luma_y = macroblock_y * 16;
        if filter_left {
            let previous = left.expect("filter_left requires a neighbor");
            for block_row in 0..4 {
                filter_vertical_edge(
                    luma,
                    width,
                    luma_x,
                    luma_y + block_row * 4,
                    4,
                    edge_parameters(previous, block_row * 4 + 3, current, block_row * 4, true, 0),
                )?;
            }
        }
        for block_column in 1..4 {
            if block_column == 2 || !current.transform_8x8 {
                for block_row in 0..4 {
                    let q = block_row * 4 + block_column;
                    filter_vertical_edge(
                        luma,
                        width,
                        luma_x + block_column * 4,
                        luma_y + block_row * 4,
                        4,
                        edge_parameters(current, q - 1, current, q, false, 0),
                    )?;
                }
            }
        }
        if filter_top {
            let previous = top.expect("filter_top requires a neighbor");
            for block_column in 0..4 {
                filter_horizontal_edge(
                    luma,
                    width,
                    luma_x + block_column * 4,
                    luma_y,
                    4,
                    edge_parameters(previous, 12 + block_column, current, block_column, true, 0),
                )?;
            }
        }
        for block_row in 1..4 {
            if block_row == 2 || !current.transform_8x8 {
                for block_column in 0..4 {
                    let q = block_row * 4 + block_column;
                    filter_horizontal_edge(
                        luma,
                        width,
                        luma_x + block_column * 4,
                        luma_y + block_row * 4,
                        4,
                        edge_parameters(current, q - 4, current, q, false, 0),
                    )?;
                }
            }
        }

        let chroma_x = macroblock_x * 8;
        let chroma_y = macroblock_y * 8;
        for (plane, component) in [(&mut *cb, 1), (&mut *cr, 2)] {
            if filter_left {
                let previous = left.expect("filter_left requires a neighbor");
                for block_row in 0..4 {
                    filter_vertical_edge(
                        plane,
                        chroma_stride,
                        chroma_x,
                        chroma_y + block_row * 2,
                        2,
                        edge_parameters(
                            previous,
                            block_row * 4 + 3,
                            current,
                            block_row * 4,
                            true,
                            component,
                        ),
                    )?;
                }
            }
            for block_row in 0..4 {
                filter_vertical_edge(
                    plane,
                    chroma_stride,
                    chroma_x + 4,
                    chroma_y + block_row * 2,
                    2,
                    edge_parameters(
                        current,
                        block_row * 4 + 1,
                        current,
                        block_row * 4 + 2,
                        false,
                        component,
                    ),
                )?;
            }
            if filter_top {
                let previous = top.expect("filter_top requires a neighbor");
                for block_column in 0..4 {
                    filter_horizontal_edge(
                        plane,
                        chroma_stride,
                        chroma_x + block_column * 2,
                        chroma_y,
                        2,
                        edge_parameters(
                            previous,
                            12 + block_column,
                            current,
                            block_column,
                            true,
                            component,
                        ),
                    )?;
                }
            }
            for block_column in 0..4 {
                filter_horizontal_edge(
                    plane,
                    chroma_stride,
                    chroma_x + block_column * 2,
                    chroma_y + 4,
                    2,
                    edge_parameters(
                        current,
                        4 + block_column,
                        current,
                        8 + block_column,
                        false,
                        component,
                    ),
                )?;
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

fn edge_parameters(
    previous: MacroblockDeblockInfo,
    previous_cell: usize,
    current: MacroblockDeblockInfo,
    current_cell: usize,
    external: bool,
    component: u8,
) -> EdgeParameters {
    let qp = |macroblock: MacroblockDeblockInfo| match component {
        0 => macroblock.luma_qp,
        1 => macroblock.cb_qp,
        _ => macroblock.cr_qp,
    };
    EdgeParameters {
        boundary_strength: boundary_strength(
            previous,
            previous_cell,
            current,
            current_cell,
            external,
        ),
        qp_p: qp(previous),
        qp_q: qp(current),
        alpha_offset_div2: current.filter.alpha_c0_offset_div2,
        beta_offset_div2: current.filter.beta_offset_div2,
        chroma_style: component != 0,
    }
}

fn boundary_strength(
    previous: MacroblockDeblockInfo,
    previous_cell: usize,
    current: MacroblockDeblockInfo,
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
    let same_order_differs = list_motion_differs(previous.list0, current.list0)
        || list_motion_differs(previous.list1, current.list1);
    if !same_order_differs {
        return false;
    }

    let references_are_swapped = previous.list0.reference_id == current.list1.reference_id
        && previous.list1.reference_id == current.list0.reference_id;
    if !references_are_swapped {
        return true;
    }

    list_motion_differs(previous.list0, current.list1)
        || list_motion_differs(previous.list1, current.list0)
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
    parameters: EdgeParameters,
) -> Result<()> {
    validate_edge_parameters(parameters)?;
    if parameters.boundary_strength == 0 {
        return Ok(());
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
    Ok(())
}

fn filter_horizontal_edge(
    plane: &mut [u8],
    stride: usize,
    x: usize,
    y: usize,
    length: usize,
    parameters: EdgeParameters,
) -> Result<()> {
    validate_edge_parameters(parameters)?;
    if parameters.boundary_strength == 0 {
        return Ok(());
    }
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
    Ok(())
}

fn apply_parameters(
    samples: DeblockEdgeSamples,
    parameters: EdgeParameters,
) -> FilteredDeblockEdge {
    filter_deblock_edge_validated(
        samples,
        parameters.boundary_strength,
        parameters.qp_p,
        parameters.qp_q,
        parameters.alpha_offset_div2,
        parameters.beta_offset_div2,
        parameters.chroma_style,
    )
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

    fn inter_macroblock(reference_id: usize, vector: MotionVector) -> MacroblockDeblockInfo {
        MacroblockDeblockInfo {
            is_intra: false,
            motion: [DeblockMotion {
                list0: DeblockListMotion {
                    reference_id,
                    vector,
                },
                list1: DeblockListMotion::default(),
            }; 16],
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
        assert_eq!(boundary_strength(same, 3, same, 0, true), 0);

        let different_reference = inter_macroblock(8, same.motion[0].list0.vector);
        assert_eq!(boundary_strength(same, 3, different_reference, 0, true), 1);

        let different_vector = inter_macroblock(7, MotionVector { x: 7, y: -2 });
        assert_eq!(boundary_strength(same, 3, different_vector, 0, true), 1);

        let mut residual = same;
        residual.luma_nonzero[3] = true;
        assert_eq!(boundary_strength(residual, 3, same, 0, true), 2);
        assert_eq!(boundary_strength(macroblock(1, 0), 3, same, 0, true), 4);
        assert_eq!(boundary_strength(macroblock(1, 0), 3, same, 0, false), 3);
    }

    #[test]
    fn derives_bidirectional_boundary_strength_with_swapped_lists() {
        let list = |reference_id, x| DeblockListMotion {
            reference_id,
            vector: MotionVector { x, y: 0 },
        };
        let bidirectional = |list0, list1| MacroblockDeblockInfo {
            is_intra: false,
            motion: [DeblockMotion { list0, list1 }; 16],
            ..macroblock(1, 0)
        };

        let previous = bidirectional(list(7, 2), list(8, 9));
        let swapped = bidirectional(list(8, 9), list(7, 2));
        assert_eq!(boundary_strength(previous, 3, swapped, 0, true), 0);

        let crossed_difference = bidirectional(list(8, 13), list(7, 2));
        assert_eq!(
            boundary_strength(previous, 3, crossed_difference, 0, true),
            1
        );

        let mismatched_pair = bidirectional(list(7, 2), list(9, 9));
        assert_eq!(boundary_strength(previous, 3, mismatched_pair, 0, true), 1);
    }
}
