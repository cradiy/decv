//! Normative H.264 in-loop deblocking.
//!
//! This module starts with the sample-level filtering processes from clauses
//! 8.7.2.2 through 8.7.2.4. Picture traversal and boundary-strength derivation
//! are deliberately kept separate: callers provide the already-derived
//! boundary strength and the QPs on both sides of one edge.

use crate::{H264Error, Result};

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

    let unchanged = samples.unchanged();
    if boundary_strength == 0 {
        return Ok(unchanged);
    }

    let qp_average = (i16::from(qp_p) + i16::from(qp_q) + 1) >> 1;
    let index_a = (qp_average + i16::from(alpha_offset_div2) * 2).clamp(0, 51) as usize;
    let index_b = (qp_average + i16::from(beta_offset_div2) * 2).clamp(0, 51) as usize;
    let alpha = i16::from(ALPHA[index_a]);
    let beta = i16::from(BETA[index_b]);

    let [p0, p1, p2, p3] = samples.p.map(i16::from);
    let [q0, q1, q2, q3] = samples.q.map(i16::from);
    if (p0 - q0).abs() >= alpha || (p1 - p0).abs() >= beta || (q1 - q0).abs() >= beta {
        return Ok(unchanged);
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

        return Ok(FilteredDeblockEdge {
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
        });
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
    Ok(FilteredDeblockEdge { p, q })
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
    use super::{ALPHA, BETA, DeblockEdgeSamples, FilteredDeblockEdge, TC0, filter_deblock_edge};
    use crate::H264Error;

    const SMOOTH_EDGE: DeblockEdgeSamples = DeblockEdgeSamples {
        p: [100, 99, 98, 97],
        q: [110, 111, 112, 113],
    };

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
}
