//! Motion-vector prediction state for frame-coded P slices.
//!
//! H.264 predicts each inter partition from already decoded neighbouring
//! partitions. Storing the result at 4x4 granularity makes the A/B/C/D lookup
//! rules work uniformly for macroblock and sub-macroblock partitions.

use crate::{
    H264Error, MotionVectorDifference, PInterMacroblockHeader, PPartitionMode, PSubMacroblockType,
    Result,
};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct MotionVector {
    /// Horizontal displacement in quarter-luma-sample units.
    pub x: i16,
    /// Vertical displacement in quarter-luma-sample units.
    pub y: i16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResolvedPPartition {
    /// Horizontal luma-sample offset inside the macroblock.
    pub x: u8,
    /// Vertical luma-sample offset inside the macroblock.
    pub y: u8,
    pub width: u8,
    pub height: u8,
    pub reference_index: u8,
    pub motion_vector: MotionVector,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedPMacroblock {
    pub skipped: bool,
    /// Partitions are in macroblock/sub-macroblock decoding order.
    pub partitions: Vec<ResolvedPPartition>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResolvedBListMotion {
    pub reference_index: u8,
    pub motion_vector: MotionVector,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResolvedBPartition {
    pub x: u8,
    pub y: u8,
    pub width: u8,
    pub height: u8,
    pub list0: Option<ResolvedBListMotion>,
    pub list1: Option<ResolvedBListMotion>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedBMacroblock {
    pub direct: bool,
    pub partitions: Vec<ResolvedBPartition>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MotionKind {
    Intra,
    Inter {
        reference_index: u8,
        vector: MotionVector,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MotionCell {
    slice_id: u32,
    kind: MotionKind,
}

#[derive(Debug, Clone, Copy)]
struct NeighbourMotion {
    available: bool,
    reference_index: i8,
    vector: MotionVector,
}

impl NeighbourMotion {
    const UNAVAILABLE: Self = Self {
        available: false,
        reference_index: -1,
        vector: MotionVector { x: 0, y: 0 },
    };
}

#[derive(Debug, Clone, Copy)]
struct PartitionGeometry {
    x: u8,
    y: u8,
    width: u8,
    height: u8,
    macroblock_partition_index: usize,
}

/// Per-picture List-0 motion state for frame-coded P slices.
///
/// The state is deliberately independent from pixel reconstruction. It can be
/// populated in slice decoding order, including arbitrary macroblock
/// addresses, while neighbour availability is restricted to the same slice.
#[derive(Debug, Clone)]
pub struct PMotionState {
    width_in_macroblocks: usize,
    height_in_macroblocks: usize,
    cells: Vec<Option<MotionCell>>,
}

impl PMotionState {
    pub fn new(width_in_macroblocks: usize, height_in_macroblocks: usize) -> Result<Self> {
        if width_in_macroblocks == 0 || height_in_macroblocks == 0 {
            return Err(H264Error::InvalidSyntax(
                "motion-state picture dimensions must be non-zero",
            ));
        }
        let cell_count = width_in_macroblocks
            .checked_mul(height_in_macroblocks)
            .and_then(|value| value.checked_mul(16))
            .ok_or(H264Error::IntegerOverflow)?;
        Ok(Self {
            width_in_macroblocks,
            height_in_macroblocks,
            cells: vec![None; cell_count],
        })
    }

    #[inline]
    pub const fn width_in_macroblocks(&self) -> usize {
        self.width_in_macroblocks
    }

    #[inline]
    pub const fn height_in_macroblocks(&self) -> usize {
        self.height_in_macroblocks
    }

    /// Records an intra macroblock so it remains spatially available while
    /// contributing the inferred `(refIdxL0, mvL0) = (-1, (0, 0))`.
    pub fn record_intra_macroblock(
        &mut self,
        macroblock_address: usize,
        slice_id: u32,
    ) -> Result<()> {
        self.ensure_macroblock_available_for_write(macroblock_address)?;
        let cell = MotionCell {
            slice_id,
            kind: MotionKind::Intra,
        };
        self.commit_local_cells(macroblock_address, [Some(cell); 16]);
        Ok(())
    }

    /// Derives and records all List-0 motion vectors of one non-skipped P
    /// macroblock.
    ///
    /// The operation is transactional: malformed public input or vector
    /// overflow leaves the picture state unchanged.
    pub fn resolve_inter_macroblock(
        &mut self,
        macroblock_address: usize,
        slice_id: u32,
        header: &PInterMacroblockHeader,
    ) -> Result<ResolvedPMacroblock> {
        self.ensure_macroblock_available_for_write(macroblock_address)?;
        let plans = partition_plans(header)?;
        let mut local = [None; 16];
        let mut resolved = Vec::with_capacity(plans.len());

        for (geometry, reference_index, difference) in plans {
            let predictor = self.predict_motion_vector(
                macroblock_address,
                slice_id,
                &local,
                geometry,
                reference_index,
                &header.partition_mode,
            );
            let motion_vector = add_motion_vector_difference(predictor, difference)?;
            let partition = ResolvedPPartition {
                x: geometry.x,
                y: geometry.y,
                width: geometry.width,
                height: geometry.height,
                reference_index,
                motion_vector,
            };
            fill_partition_cells(
                &mut local,
                slice_id,
                partition,
                MotionKind::Inter {
                    reference_index,
                    vector: motion_vector,
                },
            )?;
            resolved.push(partition);
        }

        self.commit_local_cells(macroblock_address, local);
        Ok(ResolvedPMacroblock {
            skipped: false,
            partitions: resolved,
        })
    }

    /// Derives and records the inferred List-0 motion of one P_Skip
    /// macroblock.
    pub fn resolve_skip_macroblock(
        &mut self,
        macroblock_address: usize,
        slice_id: u32,
    ) -> Result<ResolvedPMacroblock> {
        self.ensure_macroblock_available_for_write(macroblock_address)?;
        let local = [None; 16];
        let geometry = PartitionGeometry {
            x: 0,
            y: 0,
            width: 16,
            height: 16,
            macroblock_partition_index: 0,
        };
        let [a, b, _, _] = self.neighbours(macroblock_address, slice_id, &local, geometry);
        let zero = MotionVector::default();
        let vector = if !a.available
            || !b.available
            || (a.reference_index == 0 && a.vector == zero)
            || (b.reference_index == 0 && b.vector == zero)
        {
            zero
        } else {
            self.predict_motion_vector(
                macroblock_address,
                slice_id,
                &local,
                geometry,
                0,
                &PPartitionMode::L0_16x16,
            )
        };
        let partition = ResolvedPPartition {
            x: 0,
            y: 0,
            width: 16,
            height: 16,
            reference_index: 0,
            motion_vector: vector,
        };
        let mut completed = [None; 16];
        fill_partition_cells(
            &mut completed,
            slice_id,
            partition,
            MotionKind::Inter {
                reference_index: 0,
                vector,
            },
        )?;
        self.commit_local_cells(macroblock_address, completed);
        Ok(ResolvedPMacroblock {
            skipped: true,
            partitions: vec![partition],
        })
    }

    pub(crate) fn clear_macroblock(&mut self, macroblock_address: usize) -> Result<()> {
        if macroblock_address >= self.width_in_macroblocks * self.height_in_macroblocks {
            return Err(H264Error::InvalidSyntax(
                "motion-state macroblock address exceeds the picture",
            ));
        }
        let start = macroblock_address * 16;
        self.cells[start..start + 16].fill(None);
        Ok(())
    }

    fn predict_motion_vector(
        &self,
        macroblock_address: usize,
        slice_id: u32,
        local: &[Option<MotionCell>; 16],
        geometry: PartitionGeometry,
        reference_index: u8,
        mode: &PPartitionMode,
    ) -> MotionVector {
        let [a, b, mut c, d] = self.neighbours(macroblock_address, slice_id, local, geometry);
        if !c.available {
            c = d;
        }

        match mode {
            PPartitionMode::L0_16x8
                if geometry.macroblock_partition_index == 0
                    && b.reference_index == reference_index as i8 =>
            {
                b.vector
            }
            PPartitionMode::L0_16x8
                if geometry.macroblock_partition_index == 1
                    && a.reference_index == reference_index as i8 =>
            {
                a.vector
            }
            PPartitionMode::L0_8x16
                if geometry.macroblock_partition_index == 0
                    && a.reference_index == reference_index as i8 =>
            {
                a.vector
            }
            PPartitionMode::L0_8x16
                if geometry.macroblock_partition_index == 1
                    && c.reference_index == reference_index as i8 =>
            {
                c.vector
            }
            _ => median_motion_vector_prediction(a, b, c, reference_index),
        }
    }

    fn neighbours(
        &self,
        macroblock_address: usize,
        slice_id: u32,
        local: &[Option<MotionCell>; 16],
        geometry: PartitionGeometry,
    ) -> [NeighbourMotion; 4] {
        let macroblock_x = macroblock_address % self.width_in_macroblocks;
        let macroblock_y = macroblock_address / self.width_in_macroblocks;
        let x = (macroblock_x * 16 + usize::from(geometry.x)) as isize;
        let y = (macroblock_y * 16 + usize::from(geometry.y)) as isize;
        [
            self.neighbour_at(x - 1, y, macroblock_address, slice_id, local),
            self.neighbour_at(x, y - 1, macroblock_address, slice_id, local),
            self.neighbour_at(
                x + isize::from(geometry.width),
                y - 1,
                macroblock_address,
                slice_id,
                local,
            ),
            self.neighbour_at(x - 1, y - 1, macroblock_address, slice_id, local),
        ]
    }

    fn neighbour_at(
        &self,
        x: isize,
        y: isize,
        current_macroblock_address: usize,
        slice_id: u32,
        local: &[Option<MotionCell>; 16],
    ) -> NeighbourMotion {
        if x < 0
            || y < 0
            || x >= (self.width_in_macroblocks * 16) as isize
            || y >= (self.height_in_macroblocks * 16) as isize
        {
            return NeighbourMotion::UNAVAILABLE;
        }

        let x = x as usize;
        let y = y as usize;
        let macroblock_x = x / 16;
        let macroblock_y = y / 16;
        let macroblock_address = macroblock_y * self.width_in_macroblocks + macroblock_x;
        let local_index = (y % 16) / 4 * 4 + (x % 16) / 4;
        let cell = if macroblock_address == current_macroblock_address {
            local[local_index]
        } else {
            self.cells[macroblock_address * 16 + local_index]
        };

        let Some(cell) = cell.filter(|cell| cell.slice_id == slice_id) else {
            return NeighbourMotion::UNAVAILABLE;
        };
        match cell.kind {
            MotionKind::Intra => NeighbourMotion {
                available: true,
                reference_index: -1,
                vector: MotionVector::default(),
            },
            MotionKind::Inter {
                reference_index,
                vector,
            } => NeighbourMotion {
                available: true,
                reference_index: reference_index as i8,
                vector,
            },
        }
    }

    fn ensure_macroblock_available_for_write(&self, macroblock_address: usize) -> Result<()> {
        if macroblock_address >= self.width_in_macroblocks * self.height_in_macroblocks {
            return Err(H264Error::InvalidSyntax(
                "motion-state macroblock address exceeds the picture",
            ));
        }
        let start = macroblock_address * 16;
        if self.cells[start..start + 16].iter().any(Option::is_some) {
            return Err(H264Error::InvalidSyntax(
                "motion-state macroblock was already recorded",
            ));
        }
        Ok(())
    }

    fn commit_local_cells(&mut self, macroblock_address: usize, local: [Option<MotionCell>; 16]) {
        let start = macroblock_address * 16;
        self.cells[start..start + 16].copy_from_slice(&local);
    }
}

fn partition_plans(
    header: &PInterMacroblockHeader,
) -> Result<Vec<(PartitionGeometry, u8, MotionVectorDifference)>> {
    let mut plans = Vec::new();
    match &header.partition_mode {
        PPartitionMode::L0_16x16 => {
            validate_partition_count(header, 1, true)?;
            push_partition_plan(&mut plans, header, 0, 0, 0, 16, 16, 0)?;
        }
        PPartitionMode::L0_16x8 => {
            validate_partition_count(header, 2, true)?;
            push_partition_plan(&mut plans, header, 0, 0, 0, 16, 8, 0)?;
            push_partition_plan(&mut plans, header, 1, 0, 8, 16, 8, 0)?;
        }
        PPartitionMode::L0_8x16 => {
            validate_partition_count(header, 2, true)?;
            push_partition_plan(&mut plans, header, 0, 0, 0, 8, 16, 0)?;
            push_partition_plan(&mut plans, header, 1, 8, 0, 8, 16, 0)?;
        }
        PPartitionMode::L0_8x8 {
            sub_macroblocks,
            reference_index_forced_zero,
        } => {
            validate_partition_count(header, 4, false)?;
            for (macroblock_partition_index, sub_type) in
                sub_macroblocks.iter().copied().enumerate()
            {
                let motion = &header.partitions[macroblock_partition_index];
                if *reference_index_forced_zero && motion.reference_index != 0 {
                    return Err(H264Error::InvalidSyntax(
                        "P_8x8ref0 contains a non-zero reference index",
                    ));
                }
                if motion.differences.len() != sub_type.partition_count() {
                    return Err(H264Error::InvalidSyntax(
                        "P sub-macroblock motion-vector count is inconsistent",
                    ));
                }
                let base_x = (macroblock_partition_index % 2 * 8) as u8;
                let base_y = (macroblock_partition_index / 2 * 8) as u8;
                let (width, height) = sub_type.partition_size();
                for sub_partition_index in 0..sub_type.partition_count() {
                    let (sub_x, sub_y) = sub_partition_offset(sub_type, sub_partition_index);
                    push_partition_plan(
                        &mut plans,
                        header,
                        macroblock_partition_index,
                        base_x + sub_x,
                        base_y + sub_y,
                        width,
                        height,
                        sub_partition_index,
                    )?;
                }
            }
        }
    }
    Ok(plans)
}

fn validate_partition_count(
    header: &PInterMacroblockHeader,
    expected: usize,
    one_difference_per_partition: bool,
) -> Result<()> {
    if header.partitions.len() != expected {
        return Err(H264Error::InvalidSyntax(
            "P macroblock partition count is inconsistent",
        ));
    }
    if one_difference_per_partition {
        for motion in &header.partitions {
            if motion.differences.len() != 1 {
                return Err(H264Error::InvalidSyntax(
                    "P macroblock motion-vector count is inconsistent",
                ));
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn push_partition_plan(
    plans: &mut Vec<(PartitionGeometry, u8, MotionVectorDifference)>,
    header: &PInterMacroblockHeader,
    macroblock_partition_index: usize,
    x: u8,
    y: u8,
    width: u8,
    height: u8,
    difference_index: usize,
) -> Result<()> {
    let motion = &header.partitions[macroblock_partition_index];
    if motion.reference_index > 31 {
        return Err(H264Error::InvalidSyntax(
            "P macroblock reference index exceeds 31",
        ));
    }
    let difference = *motion
        .differences
        .get(difference_index)
        .ok_or(H264Error::InvalidSyntax(
            "P macroblock motion-vector difference is missing",
        ))?;
    plans.push((
        PartitionGeometry {
            x,
            y,
            width,
            height,
            macroblock_partition_index,
        },
        motion.reference_index,
        difference,
    ));
    Ok(())
}

const fn sub_partition_offset(
    sub_type: PSubMacroblockType,
    sub_partition_index: usize,
) -> (u8, u8) {
    match sub_type {
        PSubMacroblockType::L0_8x8 => (0, 0),
        PSubMacroblockType::L0_8x4 => (0, (sub_partition_index * 4) as u8),
        PSubMacroblockType::L0_4x8 => ((sub_partition_index * 4) as u8, 0),
        PSubMacroblockType::L0_4x4 => (
            (sub_partition_index % 2 * 4) as u8,
            (sub_partition_index / 2 * 4) as u8,
        ),
    }
}

fn fill_partition_cells(
    cells: &mut [Option<MotionCell>; 16],
    slice_id: u32,
    partition: ResolvedPPartition,
    kind: MotionKind,
) -> Result<()> {
    for y in (partition.y..partition.y + partition.height).step_by(4) {
        for x in (partition.x..partition.x + partition.width).step_by(4) {
            let index = usize::from(y / 4) * 4 + usize::from(x / 4);
            if cells[index].is_some() {
                return Err(H264Error::InvalidSyntax("P macroblock partitions overlap"));
            }
            cells[index] = Some(MotionCell { slice_id, kind });
        }
    }
    Ok(())
}

fn add_motion_vector_difference(
    predictor: MotionVector,
    difference: MotionVectorDifference,
) -> Result<MotionVector> {
    let x = i32::from(predictor.x) + i32::from(difference.x);
    let y = i32::from(predictor.y) + i32::from(difference.y);
    Ok(MotionVector {
        x: i16::try_from(x).map_err(|_| {
            H264Error::InvalidSyntax("horizontal P motion vector exceeds the supported range")
        })?,
        y: i16::try_from(y).map_err(|_| {
            H264Error::InvalidSyntax("vertical P motion vector exceeds the supported range")
        })?,
    })
}

fn median_motion_vector_prediction(
    a: NeighbourMotion,
    mut b: NeighbourMotion,
    mut c: NeighbourMotion,
    reference_index: u8,
) -> MotionVector {
    if a.available && !b.available && !c.available {
        b = a;
        c = a;
    }
    let reference_index = reference_index as i8;
    let matches = [
        a.reference_index == reference_index,
        b.reference_index == reference_index,
        c.reference_index == reference_index,
    ];
    if matches.iter().filter(|&&matches| matches).count() == 1 {
        if matches[0] {
            return a.vector;
        }
        if matches[1] {
            return b.vector;
        }
        return c.vector;
    }
    MotionVector {
        x: median(a.vector.x, b.vector.x, c.vector.x),
        y: median(a.vector.y, b.vector.y, c.vector.y),
    }
}

#[inline]
fn median(a: i16, b: i16, c: i16) -> i16 {
    a.wrapping_add(b)
        .wrapping_add(c)
        .wrapping_sub(a.min(b).min(c))
        .wrapping_sub(a.max(b).max(c))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CodedBlockPattern, PPartitionMotion};

    fn difference(x: i16, y: i16) -> MotionVectorDifference {
        MotionVectorDifference { x, y }
    }

    fn motion(reference_index: u8, differences: &[(i16, i16)]) -> PPartitionMotion {
        PPartitionMotion {
            reference_index,
            differences: differences.iter().map(|&(x, y)| difference(x, y)).collect(),
        }
    }

    fn header(
        partition_mode: PPartitionMode,
        partitions: Vec<PPartitionMotion>,
    ) -> PInterMacroblockHeader {
        PInterMacroblockHeader {
            partition_mode,
            partitions,
            coded_block_pattern: CodedBlockPattern { luma: 0, chroma: 0 },
            transform_size_8x8: false,
            qp_delta: 0,
        }
    }

    fn neighbour(available: bool, reference_index: i8, x: i16, y: i16) -> NeighbourMotion {
        NeighbourMotion {
            available,
            reference_index,
            vector: MotionVector { x, y },
        }
    }

    #[test]
    fn median_prediction_uses_the_only_matching_reference() {
        let predicted = median_motion_vector_prediction(
            neighbour(true, 0, 10, 20),
            neighbour(true, 1, 30, 40),
            neighbour(true, 2, 50, 60),
            1,
        );
        assert_eq!(predicted, MotionVector { x: 30, y: 40 });
    }

    #[test]
    fn median_prediction_replicates_a_when_b_and_c_are_unavailable() {
        let predicted = median_motion_vector_prediction(
            neighbour(true, 2, 11, -7),
            NeighbourMotion::UNAVAILABLE,
            NeighbourMotion::UNAVAILABLE,
            0,
        );
        assert_eq!(predicted, MotionVector { x: 11, y: -7 });
    }

    #[test]
    fn median_prediction_uses_component_wise_medians() {
        let predicted = median_motion_vector_prediction(
            neighbour(true, 0, -5, 30),
            neighbour(true, 0, 20, -10),
            neighbour(true, 0, 3, 4),
            0,
        );
        assert_eq!(predicted, MotionVector { x: 3, y: 4 });
    }

    #[test]
    fn first_inter_macroblock_uses_zero_predictor() {
        let mut state = PMotionState::new(2, 1).unwrap();
        let resolved = state
            .resolve_inter_macroblock(
                0,
                7,
                &header(PPartitionMode::L0_16x16, vec![motion(0, &[(5, -3)])]),
            )
            .unwrap();
        assert_eq!(
            resolved.partitions[0].motion_vector,
            MotionVector { x: 5, y: -3 }
        );
    }

    #[test]
    fn left_macroblock_predicts_the_next_macroblock() {
        let mut state = PMotionState::new(2, 1).unwrap();
        state
            .resolve_inter_macroblock(
                0,
                1,
                &header(PPartitionMode::L0_16x16, vec![motion(0, &[(6, 2)])]),
            )
            .unwrap();
        let resolved = state
            .resolve_inter_macroblock(
                1,
                1,
                &header(PPartitionMode::L0_16x16, vec![motion(0, &[(1, -1)])]),
            )
            .unwrap();
        assert_eq!(
            resolved.partitions[0].motion_vector,
            MotionVector { x: 7, y: 1 }
        );
    }

    #[test]
    fn neighbours_from_other_slices_are_unavailable() {
        let mut state = PMotionState::new(2, 1).unwrap();
        state
            .resolve_inter_macroblock(
                0,
                1,
                &header(PPartitionMode::L0_16x16, vec![motion(0, &[(6, 2)])]),
            )
            .unwrap();
        let resolved = state
            .resolve_inter_macroblock(
                1,
                2,
                &header(PPartitionMode::L0_16x16, vec![motion(0, &[(1, -1)])]),
            )
            .unwrap();
        assert_eq!(
            resolved.partitions[0].motion_vector,
            MotionVector { x: 1, y: -1 }
        );
    }

    #[test]
    fn intra_neighbour_is_available_but_has_no_list_zero_motion() {
        let mut state = PMotionState::new(2, 2).unwrap();
        state.record_intra_macroblock(0, 3).unwrap();
        state
            .resolve_inter_macroblock(
                1,
                3,
                &header(PPartitionMode::L0_16x16, vec![motion(0, &[(12, 4)])]),
            )
            .unwrap();
        let resolved = state
            .resolve_inter_macroblock(
                2,
                3,
                &header(PPartitionMode::L0_16x16, vec![motion(0, &[(0, 0)])]),
            )
            .unwrap();
        assert_eq!(
            resolved.partitions[0].motion_vector,
            MotionVector { x: 12, y: 4 }
        );
    }

    #[test]
    fn sixteen_by_eight_uses_directional_predictors() {
        let mut state = PMotionState::new(2, 2).unwrap();
        state
            .resolve_inter_macroblock(
                0,
                1,
                &header(PPartitionMode::L0_16x16, vec![motion(0, &[(4, 5)])]),
            )
            .unwrap();
        let resolved = state
            .resolve_inter_macroblock(
                2,
                1,
                &header(
                    PPartitionMode::L0_16x8,
                    vec![motion(0, &[(1, 2)]), motion(0, &[(3, 4)])],
                ),
            )
            .unwrap();
        assert_eq!(
            resolved
                .partitions
                .iter()
                .map(|partition| partition.motion_vector)
                .collect::<Vec<_>>(),
            vec![MotionVector { x: 5, y: 7 }, MotionVector { x: 8, y: 11 }]
        );
    }

    #[test]
    fn subpartitions_use_earlier_motion_within_the_macroblock() {
        let mut state = PMotionState::new(1, 1).unwrap();
        let resolved = state
            .resolve_inter_macroblock(
                0,
                1,
                &header(
                    PPartitionMode::L0_8x8 {
                        sub_macroblocks: [
                            PSubMacroblockType::L0_4x4,
                            PSubMacroblockType::L0_8x8,
                            PSubMacroblockType::L0_8x8,
                            PSubMacroblockType::L0_8x8,
                        ],
                        reference_index_forced_zero: false,
                    },
                    vec![
                        motion(0, &[(2, 1), (3, 1), (4, 1), (5, 1)]),
                        motion(0, &[(0, 0)]),
                        motion(0, &[(0, 0)]),
                        motion(0, &[(0, 0)]),
                    ],
                ),
            )
            .unwrap();
        assert_eq!(
            resolved
                .partitions
                .iter()
                .take(4)
                .map(|partition| partition.motion_vector)
                .collect::<Vec<_>>(),
            vec![
                MotionVector { x: 2, y: 1 },
                MotionVector { x: 5, y: 2 },
                MotionVector { x: 6, y: 2 },
                MotionVector { x: 10, y: 3 },
            ]
        );
    }

    #[test]
    fn skip_is_zero_at_the_picture_boundary() {
        let mut state = PMotionState::new(1, 1).unwrap();
        let resolved = state.resolve_skip_macroblock(0, 9).unwrap();
        assert!(resolved.skipped);
        assert_eq!(
            resolved.partitions[0].motion_vector,
            MotionVector::default()
        );
    }

    #[test]
    fn skip_uses_prediction_when_a_and_b_are_nonzero() {
        let mut state = PMotionState::new(2, 2).unwrap();
        state
            .resolve_inter_macroblock(
                0,
                4,
                &header(PPartitionMode::L0_16x16, vec![motion(0, &[(8, 2)])]),
            )
            .unwrap();
        state
            .resolve_inter_macroblock(
                1,
                4,
                &header(PPartitionMode::L0_16x16, vec![motion(0, &[(2, 4)])]),
            )
            .unwrap();
        state
            .resolve_inter_macroblock(
                2,
                4,
                &header(PPartitionMode::L0_16x16, vec![motion(0, &[(4, 6)])]),
            )
            .unwrap();
        let resolved = state.resolve_skip_macroblock(3, 4).unwrap();
        assert_eq!(
            resolved.partitions[0].motion_vector,
            MotionVector { x: 10, y: 6 }
        );
    }

    #[test]
    fn vector_overflow_does_not_commit_partial_macroblock() {
        let mut state = PMotionState::new(2, 1).unwrap();
        state
            .resolve_inter_macroblock(
                0,
                1,
                &header(PPartitionMode::L0_16x16, vec![motion(0, &[(i16::MAX, 0)])]),
            )
            .unwrap();
        let overflowing = header(
            PPartitionMode::L0_16x8,
            vec![motion(0, &[(1, 0)]), motion(0, &[(0, 0)])],
        );
        assert!(matches!(
            state.resolve_inter_macroblock(1, 1, &overflowing),
            Err(H264Error::InvalidSyntax(_))
        ));
        assert!(state.cells[16..32].iter().all(Option::is_none));
    }

    #[test]
    fn malformed_public_header_does_not_modify_state() {
        let mut state = PMotionState::new(1, 1).unwrap();
        let malformed = header(PPartitionMode::L0_16x8, vec![motion(0, &[(0, 0)])]);
        assert!(matches!(
            state.resolve_inter_macroblock(0, 1, &malformed),
            Err(H264Error::InvalidSyntax(_))
        ));
        assert!(state.cells.iter().all(Option::is_none));
    }
}
