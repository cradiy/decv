//! Spatial motion-vector prediction for explicit progressive B partitions.

use crate::{
    BInterMacroblockHeader, BPartitionMode, BPartitionMotion, BPredictionMode, BSubMacroblockType,
    H264Error, MotionVector, MotionVectorDifference, ResolvedBListMotion, ResolvedBMacroblock,
    ResolvedBPartition, Result,
};

#[derive(Debug, Clone, Copy)]
struct MotionCell {
    slice_id: u32,
    list0: Option<ResolvedBListMotion>,
    list1: Option<ResolvedBListMotion>,
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

#[derive(Debug, Clone, Copy)]
enum DirectionalMode {
    None,
    SixteenByEight,
    EightBySixteen,
}

#[derive(Debug, Clone, Copy)]
struct ListPlan {
    reference_index: u8,
    difference: MotionVectorDifference,
}

#[derive(Debug, Clone, Copy)]
struct PartitionPlan {
    geometry: PartitionGeometry,
    directional_mode: DirectionalMode,
    list0: Option<ListPlan>,
    list1: Option<ListPlan>,
}

/// Per-picture spatial motion state for explicit frame-coded B partitions.
///
/// Direct prediction is deliberately rejected here because it additionally
/// requires co-located motion and POC-distance scaling. List0, List1, and Bi
/// partitions use the same A/B/C/D spatial prediction rules independently for
/// each list.
#[derive(Debug, Clone)]
pub struct BMotionState {
    width_in_macroblocks: usize,
    height_in_macroblocks: usize,
    cells: Vec<Option<MotionCell>>,
}

impl BMotionState {
    pub fn new(width_in_macroblocks: usize, height_in_macroblocks: usize) -> Result<Self> {
        if width_in_macroblocks == 0 || height_in_macroblocks == 0 {
            return Err(H264Error::InvalidSyntax(
                "B motion-state picture dimensions must be non-zero",
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

    pub fn record_intra_macroblock(
        &mut self,
        macroblock_address: usize,
        slice_id: u32,
    ) -> Result<()> {
        self.ensure_macroblock_available_for_write(macroblock_address)?;
        self.commit_local_cells(
            macroblock_address,
            [Some(MotionCell {
                slice_id,
                list0: None,
                list1: None,
            }); 16],
        );
        Ok(())
    }

    /// Resolves and records an explicit non-Direct B macroblock atomically.
    pub fn resolve_inter_macroblock(
        &mut self,
        macroblock_address: usize,
        slice_id: u32,
        header: &BInterMacroblockHeader,
    ) -> Result<ResolvedBMacroblock> {
        self.ensure_macroblock_available_for_write(macroblock_address)?;
        let plans = partition_plans(header)?;
        let mut local = [None; 16];
        let mut resolved = Vec::with_capacity(plans.len());
        for plan in plans {
            let list0 = self.resolve_list(
                macroblock_address,
                slice_id,
                &local,
                plan,
                plan.list0,
                false,
            )?;
            let list1 =
                self.resolve_list(macroblock_address, slice_id, &local, plan, plan.list1, true)?;
            let partition = ResolvedBPartition {
                x: plan.geometry.x,
                y: plan.geometry.y,
                width: plan.geometry.width,
                height: plan.geometry.height,
                list0,
                list1,
            };
            fill_partition_cells(&mut local, slice_id, partition)?;
            resolved.push(partition);
        }
        self.commit_local_cells(macroblock_address, local);
        Ok(ResolvedBMacroblock {
            direct: false,
            partitions: resolved,
        })
    }

    pub(crate) fn clear_macroblock(&mut self, macroblock_address: usize) -> Result<()> {
        if macroblock_address >= self.width_in_macroblocks * self.height_in_macroblocks {
            return Err(H264Error::InvalidSyntax(
                "B motion-state macroblock address exceeds the picture",
            ));
        }
        let start = macroblock_address * 16;
        self.cells[start..start + 16].fill(None);
        Ok(())
    }

    fn resolve_list(
        &self,
        macroblock_address: usize,
        slice_id: u32,
        local: &[Option<MotionCell>; 16],
        plan: PartitionPlan,
        list: Option<ListPlan>,
        list1: bool,
    ) -> Result<Option<ResolvedBListMotion>> {
        let Some(list) = list else {
            return Ok(None);
        };
        let predictor = self.predict_motion_vector(
            macroblock_address,
            slice_id,
            local,
            plan.geometry,
            plan.directional_mode,
            list.reference_index,
            list1,
        );
        Ok(Some(ResolvedBListMotion {
            reference_index: list.reference_index,
            motion_vector: add_motion_vector_difference(predictor, list.difference)?,
        }))
    }

    #[allow(clippy::too_many_arguments)]
    fn predict_motion_vector(
        &self,
        macroblock_address: usize,
        slice_id: u32,
        local: &[Option<MotionCell>; 16],
        geometry: PartitionGeometry,
        mode: DirectionalMode,
        reference_index: u8,
        list1: bool,
    ) -> MotionVector {
        let [a, b, mut c, d] =
            self.neighbours(macroblock_address, slice_id, local, geometry, list1);
        if !c.available {
            c = d;
        }
        match mode {
            DirectionalMode::SixteenByEight
                if geometry.macroblock_partition_index == 0
                    && b.reference_index == reference_index as i8 =>
            {
                b.vector
            }
            DirectionalMode::SixteenByEight
                if geometry.macroblock_partition_index == 1
                    && a.reference_index == reference_index as i8 =>
            {
                a.vector
            }
            DirectionalMode::EightBySixteen
                if geometry.macroblock_partition_index == 0
                    && a.reference_index == reference_index as i8 =>
            {
                a.vector
            }
            DirectionalMode::EightBySixteen
                if geometry.macroblock_partition_index == 1
                    && c.reference_index == reference_index as i8 =>
            {
                c.vector
            }
            _ => median_prediction(a, b, c, reference_index),
        }
    }

    fn neighbours(
        &self,
        macroblock_address: usize,
        slice_id: u32,
        local: &[Option<MotionCell>; 16],
        geometry: PartitionGeometry,
        list1: bool,
    ) -> [NeighbourMotion; 4] {
        let macroblock_x = macroblock_address % self.width_in_macroblocks;
        let macroblock_y = macroblock_address / self.width_in_macroblocks;
        let x = (macroblock_x * 16 + usize::from(geometry.x)) as isize;
        let y = (macroblock_y * 16 + usize::from(geometry.y)) as isize;
        [
            self.neighbour_at(x - 1, y, macroblock_address, slice_id, local, list1),
            self.neighbour_at(x, y - 1, macroblock_address, slice_id, local, list1),
            self.neighbour_at(
                x + isize::from(geometry.width),
                y - 1,
                macroblock_address,
                slice_id,
                local,
                list1,
            ),
            self.neighbour_at(x - 1, y - 1, macroblock_address, slice_id, local, list1),
        ]
    }

    #[allow(clippy::too_many_arguments)]
    fn neighbour_at(
        &self,
        x: isize,
        y: isize,
        current_macroblock_address: usize,
        slice_id: u32,
        local: &[Option<MotionCell>; 16],
        list1: bool,
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
        let macroblock_address = (y / 16) * self.width_in_macroblocks + x / 16;
        let local_index = (y % 16) / 4 * 4 + (x % 16) / 4;
        let cell = if macroblock_address == current_macroblock_address {
            local[local_index]
        } else {
            self.cells[macroblock_address * 16 + local_index]
        };
        let Some(cell) = cell.filter(|cell| cell.slice_id == slice_id) else {
            return NeighbourMotion::UNAVAILABLE;
        };
        let motion = if list1 { cell.list1 } else { cell.list0 };
        NeighbourMotion {
            available: true,
            reference_index: motion.map_or(-1, |motion| motion.reference_index as i8),
            vector: motion.map_or_else(MotionVector::default, |motion| motion.motion_vector),
        }
    }

    fn ensure_macroblock_available_for_write(&self, macroblock_address: usize) -> Result<()> {
        if macroblock_address >= self.width_in_macroblocks * self.height_in_macroblocks {
            return Err(H264Error::InvalidSyntax(
                "B motion-state macroblock address exceeds the picture",
            ));
        }
        let start = macroblock_address * 16;
        if self.cells[start..start + 16].iter().any(Option::is_some) {
            return Err(H264Error::InvalidSyntax(
                "B motion-state macroblock was already recorded",
            ));
        }
        Ok(())
    }

    fn commit_local_cells(&mut self, macroblock_address: usize, local: [Option<MotionCell>; 16]) {
        let start = macroblock_address * 16;
        self.cells[start..start + 16].copy_from_slice(&local);
    }
}

fn partition_plans(header: &BInterMacroblockHeader) -> Result<Vec<PartitionPlan>> {
    match &header.partition_mode {
        BPartitionMode::Direct16x16 => Err(H264Error::UnsupportedFeature(
            "Direct B motion-vector derivation",
        )),
        BPartitionMode::SixteenBySixteen => {
            ordinary_partition_plans(header, &[(0, 0, 16, 16)], DirectionalMode::None)
        }
        BPartitionMode::SixteenByEight => ordinary_partition_plans(
            header,
            &[(0, 0, 16, 8), (0, 8, 16, 8)],
            DirectionalMode::SixteenByEight,
        ),
        BPartitionMode::EightBySixteen => ordinary_partition_plans(
            header,
            &[(0, 0, 8, 16), (8, 0, 8, 16)],
            DirectionalMode::EightBySixteen,
        ),
        BPartitionMode::EightByEight { sub_macroblocks } => {
            if header.partitions.len() != 4 {
                return Err(H264Error::InvalidSyntax(
                    "B_8x8 partition count is inconsistent",
                ));
            }
            let mut plans = Vec::new();
            for (index, sub_type) in sub_macroblocks.iter().copied().enumerate() {
                if sub_type == BSubMacroblockType::Direct8x8 {
                    return Err(H264Error::UnsupportedFeature(
                        "Direct B sub-macroblock motion-vector derivation",
                    ));
                }
                let syntax = &header.partitions[index];
                if syntax.prediction != sub_type.prediction() {
                    return Err(H264Error::InvalidSyntax(
                        "B sub-macroblock prediction mode is inconsistent",
                    ));
                }
                validate_motion_syntax(syntax, sub_type.partition_count())?;
                let base_x = (index % 2 * 8) as u8;
                let base_y = (index / 2 * 8) as u8;
                let (width, height) = sub_type.partition_size();
                for sub_index in 0..sub_type.partition_count() {
                    let (offset_x, offset_y) = sub_partition_offset(sub_type, sub_index);
                    plans.push(make_plan(
                        syntax,
                        sub_index,
                        PartitionGeometry {
                            x: base_x + offset_x,
                            y: base_y + offset_y,
                            width,
                            height,
                            macroblock_partition_index: index,
                        },
                        DirectionalMode::None,
                    )?);
                }
            }
            Ok(plans)
        }
    }
}

fn ordinary_partition_plans(
    header: &BInterMacroblockHeader,
    geometry: &[(u8, u8, u8, u8)],
    directional_mode: DirectionalMode,
) -> Result<Vec<PartitionPlan>> {
    if header.partitions.len() != geometry.len() {
        return Err(H264Error::InvalidSyntax(
            "B macroblock partition count is inconsistent",
        ));
    }
    header
        .partitions
        .iter()
        .zip(geometry)
        .enumerate()
        .map(|(index, (syntax, &(x, y, width, height)))| {
            if syntax.prediction == BPredictionMode::Direct {
                return Err(H264Error::UnsupportedFeature(
                    "Direct B motion-vector derivation",
                ));
            }
            validate_motion_syntax(syntax, 1)?;
            make_plan(
                syntax,
                0,
                PartitionGeometry {
                    x,
                    y,
                    width,
                    height,
                    macroblock_partition_index: index,
                },
                directional_mode,
            )
        })
        .collect()
}

fn validate_motion_syntax(syntax: &BPartitionMotion, difference_count: usize) -> Result<()> {
    validate_list_syntax(
        syntax.prediction.uses_list0(),
        syntax.reference_index_l0,
        &syntax.differences_l0,
        difference_count,
    )?;
    validate_list_syntax(
        syntax.prediction.uses_list1(),
        syntax.reference_index_l1,
        &syntax.differences_l1,
        difference_count,
    )
}

fn validate_list_syntax(
    used: bool,
    reference_index: Option<u8>,
    differences: &[MotionVectorDifference],
    expected_count: usize,
) -> Result<()> {
    if used {
        if reference_index.is_none_or(|index| index > 31) || differences.len() != expected_count {
            return Err(H264Error::InvalidSyntax(
                "B list motion syntax is inconsistent",
            ));
        }
    } else if reference_index.is_some() || !differences.is_empty() {
        return Err(H264Error::InvalidSyntax(
            "unused B reference list contains motion syntax",
        ));
    }
    Ok(())
}

fn make_plan(
    syntax: &BPartitionMotion,
    difference_index: usize,
    geometry: PartitionGeometry,
    directional_mode: DirectionalMode,
) -> Result<PartitionPlan> {
    let list0 = syntax.reference_index_l0.map(|reference_index| ListPlan {
        reference_index,
        difference: syntax.differences_l0[difference_index],
    });
    let list1 = syntax.reference_index_l1.map(|reference_index| ListPlan {
        reference_index,
        difference: syntax.differences_l1[difference_index],
    });
    if list0.is_none() && list1.is_none() {
        return Err(H264Error::InvalidSyntax(
            "explicit B partition uses neither reference list",
        ));
    }
    Ok(PartitionPlan {
        geometry,
        directional_mode,
        list0,
        list1,
    })
}

const fn sub_partition_offset(
    sub_type: BSubMacroblockType,
    sub_partition_index: usize,
) -> (u8, u8) {
    match sub_type.partition_size() {
        (8, 8) => (0, 0),
        (8, 4) => (0, (sub_partition_index * 4) as u8),
        (4, 8) => ((sub_partition_index * 4) as u8, 0),
        (4, 4) => (
            (sub_partition_index % 2 * 4) as u8,
            (sub_partition_index / 2 * 4) as u8,
        ),
        _ => (0, 0),
    }
}

fn fill_partition_cells(
    cells: &mut [Option<MotionCell>; 16],
    slice_id: u32,
    partition: ResolvedBPartition,
) -> Result<()> {
    for y in (partition.y..partition.y + partition.height).step_by(4) {
        for x in (partition.x..partition.x + partition.width).step_by(4) {
            let index = usize::from(y / 4) * 4 + usize::from(x / 4);
            if index >= cells.len() || cells[index].is_some() {
                return Err(H264Error::InvalidSyntax(
                    "B macroblock partitions overlap or exceed bounds",
                ));
            }
            cells[index] = Some(MotionCell {
                slice_id,
                list0: partition.list0,
                list1: partition.list1,
            });
        }
    }
    Ok(())
}

fn add_motion_vector_difference(
    predictor: MotionVector,
    difference: MotionVectorDifference,
) -> Result<MotionVector> {
    Ok(MotionVector {
        x: i16::try_from(i32::from(predictor.x) + i32::from(difference.x)).map_err(|_| {
            H264Error::InvalidSyntax("horizontal B motion vector exceeds the supported range")
        })?,
        y: i16::try_from(i32::from(predictor.y) + i32::from(difference.y)).map_err(|_| {
            H264Error::InvalidSyntax("vertical B motion vector exceeds the supported range")
        })?,
    })
}

fn median_prediction(
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
    if matches.iter().filter(|&&matched| matched).count() == 1 {
        return if matches[0] {
            a.vector
        } else if matches[1] {
            b.vector
        } else {
            c.vector
        };
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
    use crate::{BPartitionMotion, CodedBlockPattern};

    fn difference(x: i16, y: i16) -> MotionVectorDifference {
        MotionVectorDifference { x, y }
    }

    fn partition(
        prediction: BPredictionMode,
        reference_index_l0: Option<u8>,
        reference_index_l1: Option<u8>,
        differences_l0: Vec<MotionVectorDifference>,
        differences_l1: Vec<MotionVectorDifference>,
    ) -> BPartitionMotion {
        BPartitionMotion {
            prediction,
            reference_index_l0,
            reference_index_l1,
            differences_l0,
            differences_l1,
        }
    }

    fn header(
        partition_mode: BPartitionMode,
        partitions: Vec<BPartitionMotion>,
    ) -> BInterMacroblockHeader {
        BInterMacroblockHeader {
            partition_mode,
            partitions,
            coded_block_pattern: CodedBlockPattern { luma: 0, chroma: 0 },
            transform_size_8x8: false,
            qp_delta: 0,
        }
    }

    #[test]
    fn resolves_both_lists_independently_at_the_picture_boundary() {
        let mut state = BMotionState::new(1, 1).unwrap();
        let resolved = state
            .resolve_inter_macroblock(
                0,
                1,
                &header(
                    BPartitionMode::SixteenBySixteen,
                    vec![partition(
                        BPredictionMode::Bi,
                        Some(0),
                        Some(1),
                        vec![difference(2, 3)],
                        vec![difference(-1, 4)],
                    )],
                ),
            )
            .unwrap();
        assert_eq!(
            resolved.partitions[0].list0,
            Some(ResolvedBListMotion {
                reference_index: 0,
                motion_vector: MotionVector { x: 2, y: 3 },
            })
        );
        assert_eq!(
            resolved.partitions[0].list1,
            Some(ResolvedBListMotion {
                reference_index: 1,
                motion_vector: MotionVector { x: -1, y: 4 },
            })
        );
    }

    #[test]
    fn predicts_each_list_from_the_completed_left_macroblock() {
        let mut state = BMotionState::new(2, 1).unwrap();
        state
            .resolve_inter_macroblock(
                0,
                1,
                &header(
                    BPartitionMode::SixteenBySixteen,
                    vec![partition(
                        BPredictionMode::Bi,
                        Some(0),
                        Some(1),
                        vec![difference(4, 0)],
                        vec![difference(0, 6)],
                    )],
                ),
            )
            .unwrap();
        let resolved = state
            .resolve_inter_macroblock(
                1,
                1,
                &header(
                    BPartitionMode::SixteenBySixteen,
                    vec![partition(
                        BPredictionMode::Bi,
                        Some(0),
                        Some(1),
                        vec![difference(0, 0)],
                        vec![difference(0, 0)],
                    )],
                ),
            )
            .unwrap();
        assert_eq!(
            resolved.partitions[0].list0.unwrap().motion_vector,
            MotionVector { x: 4, y: 0 }
        );
        assert_eq!(
            resolved.partitions[0].list1.unwrap().motion_vector,
            MotionVector { x: 0, y: 6 }
        );
    }

    #[test]
    fn expands_all_explicit_b_subpartition_shapes() {
        let sub_macroblocks = [
            BSubMacroblockType::List0_8x8,
            BSubMacroblockType::List0_8x4,
            BSubMacroblockType::List0_4x8,
            BSubMacroblockType::List0_4x4,
        ];
        let partitions = sub_macroblocks
            .iter()
            .map(|sub_type| {
                partition(
                    BPredictionMode::List0,
                    Some(0),
                    None,
                    vec![difference(0, 0); sub_type.partition_count()],
                    Vec::new(),
                )
            })
            .collect();
        let mut state = BMotionState::new(1, 1).unwrap();
        let resolved = state
            .resolve_inter_macroblock(
                0,
                1,
                &header(BPartitionMode::EightByEight { sub_macroblocks }, partitions),
            )
            .unwrap();
        assert_eq!(resolved.partitions.len(), 9);
        assert_eq!(
            resolved
                .partitions
                .iter()
                .map(|partition| (partition.width, partition.height))
                .collect::<Vec<_>>(),
            [
                (8, 8),
                (8, 4),
                (8, 4),
                (4, 8),
                (4, 8),
                (4, 4),
                (4, 4),
                (4, 4),
                (4, 4)
            ]
        );
    }

    #[test]
    fn direct_failure_does_not_commit_the_macroblock() {
        let mut state = BMotionState::new(1, 1).unwrap();
        assert!(matches!(
            state.resolve_inter_macroblock(
                0,
                1,
                &header(
                    BPartitionMode::Direct16x16,
                    vec![partition(
                        BPredictionMode::Direct,
                        None,
                        None,
                        Vec::new(),
                        Vec::new(),
                    )],
                ),
            ),
            Err(H264Error::UnsupportedFeature(_))
        ));
        assert!(
            state
                .resolve_inter_macroblock(
                    0,
                    1,
                    &header(
                        BPartitionMode::SixteenBySixteen,
                        vec![partition(
                            BPredictionMode::List0,
                            Some(0),
                            None,
                            vec![difference(0, 0)],
                            Vec::new(),
                        )],
                    ),
                )
                .is_ok()
        );
    }

    #[test]
    fn hides_motion_from_other_slices_and_records_intra_availability() {
        let mut state = BMotionState::new(2, 1).unwrap();
        state
            .resolve_inter_macroblock(
                0,
                1,
                &header(
                    BPartitionMode::SixteenBySixteen,
                    vec![partition(
                        BPredictionMode::List0,
                        Some(0),
                        None,
                        vec![difference(7, 9)],
                        Vec::new(),
                    )],
                ),
            )
            .unwrap();
        let resolved = state
            .resolve_inter_macroblock(
                1,
                2,
                &header(
                    BPartitionMode::SixteenBySixteen,
                    vec![partition(
                        BPredictionMode::List0,
                        Some(0),
                        None,
                        vec![difference(1, 2)],
                        Vec::new(),
                    )],
                ),
            )
            .unwrap();
        assert_eq!(
            resolved.partitions[0].list0.unwrap().motion_vector,
            MotionVector { x: 1, y: 2 }
        );

        let mut intra = BMotionState::new(2, 1).unwrap();
        intra.record_intra_macroblock(0, 2).unwrap();
        let resolved = intra
            .resolve_inter_macroblock(
                1,
                2,
                &header(
                    BPartitionMode::SixteenBySixteen,
                    vec![partition(
                        BPredictionMode::List0,
                        Some(0),
                        None,
                        vec![difference(3, 4)],
                        Vec::new(),
                    )],
                ),
            )
            .unwrap();
        assert_eq!(
            resolved.partitions[0].list0.unwrap().motion_vector,
            MotionVector { x: 3, y: 4 }
        );
    }
}
