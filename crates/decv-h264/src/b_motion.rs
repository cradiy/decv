//! Spatial motion-vector prediction for explicit progressive B partitions.

use smallvec::SmallVec;

use crate::{
    BInterMacroblockHeader, BPartitionMode, BPartitionMotion, BPredictionMode, BSubMacroblockType,
    H264Error, MotionVector, MotionVectorDifference, ReferenceMotionField, ResolvedBListMotion,
    ResolvedBMacroblock, ResolvedBPartition, Result,
};

#[derive(Debug, Clone, Copy)]
pub struct DirectReference<'a> {
    pub id: crate::ReferenceId,
    pub picture_order_count: i32,
    pub long_term: bool,
    pub motion: &'a ReferenceMotionField,
}

#[derive(Debug, Clone, Copy)]
pub struct SpatialDirectContext<'a> {
    pub colocated_motion: &'a ReferenceMotionField,
    pub colocated_long_term: bool,
    pub direct_8x8_inference: bool,
    pub num_ref_idx_l0_active: u8,
    pub num_ref_idx_l1_active: u8,
}

#[derive(Debug, Clone, Copy)]
pub struct TemporalDirectContext<'a> {
    pub current_picture_order_count: i32,
    pub colocated: DirectReference<'a>,
    pub references_l0: &'a [Option<DirectReference<'a>>],
    pub direct_8x8_inference: bool,
    pub num_ref_idx_l1_active: u8,
}

#[derive(Debug, Clone, Copy)]
pub enum DirectMotionContext<'a> {
    Spatial(SpatialDirectContext<'a>),
    Temporal(TemporalDirectContext<'a>),
}

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

/// Per-picture spatial motion state for frame-coded B partitions.
///
/// List0, List1, and Bi partitions use the same A/B/C/D spatial prediction
/// rules independently for each list. Direct partitions additionally consume
/// co-located motion and, for temporal Direct, POC-distance scaling metadata.
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
        let mut resolved = SmallVec::with_capacity(plans.len());
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

    pub fn resolve_spatial_direct_macroblock(
        &mut self,
        macroblock_address: usize,
        slice_id: u32,
        context: SpatialDirectContext<'_>,
    ) -> Result<ResolvedBMacroblock> {
        self.ensure_macroblock_available_for_write(macroblock_address)?;
        let geometry = PartitionGeometry {
            x: 0,
            y: 0,
            width: 16,
            height: 16,
            macroblock_partition_index: 0,
        };
        let empty = [None; 16];
        let neighbour_cells = self.neighbour_cells(macroblock_address, slice_id, &empty, geometry);
        let neighbours_l0 = neighbour_motions(neighbour_cells, false);
        let neighbours_l1 = neighbour_motions(neighbour_cells, true);
        let mut reference_l0 = spatial_direct_reference_index_from(neighbours_l0);
        let mut reference_l1 = spatial_direct_reference_index_from(neighbours_l1);
        if reference_l0.is_none() && reference_l1.is_none() {
            reference_l0 = Some(0);
            reference_l1 = Some(0);
        }
        validate_direct_reference(reference_l0, context.num_ref_idx_l0_active, "List 0")?;
        validate_direct_reference(reference_l1, context.num_ref_idx_l1_active, "List 1")?;

        let predicted_l0 = reference_l0.map(|reference_index| {
            (
                reference_index,
                predict_motion_vector_from(
                    neighbours_l0,
                    geometry,
                    DirectionalMode::None,
                    reference_index,
                ),
            )
        });
        let predicted_l1 = reference_l1.map(|reference_index| {
            (
                reference_index,
                predict_motion_vector_from(
                    neighbours_l1,
                    geometry,
                    DirectionalMode::None,
                    reference_index,
                ),
            )
        });

        let partition_size = if context.direct_8x8_inference { 8 } else { 4 };
        let macroblock_x = macroblock_address % self.width_in_macroblocks;
        let macroblock_y = macroblock_address / self.width_in_macroblocks;
        let col_zero_can_change = |prediction: Option<(u8, MotionVector)>| {
            prediction.is_some_and(|(reference_index, vector)| {
                reference_index == 0 && vector != MotionVector::default()
            })
        };
        if !col_zero_can_change(predicted_l0) && !col_zero_can_change(predicted_l1) {
            let last_colocated_x = macroblock_x * 4 + 3;
            let last_colocated_y = macroblock_y * 4 + 3;
            if last_colocated_x >= context.colocated_motion.width_in_4x4_blocks()
                || last_colocated_y >= context.colocated_motion.height_in_4x4_blocks()
            {
                return Err(H264Error::InvalidSyntax(
                    "spatial Direct co-located block lies outside the reference motion field",
                ));
            }
            let list0 = predicted_l0.map(|(reference_index, motion_vector)| ResolvedBListMotion {
                reference_index,
                motion_vector,
            });
            let list1 = predicted_l1.map(|(reference_index, motion_vector)| ResolvedBListMotion {
                reference_index,
                motion_vector,
            });
            let partition = ResolvedBPartition {
                x: 0,
                y: 0,
                width: 16,
                height: 16,
                list0,
                list1,
            };
            let mut partitions = SmallVec::new();
            partitions.push(partition);
            self.commit_local_cells(
                macroblock_address,
                [Some(MotionCell {
                    slice_id,
                    list0,
                    list1,
                }); 16],
            );
            return Ok(ResolvedBMacroblock {
                direct: true,
                partitions,
            });
        }
        let mut local = [None; 16];
        let mut partitions = SmallVec::with_capacity((16 / partition_size) * (16 / partition_size));
        for y in (0..16).step_by(partition_size) {
            for x in (0..16).step_by(partition_size) {
                let colocated = context
                    .colocated_motion
                    .cell(macroblock_x * 4 + x / 4, macroblock_y * 4 + y / 4)
                    .ok_or(H264Error::InvalidSyntax(
                        "spatial Direct co-located block lies outside the reference motion field",
                    ))?;
                let col_zero = colocated_zero_flag(colocated, context.colocated_long_term);
                let list0 = predicted_l0.map(|(reference_index, vector)| ResolvedBListMotion {
                    reference_index,
                    motion_vector: if col_zero && reference_index == 0 {
                        MotionVector::default()
                    } else {
                        vector
                    },
                });
                let list1 = predicted_l1.map(|(reference_index, vector)| ResolvedBListMotion {
                    reference_index,
                    motion_vector: if col_zero && reference_index == 0 {
                        MotionVector::default()
                    } else {
                        vector
                    },
                });
                let partition = ResolvedBPartition {
                    x: x as u8,
                    y: y as u8,
                    width: partition_size as u8,
                    height: partition_size as u8,
                    list0,
                    list1,
                };
                fill_direct_partition_cells(&mut local, slice_id, partition);
                partitions.push(partition);
            }
        }
        coalesce_uniform_direct_grid(&mut partitions);
        self.commit_local_cells(macroblock_address, local);
        Ok(ResolvedBMacroblock {
            direct: true,
            partitions,
        })
    }

    pub fn resolve_temporal_direct_macroblock(
        &mut self,
        macroblock_address: usize,
        slice_id: u32,
        context: TemporalDirectContext<'_>,
    ) -> Result<ResolvedBMacroblock> {
        self.ensure_macroblock_available_for_write(macroblock_address)?;
        if context.num_ref_idx_l1_active == 0 {
            return Err(H264Error::InvalidSyntax(
                "temporal Direct requires an active List 1 reference",
            ));
        }
        let partition_size = if context.direct_8x8_inference { 8 } else { 4 };
        let macroblock_x = macroblock_address % self.width_in_macroblocks;
        let macroblock_y = macroblock_address / self.width_in_macroblocks;
        let mut local = [None; 16];
        let mut partitions = SmallVec::with_capacity((16 / partition_size) * (16 / partition_size));
        for y in (0..16).step_by(partition_size) {
            for x in (0..16).step_by(partition_size) {
                let colocated = context
                    .colocated
                    .motion
                    .cell(macroblock_x * 4 + x / 4, macroblock_y * 4 + y / 4)
                    .ok_or(H264Error::InvalidSyntax(
                        "temporal Direct co-located block lies outside the reference motion field",
                    ))?;
                let (list0, list1) = temporal_direct_motion(colocated, context)?;
                let partition = ResolvedBPartition {
                    x: x as u8,
                    y: y as u8,
                    width: partition_size as u8,
                    height: partition_size as u8,
                    list0: Some(list0),
                    list1: Some(list1),
                };
                fill_partition_cells(&mut local, slice_id, partition)?;
                partitions.push(partition);
            }
        }
        coalesce_uniform_direct_grid(&mut partitions);
        self.commit_local_cells(macroblock_address, local);
        Ok(ResolvedBMacroblock {
            direct: true,
            partitions,
        })
    }

    pub fn resolve_mixed_direct_8x8_macroblock(
        &mut self,
        macroblock_address: usize,
        slice_id: u32,
        header: &BInterMacroblockHeader,
        direct_context: DirectMotionContext<'_>,
    ) -> Result<ResolvedBMacroblock> {
        self.ensure_macroblock_available_for_write(macroblock_address)?;
        let BPartitionMode::EightByEight { sub_macroblocks } = &header.partition_mode else {
            return Err(H264Error::InvalidSyntax(
                "mixed Direct resolution requires a B_8x8 macroblock",
            ));
        };
        if header.partitions.len() != 4 {
            return Err(H264Error::InvalidSyntax(
                "B_8x8 partition count is inconsistent",
            ));
        }
        let mut local = [None; 16];
        let mut resolved = SmallVec::new();
        for (index, (&sub_type, syntax)) in
            sub_macroblocks.iter().zip(&header.partitions).enumerate()
        {
            let base_x = (index % 2 * 8) as u8;
            let base_y = (index / 2 * 8) as u8;
            if sub_type == BSubMacroblockType::Direct8x8 {
                for partition in self.resolve_direct_region(
                    macroblock_address,
                    slice_id,
                    base_x,
                    base_y,
                    direct_context,
                )? {
                    fill_partition_cells(&mut local, slice_id, partition)?;
                    resolved.push(partition);
                }
                continue;
            }

            if syntax.prediction != sub_type.prediction() {
                return Err(H264Error::InvalidSyntax(
                    "B sub-macroblock prediction mode is inconsistent",
                ));
            }
            validate_motion_syntax(syntax, sub_type.partition_count())?;
            let (width, height) = sub_type.partition_size();
            for sub_index in 0..sub_type.partition_count() {
                let (offset_x, offset_y) = sub_partition_offset(sub_type, sub_index);
                let plan = make_plan(
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
                )?;
                let list0 = self.resolve_list(
                    macroblock_address,
                    slice_id,
                    &local,
                    plan,
                    plan.list0,
                    false,
                )?;
                let list1 = self.resolve_list(
                    macroblock_address,
                    slice_id,
                    &local,
                    plan,
                    plan.list1,
                    true,
                )?;
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
        }
        self.commit_local_cells(macroblock_address, local);
        Ok(ResolvedBMacroblock {
            direct: true,
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

    fn spatial_direct_reference_index(
        &self,
        macroblock_address: usize,
        slice_id: u32,
        local: &[Option<MotionCell>; 16],
        geometry: PartitionGeometry,
        list1: bool,
    ) -> Option<u8> {
        spatial_direct_reference_index_from(self.neighbours(
            macroblock_address,
            slice_id,
            local,
            geometry,
            list1,
        ))
    }

    fn resolve_direct_region(
        &self,
        macroblock_address: usize,
        slice_id: u32,
        base_x: u8,
        base_y: u8,
        context: DirectMotionContext<'_>,
    ) -> Result<SmallVec<[ResolvedBPartition; 4]>> {
        let partition_size = match context {
            DirectMotionContext::Spatial(context) if context.direct_8x8_inference => 8,
            DirectMotionContext::Temporal(context) if context.direct_8x8_inference => 8,
            DirectMotionContext::Spatial(_) | DirectMotionContext::Temporal(_) => 4,
        };
        let macroblock_x = macroblock_address % self.width_in_macroblocks;
        let macroblock_y = macroblock_address / self.width_in_macroblocks;
        let mut partitions = SmallVec::with_capacity((8 / partition_size) * (8 / partition_size));

        let spatial_prediction = if let DirectMotionContext::Spatial(context) = context {
            let geometry = PartitionGeometry {
                x: 0,
                y: 0,
                width: 16,
                height: 16,
                macroblock_partition_index: 0,
            };
            let empty = [None; 16];
            let mut reference_l0 = self.spatial_direct_reference_index(
                macroblock_address,
                slice_id,
                &empty,
                geometry,
                false,
            );
            let mut reference_l1 = self.spatial_direct_reference_index(
                macroblock_address,
                slice_id,
                &empty,
                geometry,
                true,
            );
            if reference_l0.is_none() && reference_l1.is_none() {
                reference_l0 = Some(0);
                reference_l1 = Some(0);
            }
            validate_direct_reference(reference_l0, context.num_ref_idx_l0_active, "List 0")?;
            validate_direct_reference(reference_l1, context.num_ref_idx_l1_active, "List 1")?;
            Some((
                reference_l0.map(|reference_index| {
                    (
                        reference_index,
                        self.predict_motion_vector(
                            macroblock_address,
                            slice_id,
                            &empty,
                            geometry,
                            DirectionalMode::None,
                            reference_index,
                            false,
                        ),
                    )
                }),
                reference_l1.map(|reference_index| {
                    (
                        reference_index,
                        self.predict_motion_vector(
                            macroblock_address,
                            slice_id,
                            &empty,
                            geometry,
                            DirectionalMode::None,
                            reference_index,
                            true,
                        ),
                    )
                }),
            ))
        } else {
            None
        };

        for y in (usize::from(base_y)..usize::from(base_y) + 8).step_by(partition_size) {
            for x in (usize::from(base_x)..usize::from(base_x) + 8).step_by(partition_size) {
                let (list0, list1) = match context {
                    DirectMotionContext::Spatial(context) => {
                        let colocated = context
                            .colocated_motion
                            .cell(macroblock_x * 4 + x / 4, macroblock_y * 4 + y / 4)
                            .ok_or(H264Error::InvalidSyntax(
                                "spatial Direct co-located block lies outside the reference motion field",
                            ))?;
                        let col_zero = colocated_zero_flag(colocated, context.colocated_long_term);
                        let (predicted_l0, predicted_l1) =
                            spatial_prediction.expect("spatial prediction is prepared above");
                        (
                            predicted_l0.map(|(reference_index, vector)| ResolvedBListMotion {
                                reference_index,
                                motion_vector: if col_zero && reference_index == 0 {
                                    MotionVector::default()
                                } else {
                                    vector
                                },
                            }),
                            predicted_l1.map(|(reference_index, vector)| ResolvedBListMotion {
                                reference_index,
                                motion_vector: if col_zero && reference_index == 0 {
                                    MotionVector::default()
                                } else {
                                    vector
                                },
                            }),
                        )
                    }
                    DirectMotionContext::Temporal(context) => {
                        let colocated = context
                            .colocated
                            .motion
                            .cell(macroblock_x * 4 + x / 4, macroblock_y * 4 + y / 4)
                            .ok_or(H264Error::InvalidSyntax(
                                "temporal Direct co-located block lies outside the reference motion field",
                            ))?;
                        let (list0, list1) = temporal_direct_motion(colocated, context)?;
                        (Some(list0), Some(list1))
                    }
                };
                partitions.push(ResolvedBPartition {
                    x: x as u8,
                    y: y as u8,
                    width: partition_size as u8,
                    height: partition_size as u8,
                    list0,
                    list1,
                });
            }
        }
        Ok(partitions)
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
        predict_motion_vector_from(
            self.neighbours(macroblock_address, slice_id, local, geometry, list1),
            geometry,
            mode,
            reference_index,
        )
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

    fn neighbour_cells(
        &self,
        macroblock_address: usize,
        slice_id: u32,
        local: &[Option<MotionCell>; 16],
        geometry: PartitionGeometry,
    ) -> [Option<MotionCell>; 4] {
        let macroblock_x = macroblock_address % self.width_in_macroblocks;
        let macroblock_y = macroblock_address / self.width_in_macroblocks;
        let x = (macroblock_x * 16 + usize::from(geometry.x)) as isize;
        let y = (macroblock_y * 16 + usize::from(geometry.y)) as isize;
        [
            self.neighbour_cell_at(x - 1, y, macroblock_address, slice_id, local),
            self.neighbour_cell_at(x, y - 1, macroblock_address, slice_id, local),
            self.neighbour_cell_at(
                x + isize::from(geometry.width),
                y - 1,
                macroblock_address,
                slice_id,
                local,
            ),
            self.neighbour_cell_at(x - 1, y - 1, macroblock_address, slice_id, local),
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
        let Some(cell) = self.neighbour_cell_at(x, y, current_macroblock_address, slice_id, local)
        else {
            return NeighbourMotion::UNAVAILABLE;
        };
        let motion = if list1 { cell.list1 } else { cell.list0 };
        NeighbourMotion {
            available: true,
            reference_index: motion.map_or(-1, |motion| motion.reference_index as i8),
            vector: motion.map_or_else(MotionVector::default, |motion| motion.motion_vector),
        }
    }

    fn neighbour_cell_at(
        &self,
        x: isize,
        y: isize,
        current_macroblock_address: usize,
        slice_id: u32,
        local: &[Option<MotionCell>; 16],
    ) -> Option<MotionCell> {
        if x < 0
            || y < 0
            || x >= (self.width_in_macroblocks * 16) as isize
            || y >= (self.height_in_macroblocks * 16) as isize
        {
            return None;
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
        cell.filter(|cell| cell.slice_id == slice_id)
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

fn neighbour_motions(cells: [Option<MotionCell>; 4], list1: bool) -> [NeighbourMotion; 4] {
    [
        neighbour_motion(cells[0], list1),
        neighbour_motion(cells[1], list1),
        neighbour_motion(cells[2], list1),
        neighbour_motion(cells[3], list1),
    ]
}

#[inline(always)]
fn neighbour_motion(cell: Option<MotionCell>, list1: bool) -> NeighbourMotion {
    let Some(cell) = cell else {
        return NeighbourMotion::UNAVAILABLE;
    };
    let motion = if list1 { cell.list1 } else { cell.list0 };
    NeighbourMotion {
        available: true,
        reference_index: motion.map_or(-1, |motion| motion.reference_index as i8),
        vector: motion.map_or_else(MotionVector::default, |motion| motion.motion_vector),
    }
}

fn spatial_direct_reference_index_from([a, b, mut c, d]: [NeighbourMotion; 4]) -> Option<u8> {
    if !c.available {
        c = d;
    }
    [a, b, c]
        .into_iter()
        .filter_map(|neighbour| {
            (neighbour.available && neighbour.reference_index >= 0)
                .then_some(neighbour.reference_index as u8)
        })
        .min()
}

fn predict_motion_vector_from(
    [a, b, mut c, d]: [NeighbourMotion; 4],
    geometry: PartitionGeometry,
    mode: DirectionalMode,
    reference_index: u8,
) -> MotionVector {
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

fn temporal_direct_motion(
    colocated: crate::MotionFieldCell,
    context: TemporalDirectContext<'_>,
) -> Result<(ResolvedBListMotion, ResolvedBListMotion)> {
    if colocated.intra {
        return Ok((
            ResolvedBListMotion {
                reference_index: 0,
                motion_vector: MotionVector::default(),
            },
            ResolvedBListMotion {
                reference_index: 0,
                motion_vector: MotionVector::default(),
            },
        ));
    }
    let Some(colocated_motion) = colocated.list0.or(colocated.list1) else {
        return Err(H264Error::InvalidSyntax(
            "temporal Direct co-located inter block has no motion",
        ));
    };
    let reference_id = colocated_motion
        .reference_id
        .ok_or(H264Error::InvalidSyntax(
            "temporal Direct co-located motion has no stable reference identity",
        ))?;
    let (reference_index, reference) = context
        .references_l0
        .iter()
        .enumerate()
        .find_map(|(index, reference)| {
            reference
                .filter(|reference| reference.id == reference_id)
                .map(|reference| (index, reference))
        })
        .ok_or(H264Error::InvalidSyntax(
            "temporal Direct co-located reference is absent from current List 0",
        ))?;
    let reference_index = u8::try_from(reference_index).map_err(|_| H264Error::IntegerOverflow)?;
    let scale = temporal_direct_scale(
        context.current_picture_order_count,
        context.colocated.picture_order_count,
        reference,
    );
    let motion_l0 = MotionVector {
        x: scale_temporal_component(colocated_motion.vector.x, scale)?,
        y: scale_temporal_component(colocated_motion.vector.y, scale)?,
    };
    let motion_l1 = MotionVector {
        x: i16::try_from(i32::from(motion_l0.x) - i32::from(colocated_motion.vector.x)).map_err(
            |_| H264Error::InvalidSyntax("temporal Direct List-1 horizontal MV overflow"),
        )?,
        y: i16::try_from(i32::from(motion_l0.y) - i32::from(colocated_motion.vector.y))
            .map_err(|_| H264Error::InvalidSyntax("temporal Direct List-1 vertical MV overflow"))?,
    };
    Ok((
        ResolvedBListMotion {
            reference_index,
            motion_vector: motion_l0,
        },
        ResolvedBListMotion {
            reference_index: 0,
            motion_vector: motion_l1,
        },
    ))
}

fn temporal_direct_scale(
    current_picture_order_count: i32,
    colocated_picture_order_count: i32,
    reference_l0: DirectReference<'_>,
) -> i32 {
    let td = (i64::from(colocated_picture_order_count)
        - i64::from(reference_l0.picture_order_count))
    .clamp(-128, 127);
    if td == 0 || reference_l0.long_term {
        return 256;
    }
    let tb = (i64::from(current_picture_order_count) - i64::from(reference_l0.picture_order_count))
        .clamp(-128, 127);
    let tx = (16_384 + td.abs() / 2) / td;
    ((tb * tx + 32) >> 6).clamp(-1024, 1023) as i32
}

fn scale_temporal_component(component: i16, scale: i32) -> Result<i16> {
    i16::try_from((scale * i32::from(component) + 128) >> 8)
        .map_err(|_| H264Error::InvalidSyntax("scaled temporal Direct motion vector exceeds i16"))
}

fn validate_direct_reference(
    reference_index: Option<u8>,
    active_count: u8,
    list_name: &'static str,
) -> Result<()> {
    if reference_index.is_some_and(|index| index >= active_count) {
        return Err(H264Error::InvalidSyntax(match list_name {
            "List 0" => "spatial Direct List-0 index exceeds the active list",
            _ => "spatial Direct List-1 index exceeds the active list",
        }));
    }
    Ok(())
}

fn colocated_zero_flag(cell: crate::MotionFieldCell, colocated_long_term: bool) -> bool {
    if cell.intra || colocated_long_term {
        return false;
    }
    let Some(motion) = cell.list0.or(cell.list1) else {
        return false;
    };
    motion.reference_index == 0
        && motion.vector.x.unsigned_abs() <= 1
        && motion.vector.y.unsigned_abs() <= 1
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

/// Records a partition produced by the Direct grid derivation.
///
/// Unlike syntax-provided explicit partitions, Direct partitions are generated
/// internally on a regular 4x4 or 8x8 grid, so bounds and overlap have already
/// been established by construction.
#[inline]
fn fill_direct_partition_cells(
    cells: &mut [Option<MotionCell>; 16],
    slice_id: u32,
    partition: ResolvedBPartition,
) {
    let cell = Some(MotionCell {
        slice_id,
        list0: partition.list0,
        list1: partition.list1,
    });
    let start_x = usize::from(partition.x / 4);
    let end_x = start_x + usize::from(partition.width / 4);
    let start_y = usize::from(partition.y / 4);
    let end_y = start_y + usize::from(partition.height / 4);
    for y in start_y..end_y {
        cells[y * 4 + start_x..y * 4 + end_x].fill(cell);
    }
}

fn coalesce_uniform_direct_grid(partitions: &mut SmallVec<[ResolvedBPartition; 4]>) {
    let Some(first) = partitions.first().copied() else {
        return;
    };
    if partitions.len() > 1
        && partitions
            .iter()
            .all(|partition| partition.list0 == first.list0 && partition.list1 == first.list1)
    {
        partitions.clear();
        partitions.push(ResolvedBPartition {
            x: 0,
            y: 0,
            width: 16,
            height: 16,
            list0: first.list0,
            list1: first.list1,
        });
    }
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
    use crate::motion_field::MotionFieldBuilder;
    use crate::{BPartitionMotion, CodedBlockPattern, ResolvedPMacroblock, ResolvedPPartition};
    use decv_core::Size;

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

    fn spatial_direct_context(motion: &ReferenceMotionField) -> SpatialDirectContext<'_> {
        SpatialDirectContext {
            colocated_motion: motion,
            colocated_long_term: false,
            direct_8x8_inference: true,
            num_ref_idx_l0_active: 2,
            num_ref_idx_l1_active: 2,
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
    fn resolves_mixed_direct_and_explicit_eight_by_eight_partitions() {
        let colocated = ReferenceMotionField::all_intra(Size::new(16, 16)).unwrap();
        let mut state = BMotionState::new(1, 1).unwrap();
        let resolved = state
            .resolve_mixed_direct_8x8_macroblock(
                0,
                1,
                &header(
                    BPartitionMode::EightByEight {
                        sub_macroblocks: [
                            BSubMacroblockType::Direct8x8,
                            BSubMacroblockType::List0_8x8,
                            BSubMacroblockType::List1_8x8,
                            BSubMacroblockType::Bi8x8,
                        ],
                    },
                    vec![
                        partition(BPredictionMode::Direct, None, None, Vec::new(), Vec::new()),
                        partition(
                            BPredictionMode::List0,
                            Some(0),
                            None,
                            vec![difference(4, 0)],
                            Vec::new(),
                        ),
                        partition(
                            BPredictionMode::List1,
                            None,
                            Some(0),
                            Vec::new(),
                            vec![difference(0, 4)],
                        ),
                        partition(
                            BPredictionMode::Bi,
                            Some(0),
                            Some(0),
                            vec![difference(0, 0)],
                            vec![difference(0, 0)],
                        ),
                    ],
                ),
                DirectMotionContext::Spatial(SpatialDirectContext {
                    colocated_motion: &colocated,
                    colocated_long_term: false,
                    direct_8x8_inference: true,
                    num_ref_idx_l0_active: 1,
                    num_ref_idx_l1_active: 1,
                }),
            )
            .unwrap();

        assert!(resolved.direct);
        assert_eq!(resolved.partitions.len(), 4);
        assert_eq!(
            (resolved.partitions[0].list0, resolved.partitions[0].list1),
            (
                Some(ResolvedBListMotion {
                    reference_index: 0,
                    motion_vector: MotionVector::default(),
                }),
                Some(ResolvedBListMotion {
                    reference_index: 0,
                    motion_vector: MotionVector::default(),
                }),
            )
        );
        assert_eq!(
            resolved.partitions[1].list0.unwrap().motion_vector,
            MotionVector { x: 4, y: 0 }
        );
        assert!(resolved.partitions[1].list1.is_none());
        assert!(resolved.partitions[2].list0.is_none());
        assert_eq!(
            resolved.partitions[2].list1.unwrap().motion_vector,
            MotionVector { x: 0, y: 4 }
        );
        assert!(resolved.partitions[3].list0.is_some());
        assert!(resolved.partitions[3].list1.is_some());
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
    fn zero_vector_spatial_direct_still_validates_the_colocated_grid() {
        let mut state = BMotionState::new(2, 1).unwrap();
        let too_small = ReferenceMotionField::all_intra(Size::new(16, 16)).unwrap();
        assert!(matches!(
            state.resolve_spatial_direct_macroblock(1, 1, spatial_direct_context(&too_small),),
            Err(H264Error::InvalidSyntax(
                "spatial Direct co-located block lies outside the reference motion field"
            ))
        ));

        let complete = ReferenceMotionField::all_intra(Size::new(32, 16)).unwrap();
        assert!(
            state
                .resolve_spatial_direct_macroblock(1, 1, spatial_direct_context(&complete),)
                .is_ok()
        );
    }

    #[test]
    fn resolves_spatial_direct_from_neighbour_reference_minima() {
        let mut state = BMotionState::new(2, 1).unwrap();
        state
            .resolve_inter_macroblock(
                0,
                1,
                &header(
                    BPartitionMode::SixteenBySixteen,
                    vec![partition(
                        BPredictionMode::Bi,
                        Some(1),
                        Some(0),
                        vec![difference(4, 2)],
                        vec![difference(-2, 6)],
                    )],
                ),
            )
            .unwrap();
        let resolved = state
            .resolve_spatial_direct_macroblock(
                1,
                1,
                spatial_direct_context(
                    &ReferenceMotionField::all_intra(Size::new(32, 16)).unwrap(),
                ),
            )
            .unwrap();
        assert!(resolved.direct);
        assert_eq!(resolved.partitions.len(), 1);
        assert!(!resolved.partitions.spilled());
        assert_eq!(
            (
                resolved.partitions[0].x,
                resolved.partitions[0].y,
                resolved.partitions[0].width,
                resolved.partitions[0].height,
            ),
            (0, 0, 16, 16)
        );
        assert!(resolved.partitions.iter().all(|partition| {
            partition.list0
                == Some(ResolvedBListMotion {
                    reference_index: 1,
                    motion_vector: MotionVector { x: 4, y: 2 },
                })
                && partition.list1
                    == Some(ResolvedBListMotion {
                        reference_index: 0,
                        motion_vector: MotionVector { x: -2, y: 6 },
                    })
        }));
    }

    #[test]
    fn spatial_direct_applies_the_colocated_zero_flag_per_eight_by_eight() {
        let mut colocated = MotionFieldBuilder::new(Size::new(32, 16)).unwrap();
        let zero = ResolvedPMacroblock {
            skipped: true,
            partitions: vec![ResolvedPPartition {
                x: 0,
                y: 0,
                width: 16,
                height: 16,
                reference_index: 0,
                motion_vector: MotionVector::default(),
            }],
        };
        colocated.record_p(0, &zero, None).unwrap();
        colocated.record_p(1, &zero, None).unwrap();
        let colocated = colocated.finish().unwrap();

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
                        Some(0),
                        vec![difference(4, 2)],
                        vec![difference(-2, 6)],
                    )],
                ),
            )
            .unwrap();
        let resolved = state
            .resolve_spatial_direct_macroblock(
                1,
                1,
                SpatialDirectContext {
                    colocated_motion: &colocated,
                    colocated_long_term: false,
                    direct_8x8_inference: true,
                    num_ref_idx_l0_active: 1,
                    num_ref_idx_l1_active: 1,
                },
            )
            .unwrap();
        assert!(resolved.partitions.iter().all(|partition| {
            partition.list0.unwrap().motion_vector == MotionVector::default()
                && partition.list1.unwrap().motion_vector == MotionVector::default()
        }));
    }

    #[test]
    fn temporal_direct_maps_stable_identity_and_scales_colocated_motion() {
        let reference_id = crate::ReferenceId(7);
        let mut colocated = MotionFieldBuilder::new(Size::new(16, 16)).unwrap();
        colocated
            .record_p(
                0,
                &ResolvedPMacroblock {
                    skipped: false,
                    partitions: vec![ResolvedPPartition {
                        x: 0,
                        y: 0,
                        width: 16,
                        height: 16,
                        reference_index: 0,
                        motion_vector: MotionVector { x: 8, y: 4 },
                    }],
                },
                Some(&[Some(reference_id)]),
            )
            .unwrap();
        let colocated_motion = colocated.finish().unwrap();
        let reference_motion = ReferenceMotionField::all_intra(Size::new(16, 16)).unwrap();
        let references_l0 = [Some(DirectReference {
            id: reference_id,
            picture_order_count: 0,
            long_term: false,
            motion: &reference_motion,
        })];
        let mut state = BMotionState::new(1, 1).unwrap();
        let resolved = state
            .resolve_temporal_direct_macroblock(
                0,
                1,
                TemporalDirectContext {
                    current_picture_order_count: 2,
                    colocated: DirectReference {
                        id: crate::ReferenceId(8),
                        picture_order_count: 8,
                        long_term: false,
                        motion: &colocated_motion,
                    },
                    references_l0: &references_l0,
                    direct_8x8_inference: true,
                    num_ref_idx_l1_active: 1,
                },
            )
            .unwrap();
        assert!(resolved.direct);
        assert!(resolved.partitions.iter().all(|partition| {
            partition.list0
                == Some(ResolvedBListMotion {
                    reference_index: 0,
                    motion_vector: MotionVector { x: 2, y: 1 },
                })
                && partition.list1
                    == Some(ResolvedBListMotion {
                        reference_index: 0,
                        motion_vector: MotionVector { x: -6, y: -3 },
                    })
        }));
    }

    #[test]
    fn temporal_direct_uses_zero_motion_for_colocated_intra_blocks() {
        let colocated_motion = ReferenceMotionField::all_intra(Size::new(16, 16)).unwrap();
        let reference_motion = ReferenceMotionField::all_intra(Size::new(16, 16)).unwrap();
        let references_l0 = [Some(DirectReference {
            id: crate::ReferenceId(7),
            picture_order_count: 0,
            long_term: false,
            motion: &reference_motion,
        })];
        let mut state = BMotionState::new(1, 1).unwrap();
        let resolved = state
            .resolve_temporal_direct_macroblock(
                0,
                1,
                TemporalDirectContext {
                    current_picture_order_count: 2,
                    colocated: DirectReference {
                        id: crate::ReferenceId(8),
                        picture_order_count: 4,
                        long_term: false,
                        motion: &colocated_motion,
                    },
                    references_l0: &references_l0,
                    direct_8x8_inference: false,
                    num_ref_idx_l1_active: 1,
                },
            )
            .unwrap();
        assert_eq!(resolved.partitions.len(), 1);
        assert_eq!(
            (
                resolved.partitions[0].x,
                resolved.partitions[0].y,
                resolved.partitions[0].width,
                resolved.partitions[0].height,
            ),
            (0, 0, 16, 16)
        );
        assert!(resolved.partitions.iter().all(|partition| {
            partition.list0.unwrap().motion_vector == MotionVector::default()
                && partition.list1.unwrap().motion_vector == MotionVector::default()
        }));
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
