//! Immutable per-4x4 motion metadata retained with reference pictures.

use std::mem::{ManuallyDrop, MaybeUninit, size_of};

use decv_core::Size;

use crate::{
    H264Error, MotionVector, ReferenceId, ResolvedBMacroblock, ResolvedPMacroblock, Result,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StoredListMotion {
    pub reference_index: u8,
    /// Stable DPB identity. `None` is allowed only for callers that decode
    /// pixels without supplying reference metadata.
    pub reference_id: Option<ReferenceId>,
    pub vector: MotionVector,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MotionFieldCell {
    pub intra: bool,
    pub list0: Option<StoredListMotion>,
    pub list1: Option<StoredListMotion>,
}

impl MotionFieldCell {
    const INTRA: Self = Self {
        intra: true,
        list0: None,
        list1: None,
    };

    #[inline]
    pub(crate) fn colocated_zero_flag(self, colocated_long_term: bool) -> bool {
        if self.intra || colocated_long_term {
            return false;
        }
        let Some(motion) = self.list0.or(self.list1) else {
            return false;
        };
        motion.reference_index == 0
            && motion.vector.x.unsigned_abs() <= 1
            && motion.vector.y.unsigned_abs() <= 1
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReferenceMotionField {
    width_in_4x4_blocks: usize,
    height_in_4x4_blocks: usize,
    cells: Vec<MotionFieldCell>,
}

impl ReferenceMotionField {
    pub fn all_intra(coded_size: Size) -> Result<Self> {
        let (width, height) = field_dimensions(coded_size)?;
        let count = width
            .checked_mul(height)
            .ok_or(H264Error::IntegerOverflow)?;
        Ok(Self {
            width_in_4x4_blocks: width,
            height_in_4x4_blocks: height,
            cells: vec![MotionFieldCell::INTRA; count],
        })
    }

    #[inline]
    pub const fn width_in_4x4_blocks(&self) -> usize {
        self.width_in_4x4_blocks
    }

    #[inline]
    pub const fn height_in_4x4_blocks(&self) -> usize {
        self.height_in_4x4_blocks
    }

    pub(crate) fn retained_allocation_bytes(&self) -> usize {
        self.cells
            .capacity()
            .saturating_mul(size_of::<MotionFieldCell>())
    }

    pub(crate) fn spatial_direct_col_zero_mask(
        &self,
        macroblock_x: usize,
        macroblock_y: usize,
        direct_8x8_inference: bool,
        colocated_long_term: bool,
    ) -> Option<u16> {
        let base_x = macroblock_x.checked_mul(4)?;
        let base_y = macroblock_y.checked_mul(4)?;
        if base_x.checked_add(4)? > self.width_in_4x4_blocks
            || base_y.checked_add(4)? > self.height_in_4x4_blocks
        {
            return None;
        }
        if colocated_long_term {
            return Some(0);
        }

        let stride = self.width_in_4x4_blocks;
        let base = base_y * stride + base_x;
        if direct_8x8_inference {
            let indices = [base, base + 2, base + 2 * stride, base + 2 * stride + 2];
            return Some(
                indices
                    .into_iter()
                    .enumerate()
                    .fold(0, |mask, (index, cell)| {
                        mask | (u16::from(self.cells[cell].colocated_zero_flag(false)) << index)
                    }),
            );
        }

        let mut mask = 0;
        for y in 0..4 {
            for x in 0..4 {
                let index = y * 4 + x;
                if self.cells[base + y * stride + x].colocated_zero_flag(false) {
                    mask |= 1 << index;
                }
            }
        }
        Some(mask)
    }

    pub fn cell(&self, x: usize, y: usize) -> Option<MotionFieldCell> {
        if x >= self.width_in_4x4_blocks || y >= self.height_in_4x4_blocks {
            return None;
        }
        Some(self.cells[y * self.width_in_4x4_blocks + x])
    }
}

#[derive(Debug, Clone)]
pub(crate) struct MotionFieldBuilder {
    width_in_macroblocks: usize,
    cells: Vec<MaybeUninit<MotionFieldCell>>,
    completed: Vec<u8>,
}

impl MotionFieldBuilder {
    pub(crate) fn new(coded_size: Size) -> Result<Self> {
        Self::new_with_retention(coded_size, true)
    }

    pub(crate) fn new_discarding(coded_size: Size) -> Result<Self> {
        Self::new_with_retention(coded_size, false)
    }

    fn new_with_retention(coded_size: Size, retain_cells: bool) -> Result<Self> {
        let (width, height) = field_dimensions(coded_size)?;
        let count = width
            .checked_mul(height)
            .ok_or(H264Error::IntegerOverflow)?;
        let mut cells = if retain_cells {
            Vec::with_capacity(count)
        } else {
            Vec::new()
        };
        if retain_cells {
            // SAFETY: `MaybeUninit<MotionFieldCell>` may hold uninitialized
            // bytes. No cell is read before its macroblock is recorded, and
            // `finish` proves every macroblock complete before converting the
            // allocation.
            unsafe {
                cells.set_len(count);
            }
        }
        Ok(Self {
            width_in_macroblocks: width / 4,
            cells,
            completed: vec![0; count / 16],
        })
    }

    pub(crate) fn record_intra(&mut self, macroblock_address: usize) -> Result<()> {
        self.record_uniform_macroblock(macroblock_address, MotionFieldCell::INTRA)
    }

    pub(crate) fn record_p(
        &mut self,
        macroblock_address: usize,
        motion: &ResolvedPMacroblock,
        reference_ids_l0: Option<&[Option<ReferenceId>]>,
    ) -> Result<()> {
        if self.cells.is_empty() {
            return self.record_discarded_inter_macroblock(
                macroblock_address,
                motion
                    .partitions
                    .iter()
                    .map(|partition| (partition.x, partition.y, partition.width, partition.height)),
            );
        }
        if let [partition] = motion.partitions.as_slice()
            && is_full_macroblock_partition(
                partition.x,
                partition.y,
                partition.width,
                partition.height,
            )
        {
            let reference_id = reference_ids_l0
                .and_then(|ids| ids.get(usize::from(partition.reference_index)))
                .copied()
                .flatten();
            return self.record_uniform_macroblock(
                macroblock_address,
                MotionFieldCell {
                    intra: false,
                    list0: Some(StoredListMotion {
                        reference_index: partition.reference_index,
                        reference_id,
                        vector: partition.motion_vector,
                    }),
                    list1: None,
                },
            );
        }

        let mut cells = [MotionFieldCell::INTRA; 16];
        let mut coverage = 0u16;
        for partition in &motion.partitions {
            let reference_id = reference_ids_l0
                .and_then(|ids| ids.get(usize::from(partition.reference_index)))
                .copied()
                .flatten();
            fill_partition(
                &mut cells,
                &mut coverage,
                partition.x,
                partition.y,
                partition.width,
                partition.height,
                MotionFieldCell {
                    intra: false,
                    list0: Some(StoredListMotion {
                        reference_index: partition.reference_index,
                        reference_id,
                        vector: partition.motion_vector,
                    }),
                    list1: None,
                },
            )?;
        }
        self.record_complete_inter_macroblock(macroblock_address, cells, coverage)
    }

    #[cfg(test)]
    pub(crate) fn record_b(
        &mut self,
        macroblock_address: usize,
        motion: &ResolvedBMacroblock,
        reference_ids_l0: Option<&[Option<ReferenceId>]>,
        reference_ids_l1: Option<&[Option<ReferenceId>]>,
    ) -> Result<()> {
        if macroblock_address >= self.completed.len() {
            return Err(H264Error::InvalidSyntax(
                "reference motion-field macroblock exceeds the picture",
            ));
        }
        let macroblock_x = macroblock_address % self.width_in_macroblocks;
        let macroblock_y = macroblock_address / self.width_in_macroblocks;
        self.record_b_at(
            macroblock_address,
            macroblock_x,
            macroblock_y,
            motion,
            reference_ids_l0,
            reference_ids_l1,
        )
    }

    pub(crate) fn record_b_at(
        &mut self,
        macroblock_address: usize,
        macroblock_x: usize,
        macroblock_y: usize,
        motion: &ResolvedBMacroblock,
        reference_ids_l0: Option<&[Option<ReferenceId>]>,
        reference_ids_l1: Option<&[Option<ReferenceId>]>,
    ) -> Result<()> {
        if self.cells.is_empty() {
            return self.record_discarded_inter_macroblock(
                macroblock_address,
                motion
                    .partitions
                    .iter()
                    .map(|partition| (partition.x, partition.y, partition.width, partition.height)),
            );
        }
        if let [partition] = motion.partitions.as_slice()
            && is_full_macroblock_partition(
                partition.x,
                partition.y,
                partition.width,
                partition.height,
            )
        {
            let list0 = partition.list0.map(|list| StoredListMotion {
                reference_index: list.reference_index,
                reference_id: reference_ids_l0
                    .and_then(|ids| ids.get(usize::from(list.reference_index)))
                    .copied()
                    .flatten(),
                vector: list.motion_vector,
            });
            let list1 = partition.list1.map(|list| StoredListMotion {
                reference_index: list.reference_index,
                reference_id: reference_ids_l1
                    .and_then(|ids| ids.get(usize::from(list.reference_index)))
                    .copied()
                    .flatten(),
                vector: list.motion_vector,
            });
            return self.record_uniform_macroblock_at(
                macroblock_address,
                macroblock_x,
                macroblock_y,
                MotionFieldCell {
                    intra: false,
                    list0,
                    list1,
                },
            );
        }

        let mut cells = [MotionFieldCell::INTRA; 16];
        let mut coverage = 0u16;
        for partition in &motion.partitions {
            let list0 = partition.list0.map(|list| StoredListMotion {
                reference_index: list.reference_index,
                reference_id: reference_ids_l0
                    .and_then(|ids| ids.get(usize::from(list.reference_index)))
                    .copied()
                    .flatten(),
                vector: list.motion_vector,
            });
            let list1 = partition.list1.map(|list| StoredListMotion {
                reference_index: list.reference_index,
                reference_id: reference_ids_l1
                    .and_then(|ids| ids.get(usize::from(list.reference_index)))
                    .copied()
                    .flatten(),
                vector: list.motion_vector,
            });
            fill_partition(
                &mut cells,
                &mut coverage,
                partition.x,
                partition.y,
                partition.width,
                partition.height,
                MotionFieldCell {
                    intra: false,
                    list0,
                    list1,
                },
            )?;
        }
        self.record_complete_inter_macroblock(macroblock_address, cells, coverage)
    }

    pub(crate) fn clear_macroblock(&mut self, macroblock_address: usize) -> Result<()> {
        self.macroblock_indices(macroblock_address)?;
        // The old cells can remain in place: an incomplete macroblock is never
        // exposed, and the next successful record overwrites all 16 cells.
        self.completed[macroblock_address] = 0;
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn finish(self) -> Result<ReferenceMotionField> {
        self.finish_optional()?.ok_or(H264Error::InvalidSyntax(
            "reference motion field was not retained",
        ))
    }

    pub(crate) fn finish_optional(self) -> Result<Option<ReferenceMotionField>> {
        if self.completed.contains(&0) {
            return Err(H264Error::InvalidSyntax(
                "reference motion field is incomplete",
            ));
        }
        if self.cells.is_empty() {
            return Ok(None);
        }
        let mut cells = ManuallyDrop::new(self.cells);
        let cell_len = cells.len();
        let cell_capacity = cells.capacity();
        // SAFETY: every macroblock is complete, and every successful record
        // writes all sixteen of its cells. `MaybeUninit<T>` has the same
        // layout as `T`; ManuallyDrop transfers allocation ownership exactly
        // once to the reconstructed Vec.
        let cells = unsafe {
            Vec::from_raw_parts(
                cells.as_mut_ptr().cast::<MotionFieldCell>(),
                cell_len,
                cell_capacity,
            )
        };
        Ok(Some(ReferenceMotionField {
            width_in_4x4_blocks: self.width_in_macroblocks * 4,
            height_in_4x4_blocks: cells.len() / (self.width_in_macroblocks * 4),
            cells,
        }))
    }

    fn record_complete_inter_macroblock(
        &mut self,
        macroblock_address: usize,
        cells: [MotionFieldCell; 16],
        coverage: u16,
    ) -> Result<()> {
        if coverage != u16::MAX {
            return Err(H264Error::InvalidSyntax(
                "motion partitions do not cover the macroblock",
            ));
        }
        self.record_macroblock(macroblock_address, cells)
    }

    fn record_discarded_inter_macroblock(
        &mut self,
        macroblock_address: usize,
        partitions: impl Iterator<Item = (u8, u8, u8, u8)>,
    ) -> Result<()> {
        let mut coverage = 0u16;
        for (x, y, width, height) in partitions {
            let partition = partition_mask(x, y, width, height)?;
            if coverage & partition != 0 {
                return Err(H264Error::InvalidSyntax(
                    "motion partitions overlap in the 4x4 field",
                ));
            }
            coverage |= partition;
        }
        if coverage != u16::MAX {
            return Err(H264Error::InvalidSyntax(
                "motion partitions do not cover the macroblock",
            ));
        }
        self.record_uniform_macroblock(macroblock_address, MotionFieldCell::INTRA)
    }

    fn record_macroblock(
        &mut self,
        macroblock_address: usize,
        local: [MotionFieldCell; 16],
    ) -> Result<()> {
        if macroblock_address >= self.completed.len() {
            return Err(H264Error::InvalidSyntax(
                "reference motion-field macroblock exceeds the picture",
            ));
        }
        if self.completed[macroblock_address] != 0 {
            return Err(H264Error::InvalidSyntax(
                "reference motion-field macroblock was already recorded",
            ));
        }
        if !self.cells.is_empty() {
            let indices = self.macroblock_indices(macroblock_address)?;
            for (index, cell) in indices.into_iter().zip(local) {
                self.cells[index].write(cell);
            }
        }
        self.completed[macroblock_address] = 1;
        Ok(())
    }

    fn record_uniform_macroblock(
        &mut self,
        macroblock_address: usize,
        cell: MotionFieldCell,
    ) -> Result<()> {
        let macroblock_count = self.completed.len();
        if macroblock_address >= macroblock_count {
            return Err(H264Error::InvalidSyntax(
                "reference motion-field macroblock exceeds the picture",
            ));
        }
        if self.completed[macroblock_address] != 0 {
            return Err(H264Error::InvalidSyntax(
                "reference motion-field macroblock was already recorded",
            ));
        }
        if !self.cells.is_empty() {
            let macroblock_x = macroblock_address % self.width_in_macroblocks;
            let macroblock_y = macroblock_address / self.width_in_macroblocks;
            let field_width = self.width_in_macroblocks * 4;
            let first = macroblock_y * 4 * field_width + macroblock_x * 4;
            for row in 0..4 {
                let row_start = first + row * field_width;
                for destination in &mut self.cells[row_start..row_start + 4] {
                    destination.write(cell);
                }
            }
        }
        self.completed[macroblock_address] = 1;
        Ok(())
    }

    #[inline(never)]
    fn record_uniform_macroblock_at(
        &mut self,
        macroblock_address: usize,
        macroblock_x: usize,
        macroblock_y: usize,
        cell: MotionFieldCell,
    ) -> Result<()> {
        let macroblock_count = self.completed.len();
        if macroblock_address >= macroblock_count {
            return Err(H264Error::InvalidSyntax(
                "reference motion-field macroblock exceeds the picture",
            ));
        }
        if self.completed[macroblock_address] != 0 {
            return Err(H264Error::InvalidSyntax(
                "reference motion-field macroblock was already recorded",
            ));
        }
        debug_assert!(macroblock_x < self.width_in_macroblocks);
        debug_assert_eq!(
            macroblock_address,
            macroblock_y * self.width_in_macroblocks + macroblock_x
        );
        let field_width = self.width_in_macroblocks * 4;
        let first = macroblock_y * 4 * field_width + macroblock_x * 4;
        for row in 0..4 {
            let row_start = first + row * field_width;
            for destination in &mut self.cells[row_start..row_start + 4] {
                destination.write(cell);
            }
        }
        self.completed[macroblock_address] = 1;
        Ok(())
    }

    fn macroblock_indices(&self, macroblock_address: usize) -> Result<[usize; 16]> {
        let macroblock_count = self.completed.len();
        if macroblock_address >= macroblock_count {
            return Err(H264Error::InvalidSyntax(
                "reference motion-field macroblock exceeds the picture",
            ));
        }
        let macroblock_x = macroblock_address % self.width_in_macroblocks;
        let macroblock_y = macroblock_address / self.width_in_macroblocks;
        let field_width = self.width_in_macroblocks * 4;
        Ok(std::array::from_fn(|index| {
            let x = macroblock_x * 4 + index % 4;
            let y = macroblock_y * 4 + index / 4;
            y * field_width + x
        }))
    }
}

#[inline]
const fn is_full_macroblock_partition(x: u8, y: u8, width: u8, height: u8) -> bool {
    x == 0 && y == 0 && width == 16 && height == 16
}

fn field_dimensions(coded_size: Size) -> Result<(usize, usize)> {
    let width = usize::try_from(coded_size.width).map_err(|_| H264Error::IntegerOverflow)?;
    let height = usize::try_from(coded_size.height).map_err(|_| H264Error::IntegerOverflow)?;
    if width == 0 || height == 0 || !width.is_multiple_of(16) || !height.is_multiple_of(16) {
        return Err(H264Error::InvalidSyntax(
            "motion-field dimensions must be non-zero and macroblock aligned",
        ));
    }
    Ok((width / 4, height / 4))
}

fn fill_partition(
    cells: &mut [MotionFieldCell; 16],
    coverage: &mut u16,
    x: u8,
    y: u8,
    width: u8,
    height: u8,
    value: MotionFieldCell,
) -> Result<()> {
    let partition = partition_mask(x, y, width, height)?;
    if *coverage & partition != 0 {
        return Err(H264Error::InvalidSyntax(
            "motion partitions overlap in the 4x4 field",
        ));
    }
    *coverage |= partition;
    for cell_y in y / 4..(y + height) / 4 {
        for cell_x in x / 4..(x + width) / 4 {
            let index = usize::from(cell_y * 4 + cell_x);
            cells[index] = value;
        }
    }
    Ok(())
}

fn partition_mask(x: u8, y: u8, width: u8, height: u8) -> Result<u16> {
    if !x.is_multiple_of(4)
        || !y.is_multiple_of(4)
        || !width.is_multiple_of(4)
        || !height.is_multiple_of(4)
        || x.checked_add(width).is_none_or(|end| end > 16)
        || y.checked_add(height).is_none_or(|end| end > 16)
    {
        return Err(H264Error::InvalidSyntax(
            "motion partition is not aligned to the 4x4 field",
        ));
    }
    let mut mask = 0u16;
    for cell_y in y / 4..(y + height) / 4 {
        for cell_x in x / 4..(x + width) / 4 {
            let index = usize::from(cell_y * 4 + cell_x);
            mask |= 1u16 << index;
        }
    }
    Ok(mask)
}

#[cfg(test)]
mod tests {
    use decv_core::Size;

    use super::{MotionFieldBuilder, MotionFieldCell, ReferenceMotionField, StoredListMotion};
    use crate::{
        MotionVector, ReferenceId, ResolvedBListMotion, ResolvedBMacroblock, ResolvedBPartition,
        ResolvedPMacroblock, ResolvedPPartition,
    };

    #[test]
    fn derives_spatial_direct_col_zero_masks_with_one_bounds_check() {
        let zero = MotionFieldCell {
            intra: false,
            list0: Some(StoredListMotion {
                reference_index: 0,
                reference_id: None,
                vector: MotionVector { x: 1, y: -1 },
            }),
            list1: None,
        };
        let mut field = ReferenceMotionField {
            width_in_4x4_blocks: 4,
            height_in_4x4_blocks: 4,
            cells: vec![zero; 16],
        };
        assert_eq!(
            field.spatial_direct_col_zero_mask(0, 0, true, false),
            Some(0b1111)
        );
        assert_eq!(
            field.spatial_direct_col_zero_mask(0, 0, false, false),
            Some(u16::MAX)
        );

        field.cells[2].list0.as_mut().unwrap().vector.x = 2;
        assert_eq!(
            field.spatial_direct_col_zero_mask(0, 0, true, false),
            Some(0b1101)
        );
        assert_eq!(
            field.spatial_direct_col_zero_mask(0, 0, false, false),
            Some(!(1 << 2))
        );
        assert_eq!(
            field.spatial_direct_col_zero_mask(0, 0, true, true),
            Some(0)
        );
        assert_eq!(field.spatial_direct_col_zero_mask(1, 0, true, false), None);
    }

    #[test]
    fn snapshots_intra_p_and_bidirectional_cells_in_raster_order() {
        let mut builder = MotionFieldBuilder::new(Size::new(32, 16)).unwrap();
        builder.record_intra(0).unwrap();
        builder
            .record_b(
                1,
                &ResolvedBMacroblock {
                    direct: false,
                    partitions: vec![
                        ResolvedBPartition {
                            x: 0,
                            y: 0,
                            width: 16,
                            height: 8,
                            list0: Some(ResolvedBListMotion {
                                reference_index: 0,
                                motion_vector: MotionVector { x: 2, y: 3 },
                            }),
                            list1: None,
                        },
                        ResolvedBPartition {
                            x: 0,
                            y: 8,
                            width: 16,
                            height: 8,
                            list0: None,
                            list1: Some(ResolvedBListMotion {
                                reference_index: 0,
                                motion_vector: MotionVector { x: -1, y: 4 },
                            }),
                        },
                    ]
                    .into(),
                },
                Some(&[Some(ReferenceId::new(7))]),
                Some(&[Some(ReferenceId::new(8))]),
            )
            .unwrap();
        let field = builder.finish().unwrap();
        assert!(field.cell(0, 0).unwrap().intra);
        assert_eq!(
            field.cell(4, 0).unwrap().list0.unwrap().reference_id,
            Some(ReferenceId::new(7))
        );
        assert_eq!(
            field.cell(7, 3).unwrap().list1.unwrap().reference_id,
            Some(ReferenceId::new(8))
        );
    }

    #[test]
    fn coordinate_preserving_b_record_matches_the_address_entry_point() {
        let motion = ResolvedBMacroblock {
            direct: true,
            partitions: vec![ResolvedBPartition {
                x: 0,
                y: 0,
                width: 16,
                height: 16,
                list0: Some(ResolvedBListMotion {
                    reference_index: 0,
                    motion_vector: MotionVector { x: 2, y: -1 },
                }),
                list1: Some(ResolvedBListMotion {
                    reference_index: 0,
                    motion_vector: MotionVector { x: -3, y: 4 },
                }),
            }]
            .into(),
        };
        let references_l0 = [Some(ReferenceId::new(7))];
        let references_l1 = [Some(ReferenceId::new(8))];

        let mut address_only = MotionFieldBuilder::new(Size::new(32, 16)).unwrap();
        address_only.record_intra(0).unwrap();
        address_only
            .record_b(
                1,
                &motion,
                Some(&references_l0),
                Some(&references_l1),
            )
            .unwrap();

        let mut coordinate_preserving = MotionFieldBuilder::new(Size::new(32, 16)).unwrap();
        coordinate_preserving.record_intra(0).unwrap();
        coordinate_preserving
            .record_b_at(
                1,
                1,
                0,
                &motion,
                Some(&references_l0),
                Some(&references_l1),
            )
            .unwrap();

        let address_only = address_only.finish().unwrap();
        let coordinate_preserving = coordinate_preserving.finish().unwrap();
        for y in 0..4 {
            for x in 0..8 {
                assert_eq!(
                    coordinate_preserving.cell(x, y),
                    address_only.cell(x, y)
                );
            }
        }
    }

    #[test]
    fn rejects_incomplete_and_overlapping_macroblocks() {
        let mut builder = MotionFieldBuilder::new(Size::new(16, 16)).unwrap();
        assert!(
            builder
                .record_p(
                    0,
                    &ResolvedPMacroblock {
                        skipped: false,
                        partitions: vec![ResolvedPPartition {
                            x: 0,
                            y: 0,
                            width: 8,
                            height: 16,
                            reference_index: 0,
                            motion_vector: MotionVector::default(),
                        }]
                        .into(),
                    },
                    None,
                )
                .is_err()
        );
        assert!(builder.finish().is_err());
    }

    #[test]
    fn clearing_requires_a_complete_re_record() {
        let mut incomplete = MotionFieldBuilder::new(Size::new(16, 16)).unwrap();
        incomplete.record_intra(0).unwrap();
        incomplete.clear_macroblock(0).unwrap();
        assert!(incomplete.finish().is_err());

        let mut builder = MotionFieldBuilder::new(Size::new(16, 16)).unwrap();
        builder.record_intra(0).unwrap();
        assert!(builder.record_intra(0).is_err());
        builder.clear_macroblock(0).unwrap();
        builder
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
                        motion_vector: MotionVector { x: 3, y: -2 },
                    }]
                    .into(),
                },
                Some(&[Some(ReferenceId::new(9))]),
            )
            .unwrap();

        let cell = builder.finish().unwrap().cell(3, 3).unwrap();
        assert!(!cell.intra);
        assert_eq!(cell.list0.unwrap().reference_id, Some(ReferenceId::new(9)));
        assert_eq!(cell.list0.unwrap().vector, MotionVector { x: 3, y: -2 });
    }

    #[test]
    fn discarding_builder_preserves_validation_without_retaining_cells() {
        let mut builder = MotionFieldBuilder::new_discarding(Size::new(32, 16)).unwrap();
        builder.record_intra(0).unwrap();
        builder
            .record_p(
                1,
                &ResolvedPMacroblock {
                    skipped: false,
                    partitions: vec![ResolvedPPartition {
                        x: 0,
                        y: 0,
                        width: 16,
                        height: 16,
                        reference_index: 0,
                        motion_vector: MotionVector { x: 3, y: -2 },
                    }]
                    .into(),
                },
                Some(&[Some(ReferenceId::new(9))]),
            )
            .unwrap();
        assert!(builder.finish_optional().unwrap().is_none());

        let mut duplicate = MotionFieldBuilder::new_discarding(Size::new(16, 16)).unwrap();
        duplicate.record_intra(0).unwrap();
        assert!(duplicate.record_intra(0).is_err());

        let mut incomplete = MotionFieldBuilder::new_discarding(Size::new(16, 16)).unwrap();
        assert!(
            incomplete
                .record_p(
                    0,
                    &ResolvedPMacroblock {
                        skipped: false,
                        partitions: vec![ResolvedPPartition {
                            x: 0,
                            y: 0,
                            width: 8,
                            height: 16,
                            reference_index: 0,
                            motion_vector: MotionVector::default(),
                        }]
                        .into(),
                    },
                    None,
                )
                .is_err()
        );
        assert!(incomplete.finish_optional().is_err());
    }

    #[test]
    fn cloning_a_partial_builder_preserves_its_initialized_cells() {
        let mut partial = MotionFieldBuilder::new(Size::new(32, 16)).unwrap();
        partial.record_intra(0).unwrap();

        let mut completed_clone = partial.clone();
        completed_clone.record_intra(1).unwrap();
        let field = completed_clone.finish().unwrap();
        assert!((0..8).all(|x| field.cell(x, 0).unwrap().intra));

        assert!(partial.finish().is_err());
    }
}
