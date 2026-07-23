//! CABAC reference-index and motion-vector-difference context state.

use crate::{CabacSyntaxDecoder, H264Error, MotionVectorDifference, Result};

const MAXIMUM_MVD_ESCAPE_ORDER: u8 = 24;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MotionSyntaxCell {
    slice_id: u32,
    reference_index: i8,
    direct: bool,
    mvd_absolute: [u8; 2],
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct MacroblockMotionSyntaxSnapshot {
    cells: [Option<MotionSyntaxCell>; 16],
}

/// One rectangular motion partition in luma-sample coordinates relative to
/// the current macroblock.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CabacMotionPartition {
    pub x: u8,
    pub y: u8,
    pub width: u8,
    pub height: u8,
}

/// Per-list CABAC syntax state stored at 4x4 motion-cell granularity.
#[derive(Debug, Clone)]
pub struct CabacMotionSyntaxState {
    width_in_macroblocks: usize,
    height_in_macroblocks: usize,
    cells: Vec<Option<MotionSyntaxCell>>,
}

impl CabacMotionSyntaxState {
    pub fn new(width_in_macroblocks: usize, height_in_macroblocks: usize) -> Result<Self> {
        if width_in_macroblocks == 0 || height_in_macroblocks == 0 {
            return Err(H264Error::InvalidSyntax(
                "CABAC motion-state dimensions must be non-zero",
            ));
        }
        let cell_count = width_in_macroblocks
            .checked_mul(height_in_macroblocks)
            .and_then(|count| count.checked_mul(16))
            .ok_or(H264Error::IntegerOverflow)?;
        Ok(Self {
            width_in_macroblocks,
            height_in_macroblocks,
            cells: vec![None; cell_count],
        })
    }

    /// Decodes and records `ref_idx_lX`. A single active reference is inferred
    /// as zero and consumes no bin.
    pub fn decode_reference_index(
        &mut self,
        syntax: &mut CabacSyntaxDecoder<'_, '_>,
        macroblock_address: usize,
        slice_id: u32,
        partition: CabacMotionPartition,
        active_reference_count: u8,
        ignore_direct_neighbours: bool,
    ) -> Result<u8> {
        if active_reference_count == 0 {
            return Err(H264Error::InvalidSyntax(
                "CABAC reference list has no active entries",
            ));
        }
        let context_increment = self.reference_context_increment(
            macroblock_address,
            slice_id,
            partition,
            ignore_direct_neighbours,
        )?;
        let reference_index =
            decode_reference_index_with(context_increment, active_reference_count, |index| {
                syntax.decision_known(index)
            })?;
        self.fill_partition(
            macroblock_address,
            partition,
            MotionSyntaxCell {
                slice_id,
                reference_index: i8::try_from(reference_index)
                    .map_err(|_| H264Error::IntegerOverflow)?,
                direct: false,
                mvd_absolute: [0; 2],
            },
        )?;
        Ok(reference_index)
    }

    /// Decodes both components of `mvd_lX` and records their clipped absolute
    /// values for following partition contexts.
    pub fn decode_motion_vector_difference(
        &mut self,
        syntax: &mut CabacSyntaxDecoder<'_, '_>,
        macroblock_address: usize,
        slice_id: u32,
        partition: CabacMotionPartition,
    ) -> Result<MotionVectorDifference> {
        let [horizontal_sum, vertical_sum] =
            self.neighbour_mvd_sums(macroblock_address, slice_id, partition)?;
        let (x, x_absolute) = decode_mvd_component_with(40, horizontal_sum, |request| {
            decode_request(syntax, request)
        })?;
        let (y, y_absolute) =
            decode_mvd_component_with(47, vertical_sum, |request| decode_request(syntax, request))?;
        self.update_partition_mvd(
            macroblock_address,
            slice_id,
            partition,
            [x_absolute, y_absolute],
        )?;
        Ok(MotionVectorDifference { x, y })
    }

    pub fn record_intra_macroblock(
        &mut self,
        macroblock_address: usize,
        slice_id: u32,
    ) -> Result<()> {
        self.fill_macroblock(
            macroblock_address,
            MotionSyntaxCell {
                slice_id,
                reference_index: -1,
                direct: false,
                mvd_absolute: [0; 2],
            },
        )
    }

    pub fn record_skip_macroblock(
        &mut self,
        macroblock_address: usize,
        slice_id: u32,
    ) -> Result<()> {
        self.fill_macroblock(
            macroblock_address,
            MotionSyntaxCell {
                slice_id,
                reference_index: 0,
                direct: false,
                mvd_absolute: [0; 2],
            },
        )
    }

    /// Records a B_Direct partition. Reference-index context derivation can
    /// explicitly ignore these cells while their inferred MVD remains zero.
    pub fn record_direct_partition(
        &mut self,
        macroblock_address: usize,
        slice_id: u32,
        partition: CabacMotionPartition,
    ) -> Result<()> {
        self.fill_partition(
            macroblock_address,
            partition,
            MotionSyntaxCell {
                slice_id,
                reference_index: 0,
                direct: true,
                mvd_absolute: [0; 2],
            },
        )
    }

    pub fn record_direct_macroblock(
        &mut self,
        macroblock_address: usize,
        slice_id: u32,
    ) -> Result<()> {
        self.record_direct_partition(
            macroblock_address,
            slice_id,
            CabacMotionPartition {
                x: 0,
                y: 0,
                width: 16,
                height: 16,
            },
        )
    }

    /// Records a partition that does not use this reference list.
    pub fn record_unused_partition(
        &mut self,
        macroblock_address: usize,
        slice_id: u32,
        partition: CabacMotionPartition,
    ) -> Result<()> {
        self.fill_partition(
            macroblock_address,
            partition,
            MotionSyntaxCell {
                slice_id,
                reference_index: -1,
                direct: false,
                mvd_absolute: [0; 2],
            },
        )
    }

    pub(crate) fn snapshot_macroblock(
        &self,
        macroblock_address: usize,
    ) -> Result<MacroblockMotionSyntaxSnapshot> {
        let base = self.macroblock_cell_base(macroblock_address)?;
        let picture_width = self.width_in_macroblocks * 4;
        let mut cells = [None; 16];
        for local_y in 0..4 {
            for local_x in 0..4 {
                cells[local_y * 4 + local_x] = self.cells[base + local_y * picture_width + local_x];
            }
        }
        Ok(MacroblockMotionSyntaxSnapshot { cells })
    }

    pub(crate) fn restore_macroblock(
        &mut self,
        macroblock_address: usize,
        snapshot: MacroblockMotionSyntaxSnapshot,
    ) -> Result<()> {
        let base = self.macroblock_cell_base(macroblock_address)?;
        let picture_width = self.width_in_macroblocks * 4;
        for local_y in 0..4 {
            for local_x in 0..4 {
                self.cells[base + local_y * picture_width + local_x] =
                    snapshot.cells[local_y * 4 + local_x];
            }
        }
        Ok(())
    }

    fn reference_context_increment(
        &self,
        macroblock_address: usize,
        slice_id: u32,
        partition: CabacMotionPartition,
        ignore_direct_neighbours: bool,
    ) -> Result<u8> {
        let (x, y, _, _) = self.partition_cells(macroblock_address, partition)?;
        let left = self.cell(x.checked_sub(1), Some(y), slice_id);
        let top = self.cell(Some(x), y.checked_sub(1), slice_id);
        Ok(u8::from(left.is_some_and(|cell| {
            cell.reference_index > 0 && !(ignore_direct_neighbours && cell.direct)
        })) + 2 * u8::from(top.is_some_and(|cell| {
            cell.reference_index > 0 && !(ignore_direct_neighbours && cell.direct)
        })))
    }

    fn neighbour_mvd_sums(
        &self,
        macroblock_address: usize,
        slice_id: u32,
        partition: CabacMotionPartition,
    ) -> Result<[u16; 2]> {
        let (x, y, _, _) = self.partition_cells(macroblock_address, partition)?;
        let left = self.cell(x.checked_sub(1), Some(y), slice_id);
        let top = self.cell(Some(x), y.checked_sub(1), slice_id);
        Ok(std::array::from_fn(|component| {
            u16::from(left.map_or(0, |cell| cell.mvd_absolute[component]))
                + u16::from(top.map_or(0, |cell| cell.mvd_absolute[component]))
        }))
    }

    fn cell(&self, x: Option<usize>, y: Option<usize>, slice_id: u32) -> Option<MotionSyntaxCell> {
        let (x, y) = (x?, y?);
        let width = self.width_in_macroblocks * 4;
        let height = self.height_in_macroblocks * 4;
        if x >= width || y >= height {
            return None;
        }
        self.cells[y * width + x].filter(|cell| cell.slice_id == slice_id)
    }

    fn fill_macroblock(&mut self, macroblock_address: usize, cell: MotionSyntaxCell) -> Result<()> {
        self.fill_partition(
            macroblock_address,
            CabacMotionPartition {
                x: 0,
                y: 0,
                width: 16,
                height: 16,
            },
            cell,
        )
    }

    fn fill_partition(
        &mut self,
        macroblock_address: usize,
        partition: CabacMotionPartition,
        cell: MotionSyntaxCell,
    ) -> Result<()> {
        let (x, y, width, height) = self.partition_cells(macroblock_address, partition)?;
        let picture_width = self.width_in_macroblocks * 4;
        for row in y..y + height {
            for column in x..x + width {
                self.cells[row * picture_width + column] = Some(cell);
            }
        }
        Ok(())
    }

    fn update_partition_mvd(
        &mut self,
        macroblock_address: usize,
        slice_id: u32,
        partition: CabacMotionPartition,
        mvd_absolute: [u8; 2],
    ) -> Result<()> {
        let (x, y, width, height) = self.partition_cells(macroblock_address, partition)?;
        let picture_width = self.width_in_macroblocks * 4;
        for row in y..y + height {
            for column in x..x + width {
                let cell = self.cells[row * picture_width + column]
                    .as_mut()
                    .filter(|cell| cell.slice_id == slice_id)
                    .ok_or(H264Error::InvalidSyntax(
                        "CABAC MVD decoded before its reference index",
                    ))?;
                cell.mvd_absolute = mvd_absolute;
            }
        }
        Ok(())
    }

    fn partition_cells(
        &self,
        macroblock_address: usize,
        partition: CabacMotionPartition,
    ) -> Result<(usize, usize, usize, usize)> {
        if macroblock_address >= self.width_in_macroblocks * self.height_in_macroblocks
            || partition.width == 0
            || partition.height == 0
            || !partition.x.is_multiple_of(4)
            || !partition.y.is_multiple_of(4)
            || !partition.width.is_multiple_of(4)
            || !partition.height.is_multiple_of(4)
            || u16::from(partition.x) + u16::from(partition.width) > 16
            || u16::from(partition.y) + u16::from(partition.height) > 16
        {
            return Err(H264Error::InvalidSyntax(
                "CABAC motion partition is outside its macroblock",
            ));
        }
        let macroblock_x = macroblock_address % self.width_in_macroblocks;
        let macroblock_y = macroblock_address / self.width_in_macroblocks;
        Ok((
            macroblock_x * 4 + usize::from(partition.x / 4),
            macroblock_y * 4 + usize::from(partition.y / 4),
            usize::from(partition.width / 4),
            usize::from(partition.height / 4),
        ))
    }

    fn macroblock_cell_base(&self, macroblock_address: usize) -> Result<usize> {
        if macroblock_address >= self.width_in_macroblocks * self.height_in_macroblocks {
            return Err(H264Error::InvalidSyntax(
                "CABAC motion macroblock address exceeds the picture",
            ));
        }
        let macroblock_x = macroblock_address % self.width_in_macroblocks;
        let macroblock_y = macroblock_address / self.width_in_macroblocks;
        Ok(macroblock_y * 4 * self.width_in_macroblocks * 4 + macroblock_x * 4)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MvdBinRequest {
    Decision(usize),
    Bypass,
}

fn decode_request(syntax: &mut CabacSyntaxDecoder<'_, '_>, request: MvdBinRequest) -> Result<u8> {
    match request {
        MvdBinRequest::Decision(context_index) => syntax.decision_known(context_index),
        MvdBinRequest::Bypass => syntax.bypass(),
    }
}

fn decode_reference_index_with(
    context_increment: u8,
    active_reference_count: u8,
    mut decision: impl FnMut(usize) -> Result<u8>,
) -> Result<u8> {
    if active_reference_count == 0 || context_increment > 3 {
        return Err(H264Error::InvalidSyntax(
            "CABAC reference-index inputs are out of range",
        ));
    }
    if active_reference_count == 1 {
        return Ok(0);
    }
    let mut reference_index = 0u8;
    let mut context_increment = usize::from(context_increment);
    loop {
        if decision(54 + context_increment)? == 0 {
            return Ok(reference_index);
        }
        reference_index = reference_index
            .checked_add(1)
            .ok_or(H264Error::IntegerOverflow)?;
        if reference_index >= active_reference_count {
            return Err(H264Error::InvalidSyntax(
                "CABAC reference index exceeds the active list",
            ));
        }
        context_increment = (context_increment >> 2) + 4;
    }
}

fn decode_mvd_component_with(
    context_base: usize,
    neighbour_absolute_sum: u16,
    mut decode: impl FnMut(MvdBinRequest) -> Result<u8>,
) -> Result<(i16, u8)> {
    let context_increment =
        usize::from(neighbour_absolute_sum > 2) + usize::from(neighbour_absolute_sum > 32);
    if decode(MvdBinRequest::Decision(context_base + context_increment))? == 0 {
        return Ok((0, 0));
    }

    let mut magnitude = 1u32;
    let mut context_index = context_base + 3;
    while magnitude < 9 && decode(MvdBinRequest::Decision(context_index))? != 0 {
        if magnitude < 4 {
            context_index += 1;
        }
        magnitude += 1;
    }
    if magnitude >= 9 {
        let mut order = 3u8;
        while decode(MvdBinRequest::Bypass)? != 0 {
            magnitude = magnitude
                .checked_add(1u32 << order)
                .ok_or(H264Error::IntegerOverflow)?;
            order = order.checked_add(1).ok_or(H264Error::IntegerOverflow)?;
            if order > MAXIMUM_MVD_ESCAPE_ORDER {
                return Err(H264Error::InvalidSyntax(
                    "CABAC MVD escape prefix is too long",
                ));
            }
        }
        for shift in (0..order).rev() {
            magnitude = magnitude
                .checked_add(u32::from(decode(MvdBinRequest::Bypass)?) << shift)
                .ok_or(H264Error::IntegerOverflow)?;
        }
    }
    let magnitude_i16 = i16::try_from(magnitude).map_err(|_| H264Error::IntegerOverflow)?;
    let value = if decode(MvdBinRequest::Bypass)? == 0 {
        magnitude_i16
    } else {
        -magnitude_i16
    };
    Ok((
        value,
        u8::try_from(magnitude.min(70)).expect("the stored MVD magnitude is capped at 70"),
    ))
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use super::*;

    #[test]
    fn progresses_reference_index_contexts_and_checks_the_active_list() {
        let mut bins = VecDeque::from([1, 1, 0]);
        let mut visited = Vec::new();
        assert_eq!(
            decode_reference_index_with(1, 4, |context_index| {
                visited.push(context_index);
                Ok(bins.pop_front().unwrap())
            })
            .unwrap(),
            2
        );
        assert_eq!(visited, [55, 58, 59]);
        assert!(decode_reference_index_with(0, 2, |_| Ok(1)).is_err());
    }

    #[test]
    fn decodes_zero_short_and_escape_mvd_components() {
        let mut visited = Vec::new();
        assert_eq!(
            decode_mvd_component_with(40, 0, |request| {
                visited.push(request);
                Ok(0)
            })
            .unwrap(),
            (0, 0)
        );
        assert_eq!(visited, [MvdBinRequest::Decision(40)]);

        let mut bins = VecDeque::from([1, 1, 1, 0, 1]);
        let mut visited = Vec::new();
        assert_eq!(
            decode_mvd_component_with(40, 3, |request| {
                visited.push(request);
                Ok(bins.pop_front().unwrap())
            })
            .unwrap(),
            (-3, 3)
        );
        assert_eq!(
            visited,
            [
                MvdBinRequest::Decision(41),
                MvdBinRequest::Decision(43),
                MvdBinRequest::Decision(44),
                MvdBinRequest::Decision(45),
                MvdBinRequest::Bypass,
            ]
        );

        let mut bins = VecDeque::from(
            [1].into_iter()
                .chain(std::iter::repeat_n(1, 8))
                .chain([0, 0, 1, 0, 0])
                .collect::<Vec<_>>(),
        );
        assert_eq!(
            decode_mvd_component_with(47, 33, |_| Ok(bins.pop_front().unwrap())).unwrap(),
            (11, 11)
        );
        assert!(bins.is_empty());
    }

    #[test]
    fn derives_partition_contexts_from_left_top_and_prior_local_cells() {
        let mut state = CabacMotionSyntaxState::new(2, 2).unwrap();
        state
            .fill_partition(
                0,
                CabacMotionPartition {
                    x: 0,
                    y: 0,
                    width: 16,
                    height: 16,
                },
                MotionSyntaxCell {
                    slice_id: 7,
                    reference_index: 2,
                    direct: false,
                    mvd_absolute: [3, 34],
                },
            )
            .unwrap();
        let first = CabacMotionPartition {
            x: 0,
            y: 0,
            width: 16,
            height: 8,
        };
        assert_eq!(state.reference_context_increment(1, 7, first, false), Ok(1));
        assert_eq!(state.neighbour_mvd_sums(1, 7, first), Ok([3, 34]));

        state
            .fill_partition(
                1,
                first,
                MotionSyntaxCell {
                    slice_id: 7,
                    reference_index: 1,
                    direct: false,
                    mvd_absolute: [5, 6],
                },
            )
            .unwrap();
        let second = CabacMotionPartition {
            x: 0,
            y: 8,
            width: 16,
            height: 8,
        };
        assert_eq!(
            state.reference_context_increment(1, 7, second, false),
            Ok(3)
        );
        assert_eq!(state.neighbour_mvd_sums(1, 7, second), Ok([8, 40]));
        assert_eq!(
            state.reference_context_increment(1, 8, second, false),
            Ok(0)
        );
    }

    #[test]
    fn restores_only_the_snapshotted_macroblock() {
        let mut state = CabacMotionSyntaxState::new(2, 1).unwrap();
        state.record_skip_macroblock(0, 3).unwrap();
        state.record_intra_macroblock(1, 3).unwrap();
        let first_before = state.snapshot_macroblock(0).unwrap();
        let second_before = state.snapshot_macroblock(1).unwrap();

        state.record_intra_macroblock(0, 4).unwrap();
        state.restore_macroblock(0, first_before).unwrap();

        assert_eq!(
            state.snapshot_macroblock(0).unwrap().cells,
            first_before.cells
        );
        assert_eq!(
            state.snapshot_macroblock(1).unwrap().cells,
            second_before.cells
        );
    }

    #[test]
    fn distinguishes_direct_and_unused_list_cells() {
        let mut state = CabacMotionSyntaxState::new(2, 1).unwrap();
        let whole = CabacMotionPartition {
            x: 0,
            y: 0,
            width: 16,
            height: 16,
        };
        state.record_direct_partition(0, 4, whole).unwrap();
        let current = CabacMotionPartition {
            x: 0,
            y: 0,
            width: 8,
            height: 16,
        };
        assert_eq!(
            state.reference_context_increment(1, 4, current, true),
            Ok(0)
        );
        assert_eq!(state.neighbour_mvd_sums(1, 4, current), Ok([0, 0]));
        assert!(
            state
                .snapshot_macroblock(0)
                .unwrap()
                .cells
                .iter()
                .all(|cell| cell.is_some_and(|cell| cell.direct && cell.reference_index == 0))
        );

        state.record_unused_partition(0, 5, whole).unwrap();
        assert_eq!(
            state.reference_context_increment(1, 5, current, false),
            Ok(0)
        );
        assert!(
            state
                .snapshot_macroblock(0)
                .unwrap()
                .cells
                .iter()
                .all(|cell| cell.is_some_and(|cell| !cell.direct && cell.reference_index == -1))
        );
    }
}
