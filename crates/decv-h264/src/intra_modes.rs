//! Stateful derivation of Intra4x4 and Intra8x8 prediction modes.

use crate::{H264Error, IntraPredictionModeSyntax, Result};

const LUMA_4X4_COORDINATES: [(usize, usize); 16] = [
    (0, 0),
    (1, 0),
    (0, 1),
    (1, 1),
    (2, 0),
    (3, 0),
    (2, 1),
    (3, 1),
    (0, 2),
    (1, 2),
    (0, 3),
    (1, 3),
    (2, 2),
    (3, 2),
    (2, 3),
    (3, 3),
];

const LUMA_8X8_COORDINATES: [(usize, usize); 4] = [(0, 0), (2, 0), (0, 2), (2, 2)];

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
enum ModeCell {
    #[default]
    Unavailable,
    IntraNxn {
        slice_id: u32,
        mode: u8,
    },
    OtherIntra {
        slice_id: u32,
    },
    Inter {
        slice_id: u32,
    },
}

impl ModeCell {
    #[inline]
    const fn slice_id(self) -> Option<u32> {
        match self {
            Self::Unavailable => None,
            Self::IntraNxn { slice_id, .. }
            | Self::OtherIntra { slice_id }
            | Self::Inter { slice_id } => Some(slice_id),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct NeighborMode {
    available: bool,
    constrained_inter: bool,
    mode: Option<u8>,
}

/// Four-by-four mode metadata for one progressively decoded picture.
///
/// The state uses a 4x4-cell grid so Intra4x4 and Intra8x8 neighbours share
/// one lookup path. Different `slice_id` values are treated as unavailable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntraModeState {
    width_in_macroblocks: usize,
    height_in_macroblocks: usize,
    first_slice_id: u32,
    cells: Vec<ModeCell>,
}

impl IntraModeState {
    pub fn new(width_in_macroblocks: usize, height_in_macroblocks: usize) -> Result<Self> {
        if width_in_macroblocks == 0 || height_in_macroblocks == 0 {
            return Err(H264Error::InvalidSyntax(
                "intra mode grid dimensions must be non-zero",
            ));
        }
        let cell_count = width_in_macroblocks
            .checked_mul(height_in_macroblocks)
            .and_then(|count| count.checked_mul(16))
            .ok_or(H264Error::IntegerOverflow)?;
        Ok(Self {
            width_in_macroblocks,
            height_in_macroblocks,
            first_slice_id: 0,
            cells: vec![ModeCell::Unavailable; cell_count],
        })
    }

    pub(crate) fn reset_for_picture(
        &mut self,
        width_in_macroblocks: usize,
        height_in_macroblocks: usize,
        first_slice_id: u32,
        clear_entries: bool,
    ) -> Result<()> {
        if width_in_macroblocks == 0 || height_in_macroblocks == 0 {
            return Err(H264Error::InvalidSyntax(
                "intra mode grid dimensions must be non-zero",
            ));
        }
        let cell_count = width_in_macroblocks
            .checked_mul(height_in_macroblocks)
            .and_then(|count| count.checked_mul(16))
            .ok_or(H264Error::IntegerOverflow)?;
        self.width_in_macroblocks = width_in_macroblocks;
        self.height_in_macroblocks = height_in_macroblocks;
        self.first_slice_id = first_slice_id;
        if self.cells.len() != cell_count {
            self.cells = vec![ModeCell::Unavailable; cell_count];
        } else if clear_entries {
            self.cells.fill(ModeCell::Unavailable);
        }
        Ok(())
    }

    /// Derives and records all sixteen Intra4x4 modes as one transaction.
    pub fn derive_intra4x4(
        &mut self,
        macroblock_address: usize,
        slice_id: u32,
        syntax: &[IntraPredictionModeSyntax; 16],
        constrained_intra_prediction: bool,
    ) -> Result<[u8; 16]> {
        self.validate_empty_macroblock(macroblock_address)?;
        let (macroblock_x, macroblock_y) = self.macroblock_coordinates(macroblock_address)?;
        let mut local = [[None; 4]; 4];
        let mut modes = [0; 16];
        for (index, &(local_x, local_y)) in LUMA_4X4_COORDINATES.iter().enumerate() {
            let predicted = self.predicted_mode(
                (macroblock_x, macroblock_y),
                (local_x, local_y),
                &local,
                slice_id,
                constrained_intra_prediction,
            );
            let mode = resolve_mode(syntax[index], predicted)?;
            local[local_y][local_x] = Some(mode);
            modes[index] = mode;
        }
        self.record_local_modes(macroblock_address, slice_id, &local)?;
        Ok(modes)
    }

    /// Derives and records all four Intra8x8 modes as one transaction.
    pub fn derive_intra8x8(
        &mut self,
        macroblock_address: usize,
        slice_id: u32,
        syntax: &[IntraPredictionModeSyntax; 4],
        constrained_intra_prediction: bool,
    ) -> Result<[u8; 4]> {
        self.validate_empty_macroblock(macroblock_address)?;
        let (macroblock_x, macroblock_y) = self.macroblock_coordinates(macroblock_address)?;
        let mut local = [[None; 4]; 4];
        let mut modes = [0; 4];
        for (index, &(local_x, local_y)) in LUMA_8X8_COORDINATES.iter().enumerate() {
            let predicted = self.predicted_mode(
                (macroblock_x, macroblock_y),
                (local_x, local_y),
                &local,
                slice_id,
                constrained_intra_prediction,
            );
            let mode = resolve_mode(syntax[index], predicted)?;
            for row in local.iter_mut().skip(local_y).take(2) {
                for cell in row.iter_mut().skip(local_x).take(2) {
                    *cell = Some(mode);
                }
            }
            modes[index] = mode;
        }
        self.record_local_modes(macroblock_address, slice_id, &local)?;
        Ok(modes)
    }

    pub fn record_other_intra(&mut self, macroblock_address: usize, slice_id: u32) -> Result<()> {
        self.fill_macroblock(macroblock_address, ModeCell::OtherIntra { slice_id }, true)
    }

    pub fn record_inter(&mut self, macroblock_address: usize, slice_id: u32) -> Result<()> {
        self.fill_macroblock(macroblock_address, ModeCell::Inter { slice_id }, true)
    }

    pub(crate) fn clear_macroblock(&mut self, macroblock_address: usize) -> Result<()> {
        self.fill_macroblock(macroblock_address, ModeCell::Unavailable, false)
    }

    fn predicted_mode(
        &self,
        macroblock: (usize, usize),
        local_position: (usize, usize),
        local: &[[Option<u8>; 4]; 4],
        slice_id: u32,
        constrained_intra_prediction: bool,
    ) -> u8 {
        let (macroblock_x, macroblock_y) = macroblock;
        let (local_x, local_y) = local_position;
        let left = self.neighbor(
            macroblock,
            (local_x as isize - 1, local_y as isize),
            local,
            slice_id,
            constrained_intra_prediction,
        );
        let top = self.neighbor(
            (macroblock_x, macroblock_y),
            (local_x as isize, local_y as isize - 1),
            local,
            slice_id,
            constrained_intra_prediction,
        );
        if !left.available || !top.available || left.constrained_inter || top.constrained_inter {
            2
        } else {
            left.mode.unwrap_or(2).min(top.mode.unwrap_or(2))
        }
    }

    fn neighbor(
        &self,
        macroblock: (usize, usize),
        local_position: (isize, isize),
        local: &[[Option<u8>; 4]; 4],
        slice_id: u32,
        constrained_intra_prediction: bool,
    ) -> NeighborMode {
        let (macroblock_x, macroblock_y) = macroblock;
        let (local_x, local_y) = local_position;
        if (0..4).contains(&local_x) && (0..4).contains(&local_y) {
            return local[local_y as usize][local_x as usize].map_or(
                NeighborMode {
                    available: false,
                    constrained_inter: false,
                    mode: None,
                },
                |mode| NeighborMode {
                    available: true,
                    constrained_inter: false,
                    mode: Some(mode),
                },
            );
        }

        let global_x = macroblock_x as isize * 4 + local_x;
        let global_y = macroblock_y as isize * 4 + local_y;
        if global_x < 0
            || global_y < 0
            || global_x >= (self.width_in_macroblocks * 4) as isize
            || global_y >= (self.height_in_macroblocks * 4) as isize
        {
            return NeighborMode {
                available: false,
                constrained_inter: false,
                mode: None,
            };
        }
        let index = global_y as usize * self.width_in_macroblocks * 4 + global_x as usize;
        match self.cells[index] {
            ModeCell::IntraNxn {
                slice_id: neighbor_slice,
                mode,
            } if neighbor_slice == slice_id => NeighborMode {
                available: true,
                constrained_inter: false,
                mode: Some(mode),
            },
            ModeCell::OtherIntra {
                slice_id: neighbor_slice,
            } if neighbor_slice == slice_id => NeighborMode {
                available: true,
                constrained_inter: false,
                mode: None,
            },
            ModeCell::Inter {
                slice_id: neighbor_slice,
            } if neighbor_slice == slice_id => NeighborMode {
                available: true,
                constrained_inter: constrained_intra_prediction,
                mode: None,
            },
            _ => NeighborMode {
                available: false,
                constrained_inter: false,
                mode: None,
            },
        }
    }

    fn record_local_modes(
        &mut self,
        macroblock_address: usize,
        slice_id: u32,
        local: &[[Option<u8>; 4]; 4],
    ) -> Result<()> {
        let (macroblock_x, macroblock_y) = self.macroblock_coordinates(macroblock_address)?;
        let stride = self.width_in_macroblocks * 4;
        for (local_y, row) in local.iter().enumerate() {
            for (local_x, mode) in row.iter().enumerate() {
                let mode = mode.ok_or(H264Error::InvalidSyntax(
                    "intra mode transaction left an undecoded block",
                ))?;
                let index = (macroblock_y * 4 + local_y) * stride + macroblock_x * 4 + local_x;
                self.cells[index] = ModeCell::IntraNxn { slice_id, mode };
            }
        }
        Ok(())
    }

    fn fill_macroblock(
        &mut self,
        macroblock_address: usize,
        value: ModeCell,
        require_empty: bool,
    ) -> Result<()> {
        if require_empty {
            self.validate_empty_macroblock(macroblock_address)?;
        }
        let (macroblock_x, macroblock_y) = self.macroblock_coordinates(macroblock_address)?;
        let stride = self.width_in_macroblocks * 4;
        for local_y in 0..4 {
            let start = (macroblock_y * 4 + local_y) * stride + macroblock_x * 4;
            self.cells[start..start + 4].fill(value);
        }
        Ok(())
    }

    fn validate_empty_macroblock(&self, macroblock_address: usize) -> Result<()> {
        let (macroblock_x, macroblock_y) = self.macroblock_coordinates(macroblock_address)?;
        let stride = self.width_in_macroblocks * 4;
        for local_y in 0..4 {
            let start = (macroblock_y * 4 + local_y) * stride + macroblock_x * 4;
            if self.cells[start..start + 4].iter().any(|cell| {
                cell.slice_id()
                    .is_some_and(|slice_id| slice_id >= self.first_slice_id)
            }) {
                return Err(H264Error::InvalidSyntax(
                    "macroblock prediction modes were already recorded",
                ));
            }
        }
        Ok(())
    }

    fn macroblock_coordinates(&self, macroblock_address: usize) -> Result<(usize, usize)> {
        let count = self
            .width_in_macroblocks
            .checked_mul(self.height_in_macroblocks)
            .ok_or(H264Error::IntegerOverflow)?;
        if macroblock_address >= count {
            return Err(H264Error::InvalidSyntax(
                "macroblock address exceeds intra mode grid",
            ));
        }
        Ok((
            macroblock_address % self.width_in_macroblocks,
            macroblock_address / self.width_in_macroblocks,
        ))
    }
}

fn resolve_mode(syntax: IntraPredictionModeSyntax, predicted: u8) -> Result<u8> {
    if syntax.use_predicted {
        if syntax.remaining_mode.is_some() {
            return Err(H264Error::InvalidSyntax(
                "predicted intra mode unexpectedly has a remaining mode",
            ));
        }
        return Ok(predicted);
    }
    let remaining = syntax.remaining_mode.ok_or(H264Error::InvalidSyntax(
        "explicit intra mode is missing its remaining mode",
    ))?;
    if remaining > 7 {
        return Err(H264Error::InvalidSyntax(
            "remaining intra prediction mode exceeds 7",
        ));
    }
    Ok(if remaining < predicted {
        remaining
    } else {
        remaining + 1
    })
}

#[cfg(test)]
mod tests {
    use super::{IntraModeState, ModeCell, resolve_mode};
    use crate::{H264Error, IntraPredictionModeSyntax};

    const PREDICTED: IntraPredictionModeSyntax = IntraPredictionModeSyntax {
        use_predicted: true,
        remaining_mode: None,
    };

    fn explicit(remaining_mode: u8) -> IntraPredictionModeSyntax {
        IntraPredictionModeSyntax {
            use_predicted: false,
            remaining_mode: Some(remaining_mode),
        }
    }

    fn fill_nxn(state: &mut IntraModeState, macroblock_address: usize, slice_id: u32, mode: u8) {
        state
            .fill_macroblock(
                macroblock_address,
                ModeCell::IntraNxn { slice_id, mode },
                true,
            )
            .unwrap();
    }

    #[test]
    fn reset_reuses_storage_and_ignores_stale_picture_cells() {
        let mut state = IntraModeState::new(2, 1).unwrap();
        state.record_inter(0, 10).unwrap();
        let allocation = state.cells.as_ptr();

        state.reset_for_picture(2, 1, 11, false).unwrap();
        assert_eq!(state.cells.as_ptr(), allocation);
        state.record_inter(0, 11).unwrap();
        assert!(state.record_inter(0, 11).is_err());

        state.reset_for_picture(2, 1, 12, true).unwrap();
        assert!(
            state
                .cells
                .iter()
                .all(|cell| *cell == ModeCell::Unavailable)
        );
    }

    #[test]
    fn resolves_predicted_and_remaining_mode_syntax() {
        assert_eq!(resolve_mode(PREDICTED, 4), Ok(4));
        assert_eq!(resolve_mode(explicit(2), 4), Ok(2));
        assert_eq!(resolve_mode(explicit(4), 4), Ok(5));
        assert!(resolve_mode(explicit(8), 4).is_err());
        assert!(
            resolve_mode(
                IntraPredictionModeSyntax {
                    use_predicted: true,
                    remaining_mode: Some(0),
                },
                2
            )
            .is_err()
        );
    }

    #[test]
    fn derives_first_macroblock_modes_in_scan_order() {
        let mut predicted = IntraModeState::new(1, 1).unwrap();
        assert_eq!(
            predicted
                .derive_intra4x4(0, 7, &[PREDICTED; 16], false)
                .unwrap(),
            [2; 16]
        );

        let mut explicit_vertical = IntraModeState::new(1, 1).unwrap();
        assert_eq!(
            explicit_vertical
                .derive_intra4x4(0, 7, &[explicit(0); 16], false)
                .unwrap(),
            [0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0]
        );

        let mut eight = IntraModeState::new(1, 1).unwrap();
        assert_eq!(
            eight.derive_intra8x8(0, 7, &[PREDICTED; 4], false).unwrap(),
            [2; 4]
        );
    }

    #[test]
    fn shares_neighbour_modes_between_4x4_and_8x8() {
        let mut four = IntraModeState::new(2, 2).unwrap();
        fill_nxn(&mut four, 0, 3, 8);
        fill_nxn(&mut four, 1, 3, 4);
        fill_nxn(&mut four, 2, 3, 5);
        let modes = four.derive_intra4x4(3, 3, &[PREDICTED; 16], false).unwrap();
        assert_eq!(modes[0], 4);

        let mut eight = IntraModeState::new(2, 2).unwrap();
        fill_nxn(&mut eight, 0, 3, 8);
        fill_nxn(&mut eight, 1, 3, 6);
        fill_nxn(&mut eight, 2, 3, 1);
        let modes = eight.derive_intra8x8(3, 3, &[PREDICTED; 4], false).unwrap();
        assert_eq!(modes[0], 1);
    }

    #[test]
    fn handles_slice_boundaries_and_constrained_inter_neighbours() {
        let mut slice_boundary = IntraModeState::new(2, 2).unwrap();
        fill_nxn(&mut slice_boundary, 0, 1, 0);
        fill_nxn(&mut slice_boundary, 1, 2, 0);
        fill_nxn(&mut slice_boundary, 2, 1, 0);
        let modes = slice_boundary
            .derive_intra4x4(3, 1, &[PREDICTED; 16], false)
            .unwrap();
        assert_eq!(modes[0], 2);

        let mut unconstrained = IntraModeState::new(2, 2).unwrap();
        fill_nxn(&mut unconstrained, 0, 1, 0);
        fill_nxn(&mut unconstrained, 1, 1, 0);
        unconstrained.record_inter(2, 1).unwrap();
        let modes = unconstrained
            .derive_intra4x4(3, 1, &[PREDICTED; 16], false)
            .unwrap();
        assert_eq!(modes[0], 0);

        let mut constrained = IntraModeState::new(2, 2).unwrap();
        fill_nxn(&mut constrained, 0, 1, 0);
        fill_nxn(&mut constrained, 1, 1, 0);
        constrained.record_inter(2, 1).unwrap();
        let modes = constrained
            .derive_intra4x4(3, 1, &[PREDICTED; 16], true)
            .unwrap();
        assert_eq!(modes[0], 2);
    }

    #[test]
    fn rolls_back_invalid_mode_transactions() {
        let mut state = IntraModeState::new(1, 1).unwrap();
        let mut invalid = [PREDICTED; 16];
        invalid[8] = IntraPredictionModeSyntax {
            use_predicted: false,
            remaining_mode: None,
        };
        assert_eq!(
            state.derive_intra4x4(0, 0, &invalid, false),
            Err(H264Error::InvalidSyntax(
                "explicit intra mode is missing its remaining mode"
            ))
        );
        assert_eq!(
            state
                .derive_intra4x4(0, 0, &[PREDICTED; 16], false)
                .unwrap(),
            [2; 16]
        );
        assert!(state.record_other_intra(0, 0).is_err());
        assert!(state.record_inter(1, 0).is_err());
    }
}
