//! Macroblock and chroma quantization-parameter derivation.

use crate::{H264Error, Result};

const CHROMA_QP_30_TO_51: [u8; 22] = [
    29, 30, 31, 32, 32, 33, 34, 34, 35, 35, 36, 36, 37, 37, 37, 38, 38, 38, 39, 39, 39, 39,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MacroblockQuantizer {
    /// QP'Y for the current 8-bit macroblock.
    pub luma: u8,
    /// QP'Cb after offset, clipping, and Table 8-15 mapping.
    pub chroma_cb: u8,
    /// QP'Cr after offset, clipping, and Table 8-15 mapping.
    pub chroma_cr: u8,
    pub transform_bypass: bool,
}

/// Slice-local QPY,PREV state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MacroblockQuantizerState {
    previous_luma: u8,
    chroma_cb_offset: i8,
    chroma_cr_offset: i8,
    qpprime_y_zero_transform_bypass: bool,
}

impl MacroblockQuantizerState {
    pub fn new(
        slice_qp_y: u8,
        chroma_cb_offset: i8,
        chroma_cr_offset: i8,
        qpprime_y_zero_transform_bypass: bool,
    ) -> Result<Self> {
        if slice_qp_y > 51 {
            return Err(H264Error::InvalidSyntax("SliceQPY exceeds 51"));
        }
        if !(-12..=12).contains(&chroma_cb_offset) || !(-12..=12).contains(&chroma_cr_offset) {
            return Err(H264Error::InvalidSyntax(
                "chroma QP offset is outside -12..=12",
            ));
        }
        Ok(Self {
            previous_luma: slice_qp_y,
            chroma_cb_offset,
            chroma_cr_offset,
            qpprime_y_zero_transform_bypass,
        })
    }

    #[inline]
    pub const fn previous_luma(&self) -> u8 {
        self.previous_luma
    }

    /// Derives a macroblock QP without changing QPY,PREV.
    pub fn derive(&self, qp_delta: i8) -> Result<MacroblockQuantizer> {
        if !(-26..=25).contains(&qp_delta) {
            return Err(H264Error::InvalidSyntax(
                "mb_qp_delta is outside the 8-bit range",
            ));
        }
        let luma = (i32::from(self.previous_luma) + i32::from(qp_delta)).rem_euclid(52) as u8;
        Ok(MacroblockQuantizer {
            luma,
            chroma_cb: derive_chroma_qp(luma, self.chroma_cb_offset),
            chroma_cr: derive_chroma_qp(luma, self.chroma_cr_offset),
            transform_bypass: self.qpprime_y_zero_transform_bypass && luma == 0,
        })
    }

    /// Commits QPY only if the complete macroblock operation succeeds.
    pub fn with_macroblock<T>(
        &mut self,
        qp_delta: i8,
        operation: impl FnOnce(MacroblockQuantizer) -> Result<T>,
    ) -> Result<T> {
        let quantizer = self.derive(qp_delta)?;
        let output = operation(quantizer)?;
        self.previous_luma = quantizer.luma;
        Ok(output)
    }
}

#[inline]
pub fn derive_chroma_qp(luma_qp: u8, offset: i8) -> u8 {
    let index = (i16::from(luma_qp) + i16::from(offset)).clamp(0, 51) as u8;
    if index < 30 {
        index
    } else {
        CHROMA_QP_30_TO_51[usize::from(index - 30)]
    }
}

#[cfg(test)]
mod tests {
    use super::{MacroblockQuantizer, MacroblockQuantizerState, derive_chroma_qp};
    use crate::H264Error;

    #[test]
    fn derives_luma_wraparound_and_chroma_mapping() {
        let mut state = MacroblockQuantizerState::new(26, 0, 0, false).unwrap();
        assert_eq!(
            state.with_macroblock(25, Ok),
            Ok(MacroblockQuantizer {
                luma: 51,
                chroma_cb: 39,
                chroma_cr: 39,
                transform_bypass: false,
            })
        );
        assert_eq!(state.previous_luma(), 51);
        assert_eq!(
            state.with_macroblock(1, Ok),
            Ok(MacroblockQuantizer {
                luma: 0,
                chroma_cb: 0,
                chroma_cr: 0,
                transform_bypass: false,
            })
        );
        assert_eq!(state.previous_luma(), 0);
        assert_eq!(state.derive(-26).unwrap().luma, 26);
    }

    #[test]
    fn applies_independent_chroma_offsets_and_clipping() {
        let state = MacroblockQuantizerState::new(40, 12, -12, false).unwrap();
        assert_eq!(
            state.derive(0),
            Ok(MacroblockQuantizer {
                luma: 40,
                chroma_cb: 39,
                chroma_cr: 28,
                transform_bypass: false,
            })
        );
        assert_eq!(derive_chroma_qp(30, 0), 29);
        assert_eq!(derive_chroma_qp(44, 0), 37);
        assert_eq!(derive_chroma_qp(51, 12), 39);
        assert_eq!(derive_chroma_qp(0, -12), 0);
    }

    #[test]
    fn commits_only_after_a_successful_macroblock() {
        let mut state = MacroblockQuantizerState::new(20, 0, 0, false).unwrap();
        let result: crate::Result<()> = state.with_macroblock(5, |_| Err(H264Error::UnexpectedEof));
        assert_eq!(result, Err(H264Error::UnexpectedEof));
        assert_eq!(state.previous_luma(), 20);

        assert_eq!(state.with_macroblock(5, |_| Ok("done")), Ok("done"));
        assert_eq!(state.previous_luma(), 25);
    }

    #[test]
    fn derives_transform_bypass_at_zero_luma_qp() {
        let mut state = MacroblockQuantizerState::new(1, 0, 0, true).unwrap();
        assert!(state.with_macroblock(-1, Ok).unwrap().transform_bypass);
        assert!(!state.with_macroblock(1, Ok).unwrap().transform_bypass);
    }

    #[test]
    fn rejects_invalid_initial_values_and_delta() {
        assert!(matches!(
            MacroblockQuantizerState::new(52, 0, 0, false),
            Err(H264Error::InvalidSyntax(_))
        ));
        assert!(matches!(
            MacroblockQuantizerState::new(26, 13, 0, false),
            Err(H264Error::InvalidSyntax(_))
        ));
        let state = MacroblockQuantizerState::new(26, 0, 0, false).unwrap();
        assert!(matches!(state.derive(26), Err(H264Error::InvalidSyntax(_))));
    }
}
