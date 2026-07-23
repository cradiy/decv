//! Stateful reconstruction of progressive 8-bit 4:2:0 intra pictures.

use decv_core::{DecodedVideoFrame, MediaTime, Size, VideoFormat};

use crate::{
    ChromaPlane, DecodedIntraMacroblock, H264Error, IntraLumaPrediction, IntraMacroblock,
    IntraModeState, IntraReferenceAvailability, MacroblockQuantizer, ReconstructedIntraResidual,
    ResolvedScalingLists4x4, Result, ScanMode, Yuv420Picture, predict_intra_16x16,
    predict_intra_chroma_420, reconstruct_intra_residual,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CompletedMacroblock {
    slice_id: u32,
    is_intra: bool,
}

/// Reconstructs one progressively scanned intra picture in macroblock order.
#[derive(Debug, Clone)]
pub struct IntraPictureReconstructor {
    width_in_macroblocks: usize,
    picture: Yuv420Picture,
    modes: IntraModeState,
    completed: Vec<Option<CompletedMacroblock>>,
    scaling_lists: ResolvedScalingLists4x4,
    scan_mode: ScanMode,
    constrained_intra_prediction: bool,
}

impl IntraPictureReconstructor {
    pub fn new(
        coded_size: Size,
        scaling_lists: ResolvedScalingLists4x4,
        constrained_intra_prediction: bool,
    ) -> Result<Self> {
        let picture = Yuv420Picture::new(coded_size)?;
        let width_in_macroblocks =
            usize::try_from(coded_size.width / 16).map_err(|_| H264Error::IntegerOverflow)?;
        let height_in_macroblocks =
            usize::try_from(coded_size.height / 16).map_err(|_| H264Error::IntegerOverflow)?;
        let macroblock_count = width_in_macroblocks
            .checked_mul(height_in_macroblocks)
            .ok_or(H264Error::IntegerOverflow)?;
        Ok(Self {
            width_in_macroblocks,
            picture,
            modes: IntraModeState::new(width_in_macroblocks, height_in_macroblocks)?,
            completed: vec![None; macroblock_count],
            scaling_lists,
            scan_mode: ScanMode::Frame,
            constrained_intra_prediction,
        })
    }

    #[inline]
    pub fn picture(&self) -> &Yuv420Picture {
        &self.picture
    }

    /// Reconstructs one decoded CAVLC intra macroblock.
    ///
    /// Prediction and residual matrices are fully prepared before any picture
    /// sample is changed. Consequently malformed directional modes do not
    /// leave a partially written macroblock.
    pub fn reconstruct_macroblock(
        &mut self,
        macroblock_address: usize,
        slice_id: u32,
        decoded: &DecodedIntraMacroblock,
        quantizer: MacroblockQuantizer,
    ) -> Result<()> {
        let (macroblock_x, macroblock_y) = self.validate_new_macroblock(macroblock_address)?;
        match &decoded.macroblock {
            IntraMacroblock::Pcm(pcm) => {
                if decoded.residual.is_some() {
                    return Err(H264Error::InvalidSyntax(
                        "I_PCM macroblock unexpectedly contains residual blocks",
                    ));
                }
                self.modes
                    .record_other_intra(macroblock_address, slice_id)?;
                self.picture
                    .write_pcm_macroblock(macroblock_x, macroblock_y, pcm)?;
            }
            IntraMacroblock::Predicted(header) => {
                let residual = decoded.residual.as_ref().ok_or(H264Error::InvalidSyntax(
                    "predicted intra macroblock is missing residual blocks",
                ))?;
                let reconstructed = reconstruct_intra_residual(
                    header,
                    residual,
                    quantizer,
                    &self.scaling_lists,
                    self.scan_mode,
                )?;
                match header.luma_prediction {
                    IntraLumaPrediction::SixteenBySixteen { mode } => {
                        self.reconstruct_intra16x16(
                            macroblock_address,
                            macroblock_x,
                            macroblock_y,
                            slice_id,
                            mode,
                            header.chroma_prediction_mode,
                            &reconstructed,
                        )?;
                        self.modes
                            .record_other_intra(macroblock_address, slice_id)?;
                    }
                    IntraLumaPrediction::FourByFour(_) => {
                        return Err(H264Error::UnsupportedFeature(
                            "Intra4x4 picture reconstruction",
                        ));
                    }
                    IntraLumaPrediction::EightByEight(_) => {
                        return Err(H264Error::UnsupportedFeature(
                            "Intra8x8 picture reconstruction",
                        ));
                    }
                }
            }
        }
        self.completed[macroblock_address] = Some(CompletedMacroblock {
            slice_id,
            is_intra: true,
        });
        Ok(())
    }

    pub fn into_nv12_frame(
        self,
        id: u64,
        pts: Option<MediaTime>,
        duration: Option<MediaTime>,
        format: VideoFormat,
    ) -> Result<DecodedVideoFrame> {
        if self.completed.iter().any(Option::is_none) {
            return Err(H264Error::InvalidSyntax(
                "cannot output an incomplete reconstructed picture",
            ));
        }
        self.picture.into_nv12_frame(id, pts, duration, format)
    }

    #[allow(clippy::too_many_arguments)]
    fn reconstruct_intra16x16(
        &mut self,
        macroblock_address: usize,
        macroblock_x: usize,
        macroblock_y: usize,
        slice_id: u32,
        luma_mode: u8,
        chroma_mode: u8,
        residual: &ReconstructedIntraResidual,
    ) -> Result<()> {
        let availability = self.macroblock_availability(macroblock_address, slice_id);
        let luma_references =
            self.picture
                .intra16x16_references(macroblock_x, macroblock_y, availability)?;
        let luma_prediction = predict_intra_16x16(luma_mode, &luma_references)?;
        let cb_references = self.picture.intra_chroma_references(
            ChromaPlane::Cb,
            macroblock_x,
            macroblock_y,
            availability,
        )?;
        let cr_references = self.picture.intra_chroma_references(
            ChromaPlane::Cr,
            macroblock_x,
            macroblock_y,
            availability,
        )?;
        let cb_prediction = predict_intra_chroma_420(chroma_mode, &cb_references)?;
        let cr_prediction = predict_intra_chroma_420(chroma_mode, &cr_references)?;

        let luma_residual = assemble_luma_residual(&residual.luma);
        let cb_residual = assemble_chroma_residual(&residual.chroma_cb);
        let cr_residual = assemble_chroma_residual(&residual.chroma_cr);
        self.picture.write_luma_16x16(
            macroblock_x,
            macroblock_y,
            &luma_prediction,
            &luma_residual,
        )?;
        self.picture.write_chroma_8x8(
            ChromaPlane::Cb,
            macroblock_x,
            macroblock_y,
            &cb_prediction,
            &cb_residual,
        )?;
        self.picture.write_chroma_8x8(
            ChromaPlane::Cr,
            macroblock_x,
            macroblock_y,
            &cr_prediction,
            &cr_residual,
        )
    }

    fn macroblock_availability(
        &self,
        macroblock_address: usize,
        slice_id: u32,
    ) -> IntraReferenceAvailability {
        let macroblock_x = macroblock_address % self.width_in_macroblocks;
        let macroblock_y = macroblock_address / self.width_in_macroblocks;
        let top = (macroblock_y > 0).then_some(macroblock_address - self.width_in_macroblocks);
        let left = (macroblock_x > 0).then_some(macroblock_address - 1);
        let top_left = (macroblock_x > 0 && macroblock_y > 0)
            .then_some(macroblock_address - self.width_in_macroblocks - 1);
        IntraReferenceAvailability {
            top: top.is_some_and(|address| self.is_available(address, slice_id)),
            left: left.is_some_and(|address| self.is_available(address, slice_id)),
            top_left: top_left.is_some_and(|address| self.is_available(address, slice_id)),
            top_right: false,
        }
    }

    fn is_available(&self, address: usize, slice_id: u32) -> bool {
        self.completed[address].is_some_and(|macroblock| {
            macroblock.slice_id == slice_id
                && (!self.constrained_intra_prediction || macroblock.is_intra)
        })
    }

    fn validate_new_macroblock(&self, address: usize) -> Result<(usize, usize)> {
        if address >= self.completed.len() {
            return Err(H264Error::InvalidSyntax(
                "macroblock address exceeds reconstructed picture",
            ));
        }
        if self.completed[address].is_some() {
            return Err(H264Error::InvalidSyntax(
                "macroblock was already reconstructed",
            ));
        }
        Ok((
            address % self.width_in_macroblocks,
            address / self.width_in_macroblocks,
        ))
    }
}

fn assemble_luma_residual(blocks: &[[[i32; 4]; 4]; 16]) -> [[i32; 16]; 16] {
    let mut output = [[0; 16]; 16];
    const COORDINATES: [(usize, usize); 16] = [
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
    for (block, (block_x, block_y)) in blocks.iter().zip(COORDINATES) {
        for row in 0..4 {
            output[block_y * 4 + row][block_x * 4..block_x * 4 + 4].copy_from_slice(&block[row]);
        }
    }
    output
}

fn assemble_chroma_residual(blocks: &[[[i32; 4]; 4]; 4]) -> [[i32; 8]; 8] {
    let mut output = [[0; 8]; 8];
    for (index, block) in blocks.iter().enumerate() {
        let block_x = index % 2;
        let block_y = index / 2;
        for row in 0..4 {
            output[block_y * 4 + row][block_x * 4..block_x * 4 + 4].copy_from_slice(&block[row]);
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use decv_core::{ColorInfo, FrameStorage, PixelFormat, Rect, Size, VideoFormat};

    use super::IntraPictureReconstructor;
    use crate::{
        CodedBlockPattern, DecodedIntraMacroblock, IntraLumaPrediction, IntraMacroblock,
        IntraMacroblockHeader, IntraResidual, MacroblockQuantizer, PcmMacroblock, ResidualBlock,
        resolve_scaling_lists_4x4,
    };

    fn quantizer() -> MacroblockQuantizer {
        MacroblockQuantizer {
            luma: 0,
            chroma_cb: 0,
            chroma_cr: 0,
            transform_bypass: false,
        }
    }

    fn predicted(luma_mode: u8, chroma_mode: u8) -> DecodedIntraMacroblock {
        DecodedIntraMacroblock {
            macroblock: IntraMacroblock::Predicted(IntraMacroblockHeader {
                luma_prediction: IntraLumaPrediction::SixteenBySixteen { mode: luma_mode },
                chroma_prediction_mode: chroma_mode,
                coded_block_pattern: CodedBlockPattern { luma: 0, chroma: 0 },
                qp_delta: 0,
            }),
            residual: Some(IntraResidual {
                luma_dc: Some(ResidualBlock::empty(16)),
                luma: [ResidualBlock::empty(15); 16],
                chroma_dc: [ResidualBlock::empty(4); 2],
                chroma_ac: [[ResidualBlock::empty(15); 4]; 2],
            }),
        }
    }

    fn pcm(luma: u8, cb: u8, cr: u8) -> DecodedIntraMacroblock {
        DecodedIntraMacroblock {
            macroblock: IntraMacroblock::Pcm(PcmMacroblock {
                luma: Box::new([luma; 256]),
                chroma: Box::new(std::array::from_fn(
                    |index| if index < 64 { cb } else { cr },
                )),
            }),
            residual: None,
        }
    }

    fn reconstructor(size: Size) -> IntraPictureReconstructor {
        IntraPictureReconstructor::new(size, resolve_scaling_lists_4x4(None, None).unwrap(), false)
            .unwrap()
    }

    fn format(size: Size) -> VideoFormat {
        VideoFormat {
            coded_size: size,
            visible_rect: Rect::new(0, 0, size.width, size.height),
            display_size: size,
            pixel_format: PixelFormat::Nv12,
            color: ColorInfo::default(),
        }
    }

    #[test]
    fn reconstructs_flat_intra16_picture_to_nv12() {
        let size = Size::new(16, 16);
        let mut reconstructor = reconstructor(size);
        reconstructor
            .reconstruct_macroblock(0, 1, &predicted(2, 0), quantizer())
            .unwrap();
        let frame = reconstructor
            .into_nv12_frame(1, None, None, format(size))
            .unwrap();
        let cpu = match frame.storage {
            FrameStorage::Cpu(cpu) => cpu,
            _ => panic!("expected CPU frame"),
        };
        assert!(cpu.planes[0].bytes.iter().all(|&sample| sample == 128));
    }

    #[test]
    fn uses_completed_left_macroblock_for_horizontal_prediction() {
        let size = Size::new(32, 16);
        let mut reconstructor = reconstructor(size);
        reconstructor
            .reconstruct_macroblock(0, 1, &predicted(2, 0), quantizer())
            .unwrap();
        reconstructor
            .reconstruct_macroblock(1, 1, &predicted(1, 1), quantizer())
            .unwrap();
        assert!(
            reconstructor
                .into_nv12_frame(1, None, None, format(size))
                .is_ok()
        );
    }

    #[test]
    fn failed_directional_prediction_leaves_macroblock_retryable() {
        let size = Size::new(16, 16);
        let mut reconstructor = reconstructor(size);
        assert!(
            reconstructor
                .reconstruct_macroblock(0, 1, &predicted(0, 2), quantizer())
                .is_err()
        );
        reconstructor
            .reconstruct_macroblock(0, 1, &predicted(2, 0), quantizer())
            .unwrap();
        assert!(
            reconstructor
                .into_nv12_frame(1, None, None, format(size))
                .is_ok()
        );
    }

    #[test]
    fn hides_references_across_slice_boundaries() {
        let size = Size::new(32, 16);
        let mut reconstructor = reconstructor(size);
        reconstructor
            .reconstruct_macroblock(0, 1, &predicted(2, 0), quantizer())
            .unwrap();
        assert!(
            reconstructor
                .reconstruct_macroblock(1, 2, &predicted(1, 1), quantizer())
                .is_err()
        );
        reconstructor
            .reconstruct_macroblock(1, 2, &predicted(2, 0), quantizer())
            .unwrap();
    }

    #[test]
    fn writes_pcm_and_rejects_incomplete_output() {
        let size = Size::new(32, 16);
        let mut incomplete = reconstructor(size);
        incomplete
            .reconstruct_macroblock(0, 1, &pcm(5, 10, 20), quantizer())
            .unwrap();
        assert!(
            incomplete
                .into_nv12_frame(1, None, None, format(size))
                .is_err()
        );

        let size = Size::new(16, 16);
        let mut complete = reconstructor(size);
        complete
            .reconstruct_macroblock(0, 1, &pcm(5, 10, 20), quantizer())
            .unwrap();
        let frame = complete
            .into_nv12_frame(1, None, None, format(size))
            .unwrap();
        let cpu = match frame.storage {
            FrameStorage::Cpu(cpu) => cpu,
            _ => panic!("expected CPU frame"),
        };
        assert!(cpu.planes[0].bytes[..256].iter().all(|&sample| sample == 5));
        assert_eq!(&cpu.planes[1].bytes[256..260], &[10, 20, 10, 20]);
    }
}
