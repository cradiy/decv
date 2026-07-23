//! P-macroblock assembly from reference prediction and spatial residuals.

use crate::{
    H264Error, ReconstructedInterResidual, ReconstructedLumaResidual, ResolvedPMacroblock, Result,
    Yuv420Picture,
};

const LUMA_BLOCK_COORDINATES: [(usize, usize); 16] = [
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

/// Reconstructs one progressive 8-bit 4:2:0 P macroblock using default
/// (unweighted) List-0 prediction.
///
/// All reference selection, prediction, geometry, and bounds checks complete
/// before the current picture is modified.
pub fn reconstruct_p_macroblock_420(
    current: &mut Yuv420Picture,
    references_l0: &[&Yuv420Picture],
    macroblock_x: usize,
    macroblock_y: usize,
    motion: &ResolvedPMacroblock,
    residual: &ReconstructedInterResidual,
) -> Result<()> {
    let luma_x = macroblock_x
        .checked_mul(16)
        .ok_or(H264Error::IntegerOverflow)?;
    let luma_y = macroblock_y
        .checked_mul(16)
        .ok_or(H264Error::IntegerOverflow)?;
    let (width, height) = current.dimensions();
    if luma_x.checked_add(16).is_none_or(|right| right > width)
        || luma_y.checked_add(16).is_none_or(|bottom| bottom > height)
    {
        return Err(H264Error::InvalidSyntax(
            "P macroblock lies outside the current picture",
        ));
    }

    let mut predicted_luma = [[0u8; 16]; 16];
    let mut predicted_cb = [[0u8; 8]; 8];
    let mut predicted_cr = [[0u8; 8]; 8];
    let mut covered = [[false; 4]; 4];
    for partition in &motion.partitions {
        let reference = references_l0
            .get(usize::from(partition.reference_index))
            .ok_or(H264Error::InvalidSyntax(
                "P partition reference index exceeds List 0",
            ))?;
        if reference.coded_size() != current.coded_size() {
            return Err(H264Error::InvalidSyntax(
                "P reference picture coded size does not match",
            ));
        }
        let prediction = reference.predict_inter_420(macroblock_x, macroblock_y, *partition)?;
        for y in 0..usize::from(partition.height) {
            let destination = &mut predicted_luma[usize::from(partition.y) + y]
                [usize::from(partition.x)..usize::from(partition.x + partition.width)];
            destination.copy_from_slice(&prediction.luma[y][..usize::from(partition.width)]);
        }
        for y in 0..usize::from(partition.height / 2) {
            let start = usize::from(partition.x / 2);
            let end = usize::from((partition.x + partition.width) / 2);
            predicted_cb[usize::from(partition.y / 2) + y][start..end]
                .copy_from_slice(&prediction.cb[y][..usize::from(partition.width / 2)]);
            predicted_cr[usize::from(partition.y / 2) + y][start..end]
                .copy_from_slice(&prediction.cr[y][..usize::from(partition.width / 2)]);
        }
        for y in (partition.y..partition.y + partition.height).step_by(4) {
            for x in (partition.x..partition.x + partition.width).step_by(4) {
                let cell = &mut covered[usize::from(y / 4)][usize::from(x / 4)];
                if *cell {
                    return Err(H264Error::InvalidSyntax("P prediction partitions overlap"));
                }
                *cell = true;
            }
        }
    }
    if covered.iter().flatten().any(|covered| !covered) {
        return Err(H264Error::InvalidSyntax(
            "P prediction partitions do not cover the macroblock",
        ));
    }

    let mut residual_luma = [[0i32; 16]; 16];
    match &residual.luma {
        ReconstructedLumaResidual::FourByFour(blocks) => {
            for (index, block) in blocks.iter().enumerate() {
                let (block_x, block_y) = LUMA_BLOCK_COORDINATES[index];
                copy_residual_block(&mut residual_luma, block_x * 4, block_y * 4, block);
            }
        }
        ReconstructedLumaResidual::EightByEight(blocks) => {
            for (index, block) in blocks.iter().enumerate() {
                copy_residual_block(&mut residual_luma, index % 2 * 8, index / 2 * 8, block);
            }
        }
    }
    let mut residual_cb = [[0i32; 8]; 8];
    let mut residual_cr = [[0i32; 8]; 8];
    for index in 0..4 {
        copy_residual_block(
            &mut residual_cb,
            index % 2 * 4,
            index / 2 * 4,
            &residual.chroma_cb[index],
        );
        copy_residual_block(
            &mut residual_cr,
            index % 2 * 4,
            index / 2 * 4,
            &residual.chroma_cr[index],
        );
    }

    let chroma_x = macroblock_x * 8;
    let chroma_y = macroblock_y * 8;
    let chroma_stride = width / 2;
    let (luma, cb, cr) = current.planes_mut();
    add_prediction_and_residual(luma, width, luma_x, luma_y, &predicted_luma, &residual_luma);
    add_prediction_and_residual(
        cb,
        chroma_stride,
        chroma_x,
        chroma_y,
        &predicted_cb,
        &residual_cb,
    );
    add_prediction_and_residual(
        cr,
        chroma_stride,
        chroma_x,
        chroma_y,
        &predicted_cr,
        &residual_cr,
    );
    Ok(())
}

fn copy_residual_block<const OUTPUT: usize, const BLOCK: usize>(
    output: &mut [[i32; OUTPUT]; OUTPUT],
    x: usize,
    y: usize,
    block: &[[i32; BLOCK]; BLOCK],
) {
    for row in 0..BLOCK {
        output[y + row][x..x + BLOCK].copy_from_slice(&block[row]);
    }
}

fn add_prediction_and_residual<const SIZE: usize>(
    plane: &mut [u8],
    stride: usize,
    x: usize,
    y: usize,
    prediction: &[[u8; SIZE]; SIZE],
    residual: &[[i32; SIZE]; SIZE],
) {
    for row in 0..SIZE {
        let output = &mut plane[(y + row) * stride + x..(y + row) * stride + x + SIZE];
        for column in 0..SIZE {
            output[column] =
                (i32::from(prediction[row][column]) + residual[row][column]).clamp(0, 255) as u8;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MotionVector, ResolvedPPartition};
    use decv_core::Size;

    fn picture(value: u8) -> Yuv420Picture {
        let mut picture = Yuv420Picture::new(Size {
            width: 16,
            height: 16,
        })
        .unwrap();
        let (luma, cb, cr) = picture.planes_mut();
        luma.fill(value);
        cb.fill(value + 1);
        cr.fill(value + 2);
        picture
    }

    fn zero_residual() -> ReconstructedInterResidual {
        ReconstructedInterResidual {
            luma: ReconstructedLumaResidual::FourByFour(Box::new([[[0; 4]; 4]; 16])),
            chroma_cb: [[[0; 4]; 4]; 4],
            chroma_cr: [[[0; 4]; 4]; 4],
        }
    }

    fn partition(x: u8, y: u8, width: u8, height: u8, reference_index: u8) -> ResolvedPPartition {
        ResolvedPPartition {
            x,
            y,
            width,
            height,
            reference_index,
            motion_vector: MotionVector::default(),
        }
    }

    #[test]
    fn reconstructs_prediction_plus_residual_with_clipping() {
        let reference = picture(40);
        let mut current = picture(0);
        let mut residual = zero_residual();
        let ReconstructedLumaResidual::FourByFour(blocks) = &mut residual.luma else {
            unreachable!()
        };
        blocks[0] = [[10; 4]; 4];
        blocks[1] = [[-50; 4]; 4];
        residual.chroma_cb[0] = [[220; 4]; 4];
        reconstruct_p_macroblock_420(
            &mut current,
            &[&reference],
            0,
            0,
            &ResolvedPMacroblock {
                skipped: false,
                partitions: vec![partition(0, 0, 16, 16, 0)],
            },
            &residual,
        )
        .unwrap();
        let (luma, cb, cr) = current.planes();
        assert_eq!((luma[0], luma[4], luma[8]), (50, 0, 40));
        assert_eq!((cb[0], cb[4], cr[0]), (255, 41, 42));
    }

    #[test]
    fn selects_reference_per_partition() {
        let first = picture(20);
        let second = picture(80);
        let mut current = picture(0);
        reconstruct_p_macroblock_420(
            &mut current,
            &[&first, &second],
            0,
            0,
            &ResolvedPMacroblock {
                skipped: false,
                partitions: vec![partition(0, 0, 8, 16, 0), partition(8, 0, 8, 16, 1)],
            },
            &zero_residual(),
        )
        .unwrap();
        let (luma, cb, _) = current.planes();
        assert_eq!((luma[0], luma[8], cb[0], cb[4]), (20, 80, 21, 81));
    }

    #[test]
    fn validation_failure_leaves_current_picture_unchanged() {
        let reference = picture(20);
        let mut current = picture(7);
        let before = current.clone();
        let result = reconstruct_p_macroblock_420(
            &mut current,
            &[&reference],
            0,
            0,
            &ResolvedPMacroblock {
                skipped: false,
                partitions: vec![partition(0, 0, 8, 16, 0)],
            },
            &zero_residual(),
        );
        assert!(matches!(result, Err(H264Error::InvalidSyntax(_))));
        assert_eq!(current, before);
    }
}
