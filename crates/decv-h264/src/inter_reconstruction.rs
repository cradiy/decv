//! P-macroblock assembly from reference prediction and spatial residuals.

use decv_core::Size;

use crate::inter_prediction::copy_fixed_row;
use crate::picture_surface::{MacroblockPixels, StagedMacroblockPixels};
use crate::{
    H264Error, InterPrediction420, PredictionWeight, PredictionWeightTable,
    ReconstructedInterResidual, ReconstructedLumaResidual, ResolvedBListMotion,
    ResolvedBMacroblock, ResolvedBPartition, ResolvedPMacroblock, ResolvedPPartition, Result,
    WeightOffset, Yuv420Picture,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImplicitWeightReference {
    pub picture_order_count: i32,
    pub long_term: bool,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum BPredictionWeightMode<'a> {
    Default,
    Explicit(&'a PredictionWeightTable),
    Implicit {
        current_picture_order_count: i32,
        list0: &'a [Option<ImplicitWeightReference>],
        list1: &'a [Option<ImplicitWeightReference>],
    },
}

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

type LumaResidualSamples = [[i32; 16]; 16];
type ChromaResidualSamples = [[i32; 8]; 8];
type MacroblockResidualSamples = (
    LumaResidualSamples,
    ChromaResidualSamples,
    ChromaResidualSamples,
);

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
    let references = references_l0.iter().copied().map(Some).collect::<Vec<_>>();
    reconstruct_p_macroblock_from_list_420(
        current,
        &references,
        macroblock_x,
        macroblock_y,
        motion,
        residual,
    )
}

/// Variant that preserves explicit "no reference picture" entries in an
/// active reference list.
pub fn reconstruct_p_macroblock_from_list_420(
    current: &mut Yuv420Picture,
    references_l0: &[Option<&Yuv420Picture>],
    macroblock_x: usize,
    macroblock_y: usize,
    motion: &ResolvedPMacroblock,
    residual: &ReconstructedInterResidual,
) -> Result<()> {
    reconstruct_p_macroblock_from_list_inner(
        current,
        references_l0,
        macroblock_x,
        macroblock_y,
        motion,
        Some(residual),
        None,
    )
}

pub fn reconstruct_weighted_p_macroblock_from_list_420(
    current: &mut Yuv420Picture,
    references_l0: &[Option<&Yuv420Picture>],
    macroblock_x: usize,
    macroblock_y: usize,
    motion: &ResolvedPMacroblock,
    residual: &ReconstructedInterResidual,
    weights: &PredictionWeightTable,
) -> Result<()> {
    reconstruct_p_macroblock_from_list_inner(
        current,
        references_l0,
        macroblock_x,
        macroblock_y,
        motion,
        Some(residual),
        Some(weights),
    )
}

pub(crate) fn reconstruct_p_skip_macroblock_from_list_420(
    current: &mut Yuv420Picture,
    references_l0: &[Option<&Yuv420Picture>],
    macroblock_x: usize,
    macroblock_y: usize,
    motion: &ResolvedPMacroblock,
) -> Result<()> {
    reconstruct_p_macroblock_from_list_inner(
        current,
        references_l0,
        macroblock_x,
        macroblock_y,
        motion,
        None,
        None,
    )
}

pub(crate) fn reconstruct_weighted_p_skip_macroblock_from_list_420(
    current: &mut Yuv420Picture,
    references_l0: &[Option<&Yuv420Picture>],
    macroblock_x: usize,
    macroblock_y: usize,
    motion: &ResolvedPMacroblock,
    weights: &PredictionWeightTable,
) -> Result<()> {
    reconstruct_p_macroblock_from_list_inner(
        current,
        references_l0,
        macroblock_x,
        macroblock_y,
        motion,
        None,
        Some(weights),
    )
}

/// Reconstructs one progressive 8-bit 4:2:0 B macroblock with default
/// unweighted List-0, List-1, or bidirectional prediction.
///
/// Bidirectional samples use the normative rounded average `(p0 + p1 + 1) >>
/// 1`. All prediction and coverage validation completes before the current
/// picture is modified.
pub fn reconstruct_b_macroblock_from_lists_420(
    current: &mut Yuv420Picture,
    references_l0: &[Option<&Yuv420Picture>],
    references_l1: &[Option<&Yuv420Picture>],
    macroblock_x: usize,
    macroblock_y: usize,
    motion: &ResolvedBMacroblock,
    residual: &ReconstructedInterResidual,
) -> Result<()> {
    reconstruct_b_macroblock_from_lists_with_mode(
        current,
        references_l0,
        references_l1,
        macroblock_x,
        macroblock_y,
        motion,
        Some(residual),
        BPredictionWeightMode::Default,
    )
}

/// Reconstructs one progressive B macroblock using the explicit
/// `pred_weight_table` carried by the slice header.
#[allow(clippy::too_many_arguments)]
pub fn reconstruct_weighted_b_macroblock_from_lists_420(
    current: &mut Yuv420Picture,
    references_l0: &[Option<&Yuv420Picture>],
    references_l1: &[Option<&Yuv420Picture>],
    macroblock_x: usize,
    macroblock_y: usize,
    motion: &ResolvedBMacroblock,
    residual: &ReconstructedInterResidual,
    weights: &PredictionWeightTable,
) -> Result<()> {
    reconstruct_b_macroblock_from_lists_with_mode(
        current,
        references_l0,
        references_l1,
        macroblock_x,
        macroblock_y,
        motion,
        Some(residual),
        BPredictionWeightMode::Explicit(weights),
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn reconstruct_b_macroblock_from_lists_with_mode(
    current: &mut Yuv420Picture,
    references_l0: &[Option<&Yuv420Picture>],
    references_l1: &[Option<&Yuv420Picture>],
    macroblock_x: usize,
    macroblock_y: usize,
    motion: &ResolvedBMacroblock,
    residual: Option<&ReconstructedInterResidual>,
    weight_mode: BPredictionWeightMode<'_>,
) -> Result<()> {
    let pixels = reconstruct_b_macroblock_pixels_from_lists(
        current.coded_size(),
        references_l0,
        references_l1,
        macroblock_x,
        macroblock_y,
        motion,
        residual,
        weight_mode,
    )?;
    let width_in_macroblocks =
        usize::try_from(current.coded_size().width / 16).map_err(|_| H264Error::IntegerOverflow)?;
    let address = macroblock_y
        .checked_mul(width_in_macroblocks)
        .and_then(|address| address.checked_add(macroblock_x))
        .ok_or(H264Error::IntegerOverflow)?;
    current.commit_macroblock_batch(&[StagedMacroblockPixels::new(address, pixels)])
}

/// Reconstructs one resolved B macroblock into owned pixels without mutating
/// decoder-visible picture state.
///
/// This is the worker-side boundary for deterministic batch reconstruction.
#[allow(clippy::too_many_arguments)]
pub(crate) fn reconstruct_b_macroblock_pixels_from_lists(
    current_size: Size,
    references_l0: &[Option<&Yuv420Picture>],
    references_l1: &[Option<&Yuv420Picture>],
    macroblock_x: usize,
    macroblock_y: usize,
    motion: &ResolvedBMacroblock,
    residual: Option<&ReconstructedInterResidual>,
    weight_mode: BPredictionWeightMode<'_>,
) -> Result<MacroblockPixels> {
    let mut prediction_l0 = InterPrediction420::empty();
    let mut prediction_l1 = InterPrediction420::empty();
    reconstruct_b_macroblock_pixels_from_lists_with_scratch(
        current_size,
        references_l0,
        references_l1,
        macroblock_x,
        macroblock_y,
        motion,
        residual,
        weight_mode,
        &mut prediction_l0,
        &mut prediction_l1,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn reconstruct_b_macroblock_pixels_from_lists_with_scratch(
    current_size: Size,
    references_l0: &[Option<&Yuv420Picture>],
    references_l1: &[Option<&Yuv420Picture>],
    macroblock_x: usize,
    macroblock_y: usize,
    motion: &ResolvedBMacroblock,
    residual: Option<&ReconstructedInterResidual>,
    weight_mode: BPredictionWeightMode<'_>,
    prediction_l0: &mut InterPrediction420,
    prediction_l1: &mut InterPrediction420,
) -> Result<MacroblockPixels> {
    let luma_x = macroblock_x
        .checked_mul(16)
        .ok_or(H264Error::IntegerOverflow)?;
    let luma_y = macroblock_y
        .checked_mul(16)
        .ok_or(H264Error::IntegerOverflow)?;
    let width = usize::try_from(current_size.width).map_err(|_| H264Error::IntegerOverflow)?;
    let height = usize::try_from(current_size.height).map_err(|_| H264Error::IntegerOverflow)?;
    if luma_x.checked_add(16).is_none_or(|right| right > width)
        || luma_y.checked_add(16).is_none_or(|bottom| bottom > height)
    {
        return Err(H264Error::InvalidSyntax(
            "B macroblock lies outside the current picture",
        ));
    }

    let mut predicted_luma = [[0u8; 16]; 16];
    let mut predicted_cb = [[0u8; 8]; 8];
    let mut predicted_cr = [[0u8; 8]; 8];
    let mut covered = [[false; 4]; 4];
    for partition in &motion.partitions {
        let has_l0 = predict_b_partition_list_into(
            references_l0,
            current_size,
            macroblock_x,
            macroblock_y,
            *partition,
            partition.list0,
            "B partition selects no reference picture in List 0",
            prediction_l0,
        )?;
        let has_l1 = predict_b_partition_list_into(
            references_l1,
            current_size,
            macroblock_x,
            macroblock_y,
            *partition,
            partition.list1,
            "B partition selects no reference picture in List 1",
            prediction_l1,
        )?;
        let prediction = merge_b_predictions(
            has_l0.then_some(&mut *prediction_l0),
            has_l1.then_some(&mut *prediction_l1),
            partition.list0.map(|motion| motion.reference_index),
            partition.list1.map(|motion| motion.reference_index),
            weight_mode,
        )?;
        for y in 0..usize::from(partition.height) {
            let destination_y = usize::from(partition.y) + y;
            // SAFETY: prediction validation guarantees a 4/8/16-byte luma
            // partition within this macroblock and both fixed rows.
            unsafe {
                copy_fixed_row(
                    predicted_luma[destination_y]
                        .as_mut_ptr()
                        .add(usize::from(partition.x)),
                    prediction.luma[y].as_ptr(),
                    usize::from(partition.width),
                );
            }
        }
        for y in 0..usize::from(partition.height / 2) {
            let start = usize::from(partition.x / 2);
            let destination_y = usize::from(partition.y / 2) + y;
            let width = usize::from(partition.width / 2);
            // SAFETY: prediction validation guarantees a 2/4/8-byte chroma
            // partition within these fixed eight-byte rows.
            unsafe {
                copy_fixed_row(
                    predicted_cb[destination_y].as_mut_ptr().add(start),
                    prediction.cb[y].as_ptr(),
                    width,
                );
                copy_fixed_row(
                    predicted_cr[destination_y].as_mut_ptr().add(start),
                    prediction.cr[y].as_ptr(),
                    width,
                );
            }
        }
        for y in (partition.y..partition.y + partition.height).step_by(4) {
            for x in (partition.x..partition.x + partition.width).step_by(4) {
                let cell = covered
                    .get_mut(usize::from(y / 4))
                    .and_then(|row| row.get_mut(usize::from(x / 4)))
                    .ok_or(H264Error::InvalidSyntax(
                        "B prediction partition exceeds the macroblock",
                    ))?;
                if *cell {
                    return Err(H264Error::InvalidSyntax("B prediction partitions overlap"));
                }
                *cell = true;
            }
        }
    }
    if covered.iter().flatten().any(|covered| !covered) {
        return Err(H264Error::InvalidSyntax(
            "B prediction partitions do not cover the macroblock",
        ));
    }

    if let Some(residual) = residual {
        add_inter_residual_to_prediction(
            &mut predicted_luma,
            &mut predicted_cb,
            &mut predicted_cr,
            residual,
        );
    }
    Ok(MacroblockPixels::new(
        predicted_luma,
        predicted_cb,
        predicted_cr,
    ))
}

#[allow(clippy::too_many_arguments)]
fn predict_b_partition_list_into(
    references: &[Option<&Yuv420Picture>],
    expected_size: decv_core::Size,
    macroblock_x: usize,
    macroblock_y: usize,
    partition: ResolvedBPartition,
    list_motion: Option<ResolvedBListMotion>,
    missing_reference: &'static str,
    prediction: &mut InterPrediction420,
) -> Result<bool> {
    let Some(list_motion) = list_motion else {
        return Ok(false);
    };
    let reference = references
        .get(usize::from(list_motion.reference_index))
        .copied()
        .flatten()
        .ok_or(H264Error::InvalidSyntax(missing_reference))?;
    if reference.coded_size() != expected_size {
        return Err(H264Error::InvalidSyntax(
            "B reference picture coded size does not match",
        ));
    }
    reference.predict_inter_420_into(
        macroblock_x,
        macroblock_y,
        ResolvedPPartition {
            x: partition.x,
            y: partition.y,
            width: partition.width,
            height: partition.height,
            reference_index: list_motion.reference_index,
            motion_vector: list_motion.motion_vector,
        },
        prediction,
    )?;
    Ok(true)
}

fn merge_b_predictions<'a>(
    list0: Option<&'a mut InterPrediction420>,
    list1: Option<&'a mut InterPrediction420>,
    reference_index_l0: Option<u8>,
    reference_index_l1: Option<u8>,
    weight_mode: BPredictionWeightMode<'_>,
) -> Result<&'a InterPrediction420> {
    match (list0, list1) {
        (Some(prediction), None) => {
            if let BPredictionWeightMode::Explicit(weights) = weight_mode {
                apply_prediction_weights_for_list(
                    prediction,
                    reference_index_l0.ok_or(H264Error::InvalidSyntax(
                        "List-0 B prediction is missing its reference index",
                    ))?,
                    weights,
                    false,
                )?;
            }
            Ok(prediction)
        }
        (None, Some(prediction)) => {
            if let BPredictionWeightMode::Explicit(weights) = weight_mode {
                apply_prediction_weights_for_list(
                    prediction,
                    reference_index_l1.ok_or(H264Error::InvalidSyntax(
                        "List-1 B prediction is missing its reference index",
                    ))?,
                    weights,
                    true,
                )?;
            }
            Ok(prediction)
        }
        (Some(list0), Some(list1)) => {
            if list0.width != list1.width || list0.height != list1.height {
                return Err(H264Error::InvalidSyntax(
                    "bidirectional prediction dimensions do not match",
                ));
            }
            let reference_index_l0 = reference_index_l0.ok_or(H264Error::InvalidSyntax(
                "bidirectional List-0 prediction is missing its reference index",
            ))?;
            let reference_index_l1 = reference_index_l1.ok_or(H264Error::InvalidSyntax(
                "bidirectional List-1 prediction is missing its reference index",
            ))?;
            match weight_mode {
                BPredictionWeightMode::Explicit(weights) => {
                    apply_explicit_bipred_weights(
                        list0,
                        list1,
                        reference_index_l0,
                        reference_index_l1,
                        weights,
                    )?;
                    return Ok(list0);
                }
                BPredictionWeightMode::Implicit {
                    current_picture_order_count,
                    list0: implicit_l0,
                    list1: implicit_l1,
                } => {
                    let reference_l0 =
                        implicit_weight_reference(implicit_l0, reference_index_l0, "List 0")?;
                    let reference_l1 =
                        implicit_weight_reference(implicit_l1, reference_index_l1, "List 1")?;
                    apply_implicit_bipred_weights(
                        list0,
                        list1,
                        current_picture_order_count,
                        reference_l0,
                        reference_l1,
                    );
                    return Ok(list0);
                }
                BPredictionWeightMode::Default => {}
            }
            for y in 0..usize::from(list0.height) {
                for x in 0..usize::from(list0.width) {
                    list0.luma[y][x] = rounded_average(list0.luma[y][x], list1.luma[y][x]);
                }
            }
            for y in 0..usize::from(list0.height / 2) {
                for x in 0..usize::from(list0.width / 2) {
                    list0.cb[y][x] = rounded_average(list0.cb[y][x], list1.cb[y][x]);
                    list0.cr[y][x] = rounded_average(list0.cr[y][x], list1.cr[y][x]);
                }
            }
            Ok(list0)
        }
        (None, None) => Err(H264Error::InvalidSyntax(
            "B partition uses neither reference list",
        )),
    }
}

#[inline]
fn rounded_average(left: u8, right: u8) -> u8 {
    ((u16::from(left) + u16::from(right) + 1) >> 1) as u8
}

fn reconstruct_p_macroblock_from_list_inner(
    current: &mut Yuv420Picture,
    references_l0: &[Option<&Yuv420Picture>],
    macroblock_x: usize,
    macroblock_y: usize,
    motion: &ResolvedPMacroblock,
    residual: Option<&ReconstructedInterResidual>,
    weights: Option<&PredictionWeightTable>,
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
    let mut prediction = InterPrediction420::empty();
    for partition in &motion.partitions {
        let reference = references_l0
            .get(usize::from(partition.reference_index))
            .copied()
            .flatten()
            .ok_or(H264Error::InvalidSyntax(
                "P partition selects no reference picture in List 0",
            ))?;
        if reference.coded_size() != current.coded_size() {
            return Err(H264Error::InvalidSyntax(
                "P reference picture coded size does not match",
            ));
        }
        reference.predict_inter_420_into(
            macroblock_x,
            macroblock_y,
            *partition,
            &mut prediction,
        )?;
        if let Some(weights) = weights {
            apply_prediction_weights(&mut prediction, partition.reference_index, weights)?;
        }
        for y in 0..usize::from(partition.height) {
            let destination_y = usize::from(partition.y) + y;
            // SAFETY: P-partition validation guarantees a 4/8/16-byte luma
            // region within both fixed rows.
            unsafe {
                copy_fixed_row(
                    predicted_luma[destination_y]
                        .as_mut_ptr()
                        .add(usize::from(partition.x)),
                    prediction.luma[y].as_ptr(),
                    usize::from(partition.width),
                );
            }
        }
        for y in 0..usize::from(partition.height / 2) {
            let start = usize::from(partition.x / 2);
            let destination_y = usize::from(partition.y / 2) + y;
            let width = usize::from(partition.width / 2);
            // SAFETY: P-partition validation guarantees a 2/4/8-byte chroma
            // region within both fixed rows.
            unsafe {
                copy_fixed_row(
                    predicted_cb[destination_y].as_mut_ptr().add(start),
                    prediction.cb[y].as_ptr(),
                    width,
                );
                copy_fixed_row(
                    predicted_cr[destination_y].as_mut_ptr().add(start),
                    prediction.cr[y].as_ptr(),
                    width,
                );
            }
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

    let (residual_luma, residual_cb, residual_cr) = residual.map_or_else(
        || ([[0; 16]; 16], [[0; 8]; 8], [[0; 8]; 8]),
        assemble_residual,
    );

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

fn assemble_residual(residual: &ReconstructedInterResidual) -> MacroblockResidualSamples {
    let mut residual_luma = [[0i32; 16]; 16];
    let mut residual_cb = [[0i32; 8]; 8];
    let mut residual_cr = [[0i32; 8]; 8];
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
    (residual_luma, residual_cb, residual_cr)
}

fn add_inter_residual_to_prediction(
    luma: &mut [[u8; 16]; 16],
    cb: &mut [[u8; 8]; 8],
    cr: &mut [[u8; 8]; 8],
    residual: &ReconstructedInterResidual,
) {
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: SSE2 is part of the x86_64 baseline. All pointers come from
        // fixed-size prediction and residual matrices, and the helper visits
        // only complete normative 4x4 or 8x8 block rows.
        unsafe {
            add_inter_residual_to_prediction_sse2(luma, cb, cr, residual);
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        match &residual.luma {
            ReconstructedLumaResidual::FourByFour(blocks) => {
                for (index, block) in blocks.iter().enumerate() {
                    let (block_x, block_y) = LUMA_BLOCK_COORDINATES[index];
                    add_residual_block(luma, block_x * 4, block_y * 4, block);
                }
            }
            ReconstructedLumaResidual::EightByEight(blocks) => {
                for (index, block) in blocks.iter().enumerate() {
                    add_residual_block(luma, index % 2 * 8, index / 2 * 8, block);
                }
            }
        }
        for index in 0..4 {
            add_residual_block(cb, index % 2 * 4, index / 2 * 4, &residual.chroma_cb[index]);
            add_residual_block(cr, index % 2 * 4, index / 2 * 4, &residual.chroma_cr[index]);
        }
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn add_residual_block<const OUTPUT: usize, const BLOCK: usize>(
    prediction: &mut [[u8; OUTPUT]; OUTPUT],
    x: usize,
    y: usize,
    residual: &[[i32; BLOCK]; BLOCK],
) {
    for row in 0..BLOCK {
        for column in 0..BLOCK {
            prediction[y + row][x + column] = i32::from(prediction[y + row][x + column])
                .saturating_add(residual[row][column])
                .clamp(0, 255) as u8;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn add_inter_residual_to_prediction_sse2(
    luma: &mut [[u8; 16]; 16],
    cb: &mut [[u8; 8]; 8],
    cr: &mut [[u8; 8]; 8],
    residual: &ReconstructedInterResidual,
) {
    use std::arch::x86_64::{
        __m128i, _mm_adds_epi16, _mm_loadl_epi64, _mm_loadu_si128, _mm_packs_epi32,
        _mm_packus_epi16, _mm_setzero_si128, _mm_storel_epi64, _mm_storeu_si128, _mm_unpackhi_epi8,
        _mm_unpacklo_epi8,
    };

    const BLOCKS_4X4_BY_ROW: [[usize; 4]; 4] =
        [[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]];

    let zero = _mm_setzero_si128();
    match &residual.luma {
        ReconstructedLumaResidual::FourByFour(blocks) => {
            for (y, prediction_row) in luma.iter_mut().enumerate() {
                let block_row = BLOCKS_4X4_BY_ROW[y / 4];
                let row = y % 4;
                // SAFETY: Every selected residual row has four i32 values and
                // every prediction row has sixteen u8 values.
                unsafe {
                    let prediction = _mm_loadu_si128(prediction_row.as_ptr().cast::<__m128i>());
                    let residual_low = _mm_packs_epi32(
                        _mm_loadu_si128(blocks[block_row[0]][row].as_ptr().cast::<__m128i>()),
                        _mm_loadu_si128(blocks[block_row[1]][row].as_ptr().cast::<__m128i>()),
                    );
                    let residual_high = _mm_packs_epi32(
                        _mm_loadu_si128(blocks[block_row[2]][row].as_ptr().cast::<__m128i>()),
                        _mm_loadu_si128(blocks[block_row[3]][row].as_ptr().cast::<__m128i>()),
                    );
                    let low = _mm_adds_epi16(_mm_unpacklo_epi8(prediction, zero), residual_low);
                    let high = _mm_adds_epi16(_mm_unpackhi_epi8(prediction, zero), residual_high);
                    _mm_storeu_si128(
                        prediction_row.as_mut_ptr().cast::<__m128i>(),
                        _mm_packus_epi16(low, high),
                    );
                }
            }
        }
        ReconstructedLumaResidual::EightByEight(blocks) => {
            for (y, prediction_row) in luma.iter_mut().enumerate() {
                let first = (y / 8) * 2;
                let row = y % 8;
                // SAFETY: Each residual row has eight i32 values and each
                // prediction row has sixteen u8 values.
                unsafe {
                    let prediction = _mm_loadu_si128(prediction_row.as_ptr().cast::<__m128i>());
                    let residual_low = _mm_packs_epi32(
                        _mm_loadu_si128(blocks[first][row].as_ptr().cast::<__m128i>()),
                        _mm_loadu_si128(blocks[first][row].as_ptr().add(4).cast::<__m128i>()),
                    );
                    let residual_high = _mm_packs_epi32(
                        _mm_loadu_si128(blocks[first + 1][row].as_ptr().cast::<__m128i>()),
                        _mm_loadu_si128(blocks[first + 1][row].as_ptr().add(4).cast::<__m128i>()),
                    );
                    let low = _mm_adds_epi16(_mm_unpacklo_epi8(prediction, zero), residual_low);
                    let high = _mm_adds_epi16(_mm_unpackhi_epi8(prediction, zero), residual_high);
                    _mm_storeu_si128(
                        prediction_row.as_mut_ptr().cast::<__m128i>(),
                        _mm_packus_epi16(low, high),
                    );
                }
            }
        }
    }

    for (prediction, blocks) in [(cb, &residual.chroma_cb), (cr, &residual.chroma_cr)] {
        for (y, prediction_row) in prediction.iter_mut().enumerate() {
            let first = (y / 4) * 2;
            let row = y % 4;
            // SAFETY: Each selected residual row has four i32 values and each
            // prediction row has eight u8 values.
            unsafe {
                let packed_prediction = _mm_loadl_epi64(prediction_row.as_ptr().cast::<__m128i>());
                let packed_residual = _mm_packs_epi32(
                    _mm_loadu_si128(blocks[first][row].as_ptr().cast::<__m128i>()),
                    _mm_loadu_si128(blocks[first + 1][row].as_ptr().cast::<__m128i>()),
                );
                let sum =
                    _mm_adds_epi16(_mm_unpacklo_epi8(packed_prediction, zero), packed_residual);
                _mm_storel_epi64(
                    prediction_row.as_mut_ptr().cast::<__m128i>(),
                    _mm_packus_epi16(sum, zero),
                );
            }
        }
    }
}

fn apply_prediction_weights(
    prediction: &mut InterPrediction420,
    reference_index: u8,
    table: &PredictionWeightTable,
) -> Result<()> {
    apply_prediction_weights_for_list(prediction, reference_index, table, false)
}

fn apply_prediction_weights_for_list(
    prediction: &mut InterPrediction420,
    reference_index: u8,
    table: &PredictionWeightTable,
    list1: bool,
) -> Result<()> {
    let weights = prediction_weight(table, list1, reference_index)?;
    let luma_default = 1i32 << table.luma_log2_weight_denom;
    let luma = weights.luma.unwrap_or(WeightOffset {
        weight: luma_default,
        offset: 0,
    });
    let luma_width = usize::from(prediction.width);
    for row in prediction
        .luma
        .iter_mut()
        .take(usize::from(prediction.height))
    {
        weighted_row(&mut row[..luma_width], luma, table.luma_log2_weight_denom);
    }

    let chroma_default = 1i32 << table.chroma_log2_weight_denom;
    let chroma = weights.chroma.unwrap_or([
        WeightOffset {
            weight: chroma_default,
            offset: 0,
        },
        WeightOffset {
            weight: chroma_default,
            offset: 0,
        },
    ]);
    let chroma_height = usize::from(prediction.height / 2);
    let chroma_width = usize::from(prediction.width / 2);
    for row in prediction.cb.iter_mut().take(chroma_height) {
        weighted_row(
            &mut row[..chroma_width],
            chroma[0],
            table.chroma_log2_weight_denom,
        );
    }
    for row in prediction.cr.iter_mut().take(chroma_height) {
        weighted_row(
            &mut row[..chroma_width],
            chroma[1],
            table.chroma_log2_weight_denom,
        );
    }
    Ok(())
}

#[inline]
fn weighted_row(samples: &mut [u8], weight: WeightOffset, denominator: u8) {
    #[cfg(target_arch = "x86_64")]
    let offset = if (-128..=128).contains(&weight.weight)
        && (-128..=127).contains(&weight.offset)
        && denominator <= 7
    {
        // SAFETY: SSE2 is part of the x86_64 baseline. The guarded H.264
        // weight, offset, and denominator ranges keep every intermediate in
        // i16, and the helper only loads and stores complete chunks.
        unsafe {
            weighted_row_sse2(
                samples,
                weight.weight,
                weight.offset,
                u32::from(denominator),
            )
        }
    } else {
        0
    };
    #[cfg(not(target_arch = "x86_64"))]
    let offset = 0;
    for sample in &mut samples[offset..] {
        *sample = weighted_sample(*sample, weight, denominator);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn weighted_row_sse2(
    samples: &mut [u8],
    weight: i32,
    value_offset: i32,
    denominator: u32,
) -> usize {
    use std::arch::x86_64::{
        __m128i, _mm_add_epi16, _mm_cvtsi32_si128, _mm_loadl_epi64, _mm_loadu_si128,
        _mm_mullo_epi16, _mm_packus_epi16, _mm_set1_epi16, _mm_setzero_si128, _mm_sra_epi16,
        _mm_storel_epi64, _mm_storeu_si128, _mm_unpackhi_epi8, _mm_unpacklo_epi8,
    };

    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn apply(
        samples: __m128i,
        weight: __m128i,
        rounding: __m128i,
        shift: __m128i,
        value_offset: __m128i,
    ) -> __m128i {
        let weighted = _mm_add_epi16(_mm_mullo_epi16(samples, weight), rounding);
        _mm_add_epi16(_mm_sra_epi16(weighted, shift), value_offset)
    }

    let zero = _mm_setzero_si128();
    let weight = _mm_set1_epi16(weight as i16);
    let rounding = _mm_set1_epi16(if denominator == 0 {
        0
    } else {
        1i16 << (denominator - 1)
    });
    let shift = _mm_cvtsi32_si128(denominator as i32);
    let value_offset = _mm_set1_epi16(value_offset as i16);
    let mut offset = 0;
    while samples.len() - offset >= 16 {
        // SAFETY: The loop condition proves the 16-byte range is in-bounds.
        let packed = unsafe { _mm_loadu_si128(samples.as_ptr().add(offset).cast::<__m128i>()) };
        // SAFETY: This function and the helper are compiled with SSE2 enabled.
        let low = unsafe {
            apply(
                _mm_unpacklo_epi8(packed, zero),
                weight,
                rounding,
                shift,
                value_offset,
            )
        };
        // SAFETY: This function and the helper are compiled with SSE2 enabled.
        let high = unsafe {
            apply(
                _mm_unpackhi_epi8(packed, zero),
                weight,
                rounding,
                shift,
                value_offset,
            )
        };
        // SAFETY: The loop condition proves the destination is in-bounds.
        unsafe {
            _mm_storeu_si128(
                samples.as_mut_ptr().add(offset).cast::<__m128i>(),
                _mm_packus_epi16(low, high),
            );
        }
        offset += 16;
    }
    if samples.len() - offset >= 8 {
        // SAFETY: The branch proves the eight-byte range is in-bounds.
        let packed = unsafe { _mm_loadl_epi64(samples.as_ptr().add(offset).cast::<__m128i>()) };
        // SAFETY: This function and the helper are compiled with SSE2 enabled.
        let low = unsafe {
            apply(
                _mm_unpacklo_epi8(packed, zero),
                weight,
                rounding,
                shift,
                value_offset,
            )
        };
        // SAFETY: The branch proves the destination is in-bounds.
        unsafe {
            _mm_storel_epi64(
                samples.as_mut_ptr().add(offset).cast::<__m128i>(),
                _mm_packus_epi16(low, zero),
            );
        }
        offset += 8;
    }
    offset
}

fn apply_explicit_bipred_weights(
    list0: &mut InterPrediction420,
    list1: &InterPrediction420,
    reference_index_l0: u8,
    reference_index_l1: u8,
    table: &PredictionWeightTable,
) -> Result<()> {
    let weights_l0 = prediction_weight(table, false, reference_index_l0)?;
    let weights_l1 = prediction_weight(table, true, reference_index_l1)?;
    let default_luma = 1i32 << table.luma_log2_weight_denom;
    let luma_l0 = weights_l0.luma.unwrap_or(WeightOffset {
        weight: default_luma,
        offset: 0,
    });
    let luma_l1 = weights_l1.luma.unwrap_or(WeightOffset {
        weight: default_luma,
        offset: 0,
    });
    for y in 0..usize::from(list0.height) {
        for x in 0..usize::from(list0.width) {
            list0.luma[y][x] = weighted_bipred_sample(
                list0.luma[y][x],
                list1.luma[y][x],
                luma_l0,
                luma_l1,
                table.luma_log2_weight_denom,
            );
        }
    }

    let default_chroma = 1i32 << table.chroma_log2_weight_denom;
    let default_chroma_weights = [
        WeightOffset {
            weight: default_chroma,
            offset: 0,
        },
        WeightOffset {
            weight: default_chroma,
            offset: 0,
        },
    ];
    let chroma_l0 = weights_l0.chroma.unwrap_or(default_chroma_weights);
    let chroma_l1 = weights_l1.chroma.unwrap_or(default_chroma_weights);
    for y in 0..usize::from(list0.height / 2) {
        for x in 0..usize::from(list0.width / 2) {
            list0.cb[y][x] = weighted_bipred_sample(
                list0.cb[y][x],
                list1.cb[y][x],
                chroma_l0[0],
                chroma_l1[0],
                table.chroma_log2_weight_denom,
            );
            list0.cr[y][x] = weighted_bipred_sample(
                list0.cr[y][x],
                list1.cr[y][x],
                chroma_l0[1],
                chroma_l1[1],
                table.chroma_log2_weight_denom,
            );
        }
    }
    Ok(())
}

fn implicit_weight_reference(
    references: &[Option<ImplicitWeightReference>],
    reference_index: u8,
    list_name: &'static str,
) -> Result<ImplicitWeightReference> {
    references
        .get(usize::from(reference_index))
        .copied()
        .flatten()
        .ok_or(H264Error::InvalidSyntax(match list_name {
            "List 0" => "implicit weighting List-0 index has no reference metadata",
            _ => "implicit weighting List-1 index has no reference metadata",
        }))
}

fn apply_implicit_bipred_weights(
    list0: &mut InterPrediction420,
    list1: &InterPrediction420,
    current_picture_order_count: i32,
    reference_l0: ImplicitWeightReference,
    reference_l1: ImplicitWeightReference,
) {
    let (weight_l0, weight_l1) =
        derive_implicit_bipred_weights(current_picture_order_count, reference_l0, reference_l1);
    let weight_l0 = WeightOffset {
        weight: weight_l0,
        offset: 0,
    };
    let weight_l1 = WeightOffset {
        weight: weight_l1,
        offset: 0,
    };
    let luma_width = usize::from(list0.width);
    for y in 0..usize::from(list0.height) {
        implicit_bipred_row(
            &mut list0.luma[y][..luma_width],
            &list1.luma[y][..luma_width],
            weight_l0,
            weight_l1,
        );
    }
    let chroma_width = usize::from(list0.width / 2);
    for y in 0..usize::from(list0.height / 2) {
        implicit_bipred_row(
            &mut list0.cb[y][..chroma_width],
            &list1.cb[y][..chroma_width],
            weight_l0,
            weight_l1,
        );
        implicit_bipred_row(
            &mut list0.cr[y][..chroma_width],
            &list1.cr[y][..chroma_width],
            weight_l0,
            weight_l1,
        );
    }
}

#[inline]
fn implicit_bipred_row(
    list0: &mut [u8],
    list1: &[u8],
    weight_l0: WeightOffset,
    weight_l1: WeightOffset,
) {
    debug_assert_eq!(list0.len(), list1.len());
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: SSE2 is part of the x86_64 baseline. Both slices have equal
        // lengths, and the helper only loads/stores complete 8- or 16-byte
        // chunks before returning the scalar tail offset.
        let offset =
            unsafe { implicit_bipred_row_sse2(list0, list1, weight_l0.weight, weight_l1.weight) };
        for index in offset..list0.len() {
            list0[index] =
                weighted_bipred_sample(list0[index], list1[index], weight_l0, weight_l1, 5);
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    for (sample_l0, &sample_l1) in list0.iter_mut().zip(list1) {
        *sample_l0 = weighted_bipred_sample(*sample_l0, sample_l1, weight_l0, weight_l1, 5);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn implicit_bipred_row_sse2(
    list0: &mut [u8],
    list1: &[u8],
    weight_l0: i32,
    weight_l1: i32,
) -> usize {
    use std::arch::x86_64::{
        __m128i, _mm_add_epi16, _mm_loadl_epi64, _mm_loadu_si128, _mm_mullo_epi16,
        _mm_packus_epi16, _mm_set1_epi16, _mm_setzero_si128, _mm_srai_epi16, _mm_storel_epi64,
        _mm_storeu_si128, _mm_unpackhi_epi8, _mm_unpacklo_epi8,
    };

    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn merge(
        list0: __m128i,
        list1: __m128i,
        weight_l0: __m128i,
        weight_l1: __m128i,
        rounding: __m128i,
    ) -> __m128i {
        let weighted = _mm_add_epi16(
            _mm_mullo_epi16(list0, weight_l0),
            _mm_mullo_epi16(list1, weight_l1),
        );
        _mm_srai_epi16::<6>(_mm_add_epi16(weighted, rounding))
    }

    let zero = _mm_setzero_si128();
    let weight_l0 = _mm_set1_epi16(weight_l0 as i16);
    let weight_l1 = _mm_set1_epi16(weight_l1 as i16);
    let rounding = _mm_set1_epi16(32);
    let mut offset = 0usize;
    while list0.len() - offset >= 16 {
        // SAFETY: The loop condition proves both 16-byte ranges are in-bounds.
        let samples_l0 = unsafe { _mm_loadu_si128(list0.as_ptr().add(offset).cast::<__m128i>()) };
        // SAFETY: `list1` has the same length as `list0`.
        let samples_l1 = unsafe { _mm_loadu_si128(list1.as_ptr().add(offset).cast::<__m128i>()) };
        // SAFETY: This function is compiled with SSE2 enabled.
        let low = unsafe {
            merge(
                _mm_unpacklo_epi8(samples_l0, zero),
                _mm_unpacklo_epi8(samples_l1, zero),
                weight_l0,
                weight_l1,
                rounding,
            )
        };
        // SAFETY: This function is compiled with SSE2 enabled.
        let high = unsafe {
            merge(
                _mm_unpackhi_epi8(samples_l0, zero),
                _mm_unpackhi_epi8(samples_l1, zero),
                weight_l0,
                weight_l1,
                rounding,
            )
        };
        // SAFETY: The loop condition proves the 16-byte destination is valid.
        unsafe {
            _mm_storeu_si128(
                list0.as_mut_ptr().add(offset).cast::<__m128i>(),
                _mm_packus_epi16(low, high),
            );
        }
        offset += 16;
    }
    if list0.len() - offset >= 8 {
        // SAFETY: The branch proves both 8-byte ranges are in-bounds.
        let samples_l0 = unsafe { _mm_loadl_epi64(list0.as_ptr().add(offset).cast::<__m128i>()) };
        // SAFETY: `list1` has the same length as `list0`.
        let samples_l1 = unsafe { _mm_loadl_epi64(list1.as_ptr().add(offset).cast::<__m128i>()) };
        // SAFETY: This function is compiled with SSE2 enabled.
        let low = unsafe {
            merge(
                _mm_unpacklo_epi8(samples_l0, zero),
                _mm_unpacklo_epi8(samples_l1, zero),
                weight_l0,
                weight_l1,
                rounding,
            )
        };
        // SAFETY: The branch proves the 8-byte destination is valid.
        unsafe {
            _mm_storel_epi64(
                list0.as_mut_ptr().add(offset).cast::<__m128i>(),
                _mm_packus_epi16(low, zero),
            );
        }
        offset += 8;
    }
    offset
}

pub fn derive_implicit_bipred_weights(
    current_picture_order_count: i32,
    reference_l0: ImplicitWeightReference,
    reference_l1: ImplicitWeightReference,
) -> (i32, i32) {
    if reference_l0.long_term || reference_l1.long_term {
        return (32, 32);
    }
    let td = (i64::from(reference_l1.picture_order_count)
        - i64::from(reference_l0.picture_order_count))
    .clamp(-128, 127);
    if td == 0 {
        return (32, 32);
    }
    let tb = (i64::from(current_picture_order_count) - i64::from(reference_l0.picture_order_count))
        .clamp(-128, 127);
    let tx = (16_384 + td.abs() / 2) / td;
    let weight_l1 = (tb * tx + 32) >> 8;
    if !(-64..=128).contains(&weight_l1) {
        return (32, 32);
    }
    let weight_l1 = weight_l1 as i32;
    (64 - weight_l1, weight_l1)
}

fn prediction_weight(
    table: &PredictionWeightTable,
    list1: bool,
    reference_index: u8,
) -> Result<&PredictionWeight> {
    let list = if list1 { &table.list1 } else { &table.list0 };
    list.get(usize::from(reference_index))
        .ok_or(H264Error::InvalidSyntax(if list1 {
            "weighted partition List-1 index exceeds pred_weight_table"
        } else {
            "weighted partition List-0 index exceeds pred_weight_table"
        }))
}

#[inline]
fn weighted_sample(sample: u8, weight: WeightOffset, denominator: u8) -> u8 {
    let rounding = if denominator == 0 {
        0
    } else {
        1 << (denominator - 1)
    };
    (((weight.weight * i32::from(sample) + rounding) >> denominator) + weight.offset).clamp(0, 255)
        as u8
}

#[inline]
fn weighted_bipred_sample(
    sample_l0: u8,
    sample_l1: u8,
    weight_l0: WeightOffset,
    weight_l1: WeightOffset,
    denominator: u8,
) -> u8 {
    let rounding = 1i32 << denominator;
    let offset = (weight_l0.offset + weight_l1.offset + 1) >> 1;
    ((weight_l0.weight * i32::from(sample_l0) + weight_l1.weight * i32::from(sample_l1) + rounding)
        >> (denominator + 1))
        .saturating_add(offset)
        .clamp(0, 255) as u8
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
    #[cfg(target_arch = "x86_64")]
    if SIZE == 8 || SIZE == 16 {
        // SAFETY: SSE2 is part of the x86_64 baseline. The caller-provided
        // plane ranges were validated at the macroblock level, and the helper
        // only reads fixed-size rows from the two square input matrices.
        unsafe {
            add_prediction_and_residual_sse2(plane, stride, x, y, prediction, residual);
        }
        return;
    }
    for row in 0..SIZE {
        let output = &mut plane[(y + row) * stride + x..(y + row) * stride + x + SIZE];
        for column in 0..SIZE {
            output[column] = i32::from(prediction[row][column])
                .saturating_add(residual[row][column])
                .clamp(0, 255) as u8;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn add_prediction_and_residual_sse2<const SIZE: usize>(
    plane: &mut [u8],
    stride: usize,
    x: usize,
    y: usize,
    prediction: &[[u8; SIZE]; SIZE],
    residual: &[[i32; SIZE]; SIZE],
) {
    use std::arch::x86_64::{
        __m128i, _mm_adds_epi16, _mm_loadl_epi64, _mm_loadu_si128, _mm_packs_epi32,
        _mm_packus_epi16, _mm_setzero_si128, _mm_storel_epi64, _mm_storeu_si128, _mm_unpackhi_epi8,
        _mm_unpacklo_epi8,
    };

    let zero = _mm_setzero_si128();
    for row in 0..SIZE {
        let prediction_ptr = prediction[row].as_ptr();
        let residual_ptr = residual[row].as_ptr();
        let output_ptr = plane.as_mut_ptr().wrapping_add((y + row) * stride + x);
        if SIZE == 16 {
            // SAFETY: A 16-wide matrix row contains every loaded element, and
            // macroblock validation proves the 16-byte output row is valid.
            unsafe {
                let predicted = _mm_loadu_si128(prediction_ptr.cast::<__m128i>());
                let residual_0 = _mm_loadu_si128(residual_ptr.cast::<__m128i>());
                let residual_1 = _mm_loadu_si128(residual_ptr.add(4).cast::<__m128i>());
                let residual_2 = _mm_loadu_si128(residual_ptr.add(8).cast::<__m128i>());
                let residual_3 = _mm_loadu_si128(residual_ptr.add(12).cast::<__m128i>());
                let low = _mm_adds_epi16(
                    _mm_unpacklo_epi8(predicted, zero),
                    _mm_packs_epi32(residual_0, residual_1),
                );
                let high = _mm_adds_epi16(
                    _mm_unpackhi_epi8(predicted, zero),
                    _mm_packs_epi32(residual_2, residual_3),
                );
                _mm_storeu_si128(output_ptr.cast::<__m128i>(), _mm_packus_epi16(low, high));
            }
        } else {
            debug_assert_eq!(SIZE, 8);
            // SAFETY: An 8-wide matrix row contains every loaded element, and
            // macroblock validation proves the 8-byte output row is valid.
            unsafe {
                let predicted = _mm_loadl_epi64(prediction_ptr.cast::<__m128i>());
                let residual_0 = _mm_loadu_si128(residual_ptr.cast::<__m128i>());
                let residual_1 = _mm_loadu_si128(residual_ptr.add(4).cast::<__m128i>());
                let sum = _mm_adds_epi16(
                    _mm_unpacklo_epi8(predicted, zero),
                    _mm_packs_epi32(residual_0, residual_1),
                );
                _mm_storel_epi64(output_ptr.cast::<__m128i>(), _mm_packus_epi16(sum, zero));
            }
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

    #[test]
    fn direct_inter_residual_fusion_matches_assembled_oracle() {
        const VALUES: [i32; 12] = [
            i32::MIN,
            -40_000,
            -256,
            -1,
            0,
            1,
            127,
            255,
            256,
            40_000,
            i32::MAX,
            17,
        ];
        let chroma = |seed: usize| {
            std::array::from_fn(|block| {
                std::array::from_fn(|row| {
                    std::array::from_fn(|column| {
                        VALUES[(seed + block * 7 + row * 3 + column) % VALUES.len()]
                    })
                })
            })
        };
        let residuals = [
            ReconstructedInterResidual {
                luma: ReconstructedLumaResidual::FourByFour(Box::new(std::array::from_fn(
                    |block| {
                        std::array::from_fn(|row| {
                            std::array::from_fn(|column| {
                                VALUES[(block * 5 + row * 3 + column) % VALUES.len()]
                            })
                        })
                    },
                ))),
                chroma_cb: chroma(1),
                chroma_cr: chroma(5),
            },
            ReconstructedInterResidual {
                luma: ReconstructedLumaResidual::EightByEight(Box::new(std::array::from_fn(
                    |block| {
                        std::array::from_fn(|row| {
                            std::array::from_fn(|column| {
                                VALUES[(block * 11 + row * 5 + column) % VALUES.len()]
                            })
                        })
                    },
                ))),
                chroma_cb: chroma(2),
                chroma_cr: chroma(7),
            },
        ];

        for residual in &residuals {
            let prediction_luma = std::array::from_fn(|row| {
                std::array::from_fn(|column| ((row * 31 + column * 17) & 255) as u8)
            });
            let prediction_cb = std::array::from_fn(|row| {
                std::array::from_fn(|column| ((row * 47 + column * 13) & 255) as u8)
            });
            let prediction_cr = std::array::from_fn(|row| {
                std::array::from_fn(|column| ((row * 19 + column * 43) & 255) as u8)
            });
            let (residual_luma, residual_cb, residual_cr) = assemble_residual(residual);
            let mut expected_luma = [[0; 16]; 16];
            let mut expected_cb = [[0; 8]; 8];
            let mut expected_cr = [[0; 8]; 8];
            add_prediction_and_residual(
                expected_luma.as_flattened_mut(),
                16,
                0,
                0,
                &prediction_luma,
                &residual_luma,
            );
            add_prediction_and_residual(
                expected_cb.as_flattened_mut(),
                8,
                0,
                0,
                &prediction_cb,
                &residual_cb,
            );
            add_prediction_and_residual(
                expected_cr.as_flattened_mut(),
                8,
                0,
                0,
                &prediction_cr,
                &residual_cr,
            );

            let mut actual_luma = prediction_luma;
            let mut actual_cb = prediction_cb;
            let mut actual_cr = prediction_cr;
            add_inter_residual_to_prediction(
                &mut actual_luma,
                &mut actual_cb,
                &mut actual_cr,
                residual,
            );
            assert_eq!(actual_luma, expected_luma);
            assert_eq!(actual_cb, expected_cb);
            assert_eq!(actual_cr, expected_cr);
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

    fn b_list(reference_index: u8) -> ResolvedBListMotion {
        ResolvedBListMotion {
            reference_index,
            motion_vector: MotionVector::default(),
        }
    }

    fn b_partition(
        x: u8,
        y: u8,
        width: u8,
        height: u8,
        list0: Option<ResolvedBListMotion>,
        list1: Option<ResolvedBListMotion>,
    ) -> ResolvedBPartition {
        ResolvedBPartition {
            x,
            y,
            width,
            height,
            list0,
            list1,
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
    fn applies_explicit_and_default_p_prediction_weights() {
        let reference = picture(40);
        let mut current = picture(0);
        reconstruct_weighted_p_macroblock_from_list_420(
            &mut current,
            &[Some(&reference)],
            0,
            0,
            &ResolvedPMacroblock {
                skipped: false,
                partitions: vec![partition(0, 0, 16, 16, 0)],
            },
            &zero_residual(),
            &PredictionWeightTable {
                luma_log2_weight_denom: 1,
                chroma_log2_weight_denom: 1,
                list0: vec![crate::PredictionWeight {
                    luma: Some(WeightOffset {
                        weight: 3,
                        offset: -5,
                    }),
                    chroma: Some([
                        WeightOffset {
                            weight: 2,
                            offset: -10,
                        },
                        WeightOffset {
                            weight: 1,
                            offset: 20,
                        },
                    ]),
                }],
                list1: Vec::new(),
            },
        )
        .unwrap();
        let (luma, cb, cr) = current.planes();
        assert_eq!((luma[0], cb[0], cr[0]), (55, 31, 41));

        let mut defaulted = picture(0);
        reconstruct_weighted_p_macroblock_from_list_420(
            &mut defaulted,
            &[Some(&reference)],
            0,
            0,
            &ResolvedPMacroblock {
                skipped: false,
                partitions: vec![partition(0, 0, 16, 16, 0)],
            },
            &zero_residual(),
            &PredictionWeightTable {
                luma_log2_weight_denom: 7,
                chroma_log2_weight_denom: 7,
                list0: vec![crate::PredictionWeight {
                    luma: None,
                    chroma: None,
                }],
                list1: Vec::new(),
            },
        )
        .unwrap();
        assert_eq!(defaulted, reference);
    }

    #[test]
    fn reconstructs_list0_list1_and_default_bipred_partitions() {
        let first = picture(20);
        let second = picture(80);
        let mut current = picture(0);
        reconstruct_b_macroblock_from_lists_420(
            &mut current,
            &[Some(&first)],
            &[Some(&second)],
            0,
            0,
            &ResolvedBMacroblock {
                direct: false,
                partitions: vec![
                    b_partition(0, 0, 8, 8, Some(b_list(0)), None),
                    b_partition(8, 0, 8, 8, None, Some(b_list(0))),
                    b_partition(0, 8, 16, 8, Some(b_list(0)), Some(b_list(0))),
                ]
                .into(),
            },
            &zero_residual(),
        )
        .unwrap();
        let (luma, cb, cr) = current.planes();
        assert_eq!((luma[0], luma[8], luma[8 * 16]), (20, 80, 50));
        assert_eq!((cb[0], cb[4], cb[4 * 8]), (21, 81, 51));
        assert_eq!(cr[4 * 8], 52);
        assert_eq!(rounded_average(20, 81), 51);
    }

    #[test]
    fn omitted_b_residual_matches_explicit_zero_residual() {
        let first = picture(20);
        let second = picture(80);
        let motion = ResolvedBMacroblock {
            direct: true,
            partitions: vec![b_partition(0, 0, 16, 16, Some(b_list(0)), Some(b_list(0)))].into(),
        };
        let mut omitted = picture(0);
        reconstruct_b_macroblock_from_lists_with_mode(
            &mut omitted,
            &[Some(&first)],
            &[Some(&second)],
            0,
            0,
            &motion,
            None,
            BPredictionWeightMode::Default,
        )
        .unwrap();

        let mut explicit = picture(0);
        reconstruct_b_macroblock_from_lists_420(
            &mut explicit,
            &[Some(&first)],
            &[Some(&second)],
            0,
            0,
            &motion,
            &zero_residual(),
        )
        .unwrap();
        assert_eq!(omitted, explicit);
    }

    #[test]
    fn applies_explicit_weights_to_each_b_prediction_mode() {
        let first = picture(20);
        let second = picture(80);
        let mut current = picture(0);
        reconstruct_weighted_b_macroblock_from_lists_420(
            &mut current,
            &[Some(&first)],
            &[Some(&second)],
            0,
            0,
            &ResolvedBMacroblock {
                direct: false,
                partitions: vec![
                    b_partition(0, 0, 8, 8, Some(b_list(0)), None),
                    b_partition(8, 0, 8, 8, None, Some(b_list(0))),
                    b_partition(0, 8, 16, 8, Some(b_list(0)), Some(b_list(0))),
                ]
                .into(),
            },
            &zero_residual(),
            &PredictionWeightTable {
                luma_log2_weight_denom: 0,
                chroma_log2_weight_denom: 0,
                list0: vec![PredictionWeight {
                    luma: Some(WeightOffset {
                        weight: 1,
                        offset: 10,
                    }),
                    chroma: Some([
                        WeightOffset {
                            weight: 1,
                            offset: 5,
                        },
                        WeightOffset {
                            weight: 1,
                            offset: 5,
                        },
                    ]),
                }],
                list1: vec![PredictionWeight {
                    luma: Some(WeightOffset {
                        weight: 3,
                        offset: -2,
                    }),
                    chroma: Some([
                        WeightOffset {
                            weight: 1,
                            offset: -5,
                        },
                        WeightOffset {
                            weight: 1,
                            offset: -5,
                        },
                    ]),
                }],
            },
        )
        .unwrap();
        let (luma, cb, cr) = current.planes();
        assert_eq!((luma[0], luma[8], luma[8 * 16]), (30, 238, 134));
        assert_eq!((cb[0], cb[4], cb[4 * 8]), (26, 76, 51));
        assert_eq!((cr[0], cr[4], cr[4 * 8]), (27, 77, 52));
    }

    #[test]
    fn derives_normative_implicit_biprediction_weights() {
        let short = |picture_order_count| ImplicitWeightReference {
            picture_order_count,
            long_term: false,
        };
        assert_eq!(
            derive_implicit_bipred_weights(2, short(0), short(8)),
            (48, 16)
        );
        assert_eq!(
            derive_implicit_bipred_weights(4, short(0), short(8)),
            (32, 32)
        );
        assert_eq!(
            derive_implicit_bipred_weights(2, short(0), short(0)),
            (32, 32)
        );
        assert_eq!(
            derive_implicit_bipred_weights(
                2,
                short(0),
                ImplicitWeightReference {
                    picture_order_count: 8,
                    long_term: true,
                },
            ),
            (32, 32)
        );
        assert_eq!(
            derive_implicit_bipred_weights(127, short(0), short(1)),
            (32, 32)
        );
    }

    #[test]
    fn simd_single_prediction_weighting_matches_the_scalar_equation() {
        let mut cases = Vec::new();
        for denominator in [0, 1, 3, 7] {
            for weight in [-128, -17, 0, 1, 127, 128] {
                for offset in [-128, 0, 127] {
                    cases.push((WeightOffset { weight, offset }, denominator));
                }
            }
        }
        // Exercise the scalar fallback outside the guarded H.264 SIMD range.
        cases.extend([
            (
                WeightOffset {
                    weight: 129,
                    offset: 0,
                },
                7,
            ),
            (
                WeightOffset {
                    weight: 1,
                    offset: 128,
                },
                7,
            ),
            (
                WeightOffset {
                    weight: 1,
                    offset: 0,
                },
                8,
            ),
        ]);

        for length in 0..=32 {
            for &(weight, denominator) in &cases {
                let mut actual = (0..length)
                    .map(|index| (index * 73 + length * 19) as u8)
                    .collect::<Vec<_>>();
                let expected = actual
                    .iter()
                    .map(|&sample| weighted_sample(sample, weight, denominator))
                    .collect::<Vec<_>>();

                weighted_row(&mut actual, weight, denominator);
                assert_eq!(
                    actual, expected,
                    "length={length} weight={weight:?} denominator={denominator}"
                );
            }
        }
    }

    #[test]
    fn simd_implicit_biprediction_matches_the_scalar_equation() {
        let source_l0 = [
            0, 1, 17, 31, 63, 64, 95, 127, 128, 159, 191, 223, 239, 253, 254, 255,
        ];
        let source_l1 = [
            255, 254, 240, 224, 192, 191, 160, 128, 127, 96, 65, 32, 16, 2, 1, 0,
        ];
        for length in [4, 8, 16] {
            for (weight_l0, weight_l1) in [(48, 16), (32, 32), (128, -64), (-64, 128)] {
                let weight_l0 = WeightOffset {
                    weight: weight_l0,
                    offset: 0,
                };
                let weight_l1 = WeightOffset {
                    weight: weight_l1,
                    offset: 0,
                };
                let mut actual = source_l0;
                let mut expected = source_l0;
                for index in 0..length {
                    expected[index] = weighted_bipred_sample(
                        expected[index],
                        source_l1[index],
                        weight_l0,
                        weight_l1,
                        5,
                    );
                }
                implicit_bipred_row(
                    &mut actual[..length],
                    &source_l1[..length],
                    weight_l0,
                    weight_l1,
                );
                assert_eq!(&actual[..length], &expected[..length]);
            }
        }
    }

    #[test]
    fn simd_prediction_plus_residual_matches_scalar_saturation() {
        fn check<const SIZE: usize>() {
            let prediction: [[u8; SIZE]; SIZE] = std::array::from_fn(|row| {
                std::array::from_fn(|column| ((row * 37 + column * 19) & 255) as u8)
            });
            let values = [
                i32::MIN,
                -65_536,
                -32_768,
                -256,
                -255,
                -1,
                0,
                1,
                255,
                256,
                32_767,
                65_535,
                i32::MAX,
            ];
            let residual: [[i32; SIZE]; SIZE] = std::array::from_fn(|row| {
                std::array::from_fn(|column| values[(row * SIZE + column) % values.len()])
            });
            let expected: [[u8; SIZE]; SIZE] = std::array::from_fn(|row| {
                std::array::from_fn::<_, SIZE, _>(|column| {
                    i32::from(prediction[row][column])
                        .saturating_add(residual[row][column])
                        .clamp(0, 255) as u8
                })
            });
            let mut actual = vec![0; SIZE * SIZE];
            add_prediction_and_residual(&mut actual, SIZE, 0, 0, &prediction, &residual);
            for (row, expected) in expected.iter().enumerate() {
                assert_eq!(&actual[row * SIZE..(row + 1) * SIZE], expected);
            }
        }

        check::<8>();
        check::<16>();
    }

    #[test]
    fn applies_implicit_weights_only_to_bidirectional_partitions() {
        let first = picture(20);
        let second = picture(80);
        let mut current = picture(0);
        let list0 = [Some(ImplicitWeightReference {
            picture_order_count: 0,
            long_term: false,
        })];
        let list1 = [Some(ImplicitWeightReference {
            picture_order_count: 8,
            long_term: false,
        })];
        reconstruct_b_macroblock_from_lists_with_mode(
            &mut current,
            &[Some(&first)],
            &[Some(&second)],
            0,
            0,
            &ResolvedBMacroblock {
                direct: false,
                partitions: vec![
                    b_partition(0, 0, 8, 8, Some(b_list(0)), None),
                    b_partition(8, 0, 8, 8, None, Some(b_list(0))),
                    b_partition(0, 8, 16, 8, Some(b_list(0)), Some(b_list(0))),
                ]
                .into(),
            },
            Some(&zero_residual()),
            BPredictionWeightMode::Implicit {
                current_picture_order_count: 2,
                list0: &list0,
                list1: &list1,
            },
        )
        .unwrap();
        let (luma, cb, cr) = current.planes();
        assert_eq!((luma[0], luma[8], luma[8 * 16]), (20, 80, 35));
        assert_eq!((cb[0], cb[4], cb[4 * 8]), (21, 81, 36));
        assert_eq!(cr[4 * 8], 37);
    }

    #[test]
    fn adds_residual_after_bidirectional_prediction() {
        let first = picture(20);
        let second = picture(80);
        let mut current = picture(0);
        let mut residual = zero_residual();
        let ReconstructedLumaResidual::FourByFour(blocks) = &mut residual.luma else {
            unreachable!()
        };
        blocks[0] = [[10; 4]; 4];
        reconstruct_b_macroblock_from_lists_420(
            &mut current,
            &[Some(&first)],
            &[Some(&second)],
            0,
            0,
            &ResolvedBMacroblock {
                direct: false,
                partitions: vec![b_partition(0, 0, 16, 16, Some(b_list(0)), Some(b_list(0)))]
                    .into(),
            },
            &residual,
        )
        .unwrap();
        assert_eq!((current.planes().0[0], current.planes().0[4]), (60, 50));
    }

    #[test]
    fn b_validation_failure_leaves_current_picture_unchanged() {
        let reference = picture(20);
        let mut current = picture(7);
        let before = current.clone();
        let result = reconstruct_b_macroblock_from_lists_420(
            &mut current,
            &[Some(&reference)],
            &[Some(&reference)],
            0,
            0,
            &ResolvedBMacroblock {
                direct: false,
                partitions: vec![b_partition(0, 0, 16, 16, None, None)].into(),
            },
            &zero_residual(),
        );
        assert!(matches!(result, Err(H264Error::InvalidSyntax(_))));
        assert_eq!(current, before);
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
