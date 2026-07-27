use crate::{CompressedHeader, ProbabilityUpdate, ProbabilityUpdateKind, Result, Vp9Error, tables};

pub(crate) const COEFFICIENT_PROBABILITIES_PER_SIZE: usize = 396;
const COEFFICIENT_MODELS_PER_SIZE: usize = COEFFICIENT_PROBABILITIES_PER_SIZE / 3;

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct CoefficientModelCounts {
    eob_branches: u32,
    tokens: [u32; 4],
}

#[derive(Debug, Clone)]
pub(crate) struct CoefficientCounts {
    models: [[CoefficientModelCounts; COEFFICIENT_MODELS_PER_SIZE]; 4],
}

#[derive(Debug, Clone, Default)]
pub(crate) struct MotionVectorComponentCounts {
    pub(crate) sign: [u32; 2],
    pub(crate) classes: [u32; 11],
    pub(crate) class_zero: [u32; 2],
    pub(crate) bits: [[u32; 2]; 10],
    pub(crate) class_zero_fractional: [[u32; 4]; 2],
    pub(crate) fractional: [u32; 4],
    pub(crate) class_zero_high_precision: [u32; 2],
    pub(crate) high_precision: [u32; 2],
}

#[derive(Debug, Clone, Default)]
pub(crate) struct MotionVectorCounts {
    pub(crate) joints: [u32; 4],
    pub(crate) components: [MotionVectorComponentCounts; 2],
}

#[derive(Debug, Clone, Default)]
pub(crate) struct FrameCounts {
    pub(crate) coefficient: CoefficientCounts,
    pub(crate) transform_8x8: [[u32; 2]; 2],
    pub(crate) transform_16x16: [[u32; 3]; 2],
    pub(crate) transform_32x32: [[u32; 4]; 2],
    pub(crate) skip: [[u32; 2]; 3],
    pub(crate) intra_inter: [[u32; 2]; 4],
    pub(crate) compound_inter: [[u32; 2]; 5],
    pub(crate) single_reference: [[[u32; 2]; 2]; 5],
    pub(crate) compound_reference: [[u32; 2]; 5],
    pub(crate) inter_mode: [[u32; 4]; 7],
    pub(crate) interpolation: [[u32; 3]; 4],
    pub(crate) y_mode: [[u32; 10]; 4],
    pub(crate) uv_mode: [[u32; 10]; 10],
    pub(crate) partition: [[u32; 4]; 16],
    pub(crate) motion_vector: MotionVectorCounts,
}

impl Default for CoefficientCounts {
    fn default() -> Self {
        Self {
            models: [[CoefficientModelCounts::default(); COEFFICIENT_MODELS_PER_SIZE]; 4],
        }
    }
}

impl CoefficientCounts {
    pub(crate) fn model_mut(
        &mut self,
        transform_size: usize,
        plane_type: usize,
        reference_type: usize,
        band: usize,
        context: usize,
    ) -> &mut CoefficientModelCounts {
        let model = coefficient_model_index(plane_type, reference_type, band, context);
        &mut self.models[transform_size][model]
    }

    fn merge_from(&mut self, other: &Self) {
        for (sizes, other_sizes) in self.models.iter_mut().zip(&other.models) {
            for (model, other_model) in sizes.iter_mut().zip(other_sizes) {
                model.eob_branches += other_model.eob_branches;
                add_counts(&mut model.tokens, &other_model.tokens);
            }
        }
    }
}

impl FrameCounts {
    pub(crate) fn merge_from(&mut self, other: &Self) {
        self.coefficient.merge_from(&other.coefficient);
        add_nested_counts(&mut self.transform_8x8, &other.transform_8x8);
        add_nested_counts(&mut self.transform_16x16, &other.transform_16x16);
        add_nested_counts(&mut self.transform_32x32, &other.transform_32x32);
        add_nested_counts(&mut self.skip, &other.skip);
        add_nested_counts(&mut self.intra_inter, &other.intra_inter);
        add_nested_counts(&mut self.compound_inter, &other.compound_inter);
        for (contexts, other_contexts) in self
            .single_reference
            .iter_mut()
            .zip(&other.single_reference)
        {
            add_nested_counts(contexts, other_contexts);
        }
        add_nested_counts(&mut self.compound_reference, &other.compound_reference);
        add_nested_counts(&mut self.inter_mode, &other.inter_mode);
        add_nested_counts(&mut self.interpolation, &other.interpolation);
        add_nested_counts(&mut self.y_mode, &other.y_mode);
        add_nested_counts(&mut self.uv_mode, &other.uv_mode);
        add_nested_counts(&mut self.partition, &other.partition);
        add_counts(&mut self.motion_vector.joints, &other.motion_vector.joints);
        for (component, other_component) in self
            .motion_vector
            .components
            .iter_mut()
            .zip(&other.motion_vector.components)
        {
            add_counts(&mut component.sign, &other_component.sign);
            add_counts(&mut component.classes, &other_component.classes);
            add_counts(&mut component.class_zero, &other_component.class_zero);
            add_nested_counts(&mut component.bits, &other_component.bits);
            add_nested_counts(
                &mut component.class_zero_fractional,
                &other_component.class_zero_fractional,
            );
            add_counts(&mut component.fractional, &other_component.fractional);
            add_counts(
                &mut component.class_zero_high_precision,
                &other_component.class_zero_high_precision,
            );
            add_counts(
                &mut component.high_precision,
                &other_component.high_precision,
            );
        }
    }
}

fn add_counts<const N: usize>(target: &mut [u32; N], source: &[u32; N]) {
    for (target, source) in target.iter_mut().zip(source) {
        *target += *source;
    }
}

fn add_nested_counts<const M: usize, const N: usize>(
    target: &mut [[u32; N]; M],
    source: &[[u32; N]; M],
) {
    for (target, source) in target.iter_mut().zip(source) {
        add_counts(target, source);
    }
}

impl CoefficientModelCounts {
    pub(crate) fn record_eob_branch(&mut self) {
        self.eob_branches += 1;
    }

    pub(crate) fn record_token(&mut self, token: usize) {
        self.tokens[token] += 1;
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ProbabilityContext {
    pub(crate) transform: [u8; 12],
    pub(crate) coefficient: [[u8; COEFFICIENT_PROBABILITIES_PER_SIZE]; 4],
    pub(crate) skip: [u8; 3],
    pub(crate) inter_mode: [u8; 21],
    pub(crate) interpolation: [u8; 8],
    pub(crate) intra_inter: [u8; 4],
    pub(crate) compound_inter: [u8; 5],
    pub(crate) single_reference: [u8; 10],
    pub(crate) compound_reference: [u8; 5],
    pub(crate) y_mode: [u8; 36],
    pub(crate) uv_mode: [u8; 90],
    pub(crate) partition: [u8; 48],
    pub(crate) motion_vector: [u8; 69],
}

impl Default for ProbabilityContext {
    fn default() -> Self {
        Self {
            transform: tables::TRANSFORM,
            coefficient: [
                tables::COEFFICIENT_4X4,
                tables::COEFFICIENT_8X8,
                tables::COEFFICIENT_16X16,
                tables::COEFFICIENT_32X32,
            ],
            skip: tables::SKIP,
            inter_mode: tables::INTER_MODE,
            interpolation: tables::SWITCHABLE_INTERPOLATION,
            intra_inter: tables::INTRA_INTER,
            compound_inter: tables::COMPOUND_INTER,
            single_reference: tables::SINGLE_REFERENCE,
            compound_reference: tables::COMPOUND_REFERENCE,
            y_mode: tables::INTER_Y_MODE,
            uv_mode: tables::INTER_UV_MODE,
            partition: tables::INTER_PARTITION,
            motion_vector: tables::MV,
        }
    }
}

impl ProbabilityContext {
    pub(crate) fn apply(&mut self, header: &CompressedHeader) -> Result<()> {
        for update in &header.updates {
            self.apply_one(*update)?;
        }
        Ok(())
    }

    fn apply_one(&mut self, update: ProbabilityUpdate) -> Result<()> {
        let probability = match update.kind {
            ProbabilityUpdateKind::Transform => self.transform.get_mut(update.index),
            ProbabilityUpdateKind::Coefficient => {
                let size = update.index / COEFFICIENT_PROBABILITIES_PER_SIZE;
                let index = update.index % COEFFICIENT_PROBABILITIES_PER_SIZE;
                self.coefficient
                    .get_mut(size)
                    .and_then(|probabilities| probabilities.get_mut(index))
            }
            ProbabilityUpdateKind::Skip => self.skip.get_mut(update.index),
            ProbabilityUpdateKind::InterMode => self.inter_mode.get_mut(update.index),
            ProbabilityUpdateKind::Interpolation => self.interpolation.get_mut(update.index),
            ProbabilityUpdateKind::IntraInter => self.intra_inter.get_mut(update.index),
            ProbabilityUpdateKind::CompoundInter => self.compound_inter.get_mut(update.index),
            ProbabilityUpdateKind::SingleReference => self.single_reference.get_mut(update.index),
            ProbabilityUpdateKind::CompoundReference => {
                self.compound_reference.get_mut(update.index)
            }
            ProbabilityUpdateKind::YMode => self.y_mode.get_mut(update.index),
            ProbabilityUpdateKind::Partition => self.partition.get_mut(update.index),
            ProbabilityUpdateKind::MotionVector => self.motion_vector.get_mut(update.index),
        }
        .ok_or(Vp9Error::InvalidData(
            "compressed-header probability index is out of range",
        ))?;
        *probability = update
            .replacement
            .unwrap_or_else(|| inverse_remap_probability(update.coded_value, *probability));
        Ok(())
    }

    pub(crate) fn adapt_coefficients(
        &mut self,
        previous: &Self,
        counts: &CoefficientCounts,
        intra_only: bool,
        previous_was_key: bool,
    ) {
        let update_factor = if intra_only {
            112
        } else if previous_was_key {
            128
        } else {
            112
        };
        for transform_size in 0..4 {
            for plane_type in 0..2 {
                for reference_type in 0..2 {
                    for band in 0..6 {
                        let contexts = if band == 0 { 3 } else { 6 };
                        for context in 0..contexts {
                            let model =
                                coefficient_model_index(plane_type, reference_type, band, context);
                            let probability = model * 3;
                            let counts = counts.models[transform_size][model];
                            let branches = [
                                [
                                    counts.tokens[3],
                                    counts.eob_branches.saturating_sub(counts.tokens[3]),
                                ],
                                [counts.tokens[0], counts.tokens[1] + counts.tokens[2]],
                                [counts.tokens[1], counts.tokens[2]],
                            ];
                            for (node, branch) in branches.into_iter().enumerate() {
                                self.coefficient[transform_size][probability + node] = merge_probs(
                                    previous.coefficient[transform_size][probability + node],
                                    branch,
                                    24,
                                    update_factor,
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    pub(crate) fn adapt_modes(
        &mut self,
        previous: &Self,
        counts: &FrameCounts,
        transform_select: bool,
        switchable_interpolation: bool,
    ) {
        for context in 0..4 {
            self.intra_inter[context] =
                mode_mv_merge_probs(previous.intra_inter[context], counts.intra_inter[context]);
        }
        for context in 0..5 {
            self.compound_inter[context] = mode_mv_merge_probs(
                previous.compound_inter[context],
                counts.compound_inter[context],
            );
            self.compound_reference[context] = mode_mv_merge_probs(
                previous.compound_reference[context],
                counts.compound_reference[context],
            );
            for node in 0..2 {
                self.single_reference[context * 2 + node] = mode_mv_merge_probs(
                    previous.single_reference[context * 2 + node],
                    counts.single_reference[context][node],
                );
            }
        }
        for context in 0..7 {
            merge_tree_probabilities(
                &mut self.inter_mode[context * 3..context * 3 + 3],
                &previous.inter_mode[context * 3..context * 3 + 3],
                &INTER_MODE_TREE,
                &counts.inter_mode[context],
            );
        }
        for group in 0..4 {
            merge_tree_probabilities(
                &mut self.y_mode[group * 9..group * 9 + 9],
                &previous.y_mode[group * 9..group * 9 + 9],
                &INTRA_MODE_TREE,
                &counts.y_mode[group],
            );
        }
        for mode in 0..10 {
            merge_tree_probabilities(
                &mut self.uv_mode[mode * 9..mode * 9 + 9],
                &previous.uv_mode[mode * 9..mode * 9 + 9],
                &INTRA_MODE_TREE,
                &counts.uv_mode[mode],
            );
        }
        for context in 0..16 {
            merge_tree_probabilities(
                &mut self.partition[context * 3..context * 3 + 3],
                &previous.partition[context * 3..context * 3 + 3],
                &PARTITION_TREE,
                &counts.partition[context],
            );
        }
        if switchable_interpolation {
            for context in 0..4 {
                merge_tree_probabilities(
                    &mut self.interpolation[context * 2..context * 2 + 2],
                    &previous.interpolation[context * 2..context * 2 + 2],
                    &SWITCHABLE_INTERPOLATION_TREE,
                    &counts.interpolation[context],
                );
            }
        }
        if transform_select {
            for context in 0..2 {
                let count8 = counts.transform_8x8[context];
                self.transform[10 + context] =
                    mode_mv_merge_probs(previous.transform[10 + context], count8);

                let count16 = counts.transform_16x16[context];
                let branches16 = [
                    [count16[0], count16[1] + count16[2]],
                    [count16[1], count16[2]],
                ];
                for (node, branch) in branches16.into_iter().enumerate() {
                    self.transform[6 + context * 2 + node] =
                        mode_mv_merge_probs(previous.transform[6 + context * 2 + node], branch);
                }

                let count32 = counts.transform_32x32[context];
                let branches32 = [
                    [count32[0], count32[1] + count32[2] + count32[3]],
                    [count32[1], count32[2] + count32[3]],
                    [count32[2], count32[3]],
                ];
                for (node, branch) in branches32.into_iter().enumerate() {
                    self.transform[context * 3 + node] =
                        mode_mv_merge_probs(previous.transform[context * 3 + node], branch);
                }
            }
        }
        for context in 0..3 {
            self.skip[context] = mode_mv_merge_probs(previous.skip[context], counts.skip[context]);
        }
    }

    pub(crate) fn adapt_motion_vectors(
        &mut self,
        previous: &Self,
        counts: &FrameCounts,
        allow_high_precision: bool,
    ) {
        merge_tree_probabilities(
            &mut self.motion_vector[..3],
            &previous.motion_vector[..3],
            &MV_JOINT_TREE,
            &counts.motion_vector.joints,
        );
        for component in 0..2 {
            let offset = 3 + component * 33;
            let component_counts = &counts.motion_vector.components[component];
            self.motion_vector[offset] =
                mode_mv_merge_probs(previous.motion_vector[offset], component_counts.sign);
            merge_tree_probabilities(
                &mut self.motion_vector[offset + 1..offset + 11],
                &previous.motion_vector[offset + 1..offset + 11],
                &MV_CLASS_TREE,
                &component_counts.classes,
            );
            self.motion_vector[offset + 11] = mode_mv_merge_probs(
                previous.motion_vector[offset + 11],
                component_counts.class_zero,
            );
            for bit in 0..10 {
                self.motion_vector[offset + 12 + bit] = mode_mv_merge_probs(
                    previous.motion_vector[offset + 12 + bit],
                    component_counts.bits[bit],
                );
            }
            for class_zero in 0..2 {
                merge_tree_probabilities(
                    &mut self.motion_vector
                        [offset + 22 + class_zero * 3..offset + 25 + class_zero * 3],
                    &previous.motion_vector
                        [offset + 22 + class_zero * 3..offset + 25 + class_zero * 3],
                    &MV_FRACTIONAL_TREE,
                    &component_counts.class_zero_fractional[class_zero],
                );
            }
            merge_tree_probabilities(
                &mut self.motion_vector[offset + 28..offset + 31],
                &previous.motion_vector[offset + 28..offset + 31],
                &MV_FRACTIONAL_TREE,
                &component_counts.fractional,
            );
            if allow_high_precision {
                self.motion_vector[offset + 31] = mode_mv_merge_probs(
                    previous.motion_vector[offset + 31],
                    component_counts.class_zero_high_precision,
                );
                self.motion_vector[offset + 32] = mode_mv_merge_probs(
                    previous.motion_vector[offset + 32],
                    component_counts.high_precision,
                );
            }
        }
    }
}

const INTRA_MODE_TREE: [i16; 18] = [
    0, 2, -9, 4, -1, 6, 8, 12, -2, 10, -4, -5, -3, 14, -8, 16, -6, -7,
];
const INTER_MODE_TREE: [i16; 6] = [-2, 2, 0, 4, -1, -3];
const PARTITION_TREE: [i16; 6] = [0, 2, -1, 4, -2, -3];
const SWITCHABLE_INTERPOLATION_TREE: [i16; 4] = [0, 2, -1, -2];
const MV_JOINT_TREE: [i16; 6] = [0, 2, -1, 4, -2, -3];
const MV_CLASS_TREE: [i16; 20] = [
    0, 2, -1, 4, 6, 8, -2, -3, 10, 12, -4, -5, -6, 14, 16, 18, -7, -8, -9, -10,
];
const MV_FRACTIONAL_TREE: [i16; 6] = [0, 2, -1, 4, -2, -3];

fn merge_tree_probabilities(
    target: &mut [u8],
    previous: &[u8],
    tree: &[i16],
    symbol_counts: &[u32],
) {
    for node in 0..target.len() {
        let left = subtree_count(tree[node * 2], tree, symbol_counts);
        let right = subtree_count(tree[node * 2 + 1], tree, symbol_counts);
        target[node] = mode_mv_merge_probs(previous[node], [left, right]);
    }
}

fn subtree_count(child: i16, tree: &[i16], symbol_counts: &[u32]) -> u32 {
    if child <= 0 {
        return symbol_counts[usize::try_from(-child).unwrap()];
    }
    let index = usize::try_from(child).unwrap();
    subtree_count(tree[index], tree, symbol_counts)
        + subtree_count(tree[index + 1], tree, symbol_counts)
}

fn mode_mv_merge_probs(previous: u8, counts: [u32; 2]) -> u8 {
    const FACTORS: [u32; 21] = [
        0, 6, 12, 19, 25, 32, 38, 44, 51, 57, 64, 70, 76, 83, 89, 96, 102, 108, 115, 121, 128,
    ];
    let denominator = counts[0] + counts[1];
    if denominator == 0 {
        return previous;
    }
    let probability = ((u64::from(counts[0]) * 256 + u64::from(denominator >> 1))
        / u64::from(denominator))
    .clamp(1, 255) as u8;
    let factor = FACTORS[denominator.min(20) as usize];
    ((u32::from(previous) * (256 - factor) + u32::from(probability) * factor + 128) >> 8) as u8
}

fn coefficient_model_index(
    plane_type: usize,
    reference_type: usize,
    band: usize,
    context: usize,
) -> usize {
    let family = plane_type * 2 + reference_type;
    let band_offset = if band == 0 { 0 } else { 3 + (band - 1) * 6 };
    family * 33 + band_offset + context
}

fn merge_probs(previous: u8, counts: [u32; 2], saturation: u32, maximum_factor: u32) -> u8 {
    let denominator = counts[0] + counts[1];
    let probability = if denominator == 0 {
        128
    } else {
        let value =
            (u64::from(counts[0]) * 256 + u64::from(denominator >> 1)) / u64::from(denominator);
        value.clamp(1, 255) as u8
    };
    let count = denominator.min(saturation);
    let factor = maximum_factor * count / saturation;
    ((u32::from(previous) * (256 - factor) + u32::from(probability) * factor + 128) >> 8) as u8
}

fn inverse_remap_probability(delta: u8, probability: u8) -> u8 {
    let value = inverse_map(delta);
    let center = u16::from(probability) - 1;
    if center * 2 <= 255 {
        (1 + inverse_recenter_nonnegative(u16::from(value), center)) as u8
    } else {
        (255 - inverse_recenter_nonnegative(u16::from(value), 254 - center)) as u8
    }
}

fn inverse_recenter_nonnegative(value: u16, center: u16) -> u16 {
    if value > center * 2 {
        value
    } else if value & 1 != 0 {
        center - ((value + 1) >> 1)
    } else {
        center + (value >> 1)
    }
}

/// The normative inverse-map permutation has a compact construction: its
/// first twenty values advance by thirteen, followed by the remaining values
/// in ascending order. The final entry repeats 253.
fn inverse_map(index: u8) -> u8 {
    if index < 20 {
        return 7 + 13 * index;
    }
    let mut rank = usize::from(index) - 20;
    for value in 1..=253u16 {
        if value % 13 == 7 {
            continue;
        }
        if rank == 0 {
            return value as u8;
        }
        rank -= 1;
    }
    253
}

#[cfg(test)]
mod tests {
    use super::{inverse_map, inverse_remap_probability};

    #[test]
    fn constructs_normative_inverse_map_edges() {
        assert_eq!(
            (0..20).map(inverse_map).collect::<Vec<_>>(),
            [
                7, 20, 33, 46, 59, 72, 85, 98, 111, 124, 137, 150, 163, 176, 189, 202, 215, 228,
                241, 254
            ]
        );
        assert_eq!(inverse_map(20), 1);
        assert_eq!(inverse_map(21), 2);
        assert_eq!(inverse_map(254), 253);
    }

    #[test]
    fn probability_remap_stays_in_normative_range() {
        for probability in 1..=255 {
            for delta in 0..=254 {
                assert_ne!(inverse_remap_probability(delta, probability), 0);
            }
        }
    }
}
