use crate::{FrameHeader, Result, Vp9Error, bool_decoder::BoolDecoder};

const DIFF_UPDATE_PROBABILITY: u8 = 252;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransformMode {
    Only4x4,
    Allow8x8,
    Allow16x16,
    Allow32x32,
    Select,
}

impl TransformMode {
    fn maximum_size_index(self) -> usize {
        match self {
            Self::Only4x4 => 0,
            Self::Allow8x8 => 1,
            Self::Allow16x16 => 2,
            Self::Allow32x32 | Self::Select => 3,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReferenceMode {
    Single,
    Compound,
    Select,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbabilityUpdateKind {
    Transform,
    Coefficient,
    Skip,
    InterMode,
    Interpolation,
    IntraInter,
    CompoundInter,
    SingleReference,
    CompoundReference,
    YMode,
    Partition,
    MotionVector,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProbabilityUpdate {
    pub kind: ProbabilityUpdateKind,
    /// Sequential position within the named probability family.
    pub index: usize,
    /// Coded update value before it is remapped around the old probability.
    pub coded_value: u8,
    /// Motion-vector probabilities carry their complete replacement value.
    pub replacement: Option<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompressedHeader {
    pub transform_mode: TransformMode,
    pub reference_mode: ReferenceMode,
    pub updates: Vec<ProbabilityUpdate>,
}

impl CompressedHeader {
    pub fn parse(frame: &[u8], header: &FrameHeader) -> Result<Self> {
        let start = header.uncompressed_header_size;
        let end = start
            .checked_add(header.compressed_header_size)
            .ok_or(Vp9Error::IntegerOverflow)?;
        let partition = frame
            .get(start..end)
            .ok_or(Vp9Error::Truncated("compressed header"))?;
        let mut decoder = BoolDecoder::new(partition)?;
        let lossless = header
            .quantization
            .ok_or(Vp9Error::InvalidData("frame has no quantization header"))?
            .lossless();
        let transform_mode = read_transform_mode(&mut decoder, lossless)?;
        let mut updates = Vec::new();

        if transform_mode == TransformMode::Select {
            // The bitstream orders transform probabilities from the smallest
            // maximum transform to the largest, while ProbabilityContext
            // stores the normative p32x32, p16x16, p8x8 struct layout.
            read_updates_at(
                &mut decoder,
                ProbabilityUpdateKind::Transform,
                10,
                2,
                &mut updates,
            )?;
            read_updates_at(
                &mut decoder,
                ProbabilityUpdateKind::Transform,
                6,
                4,
                &mut updates,
            )?;
            read_updates_at(
                &mut decoder,
                ProbabilityUpdateKind::Transform,
                0,
                6,
                &mut updates,
            )?;
        }

        let coefficient_count_per_size = 2 * 2 * (3 + 5 * 6) * 3;
        for size_index in 0..=transform_mode.maximum_size_index() {
            if decoder.read_bit()? {
                read_updates_at(
                    &mut decoder,
                    ProbabilityUpdateKind::Coefficient,
                    size_index * coefficient_count_per_size,
                    coefficient_count_per_size,
                    &mut updates,
                )?;
            }
        }
        read_updates(&mut decoder, ProbabilityUpdateKind::Skip, 3, &mut updates)?;

        let frame_is_intra = header.intra_only;
        let reference_mode = if frame_is_intra {
            ReferenceMode::Single
        } else {
            read_updates(
                &mut decoder,
                ProbabilityUpdateKind::InterMode,
                7 * 3,
                &mut updates,
            )?;
            if matches!(
                header.interpolation_filter,
                crate::InterpolationFilter::Switchable
            ) {
                read_updates(
                    &mut decoder,
                    ProbabilityUpdateKind::Interpolation,
                    4 * 2,
                    &mut updates,
                )?;
            }
            read_updates(
                &mut decoder,
                ProbabilityUpdateKind::IntraInter,
                4,
                &mut updates,
            )?;
            let reference_mode = read_reference_mode(&mut decoder, header)?;
            if reference_mode == ReferenceMode::Select {
                read_updates(
                    &mut decoder,
                    ProbabilityUpdateKind::CompoundInter,
                    5,
                    &mut updates,
                )?;
            }
            if reference_mode != ReferenceMode::Compound {
                read_updates(
                    &mut decoder,
                    ProbabilityUpdateKind::SingleReference,
                    5 * 2,
                    &mut updates,
                )?;
            }
            if reference_mode != ReferenceMode::Single {
                read_updates(
                    &mut decoder,
                    ProbabilityUpdateKind::CompoundReference,
                    5,
                    &mut updates,
                )?;
            }
            reference_mode
        };

        if !frame_is_intra {
            read_updates(
                &mut decoder,
                ProbabilityUpdateKind::YMode,
                4 * 9,
                &mut updates,
            )?;
            read_updates(
                &mut decoder,
                ProbabilityUpdateKind::Partition,
                16 * 3,
                &mut updates,
            )?;
            read_motion_vector_updates(
                &mut decoder,
                header.allow_high_precision_motion_vectors,
                &mut updates,
            )?;
        }

        Ok(Self {
            transform_mode,
            reference_mode,
            updates,
        })
    }
}

fn read_transform_mode(decoder: &mut BoolDecoder<'_>, lossless: bool) -> Result<TransformMode> {
    if lossless {
        return Ok(TransformMode::Only4x4);
    }
    Ok(match decoder.read_literal(2)? {
        0 => TransformMode::Only4x4,
        1 => TransformMode::Allow8x8,
        2 => TransformMode::Allow16x16,
        _ if decoder.read_bit()? => TransformMode::Select,
        _ => TransformMode::Allow32x32,
    })
}

fn read_reference_mode(
    decoder: &mut BoolDecoder<'_>,
    header: &FrameHeader,
) -> Result<ReferenceMode> {
    let compound_allowed = header.reference_sign_bias[1] != header.reference_sign_bias[0]
        || header.reference_sign_bias[2] != header.reference_sign_bias[0];
    if !compound_allowed {
        return Ok(ReferenceMode::Single);
    }
    if !decoder.read_bit()? {
        return Ok(ReferenceMode::Single);
    }
    Ok(if decoder.read_bit()? {
        ReferenceMode::Select
    } else {
        ReferenceMode::Compound
    })
}

fn read_updates(
    decoder: &mut BoolDecoder<'_>,
    kind: ProbabilityUpdateKind,
    count: usize,
    updates: &mut Vec<ProbabilityUpdate>,
) -> Result<()> {
    read_updates_at(decoder, kind, 0, count, updates)
}

fn read_updates_at(
    decoder: &mut BoolDecoder<'_>,
    kind: ProbabilityUpdateKind,
    start_index: usize,
    count: usize,
    updates: &mut Vec<ProbabilityUpdate>,
) -> Result<()> {
    for index in 0..count {
        if decoder.read_bool(DIFF_UPDATE_PROBABILITY)? {
            updates.push(ProbabilityUpdate {
                kind,
                index: start_index + index,
                coded_value: decode_term_subexp(decoder)? as u8,
                replacement: None,
            });
        }
    }
    Ok(())
}

fn read_motion_vector_updates(
    decoder: &mut BoolDecoder<'_>,
    allow_high_precision: bool,
    updates: &mut Vec<ProbabilityUpdate>,
) -> Result<()> {
    read_mv_update_group(decoder, 0, 3, updates)?;
    for component in 0..2 {
        let component_start = 3 + component * 33;
        // sign, classes, class-zero, and integer bits.
        read_mv_update_group(decoder, component_start, 22, updates)?;
    }
    for component in 0..2 {
        let component_start = 3 + component * 33;
        // Two class-zero fractional trees followed by the general tree.
        read_mv_update_group(decoder, component_start + 22, 9, updates)?;
    }
    if allow_high_precision {
        for component in 0..2 {
            let component_start = 3 + component * 33;
            read_mv_update_group(decoder, component_start + 31, 2, updates)?;
        }
    }
    Ok(())
}

fn read_mv_update_group(
    decoder: &mut BoolDecoder<'_>,
    start_index: usize,
    count: usize,
    updates: &mut Vec<ProbabilityUpdate>,
) -> Result<()> {
    for index in start_index..start_index + count {
        if decoder.read_bool(DIFF_UPDATE_PROBABILITY)? {
            let replacement = (decoder.read_literal(7)? as u8) << 1 | 1;
            updates.push(ProbabilityUpdate {
                kind: ProbabilityUpdateKind::MotionVector,
                index,
                coded_value: replacement,
                replacement: Some(replacement),
            });
        }
    }
    Ok(())
}

fn decode_term_subexp(decoder: &mut BoolDecoder<'_>) -> Result<u32> {
    if !decoder.read_bit()? {
        return decoder.read_literal(4);
    }
    if !decoder.read_bit()? {
        return Ok(decoder.read_literal(4)? + 16);
    }
    if !decoder.read_bit()? {
        return Ok(decoder.read_literal(5)? + 32);
    }
    let value = decoder.read_literal(7)?;
    Ok(if value < 65 {
        value + 64
    } else {
        (value << 1) - 1 + u32::from(decoder.read_bit()?)
    })
}

#[cfg(test)]
mod tests {
    use super::{CompressedHeader, TransformMode};
    use crate::HeaderParser;

    #[test]
    fn parses_target_style_keyframe_compressed_header() {
        // Uncompressed keyframe header followed by a three-byte compressed
        // header containing the required zero marker and no probability
        // updates. This prefix is from the normative syntax shape used by the
        // target Profile-0 stream, not from another decoder implementation.
        let frame = [
            0x82, 0x49, 0x83, 0x42, 0x40, 0xef, 0xf0, 0x86, 0xf4, 0x10, 0x26, 0x00, 0xe0, 0x00,
            0x30, 0x70, 0x00, 0x00,
        ];
        let header = HeaderParser::new().parse(&frame).unwrap();
        let compressed = CompressedHeader::parse(&frame, &header).unwrap();
        assert_eq!(compressed.transform_mode, TransformMode::Select);
        assert!(compressed.updates.is_empty());
    }
}
