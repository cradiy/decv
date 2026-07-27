use bit_readers::BitReader;

use crate::{Result, Vp9Error};

const FRAME_MARKER: u32 = 2;
const SYNC_BYTES: [u32; 3] = [0x49, 0x83, 0x42];
const REFERENCE_SLOTS: usize = 8;
const MAX_SEGMENTS: usize = 8;
const SEGMENT_FEATURES: usize = 4;
const MAX_TILE_WIDTH_B64: u32 = 64;
const MIN_TILE_WIDTH_B64: u32 = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameType {
    Key,
    Inter,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BitDepth {
    Eight,
    Ten,
    Twelve,
}

impl BitDepth {
    #[inline]
    pub const fn bits(self) -> u8 {
        match self {
            Self::Eight => 8,
            Self::Ten => 10,
            Self::Twelve => 12,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ColorSpace {
    Bt601 = 0,
    Bt709 = 1,
    Smpte170 = 2,
    Smpte240 = 3,
    Bt2020 = 4,
    Reserved = 5,
    Smpte431 = 6,
    Srgb = 7,
}

impl ColorSpace {
    fn from_bits(value: u32) -> Self {
        match value {
            0 => Self::Bt601,
            1 => Self::Bt709,
            2 => Self::Smpte170,
            3 => Self::Smpte240,
            4 => Self::Bt2020,
            5 => Self::Reserved,
            6 => Self::Smpte431,
            _ => Self::Srgb,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChromaSubsampling {
    Cs444,
    Cs440,
    Cs422,
    Cs420,
}

impl ChromaSubsampling {
    fn new(subsampling_x: bool, subsampling_y: bool) -> Self {
        match (subsampling_x, subsampling_y) {
            (false, false) => Self::Cs444,
            (false, true) => Self::Cs440,
            (true, false) => Self::Cs422,
            (true, true) => Self::Cs420,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpolationFilter {
    EightTap,
    EightTapSmooth,
    EightTapSharp,
    Bilinear,
    Switchable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameSize {
    pub width: u32,
    pub height: u32,
    pub render_width: u32,
    pub render_height: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColorConfig {
    pub bit_depth: BitDepth,
    pub color_space: ColorSpace,
    pub full_range: bool,
    pub subsampling: ChromaSubsampling,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoopFilter {
    pub level: u8,
    pub sharpness: u8,
    pub mode_ref_delta_enabled: bool,
    pub reference_deltas: [i8; 4],
    pub mode_deltas: [i8; 2],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Quantization {
    pub base_q_idx: u8,
    pub y_dc_delta: i8,
    pub uv_dc_delta: i8,
    pub uv_ac_delta: i8,
}

impl Quantization {
    #[inline]
    pub const fn lossless(self) -> bool {
        self.base_q_idx == 0
            && self.y_dc_delta == 0
            && self.uv_dc_delta == 0
            && self.uv_ac_delta == 0
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SegmentFeature {
    pub enabled: bool,
    pub value: i16,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Segmentation {
    pub enabled: bool,
    pub update_map: bool,
    pub tree_probabilities: [u8; 7],
    pub temporal_update: bool,
    pub prediction_probabilities: [u8; 3],
    pub update_data: bool,
    pub absolute_values: bool,
    pub features: [[SegmentFeature; SEGMENT_FEATURES]; MAX_SEGMENTS],
}

impl Default for Segmentation {
    fn default() -> Self {
        Self {
            enabled: false,
            update_map: false,
            tree_probabilities: [255; 7],
            temporal_update: false,
            prediction_probabilities: [255; 3],
            update_data: false,
            absolute_values: false,
            features: [[SegmentFeature::default(); SEGMENT_FEATURES]; MAX_SEGMENTS],
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameHeader {
    pub profile: u8,
    pub show_existing_frame: Option<u8>,
    pub frame_type: FrameType,
    pub show_frame: bool,
    pub error_resilient: bool,
    pub intra_only: bool,
    pub reset_frame_context: u8,
    pub color: Option<ColorConfig>,
    pub size: Option<FrameSize>,
    pub refresh_frame_flags: u8,
    pub reference_indices: [u8; 3],
    pub reference_sign_bias: [bool; 3],
    pub allow_high_precision_motion_vectors: bool,
    pub interpolation_filter: InterpolationFilter,
    pub refresh_frame_context: bool,
    pub frame_parallel_decoding: bool,
    pub frame_context_index: u8,
    pub loop_filter: Option<LoopFilter>,
    pub quantization: Option<Quantization>,
    pub segmentation: Option<Segmentation>,
    pub tile_columns_log2: u8,
    pub tile_rows_log2: u8,
    pub uncompressed_header_size: usize,
    pub compressed_header_size: usize,
}

impl FrameHeader {
    #[inline]
    pub const fn is_visible(&self) -> bool {
        self.show_existing_frame.is_some() || self.show_frame
    }
}

#[derive(Debug, Clone)]
pub struct HeaderParser {
    reference_sizes: [Option<FrameSize>; REFERENCE_SLOTS],
    reference_colors: [Option<ColorConfig>; REFERENCE_SLOTS],
    reference_deltas: [i8; 4],
    mode_deltas: [i8; 2],
    segmentation: Segmentation,
}

impl Default for HeaderParser {
    fn default() -> Self {
        Self::new()
    }
}

impl HeaderParser {
    pub const fn new() -> Self {
        Self {
            reference_sizes: [None; REFERENCE_SLOTS],
            reference_colors: [None; REFERENCE_SLOTS],
            reference_deltas: [1, 0, -1, -1],
            mode_deltas: [0, 0],
            segmentation: Segmentation {
                enabled: false,
                update_map: false,
                tree_probabilities: [255; 7],
                temporal_update: false,
                prediction_probabilities: [255; 3],
                update_data: false,
                absolute_values: false,
                features: [[SegmentFeature {
                    enabled: false,
                    value: 0,
                }; SEGMENT_FEATURES]; MAX_SEGMENTS],
            },
        }
    }

    /// Parses one coded frame and commits persistent header state only after
    /// the whole uncompressed header has been validated.
    pub fn parse(&mut self, data: &[u8]) -> Result<FrameHeader> {
        let mut probe = self.clone();
        let header = probe.parse_inner(data)?;
        *self = probe;
        Ok(header)
    }

    fn parse_inner(&mut self, data: &[u8]) -> Result<FrameHeader> {
        let mut bits = SyntaxBits::new(data);
        if bits.read(2, "frame marker")? != FRAME_MARKER {
            return Err(Vp9Error::InvalidData("frame marker is not binary 10"));
        }
        let profile = bits.bit("profile")? | bits.bit("profile")? << 1;
        if profile == 3 && bits.bit("profile reserved bit")? != 0 {
            return Err(Vp9Error::InvalidData("profile reserved bit must be zero"));
        }

        if bits.bool("show-existing flag")? {
            let slot = bits.read(3, "show-existing reference")? as u8;
            if self.reference_sizes[usize::from(slot)].is_none() {
                return Err(Vp9Error::MissingReference(usize::from(slot)));
            }
            return Ok(FrameHeader {
                profile,
                show_existing_frame: Some(slot),
                frame_type: FrameType::Inter,
                show_frame: true,
                error_resilient: false,
                intra_only: false,
                reset_frame_context: 0,
                color: self.reference_colors[usize::from(slot)],
                size: self.reference_sizes[usize::from(slot)],
                refresh_frame_flags: 0,
                reference_indices: [0; 3],
                reference_sign_bias: [false; 3],
                allow_high_precision_motion_vectors: false,
                interpolation_filter: InterpolationFilter::EightTap,
                refresh_frame_context: false,
                frame_parallel_decoding: false,
                frame_context_index: 0,
                loop_filter: None,
                quantization: None,
                segmentation: None,
                tile_columns_log2: 0,
                tile_rows_log2: 0,
                uncompressed_header_size: bits.byte_position(),
                compressed_header_size: 0,
            });
        }

        let frame_type = if bits.bool("frame type")? {
            FrameType::Inter
        } else {
            FrameType::Key
        };
        let show_frame = bits.bool("show-frame flag")?;
        let error_resilient = bits.bool("error-resilient flag")?;
        let mut intra_only = frame_type == FrameType::Key;
        let mut reset_frame_context = 0;
        let color;
        let size;
        let refresh_frame_flags;
        let mut reference_indices = [0; 3];
        let mut reference_sign_bias = [false; 3];
        let mut allow_high_precision_motion_vectors = false;
        let mut interpolation_filter = InterpolationFilter::EightTap;

        if frame_type == FrameType::Key {
            read_sync_code(&mut bits)?;
            color = Some(read_color_config(&mut bits, profile)?);
            size = Some(read_frame_and_render_size(&mut bits)?);
            refresh_frame_flags = 0xff;
        } else {
            if !show_frame {
                intra_only = bits.bool("intra-only flag")?;
            }
            if !error_resilient {
                reset_frame_context = bits.read(2, "reset-frame-context")? as u8;
            }
            if intra_only {
                read_sync_code(&mut bits)?;
                color = Some(if profile == 0 {
                    default_profile_zero_color()
                } else {
                    read_color_config(&mut bits, profile)?
                });
                refresh_frame_flags = bits.read(8, "refresh-frame flags")? as u8;
                size = Some(read_frame_and_render_size(&mut bits)?);
            } else {
                refresh_frame_flags = bits.read(8, "refresh-frame flags")? as u8;
                for index in 0..3 {
                    reference_indices[index] = bits.read(3, "reference-frame index")? as u8;
                    reference_sign_bias[index] = bits.bool("reference sign bias")?;
                }
                size = Some(self.read_frame_size_with_refs(&mut bits, reference_indices)?);
                color = self.reference_colors[usize::from(reference_indices[0])];
                allow_high_precision_motion_vectors =
                    bits.bool("high-precision motion-vector flag")?;
                interpolation_filter = if bits.bool("switchable interpolation flag")? {
                    InterpolationFilter::Switchable
                } else {
                    match bits.read(2, "interpolation filter")? {
                        0 => InterpolationFilter::EightTapSmooth,
                        1 => InterpolationFilter::EightTap,
                        2 => InterpolationFilter::EightTapSharp,
                        _ => InterpolationFilter::Bilinear,
                    }
                };
            }
        }

        let (refresh_frame_context, frame_parallel_decoding) = if error_resilient {
            (false, true)
        } else {
            (
                bits.bool("refresh-frame-context flag")?,
                bits.bool("frame-parallel-decoding flag")?,
            )
        };
        // The index is present for every decoded frame. Independent frames
        // reset it to zero after parsing, but still consume the two syntax
        // bits.
        let parsed_frame_context_index = bits.read(2, "frame-context index")? as u8;
        let frame_context_index = if intra_only || error_resilient {
            0
        } else {
            parsed_frame_context_index
        };

        if intra_only || error_resilient {
            self.reset_independent_state();
        }
        let loop_filter =
            read_loop_filter(&mut bits, &mut self.reference_deltas, &mut self.mode_deltas)?;
        let quantization = read_quantization(&mut bits)?;
        let segmentation = read_segmentation(&mut bits, &mut self.segmentation)?;
        let frame_size = size.ok_or(Vp9Error::InvalidData("frame has no dimensions"))?;
        let (tile_columns_log2, tile_rows_log2) = read_tile_info(&mut bits, frame_size.width)?;
        let compressed_header_size = bits.read(16, "compressed-header size")? as usize;
        let uncompressed_header_size = bits.byte_position();
        let compressed_end = uncompressed_header_size
            .checked_add(compressed_header_size)
            .ok_or(Vp9Error::IntegerOverflow)?;
        if compressed_header_size == 0 {
            return Err(Vp9Error::InvalidData("compressed header is empty"));
        }
        if compressed_end > data.len() {
            return Err(Vp9Error::Truncated("compressed header"));
        }

        for slot in 0..REFERENCE_SLOTS {
            if refresh_frame_flags & (1 << slot) != 0 {
                self.reference_sizes[slot] = Some(frame_size);
                self.reference_colors[slot] = color;
            }
        }

        Ok(FrameHeader {
            profile,
            show_existing_frame: None,
            frame_type,
            show_frame,
            error_resilient,
            intra_only,
            reset_frame_context,
            color,
            size,
            refresh_frame_flags,
            reference_indices,
            reference_sign_bias,
            allow_high_precision_motion_vectors,
            interpolation_filter,
            refresh_frame_context,
            frame_parallel_decoding,
            frame_context_index,
            loop_filter: Some(loop_filter),
            quantization: Some(quantization),
            segmentation: Some(segmentation),
            tile_columns_log2,
            tile_rows_log2,
            uncompressed_header_size,
            compressed_header_size,
        })
    }

    fn read_frame_size_with_refs(
        &self,
        bits: &mut SyntaxBits<'_>,
        reference_indices: [u8; 3],
    ) -> Result<FrameSize> {
        let mut coded_size = None;
        for reference_index in reference_indices {
            if bits.bool("use-reference-size flag")? {
                let reference = self.reference_sizes[usize::from(reference_index)]
                    .ok_or(Vp9Error::MissingReference(usize::from(reference_index)))?;
                coded_size = Some((reference.width, reference.height));
                break;
            }
        }
        let (width, height) = match coded_size {
            Some(size) => size,
            None => read_frame_size(bits)?,
        };
        read_render_size(bits, width, height)
    }

    fn reset_independent_state(&mut self) {
        self.reference_deltas = [1, 0, -1, -1];
        self.mode_deltas = [0, 0];
        self.segmentation = Segmentation::default();
    }
}

fn default_profile_zero_color() -> ColorConfig {
    ColorConfig {
        bit_depth: BitDepth::Eight,
        color_space: ColorSpace::Bt601,
        full_range: false,
        subsampling: ChromaSubsampling::Cs420,
    }
}

fn read_sync_code(bits: &mut SyntaxBits<'_>) -> Result<()> {
    for expected in SYNC_BYTES {
        if bits.read(8, "sync code")? != expected {
            return Err(Vp9Error::InvalidData("incorrect frame sync code"));
        }
    }
    Ok(())
}

fn read_color_config(bits: &mut SyntaxBits<'_>, profile: u8) -> Result<ColorConfig> {
    let bit_depth = if profile >= 2 {
        if bits.bool("twelve-bit flag")? {
            BitDepth::Twelve
        } else {
            BitDepth::Ten
        }
    } else {
        BitDepth::Eight
    };
    let color_space = ColorSpace::from_bits(bits.read(3, "color space")?);
    if color_space == ColorSpace::Srgb {
        if profile == 0 || profile == 2 {
            return Err(Vp9Error::InvalidData(
                "sRGB is not valid for profile zero or two",
            ));
        }
        if bits.bit("sRGB reserved bit")? != 0 {
            return Err(Vp9Error::InvalidData("sRGB reserved bit must be zero"));
        }
        return Ok(ColorConfig {
            bit_depth,
            color_space,
            full_range: true,
            subsampling: ChromaSubsampling::Cs444,
        });
    }

    let full_range = bits.bool("color-range flag")?;
    let subsampling = if profile == 1 || profile == 3 {
        let subsampling_x = bits.bool("chroma subsampling x")?;
        let subsampling_y = bits.bool("chroma subsampling y")?;
        if bits.bit("chroma reserved bit")? != 0 {
            return Err(Vp9Error::InvalidData(
                "chroma subsampling reserved bit must be zero",
            ));
        }
        ChromaSubsampling::new(subsampling_x, subsampling_y)
    } else {
        ChromaSubsampling::Cs420
    };
    Ok(ColorConfig {
        bit_depth,
        color_space,
        full_range,
        subsampling,
    })
}

fn read_frame_size(bits: &mut SyntaxBits<'_>) -> Result<(u32, u32)> {
    let width = bits
        .read(16, "frame width")?
        .checked_add(1)
        .ok_or(Vp9Error::IntegerOverflow)?;
    let height = bits
        .read(16, "frame height")?
        .checked_add(1)
        .ok_or(Vp9Error::IntegerOverflow)?;
    Ok((width, height))
}

fn read_render_size(bits: &mut SyntaxBits<'_>, width: u32, height: u32) -> Result<FrameSize> {
    let (render_width, render_height) = if bits.bool("render-size-different flag")? {
        (
            bits.read(16, "render width")?
                .checked_add(1)
                .ok_or(Vp9Error::IntegerOverflow)?,
            bits.read(16, "render height")?
                .checked_add(1)
                .ok_or(Vp9Error::IntegerOverflow)?,
        )
    } else {
        (width, height)
    };
    Ok(FrameSize {
        width,
        height,
        render_width,
        render_height,
    })
}

fn read_frame_and_render_size(bits: &mut SyntaxBits<'_>) -> Result<FrameSize> {
    let (width, height) = read_frame_size(bits)?;
    read_render_size(bits, width, height)
}

fn read_loop_filter(
    bits: &mut SyntaxBits<'_>,
    reference_deltas: &mut [i8; 4],
    mode_deltas: &mut [i8; 2],
) -> Result<LoopFilter> {
    let level = bits.read(6, "loop-filter level")? as u8;
    let sharpness = bits.read(3, "loop-filter sharpness")? as u8;
    let mode_ref_delta_enabled = bits.bool("loop-filter delta-enabled flag")?;
    if mode_ref_delta_enabled && bits.bool("loop-filter delta-update flag")? {
        for delta in reference_deltas.iter_mut() {
            if bits.bool("reference-delta update flag")? {
                *delta = bits.signed(6, "reference delta")? as i8;
            }
        }
        for delta in mode_deltas.iter_mut() {
            if bits.bool("mode-delta update flag")? {
                *delta = bits.signed(6, "mode delta")? as i8;
            }
        }
    }
    Ok(LoopFilter {
        level,
        sharpness,
        mode_ref_delta_enabled,
        reference_deltas: *reference_deltas,
        mode_deltas: *mode_deltas,
    })
}

fn read_delta_q(bits: &mut SyntaxBits<'_>, name: &'static str) -> Result<i8> {
    if bits.bool("quantizer-delta-present flag")? {
        Ok(bits.signed(4, name)? as i8)
    } else {
        Ok(0)
    }
}

fn read_quantization(bits: &mut SyntaxBits<'_>) -> Result<Quantization> {
    Ok(Quantization {
        base_q_idx: bits.read(8, "base quantizer")? as u8,
        y_dc_delta: read_delta_q(bits, "Y DC quantizer delta")?,
        uv_dc_delta: read_delta_q(bits, "UV DC quantizer delta")?,
        uv_ac_delta: read_delta_q(bits, "UV AC quantizer delta")?,
    })
}

fn read_probability(bits: &mut SyntaxBits<'_>, name: &'static str) -> Result<u8> {
    if bits.bool("probability-update flag")? {
        Ok(bits.read(8, name)? as u8)
    } else {
        Ok(255)
    }
}

fn read_segmentation(bits: &mut SyntaxBits<'_>, state: &mut Segmentation) -> Result<Segmentation> {
    state.enabled = bits.bool("segmentation-enabled flag")?;
    state.update_map = false;
    state.temporal_update = false;
    state.update_data = false;
    if !state.enabled {
        return Ok(state.clone());
    }

    state.update_map = bits.bool("segmentation-update-map flag")?;
    if state.update_map {
        for probability in &mut state.tree_probabilities {
            *probability = read_probability(bits, "segment-tree probability")?;
        }
        state.temporal_update = bits.bool("segmentation-temporal-update flag")?;
        for probability in &mut state.prediction_probabilities {
            *probability = if state.temporal_update {
                read_probability(bits, "segment-prediction probability")?
            } else {
                255
            };
        }
    }

    state.update_data = bits.bool("segmentation-update-data flag")?;
    if state.update_data {
        state.absolute_values = bits.bool("segmentation-absolute-values flag")?;
        state.features = [[SegmentFeature::default(); SEGMENT_FEATURES]; MAX_SEGMENTS];
        const FEATURE_BITS: [u32; SEGMENT_FEATURES] = [8, 6, 2, 0];
        const FEATURE_SIGNED: [bool; SEGMENT_FEATURES] = [true, true, false, false];
        for segment in &mut state.features {
            for feature_index in 0..SEGMENT_FEATURES {
                let feature = &mut segment[feature_index];
                feature.enabled = bits.bool("segment-feature-enabled flag")?;
                if feature.enabled && FEATURE_BITS[feature_index] != 0 {
                    let magnitude =
                        bits.read(FEATURE_BITS[feature_index], "segment-feature value")? as i16;
                    feature.value =
                        if FEATURE_SIGNED[feature_index] && bits.bool("segment-feature sign")? {
                            -magnitude
                        } else {
                            magnitude
                        };
                }
            }
        }
    }
    Ok(state.clone())
}

fn read_tile_info(bits: &mut SyntaxBits<'_>, width: u32) -> Result<(u8, u8)> {
    let mi_columns = width.div_ceil(8);
    let superblock_columns = mi_columns.div_ceil(8);

    let mut minimum_columns_log2 = 0u8;
    while (MAX_TILE_WIDTH_B64 << minimum_columns_log2) < superblock_columns {
        minimum_columns_log2 += 1;
    }
    let mut maximum_columns_log2 = 1u8;
    while superblock_columns >> maximum_columns_log2 >= MIN_TILE_WIDTH_B64 {
        maximum_columns_log2 += 1;
    }
    maximum_columns_log2 = maximum_columns_log2.saturating_sub(1);

    let mut columns_log2 = minimum_columns_log2;
    while columns_log2 < maximum_columns_log2 && bits.bool("tile-column increment")? {
        columns_log2 += 1;
    }
    let mut rows_log2 = u8::from(bits.bool("tile-row flag")?);
    if rows_log2 != 0 {
        rows_log2 += u8::from(bits.bool("second tile-row flag")?);
    }
    Ok((columns_log2, rows_log2))
}

struct SyntaxBits<'a> {
    reader: BitReader<'a>,
}

impl<'a> SyntaxBits<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            reader: BitReader::new(data),
        }
    }

    fn read(&mut self, count: u32, field: &'static str) -> Result<u32> {
        self.reader
            .read_bits(count)
            .ok_or(Vp9Error::Truncated(field))
    }

    fn bit(&mut self, field: &'static str) -> Result<u8> {
        self.reader.read_bit().ok_or(Vp9Error::Truncated(field))
    }

    fn bool(&mut self, field: &'static str) -> Result<bool> {
        Ok(self.bit(field)? != 0)
    }

    fn signed(&mut self, magnitude_bits: u32, field: &'static str) -> Result<i32> {
        let magnitude = self.read(magnitude_bits, field)? as i32;
        Ok(if self.bool("signed-literal sign")? {
            -magnitude
        } else {
            magnitude
        })
    }

    fn byte_position(&self) -> usize {
        self.reader.bit_position().div_ceil(8)
    }
}

#[cfg(test)]
mod tests {
    use super::{BitDepth, ChromaSubsampling, ColorSpace, FrameType, HeaderParser};

    /// Packs syntax values in VP9's most-significant-bit-first order.
    fn pack(fields: &[(u32, u32)]) -> Vec<u8> {
        let mut bytes = Vec::new();
        let mut position = 0usize;
        for &(value, width) in fields {
            for bit in 0..width {
                if position / 8 == bytes.len() {
                    bytes.push(0);
                }
                let source_shift = width - bit - 1;
                let target_shift = 7 - (position & 7);
                bytes[position / 8] |= ((value >> source_shift) as u8 & 1) << target_shift;
                position += 1;
            }
        }
        bytes
    }

    #[test]
    fn rejects_wrong_frame_marker() {
        assert!(HeaderParser::new().parse(&[0]).is_err());
    }

    #[test]
    fn parses_minimal_profile_zero_keyframe_header() {
        let fields = [
            (2, 2), // marker
            (0, 1), // profile low
            (0, 1), // profile high
            (0, 1), // show existing
            (0, 1), // key
            (1, 1), // show frame
            (0, 1), // error resilient
            (0x49, 8),
            (0x83, 8),
            (0x42, 8),
            (1, 3),   // BT.709
            (0, 1),   // studio range
            (63, 16), // 64 wide
            (31, 16), // 32 high
            (0, 1),   // same render size
            (0, 1),   // refresh frame context
            (1, 1),   // frame parallel
            (0, 2),   // frame context index
            (0, 6),   // filter level
            (0, 3),   // sharpness
            (0, 1),   // filter deltas disabled
            (0, 8),   // base q
            (0, 1),   // y dc delta absent
            (0, 1),   // uv dc delta absent
            (0, 1),   // uv ac delta absent
            (0, 1),   // segmentation disabled
            (0, 1),   // tile rows
            (1, 16),  // compressed header size
        ];
        let mut data = pack(&fields);
        data.push(0);
        let header = HeaderParser::new().parse(&data).unwrap();
        assert_eq!(header.frame_type, FrameType::Key);
        assert!(header.show_frame);
        assert_eq!(header.size.unwrap().width, 64);
        assert_eq!(header.size.unwrap().height, 32);
        let color = header.color.unwrap();
        assert_eq!(color.bit_depth, BitDepth::Eight);
        assert_eq!(color.color_space, ColorSpace::Bt709);
        assert_eq!(color.subsampling, ChromaSubsampling::Cs420);
        assert!(header.quantization.unwrap().lossless());
    }
}
