use std::{collections::VecDeque, sync::Arc};

use decv_core::{
    BitstreamFormat, ColorInfo, ColorMatrix, ColorPrimaries, ColorRange, CpuFrame, CpuPlane,
    DecodeInputStatus, DecodeOutput, DecodedVideoFrame, EncodedVideoPacket, FrameStorage,
    PixelFormat, Rect, Size, TransferFunction, VideoCodec, VideoDecoder, VideoDecoderConfig,
    VideoFormat,
};

use crate::{
    BitDepth, ChromaSubsampling, ColorSpace, CompressedHeader, FrameHeader, FrameType,
    HeaderParser, IntraPicture, Result, Superframe, Vp9Error,
    context::{FrameCounts, ProbabilityContext},
    inter::{InterModeMap, decode_inter_picture_with_context},
    tile::decode_intra_picture_with_context,
};

#[derive(Debug)]
struct DecodedPicture {
    picture: Arc<IntraPicture>,
    format: VideoFormat,
}

/// Stateful native VP9 decoder.
///
/// Coded frames are accepted in decode order. A packet may contain a VP9
/// superframe, so one call can decode hidden reference frames before returning
/// a visible picture.
#[derive(Debug)]
pub struct Vp9Decoder {
    headers: HeaderParser,
    probability_contexts: [ProbabilityContext; 4],
    references: [Option<Arc<IntraPicture>>; 8],
    previous_frame_type: Option<FrameType>,
    previous_modes: Option<InterModeMap>,
    previous_size: Option<(u32, u32)>,
    previous_was_intra_only: bool,
    previous_was_shown: bool,
    configured: bool,
    draining: bool,
    outputs: VecDeque<DecodeOutput>,
    current_format: Option<VideoFormat>,
    next_frame_id: u64,
}

impl Default for Vp9Decoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Vp9Decoder {
    pub fn new() -> Self {
        Self {
            headers: HeaderParser::new(),
            probability_contexts: std::array::from_fn(|_| ProbabilityContext::default()),
            references: std::array::from_fn(|_| None),
            previous_frame_type: None,
            previous_modes: None,
            previous_size: None,
            previous_was_intra_only: true,
            previous_was_shown: false,
            configured: false,
            draining: false,
            outputs: VecDeque::new(),
            current_format: None,
            next_frame_id: 0,
        }
    }

    /// Decodes every coded frame in one ordinary VP9 packet or superframe and
    /// returns the pictures marked for display.
    pub fn decode_packet(&mut self, packet: &[u8]) -> Result<Vec<Arc<IntraPicture>>> {
        self.decode_packet_inner(packet).map(|pictures| {
            pictures
                .into_iter()
                .map(|decoded| decoded.picture)
                .collect()
        })
    }

    fn decode_packet_inner(&mut self, packet: &[u8]) -> Result<Vec<DecodedPicture>> {
        let superframe = Superframe::parse(packet)?;
        let mut output = Vec::with_capacity(superframe.len());
        for frame in superframe.frames(packet) {
            if let Some(picture) = self.decode_coded_frame(frame)? {
                output.push(picture);
            }
        }
        Ok(output)
    }

    fn decode_coded_frame(&mut self, frame: &[u8]) -> Result<Option<DecodedPicture>> {
        let header = self.headers.parse(frame)?;
        if let Some(slot) = header.show_existing_frame {
            let picture = self.references[usize::from(slot)]
                .clone()
                .ok_or(Vp9Error::MissingReference(usize::from(slot)))?;
            return Ok(Some(DecodedPicture {
                picture,
                format: video_format(&header)?,
            }));
        }
        if header.profile != 0
            || header.color.is_some_and(|color| {
                color.bit_depth != BitDepth::Eight || color.subsampling != ChromaSubsampling::Cs420
            })
        {
            return Err(Vp9Error::UnsupportedFeature(
                "reconstruction currently supports 8-bit VP9 Profile 0 4:2:0",
            ));
        }

        self.reset_probability_contexts(&header);
        let context_index = usize::from(header.frame_context_index);
        let previous_context = self.probability_contexts[context_index].clone();
        let compressed = CompressedHeader::parse(frame, &header)?;
        let mut current_context = previous_context.clone();
        current_context.apply(&compressed)?;
        let mut counts = FrameCounts::default();

        let size = header
            .size
            .ok_or(Vp9Error::InvalidData("frame has no dimensions"))?;
        let current_size = (size.width, size.height);
        let use_previous_modes = !header.error_resilient
            && self.previous_size == Some(current_size)
            && !self.previous_was_intra_only
            && self.previous_was_shown
            && self.previous_frame_type != Some(FrameType::Key);
        let previous_segment_ids = (self.previous_size == Some(current_size))
            .then(|| self.previous_modes.as_ref().map(InterModeMap::segment_ids))
            .flatten();

        let (picture, mode_map) = if header.intra_only {
            let (picture, decoded_segment_ids) = decode_intra_picture_with_context(
                frame,
                &header,
                &compressed,
                &current_context,
                &mut counts,
            )?;
            let segment_ids = if header
                .segmentation
                .as_ref()
                .is_some_and(|segmentation| segmentation.enabled && !segmentation.update_map)
            {
                previous_segment_ids
                    .map(<[u8]>::to_vec)
                    .unwrap_or_else(|| vec![0; decoded_segment_ids.len()])
            } else {
                decoded_segment_ids
            };
            (
                picture,
                InterModeMap::intra(current_size.0, current_size.1, segment_ids)?,
            )
        } else {
            let references = self.resolve_references(&header)?;
            decode_inter_picture_with_context(
                frame,
                &header,
                &compressed,
                &current_context,
                references,
                &mut counts,
                use_previous_modes
                    .then_some(self.previous_modes.as_ref())
                    .flatten(),
                previous_segment_ids,
            )?
        };
        let picture = Arc::new(picture);

        if !header.error_resilient && !header.frame_parallel_decoding {
            current_context.adapt_coefficients(
                &previous_context,
                &counts.coefficient,
                header.intra_only,
                self.previous_frame_type == Some(FrameType::Key),
            );
            if !header.intra_only {
                current_context.adapt_modes(
                    &previous_context,
                    &counts,
                    compressed.transform_mode == crate::TransformMode::Select,
                    header.interpolation_filter == crate::InterpolationFilter::Switchable,
                );
                current_context.adapt_motion_vectors(
                    &previous_context,
                    &counts,
                    header.allow_high_precision_motion_vectors,
                );
            }
        }
        if header.refresh_frame_context {
            self.probability_contexts[context_index] = current_context;
        }
        for slot in 0..8 {
            if header.refresh_frame_flags & (1 << slot) != 0 {
                self.references[slot] = Some(Arc::clone(&picture));
            }
        }
        self.previous_frame_type = Some(header.frame_type);
        self.previous_modes = Some(mode_map);
        self.previous_size = Some(current_size);
        self.previous_was_intra_only = header.intra_only;
        self.previous_was_shown = header.show_frame;

        if header.show_frame {
            Ok(Some(DecodedPicture {
                picture,
                format: video_format(&header)?,
            }))
        } else {
            Ok(None)
        }
    }

    fn reset_probability_contexts(&mut self, header: &FrameHeader) {
        if header.intra_only || header.error_resilient {
            let default = ProbabilityContext::default();
            if header.frame_type == FrameType::Key
                || header.error_resilient
                || header.reset_frame_context == 3
            {
                self.probability_contexts.fill(default);
            } else if header.reset_frame_context == 2 {
                self.probability_contexts[usize::from(header.frame_context_index)] = default;
            }
        }
    }

    fn resolve_references(&self, header: &FrameHeader) -> Result<[&IntraPicture; 3]> {
        let mut resolved: [Option<&IntraPicture>; 3] = [None, None, None];
        for (reference, &slot) in header.reference_indices.iter().enumerate() {
            resolved[reference] = self.references[usize::from(slot)].as_deref();
            if resolved[reference].is_none() {
                return Err(Vp9Error::MissingReference(usize::from(slot)));
            }
        }
        Ok(resolved.map(Option::unwrap))
    }

    fn reset_decode_state(&mut self) {
        self.headers = HeaderParser::new();
        self.probability_contexts = std::array::from_fn(|_| ProbabilityContext::default());
        self.references = std::array::from_fn(|_| None);
        self.previous_frame_type = None;
        self.previous_modes = None;
        self.previous_size = None;
        self.previous_was_intra_only = true;
        self.previous_was_shown = false;
        self.draining = false;
        self.outputs.clear();
        self.current_format = None;
        self.next_frame_id = 0;
    }
}

impl VideoDecoder for Vp9Decoder {
    type Error = Vp9Error;

    fn configure(&mut self, config: VideoDecoderConfig) -> Result<()> {
        config.validate()?;
        if !matches!(config.codec, VideoCodec::Vp9) {
            return Err(Vp9Error::UnsupportedFeature(
                "decoder configuration for a non-VP9 codec",
            ));
        }
        if !matches!(config.bitstream_format, BitstreamFormat::Frame) {
            return Err(Vp9Error::UnsupportedFeature(
                "VP9 input must contain complete frames or superframes",
            ));
        }
        self.reset_decode_state();
        self.configured = true;
        Ok(())
    }

    fn send_packet(&mut self, packet: EncodedVideoPacket) -> Result<DecodeInputStatus> {
        if !self.configured {
            return Err(Vp9Error::InvalidData(
                "VP9 decoder must be configured before input",
            ));
        }
        if self.draining {
            return Err(Vp9Error::InvalidData(
                "VP9 decoder cannot accept input while draining",
            ));
        }
        if !self.outputs.is_empty() {
            return Ok(DecodeInputStatus::NeedOutput(packet));
        }
        if packet.discontinuity {
            self.reset_decode_state();
        }

        let pts = packet.pts;
        let duration = packet.duration;
        for decoded in self.decode_packet_inner(&packet.data)? {
            if self.current_format != Some(decoded.format) {
                self.outputs
                    .push_back(DecodeOutput::FormatChanged(decoded.format));
                self.current_format = Some(decoded.format);
            }
            let picture = decoded.picture;
            let chroma_height = picture.height().div_ceil(2);
            let storage = FrameStorage::Cpu(CpuFrame::new(vec![
                CpuPlane::new(
                    picture.shared_plane(0),
                    0,
                    picture.stride(0),
                    picture.height(),
                ),
                CpuPlane::new(picture.shared_plane(1), 0, picture.stride(1), chroma_height),
                CpuPlane::new(picture.shared_plane(2), 0, picture.stride(2), chroma_height),
            ]));
            let frame =
                DecodedVideoFrame::new(self.next_frame_id, pts, duration, decoded.format, storage);
            frame.validate()?;
            self.next_frame_id = self
                .next_frame_id
                .checked_add(1)
                .ok_or(Vp9Error::IntegerOverflow)?;
            self.outputs.push_back(DecodeOutput::Frame(frame));
        }
        Ok(DecodeInputStatus::Accepted)
    }

    fn receive_frame(&mut self) -> Result<DecodeOutput> {
        if let Some(output) = self.outputs.pop_front() {
            return Ok(output);
        }
        if self.draining {
            Ok(DecodeOutput::EndOfStream)
        } else {
            Ok(DecodeOutput::NeedInput)
        }
    }

    fn flush(&mut self) {
        self.reset_decode_state();
    }

    fn drain(&mut self) -> Result<()> {
        if !self.configured {
            return Err(Vp9Error::InvalidData(
                "VP9 decoder must be configured before draining",
            ));
        }
        self.draining = true;
        Ok(())
    }
}

fn video_format(header: &FrameHeader) -> Result<VideoFormat> {
    let size = header
        .size
        .ok_or(Vp9Error::InvalidData("frame has no dimensions"))?;
    let coded = Size::new(size.width, size.height);
    let color = header.color.map_or_else(ColorInfo::default, |config| {
        let range = if config.full_range {
            ColorRange::Full
        } else {
            ColorRange::Limited
        };
        let (matrix, primaries, transfer) = match config.color_space {
            ColorSpace::Bt601 => (
                ColorMatrix::Bt601,
                ColorPrimaries::Unspecified,
                TransferFunction::Unspecified,
            ),
            ColorSpace::Bt709 => (
                ColorMatrix::Bt709,
                ColorPrimaries::Bt709,
                TransferFunction::Bt709,
            ),
            ColorSpace::Smpte170 => (
                ColorMatrix::Smpte170M,
                ColorPrimaries::Bt601_525,
                TransferFunction::Smpte170M,
            ),
            ColorSpace::Smpte240 => (
                ColorMatrix::Other(7),
                ColorPrimaries::Other(7),
                TransferFunction::Other(7),
            ),
            ColorSpace::Bt2020 => (
                ColorMatrix::Bt2020NonConstantLuminance,
                ColorPrimaries::Bt2020,
                TransferFunction::Bt2020TenBit,
            ),
            ColorSpace::Smpte431 => (
                ColorMatrix::Other(0),
                ColorPrimaries::Other(11),
                TransferFunction::Unspecified,
            ),
            ColorSpace::Srgb => (
                ColorMatrix::Identity,
                ColorPrimaries::Bt709,
                TransferFunction::Srgb,
            ),
            ColorSpace::Reserved => (
                ColorMatrix::Unspecified,
                ColorPrimaries::Unspecified,
                TransferFunction::Unspecified,
            ),
        };
        ColorInfo::new(range, matrix, primaries, transfer)
    });
    let format = VideoFormat::new(
        coded,
        Rect::new(0, 0, size.width, size.height),
        Size::new(size.render_width, size.render_height),
        PixelFormat::I420,
        color,
    );
    format.validate()?;
    Ok(format)
}

#[cfg(test)]
mod tests {
    use decv_core::{BitstreamFormat, DecodeOutput, VideoCodec, VideoDecoder, VideoDecoderConfig};

    use super::Vp9Decoder;

    #[test]
    fn contract_requires_vp9_frame_packets() {
        let mut decoder = Vp9Decoder::new();
        assert!(
            decoder
                .configure(VideoDecoderConfig::new(
                    VideoCodec::H264,
                    BitstreamFormat::Frame,
                ))
                .is_err()
        );
        assert!(
            decoder
                .configure(VideoDecoderConfig::new(
                    VideoCodec::Vp9,
                    BitstreamFormat::ByteStream,
                ))
                .is_err()
        );
        decoder
            .configure(VideoDecoderConfig::new(
                VideoCodec::Vp9,
                BitstreamFormat::Frame,
            ))
            .unwrap();
    }

    #[test]
    fn drain_and_flush_follow_decoder_contract() {
        let mut decoder = Vp9Decoder::new();
        decoder
            .configure(VideoDecoderConfig::new(
                VideoCodec::Vp9,
                BitstreamFormat::Frame,
            ))
            .unwrap();
        decoder.drain().unwrap();
        assert!(matches!(
            decoder.receive_frame().unwrap(),
            DecodeOutput::EndOfStream
        ));
        decoder.flush();
        assert!(matches!(
            decoder.receive_frame().unwrap(),
            DecodeOutput::NeedInput
        ));
    }
}
