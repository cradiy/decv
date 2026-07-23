//! Top-level NAL dispatch, picture-boundary detection, and frame output.

use std::{borrow::Cow, collections::VecDeque, sync::Arc};

use bit_readers::BitReader;
use decv_core::{
    BitstreamFormat, DecodeInputStatus, DecodeOutput, EncodedVideoPacket, MediaTime, PixelFormat,
    Size, VideoCodec, VideoDecoder, VideoDecoderConfig, VideoFormat,
};

use crate::{
    AnnexBNalUnit, AnnexBReader, DecodedPictureBuffer, H264Error, IntraPictureReconstructor,
    NalHeader, NalUnit, NalUnitType, ParameterSetStore, ParsedSliceHeader, PictureOrderCount,
    PictureParameterSet, ReferencePictureMarking, Result, SequenceParameterSet, SliceType,
    consume_rbsp_trailing_bits, decode_rbsp,
};

#[derive(Debug)]
struct PendingPicture {
    reconstructor: IntraPictureReconstructor,
    format: VideoFormat,
    pts: Option<MediaTime>,
    duration: Option<MediaTime>,
    nal_header: NalHeader,
    frame_num: u32,
    picture_order_count: PictureOrderCount,
    reference_picture_marking: ReferencePictureMarking,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DpbConfiguration {
    max_num_ref_frames: u32,
    log2_max_frame_num: u8,
    coded_size: Size,
}

/// Synchronous pure-Rust H.264 decoder implementing the codec-independent
/// push/pull contract.
///
/// The current reconstruction backend accepts progressive Annex-B CAVLC I and
/// unweighted P pictures. Other H.264 coding tools return explicit
/// [`H264Error::UnsupportedFeature`] errors.
#[derive(Debug)]
pub struct H264Decoder {
    configured: bool,
    parser: H264StreamParser,
    current_picture: Option<PendingPicture>,
    dpb: Option<DecodedPictureBuffer>,
    dpb_configuration: Option<DpbConfiguration>,
    outputs: VecDeque<DecodeOutput>,
    announced_format: Option<VideoFormat>,
    next_frame_id: u64,
    draining: bool,
}

impl Default for H264Decoder {
    fn default() -> Self {
        Self {
            configured: false,
            parser: H264StreamParser::new(),
            current_picture: None,
            dpb: None,
            dpb_configuration: None,
            outputs: VecDeque::new(),
            announced_format: None,
            next_frame_id: 0,
            draining: false,
        }
    }
}

impl H264Decoder {
    pub fn new() -> Self {
        Self::default()
    }

    fn process_packet(&mut self, packet: &EncodedVideoPacket) -> Result<()> {
        for unit in AnnexBReader::new(packet.data.as_ref()) {
            let event = self.parser.push_annex_b(unit?)?;
            match event {
                ParserEvent::SequenceParameterSet(_)
                | ParserEvent::PictureParameterSet(_)
                | ParserEvent::Unhandled(_) => {}
                ParserEvent::Slice {
                    parsed,
                    rbsp,
                    starts_new_picture,
                    picture_order_count,
                    nal_header,
                } => {
                    if starts_new_picture {
                        self.finish_current_picture()?;
                    }
                    if self.current_picture.is_none() {
                        self.ensure_dpb(&parsed, nal_header)?;
                        let format = video_format(&parsed)?;
                        let reconstructor =
                            IntraPictureReconstructor::from_parameter_sets(&parsed.parameter_sets)?;
                        self.current_picture = Some(PendingPicture {
                            reconstructor,
                            format,
                            pts: packet.pts,
                            duration: packet.duration,
                            nal_header,
                            frame_num: parsed.header.frame_num,
                            picture_order_count,
                            reference_picture_marking: parsed
                                .header
                                .reference_picture_marking
                                .clone(),
                        });
                    }
                    match parsed.header.slice_type {
                        SliceType::I => {
                            self.current_picture
                                .as_mut()
                                .expect("the picture is initialized above")
                                .reconstructor
                                .decode_cavlc_intra_slice(rbsp.as_ref(), &parsed)?;
                        }
                        SliceType::P => {
                            let references = self
                                .dpb
                                .as_ref()
                                .expect("the DPB is initialized with the picture")
                                .p_list0(
                                    parsed.header.frame_num,
                                    parsed.header.num_ref_idx_l0_active,
                                    &parsed.header.ref_pic_list_modifications_l0,
                                )?;
                            let borrowed = references
                                .iter()
                                .map(|picture| picture.as_deref())
                                .collect::<Vec<_>>();
                            self.current_picture
                                .as_mut()
                                .expect("the picture is initialized above")
                                .reconstructor
                                .decode_cavlc_p_slice(rbsp.as_ref(), &parsed, &borrowed)?;
                        }
                        SliceType::B | SliceType::Sp | SliceType::Si => {
                            return Err(H264Error::UnsupportedFeature(
                                "top-level reconstruction of B, SP, and SI slices",
                            ));
                        }
                    }
                }
                ParserEvent::AccessUnitDelimiter { .. } => {
                    self.finish_current_picture()?;
                }
                ParserEvent::EndOfSequence => {
                    self.finish_current_picture()?;
                    self.clear_dpb();
                }
                ParserEvent::EndOfStream => {
                    self.finish_current_picture()?;
                    self.draining = true;
                }
            }
        }
        Ok(())
    }

    fn finish_current_picture(&mut self) -> Result<()> {
        let Some(picture) = self.current_picture.take() else {
            return Ok(());
        };
        self.next_frame_id = self
            .next_frame_id
            .checked_add(1)
            .ok_or(H264Error::IntegerOverflow)?;
        let decoded = Arc::new(picture.reconstructor.into_deblocked_picture()?);
        let frame = decoded.to_nv12_frame(
            self.next_frame_id,
            picture.pts,
            picture.duration,
            picture.format,
        )?;
        if picture.nal_header.nal_ref_idc != 0 {
            let picture_order_count = picture.picture_order_count.stored.picture_order_count();
            let dpb = self
                .dpb
                .as_mut()
                .expect("a pending picture always has an initialized DPB");
            match &picture.reference_picture_marking {
                ReferencePictureMarking::Idr {
                    long_term_reference,
                    ..
                } => dpb.store_idr(picture_order_count, decoded.clone(), *long_term_reference)?,
                ReferencePictureMarking::SlidingWindow => {
                    dpb.store_short_term(picture.frame_num, picture_order_count, decoded.clone())?
                }
                ReferencePictureMarking::Adaptive(operations) => dpb.store_adaptive(
                    picture.frame_num,
                    picture_order_count,
                    decoded.clone(),
                    operations,
                )?,
                ReferencePictureMarking::None => {
                    return Err(H264Error::InvalidSyntax(
                        "reference picture is missing decoded-picture-buffer marking",
                    ));
                }
            }
        }
        if self.announced_format != Some(picture.format) {
            self.outputs
                .push_back(DecodeOutput::FormatChanged(picture.format));
            self.announced_format = Some(picture.format);
        }
        self.outputs.push_back(DecodeOutput::Frame(frame));
        Ok(())
    }

    fn reset_all_state(&mut self) {
        self.parser.reset();
        self.current_picture = None;
        self.clear_dpb();
        self.outputs.clear();
        self.announced_format = None;
        self.next_frame_id = 0;
        self.draining = false;
    }

    fn flush_timeline(&mut self) {
        self.parser.reset_picture_history();
        self.current_picture = None;
        self.clear_dpb();
        self.outputs.clear();
        self.draining = false;
    }

    fn ensure_dpb(&mut self, parsed: &ParsedSliceHeader, nal_header: NalHeader) -> Result<()> {
        let sps = &parsed.parameter_sets.sequence;
        let configuration = DpbConfiguration {
            max_num_ref_frames: sps.max_num_ref_frames,
            log2_max_frame_num: sps.log2_max_frame_num,
            coded_size: sps.coded_size,
        };
        if self.dpb_configuration == Some(configuration) {
            return Ok(());
        }
        if self.dpb.is_some() && nal_header.unit_type != NalUnitType::IdrSlice {
            return Err(H264Error::InvalidSyntax(
                "DPB parameter change requires an IDR picture",
            ));
        }
        self.dpb = Some(DecodedPictureBuffer::new(
            configuration.max_num_ref_frames,
            configuration.log2_max_frame_num,
        )?);
        self.dpb_configuration = Some(configuration);
        Ok(())
    }

    fn clear_dpb(&mut self) {
        self.dpb = None;
        self.dpb_configuration = None;
    }
}

impl VideoDecoder for H264Decoder {
    type Error = H264Error;

    fn configure(&mut self, config: VideoDecoderConfig) -> Result<()> {
        config.validate()?;
        if !matches!(config.codec, VideoCodec::H264) {
            return Err(H264Error::UnsupportedFeature(
                "decoder configuration for a non-H.264 codec",
            ));
        }
        if !matches!(config.bitstream_format, BitstreamFormat::ByteStream) {
            return Err(H264Error::UnsupportedFeature("length-prefixed H.264 input"));
        }
        if config.codec_data.is_some() {
            return Err(H264Error::UnsupportedFeature(
                "out-of-band H.264 codec configuration",
            ));
        }
        self.reset_all_state();
        self.configured = true;
        Ok(())
    }

    fn send_packet(&mut self, packet: EncodedVideoPacket) -> Result<DecodeInputStatus> {
        if !self.configured {
            return Err(H264Error::InvalidSyntax(
                "H.264 decoder must be configured before input",
            ));
        }
        if self.draining {
            return Err(H264Error::InvalidSyntax(
                "H.264 decoder cannot accept input while draining",
            ));
        }
        if !self.outputs.is_empty() {
            return Ok(DecodeInputStatus::NeedOutput(packet));
        }
        if packet.discontinuity {
            self.flush_timeline();
        }
        self.process_packet(&packet)?;
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
        self.flush_timeline();
    }

    fn drain(&mut self) -> Result<()> {
        if !self.configured {
            return Err(H264Error::InvalidSyntax(
                "H.264 decoder must be configured before draining",
            ));
        }
        self.finish_current_picture()?;
        self.draining = true;
        Ok(())
    }
}

fn video_format(parsed: &ParsedSliceHeader) -> Result<VideoFormat> {
    let sps = &parsed.parameter_sets.sequence;
    let format = VideoFormat {
        coded_size: sps.coded_size,
        visible_rect: sps.visible_rect,
        display_size: sps.display_size,
        pixel_format: PixelFormat::Nv12,
        color: sps.vui.as_ref().map(|vui| vui.color).unwrap_or_default(),
    };
    format.validate()?;
    Ok(format)
}

#[derive(Debug)]
#[non_exhaustive]
// Slice events dominate normal decoding. Keeping the parsed header inline
// avoids an otherwise unconditional heap allocation for every slice NAL.
#[allow(clippy::large_enum_variant)]
pub enum ParserEvent<'a> {
    SequenceParameterSet(Arc<SequenceParameterSet>),
    PictureParameterSet(Arc<PictureParameterSet>),
    Slice {
        parsed: ParsedSliceHeader,
        nal_header: NalHeader,
        /// Unescaped payload retained for the following slice-data parser.
        rbsp: Cow<'a, [u8]>,
        starts_new_picture: bool,
        picture_order_count: crate::PictureOrderCount,
    },
    AccessUnitDelimiter {
        primary_pic_type: u8,
    },
    EndOfSequence,
    EndOfStream,
    Unhandled(NalUnitType),
}

#[derive(Debug, Default)]
pub struct H264StreamParser {
    parameter_sets: ParameterSetStore,
    previous_vcl: Option<PictureIdentity>,
    poc_state: crate::PictureOrderCountState,
    current_picture_order_count: Option<crate::PictureOrderCount>,
}

impl H264StreamParser {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn parameter_sets(&self) -> &ParameterSetStore {
        &self.parameter_sets
    }

    pub fn push_annex_b<'a>(&mut self, unit: AnnexBNalUnit<'a>) -> Result<ParserEvent<'a>> {
        self.push_nal(NalUnit::try_from(unit)?)
    }

    pub fn push_nal<'a>(&mut self, nal: NalUnit<'a>) -> Result<ParserEvent<'a>> {
        match nal.header.unit_type {
            NalUnitType::Sps => {
                let rbsp = decode_rbsp(nal.ebsp)?;
                let sps = self.parameter_sets.parse_sps(rbsp.as_ref())?;
                Ok(ParserEvent::SequenceParameterSet(sps))
            }
            NalUnitType::Pps => {
                let rbsp = decode_rbsp(nal.ebsp)?;
                let pps = self.parameter_sets.parse_pps(rbsp.as_ref())?;
                Ok(ParserEvent::PictureParameterSet(pps))
            }
            NalUnitType::NonIdrSlice | NalUnitType::IdrSlice => {
                let rbsp = decode_rbsp(nal.ebsp)?;
                let parsed =
                    ParsedSliceHeader::parse(rbsp.as_ref(), nal.header, &self.parameter_sets)?;
                let identity = PictureIdentity::from_slice(
                    &parsed,
                    nal.header.unit_type,
                    nal.header.nal_ref_idc,
                );
                let starts_new_picture = self
                    .previous_vcl
                    .as_ref()
                    .is_none_or(|previous| identity.starts_new_picture_after(previous));
                let picture_order_count = if starts_new_picture {
                    self.poc_state.derive(&parsed, nal.header)?
                } else {
                    self.current_picture_order_count
                        .expect("a continued picture has a previously derived POC")
                };
                self.previous_vcl = Some(identity);
                self.current_picture_order_count = Some(picture_order_count);
                Ok(ParserEvent::Slice {
                    parsed,
                    nal_header: nal.header,
                    rbsp,
                    starts_new_picture,
                    picture_order_count,
                })
            }
            NalUnitType::AccessUnitDelimiter => {
                let rbsp = decode_rbsp(nal.ebsp)?;
                let primary_pic_type = parse_access_unit_delimiter(rbsp.as_ref())?;
                self.previous_vcl = None;
                self.current_picture_order_count = None;
                Ok(ParserEvent::AccessUnitDelimiter { primary_pic_type })
            }
            NalUnitType::EndOfSequence => {
                validate_empty_rbsp(nal.ebsp)?;
                self.previous_vcl = None;
                self.current_picture_order_count = None;
                self.poc_state.reset();
                Ok(ParserEvent::EndOfSequence)
            }
            NalUnitType::EndOfStream => {
                validate_empty_rbsp(nal.ebsp)?;
                self.previous_vcl = None;
                self.current_picture_order_count = None;
                self.poc_state.reset();
                Ok(ParserEvent::EndOfStream)
            }
            unit_type if unit_type.is_vcl() => Err(H264Error::UnsupportedFeature(
                "slice partitions and slice extensions are not supported",
            )),
            NalUnitType::SpsExtension | NalUnitType::SubsetSps => {
                Err(H264Error::UnsupportedFeature(
                    "SPS extensions and subset SPS NAL units are not supported",
                ))
            }
            unit_type => Ok(ParserEvent::Unhandled(unit_type)),
        }
    }

    /// Clears parameter sets and picture-boundary history.
    pub fn reset(&mut self) {
        self.parameter_sets.clear();
        self.reset_picture_history();
    }

    fn reset_picture_history(&mut self) {
        self.previous_vcl = None;
        self.poc_state.reset();
        self.current_picture_order_count = None;
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PictureIdentity {
    frame_num: u32,
    picture_parameter_set_id: u32,
    field_picture: bool,
    bottom_field: bool,
    nal_ref_idc: u8,
    is_idr: bool,
    idr_pic_id: Option<u32>,
    pic_order_cnt_lsb: Option<u32>,
    delta_pic_order_bottom: Option<i32>,
    delta_pic_order: [Option<i32>; 2],
}

impl PictureIdentity {
    fn from_slice(parsed: &ParsedSliceHeader, unit_type: NalUnitType, nal_ref_idc: u8) -> Self {
        let header = &parsed.header;
        Self {
            frame_num: header.frame_num,
            picture_parameter_set_id: header.picture_parameter_set_id,
            field_picture: header.field_picture,
            bottom_field: header.bottom_field,
            nal_ref_idc,
            is_idr: unit_type == NalUnitType::IdrSlice,
            idr_pic_id: header.idr_pic_id,
            pic_order_cnt_lsb: header.picture_order.pic_order_cnt_lsb,
            delta_pic_order_bottom: header.picture_order.delta_pic_order_bottom,
            delta_pic_order: header.picture_order.delta_pic_order,
        }
    }

    /// Implements the ordinary AVC first-VCL-NAL test from H.264 7.4.1.2.4.
    fn starts_new_picture_after(&self, previous: &Self) -> bool {
        self.frame_num != previous.frame_num
            || self.picture_parameter_set_id != previous.picture_parameter_set_id
            || self.field_picture != previous.field_picture
            || (self.field_picture
                && previous.field_picture
                && self.bottom_field != previous.bottom_field)
            || ((self.nal_ref_idc == 0) != (previous.nal_ref_idc == 0))
            || self.pic_order_cnt_lsb != previous.pic_order_cnt_lsb
            || self.delta_pic_order_bottom != previous.delta_pic_order_bottom
            || self.delta_pic_order != previous.delta_pic_order
            || self.is_idr != previous.is_idr
            || (self.is_idr && previous.is_idr && self.idr_pic_id != previous.idr_pic_id)
    }
}

fn parse_access_unit_delimiter(rbsp: &[u8]) -> Result<u8> {
    let mut reader = BitReader::new(rbsp);
    let primary_pic_type = reader
        .read_bits_const::<3>()
        .ok_or(H264Error::UnexpectedEof)? as u8;
    consume_rbsp_trailing_bits(&mut reader)?;
    Ok(primary_pic_type)
}

fn validate_empty_rbsp(ebsp: &[u8]) -> Result<()> {
    let rbsp = decode_rbsp(ebsp)?;
    if rbsp.is_empty() {
        return Ok(());
    }

    let mut reader = BitReader::new(rbsp.as_ref());
    consume_rbsp_trailing_bits(&mut reader)
}

#[cfg(test)]
mod tests {
    use decv_core::{
        BitstreamFormat, ColorInfo, DecodeInputStatus, DecodeOutput, EncodedVideoPacket,
        FrameStorage, MediaTime, PixelFormat, Rect, Size, VideoCodec, VideoDecoder,
        VideoDecoderConfig, VideoFormat,
    };

    use super::{H264Decoder, H264StreamParser, ParserEvent, PictureIdentity};
    use crate::{AnnexBReader, H264Error, IntraPictureReconstructor, NalHeader, NalUnit};

    #[test]
    fn dispatches_an_annex_b_stream_and_detects_picture_boundaries() {
        let nals = [
            (0x67, sps_rbsp()),
            (0x68, pps_rbsp()),
            (0x01, i_slice_rbsp(0, 0, 0)),
            (0x01, i_slice_rbsp(1, 0, 0)),
            (0x01, i_slice_rbsp(0, 1, 2)),
        ];
        let stream = annex_b_stream(&nals);
        let mut parser = H264StreamParser::new();
        let mut boundaries = Vec::new();
        let mut parameter_set_events = 0;

        for unit in AnnexBReader::new(&stream) {
            match parser.push_annex_b(unit.unwrap()).unwrap() {
                ParserEvent::SequenceParameterSet(sps) => {
                    assert_eq!(sps.coded_size.width, 64);
                    parameter_set_events += 1;
                }
                ParserEvent::PictureParameterSet(pps) => {
                    assert_eq!(pps.id, 0);
                    parameter_set_events += 1;
                }
                ParserEvent::Slice {
                    parsed,
                    rbsp,
                    starts_new_picture,
                    picture_order_count,
                    ..
                } => {
                    assert!(parsed.header.bit_size <= rbsp.len() * 8);
                    assert_eq!(
                        picture_order_count.stored.picture_order_count(),
                        parsed.header.picture_order.pic_order_cnt_lsb.unwrap() as i32
                    );
                    boundaries.push(starts_new_picture);
                }
                event => panic!("unexpected parser event: {event:?}"),
            }
        }

        assert_eq!(parameter_set_events, 2);
        assert_eq!(boundaries, [true, false, true]);
    }

    #[test]
    fn reconstructs_a_complete_annex_b_cavlc_idr_picture() {
        let stream = annex_b_stream(&[
            (0x67, single_macroblock_sps_rbsp()),
            (0x68, single_macroblock_pps_rbsp()),
            (0x65, single_macroblock_idr_rbsp()),
        ]);
        let mut parser = H264StreamParser::new();
        let mut reconstructor = None;
        for unit in AnnexBReader::new(&stream) {
            match parser.push_annex_b(unit.unwrap()).unwrap() {
                ParserEvent::SequenceParameterSet(sps) => {
                    assert_eq!(sps.coded_size, Size::new(16, 16));
                }
                ParserEvent::PictureParameterSet(_) => {}
                ParserEvent::Slice {
                    parsed,
                    rbsp,
                    starts_new_picture,
                    picture_order_count,
                    ..
                } => {
                    assert!(starts_new_picture);
                    assert_eq!(picture_order_count.stored.top, Some(0));
                    let mut picture =
                        IntraPictureReconstructor::from_parameter_sets(&parsed.parameter_sets)
                            .unwrap();
                    assert_eq!(
                        picture.decode_cavlc_intra_slice(rbsp.as_ref(), &parsed),
                        Ok(1)
                    );
                    reconstructor = Some(picture);
                }
                event => panic!("unexpected parser event: {event:?}"),
            }
        }

        let size = Size::new(16, 16);
        let frame = reconstructor
            .unwrap()
            .into_nv12_frame(
                1,
                None,
                None,
                VideoFormat {
                    coded_size: size,
                    visible_rect: Rect::new(0, 0, 16, 16),
                    display_size: size,
                    pixel_format: PixelFormat::Nv12,
                    color: ColorInfo::default(),
                },
            )
            .unwrap();
        let cpu = match frame.storage {
            FrameStorage::Cpu(cpu) => cpu,
            _ => panic!("expected CPU frame"),
        };
        assert_eq!(cpu.planes[0].bytes.len(), 384);
        assert!(cpu.planes[0].bytes.iter().all(|&sample| sample == 128));
    }

    #[test]
    fn exposes_cavlc_idr_through_the_video_decoder_contract() {
        let stream = annex_b_stream(&[
            (0x67, single_macroblock_sps_rbsp()),
            (0x68, single_macroblock_pps_rbsp()),
            (0x65, single_macroblock_idr_rbsp()),
        ]);
        let mut decoder = H264Decoder::new();
        decoder.configure(byte_stream_config()).unwrap();
        let mut packet = EncodedVideoPacket::new(stream);
        packet.pts = MediaTime::from_parts(10, 30);
        packet.duration = MediaTime::from_parts(1, 30);
        assert!(matches!(
            decoder.send_packet(packet).unwrap(),
            DecodeInputStatus::Accepted
        ));
        assert!(matches!(
            decoder.receive_frame().unwrap(),
            DecodeOutput::NeedInput
        ));

        decoder.drain().unwrap();
        assert!(matches!(
            decoder.receive_frame().unwrap(),
            DecodeOutput::FormatChanged(VideoFormat {
                coded_size: Size {
                    width: 16,
                    height: 16
                },
                pixel_format: PixelFormat::Nv12,
                ..
            })
        ));
        let frame = match decoder.receive_frame().unwrap() {
            DecodeOutput::Frame(frame) => frame,
            output => panic!("expected frame, got {output:?}"),
        };
        assert_eq!(frame.id, 1);
        assert_eq!(frame.pts, MediaTime::from_parts(10, 30));
        assert_eq!(frame.duration, MediaTime::from_parts(1, 30));
        assert!(matches!(frame.storage, FrameStorage::Cpu(_)));
        assert!(matches!(
            decoder.receive_frame().unwrap(),
            DecodeOutput::EndOfStream
        ));
    }

    #[test]
    fn decodes_annex_b_idr_then_reference_p_picture() {
        let stream = annex_b_stream(&[
            (0x67, single_macroblock_sps_rbsp()),
            (0x68, single_macroblock_pps_rbsp()),
            (0x65, single_macroblock_idr_rbsp()),
            (0x41, single_macroblock_p_skip_rbsp()),
        ]);
        let mut decoder = H264Decoder::new();
        decoder.configure(byte_stream_config()).unwrap();
        assert!(matches!(
            decoder
                .send_packet(EncodedVideoPacket::new(stream))
                .unwrap(),
            DecodeInputStatus::Accepted
        ));
        decoder.drain().unwrap();

        assert!(matches!(
            decoder.receive_frame().unwrap(),
            DecodeOutput::FormatChanged(_)
        ));
        for expected_id in [1, 2] {
            let frame = match decoder.receive_frame().unwrap() {
                DecodeOutput::Frame(frame) => frame,
                output => panic!("expected decoded frame, got {output:?}"),
            };
            assert_eq!(frame.id, expected_id);
            let FrameStorage::Cpu(cpu) = frame.storage else {
                panic!("expected CPU frame");
            };
            assert!(cpu.planes[0].bytes.iter().all(|&sample| sample == 128));
        }
        assert!(matches!(
            decoder.receive_frame().unwrap(),
            DecodeOutput::EndOfStream
        ));
    }

    #[test]
    fn returns_unconsumed_packets_while_output_is_pending() {
        let stream = annex_b_stream(&[
            (0x67, single_macroblock_sps_rbsp()),
            (0x68, single_macroblock_pps_rbsp()),
            (0x65, single_macroblock_idr_rbsp()),
            (0x09, vec![0x10]), // AUD, primary_pic_type=0
        ]);
        let mut decoder = H264Decoder::new();
        decoder.configure(byte_stream_config()).unwrap();
        assert!(matches!(
            decoder
                .send_packet(EncodedVideoPacket::new(stream))
                .unwrap(),
            DecodeInputStatus::Accepted
        ));

        let retry = EncodedVideoPacket::new([0, 0, 1, 0x0b]);
        match decoder.send_packet(retry).unwrap() {
            DecodeInputStatus::NeedOutput(packet) => {
                assert_eq!(packet.data.as_ref(), &[0, 0, 1, 0x0b]);
            }
            DecodeInputStatus::Accepted => panic!("packet must remain unconsumed"),
        }

        decoder.flush();
        assert!(matches!(
            decoder.receive_frame().unwrap(),
            DecodeOutput::NeedInput
        ));

        let idr_only = annex_b_stream(&[(0x65, single_macroblock_idr_rbsp())]);
        assert!(matches!(
            decoder
                .send_packet(EncodedVideoPacket::new(idr_only))
                .unwrap(),
            DecodeInputStatus::Accepted
        ));
        decoder.drain().unwrap();
        assert!(matches!(
            decoder.receive_frame().unwrap(),
            DecodeOutput::Frame(_)
        ));
    }

    #[test]
    fn rejects_unsupported_decoder_framing() {
        let mut decoder = H264Decoder::new();
        assert!(
            decoder
                .configure(VideoDecoderConfig {
                    codec: VideoCodec::H264,
                    bitstream_format: BitstreamFormat::LengthPrefixed { length_size: 4 },
                    codec_data: None,
                })
                .is_err()
        );
    }

    #[test]
    fn access_unit_delimiters_reset_picture_history() {
        let mut parser = configured_parser();
        let slice = i_slice_rbsp(0, 0, 0);
        let first = parser.push_nal(nal(0x01, slice.as_slice())).unwrap();
        assert!(matches!(
            first,
            ParserEvent::Slice {
                starts_new_picture: true,
                ..
            }
        ));

        assert!(matches!(
            parser.push_nal(nal(0x09, &[0b1011_0000])).unwrap(),
            ParserEvent::AccessUnitDelimiter {
                primary_pic_type: 5
            }
        ));

        let after_aud = parser.push_nal(nal(0x01, slice.as_slice())).unwrap();
        assert!(matches!(
            after_aud,
            ParserEvent::Slice {
                starts_new_picture: true,
                ..
            }
        ));
    }

    #[test]
    fn derives_type_zero_poc_wraparound_and_mmco5_reset() {
        let mut parser = configured_parser();

        assert_eq!(
            push_poc(&mut parser, 0x41, reference_i_slice_type0(0, 6, false)),
            (Some(6), Some(6))
        );
        assert_eq!(
            push_poc(&mut parser, 0x41, reference_i_slice_type0(1, 14, false)),
            (Some(14), Some(14))
        );
        assert_eq!(
            push_poc(&mut parser, 0x41, reference_i_slice_type0(2, 2, false)),
            (Some(18), Some(18))
        );

        let mut parser = configured_parser();
        push_poc(&mut parser, 0x41, reference_i_slice_type0(0, 6, false));
        let mmco5 =
            push_picture_order_count(&mut parser, 0x41, reference_i_slice_type0(1, 14, true));
        assert_eq!(
            (mmco5.decoding.top, mmco5.decoding.bottom),
            (Some(14), Some(14))
        );
        assert_eq!((mmco5.stored.top, mmco5.stored.bottom), (Some(0), Some(0)));
        assert_eq!(
            push_poc(&mut parser, 0x41, reference_i_slice_type0(2, 2, false)),
            (Some(2), Some(2))
        );
    }

    #[test]
    fn derives_type_one_expected_cycles_and_non_reference_offset() {
        let mut parser = configured_parser_with_sps(type1_sps_rbsp());

        assert_eq!(
            push_poc(&mut parser, 0x41, i_slice_type1(0, 0, true, false)),
            (Some(0), Some(1))
        );
        assert_eq!(
            push_poc(&mut parser, 0x41, i_slice_type1(1, 0, true, false)),
            (Some(2), Some(3))
        );
        assert_eq!(
            push_poc(&mut parser, 0x01, i_slice_type1(2, 0, false, false)),
            (Some(1), Some(2))
        );
    }

    #[test]
    fn derives_type_two_reference_non_reference_and_mmco5_values() {
        let mut parser = configured_parser_with_sps(type2_sps_rbsp());

        assert_eq!(
            push_poc(&mut parser, 0x41, i_slice_type2(0, true, false)),
            (Some(0), Some(0))
        );
        assert_eq!(
            push_poc(&mut parser, 0x01, i_slice_type2(1, false, false)),
            (Some(1), Some(1))
        );
        assert_eq!(
            push_poc(&mut parser, 0x41, i_slice_type2(2, true, false)),
            (Some(4), Some(4))
        );

        let mut parser = configured_parser_with_sps(type2_sps_rbsp());
        assert_eq!(
            push_poc(&mut parser, 0x41, i_slice_type2(15, true, true)),
            (Some(0), Some(0))
        );
        assert_eq!(
            push_poc(&mut parser, 0x41, i_slice_type2(0, true, false)),
            (Some(0), Some(0))
        );
    }

    #[test]
    fn derives_idr_top_and_complementary_bottom_field_counts() {
        let mut parser = configured_parser_with_sps(interlaced_type0_sps_rbsp());

        assert_eq!(
            push_poc(&mut parser, 0x65, idr_field_slice_type0(false, 0)),
            (Some(0), None)
        );
        assert_eq!(
            push_poc(&mut parser, 0x41, reference_field_slice_type0(true, 1)),
            (None, Some(1))
        );
    }

    #[test]
    fn rejects_missing_parameter_sets_and_unsupported_vcl_units() {
        let mut empty = H264StreamParser::new();
        let slice = i_slice_rbsp(0, 0, 0);
        assert!(matches!(
            empty.push_nal(nal(0x01, slice.as_slice())),
            Err(H264Error::MissingPps(0))
        ));

        let mut configured = configured_parser();
        assert!(matches!(
            configured.push_nal(nal(0x42, &[])),
            Err(H264Error::UnsupportedFeature(_))
        ));
    }

    #[test]
    fn recognizes_every_ordinary_first_vcl_difference() {
        let base = PictureIdentity {
            frame_num: 1,
            picture_parameter_set_id: 2,
            field_picture: true,
            bottom_field: false,
            nal_ref_idc: 2,
            is_idr: true,
            idr_pic_id: Some(3),
            pic_order_cnt_lsb: Some(4),
            delta_pic_order_bottom: Some(5),
            delta_pic_order: [Some(6), Some(7)],
        };
        assert!(!base.starts_new_picture_after(&base));

        let variants = [
            PictureIdentity {
                frame_num: 2,
                ..base.clone()
            },
            PictureIdentity {
                picture_parameter_set_id: 3,
                ..base.clone()
            },
            PictureIdentity {
                field_picture: false,
                ..base.clone()
            },
            PictureIdentity {
                bottom_field: true,
                ..base.clone()
            },
            PictureIdentity {
                nal_ref_idc: 0,
                ..base.clone()
            },
            PictureIdentity {
                is_idr: false,
                idr_pic_id: None,
                ..base.clone()
            },
            PictureIdentity {
                idr_pic_id: Some(4),
                ..base.clone()
            },
            PictureIdentity {
                pic_order_cnt_lsb: Some(8),
                ..base.clone()
            },
            PictureIdentity {
                delta_pic_order_bottom: Some(9),
                ..base.clone()
            },
            PictureIdentity {
                delta_pic_order: [Some(10), Some(7)],
                ..base.clone()
            },
        ];
        for variant in variants {
            assert!(variant.starts_new_picture_after(&base));
        }
    }

    fn configured_parser() -> H264StreamParser {
        configured_parser_with_sps(sps_rbsp())
    }

    fn byte_stream_config() -> VideoDecoderConfig {
        VideoDecoderConfig {
            codec: VideoCodec::H264,
            bitstream_format: BitstreamFormat::ByteStream,
            codec_data: None,
        }
    }

    fn configured_parser_with_sps(sps: Vec<u8>) -> H264StreamParser {
        let mut parser = H264StreamParser::new();
        let pps = pps_rbsp();
        parser.push_nal(nal(0x67, sps.as_slice())).unwrap();
        parser.push_nal(nal(0x68, pps.as_slice())).unwrap();
        parser
    }

    fn push_poc(
        parser: &mut H264StreamParser,
        nal_header: u8,
        rbsp: Vec<u8>,
    ) -> (Option<i32>, Option<i32>) {
        let picture_order_count = push_picture_order_count(parser, nal_header, rbsp);
        (
            picture_order_count.stored.top,
            picture_order_count.stored.bottom,
        )
    }

    fn push_picture_order_count(
        parser: &mut H264StreamParser,
        nal_header: u8,
        rbsp: Vec<u8>,
    ) -> crate::PictureOrderCount {
        match parser.push_nal(nal(nal_header, rbsp.as_slice())).unwrap() {
            ParserEvent::Slice {
                picture_order_count,
                ..
            } => picture_order_count,
            event => panic!("expected slice event, got {event:?}"),
        }
    }

    fn nal(header: u8, ebsp: &[u8]) -> NalUnit<'_> {
        NalUnit {
            header: NalHeader::parse(header).unwrap(),
            ebsp,
            stream_offset: 0,
        }
    }

    fn annex_b_stream(nals: &[(u8, Vec<u8>)]) -> Vec<u8> {
        let mut stream = Vec::new();
        for (header, rbsp) in nals {
            stream.extend_from_slice(&[0, 0, 1, *header]);
            stream.extend_from_slice(&encode_ebsp(rbsp));
        }
        stream
    }

    fn encode_ebsp(rbsp: &[u8]) -> Vec<u8> {
        let mut ebsp = Vec::with_capacity(rbsp.len());
        let mut zero_count = 0;
        for &byte in rbsp {
            if zero_count == 2 && byte <= 3 {
                ebsp.push(3);
                zero_count = 0;
            }
            ebsp.push(byte);
            zero_count = if byte == 0 { zero_count + 1 } else { 0 };
        }
        ebsp
    }

    fn sps_rbsp() -> Vec<u8> {
        let mut writer = BitWriter::default();
        writer.write_bits(66, 8);
        writer.write_bits(0, 8);
        writer.write_bits(30, 8);
        writer.write_ue(0);
        writer.write_ue(0);
        writer.write_ue(0);
        writer.write_ue(0);
        writer.write_ue(2);
        writer.write_flag(false);
        writer.write_ue(3);
        writer.write_ue(2);
        writer.write_flag(true);
        writer.write_flag(true);
        writer.write_flag(false);
        writer.write_flag(false);
        writer.finish_rbsp()
    }

    fn type1_sps_rbsp() -> Vec<u8> {
        let mut writer = BitWriter::default();
        write_sps_prefix(&mut writer);
        writer.write_ue(0);
        writer.write_ue(1);
        writer.write_flag(false);
        writer.write_se(-1);
        writer.write_se(1);
        writer.write_ue(2);
        writer.write_se(2);
        writer.write_se(3);
        write_sps_geometry_and_tail(&mut writer);
        writer.finish_rbsp()
    }

    fn type2_sps_rbsp() -> Vec<u8> {
        let mut writer = BitWriter::default();
        write_sps_prefix(&mut writer);
        writer.write_ue(0);
        writer.write_ue(2);
        write_sps_geometry_and_tail(&mut writer);
        writer.finish_rbsp()
    }

    fn interlaced_type0_sps_rbsp() -> Vec<u8> {
        let mut writer = BitWriter::default();
        write_sps_prefix(&mut writer);
        writer.write_ue(0);
        writer.write_ue(0);
        writer.write_ue(0);
        writer.write_ue(2);
        writer.write_flag(false);
        writer.write_ue(3);
        writer.write_ue(1);
        writer.write_flag(false);
        writer.write_flag(false);
        writer.write_flag(true);
        writer.write_flag(false);
        writer.write_flag(false);
        writer.finish_rbsp()
    }

    fn write_sps_prefix(writer: &mut BitWriter) {
        writer.write_bits(66, 8);
        writer.write_bits(0, 8);
        writer.write_bits(30, 8);
        writer.write_ue(0);
    }

    fn write_sps_geometry_and_tail(writer: &mut BitWriter) {
        writer.write_ue(2);
        writer.write_flag(false);
        writer.write_ue(3);
        writer.write_ue(2);
        writer.write_flag(true);
        writer.write_flag(true);
        writer.write_flag(false);
        writer.write_flag(false);
    }

    fn pps_rbsp() -> Vec<u8> {
        let mut writer = BitWriter::default();
        writer.write_ue(0);
        writer.write_ue(0);
        writer.write_flag(false);
        writer.write_flag(false);
        writer.write_ue(0);
        writer.write_ue(0);
        writer.write_ue(0);
        writer.write_flag(false);
        writer.write_bits(0, 2);
        writer.write_se(0);
        writer.write_se(0);
        writer.write_se(0);
        writer.write_flag(true);
        writer.write_flag(false);
        writer.write_flag(false);
        writer.finish_rbsp()
    }

    fn single_macroblock_sps_rbsp() -> Vec<u8> {
        let mut writer = BitWriter::default();
        writer.write_bits(66, 8); // Baseline profile
        writer.write_bits(0, 8); // constraints + reserved_zero_2bits
        writer.write_bits(10, 8); // level_idc
        writer.write_ue(0); // seq_parameter_set_id
        writer.write_ue(0); // log2_max_frame_num_minus4
        writer.write_ue(0); // pic_order_cnt_type
        writer.write_ue(0); // log2_max_pic_order_cnt_lsb_minus4
        writer.write_ue(0); // max_num_ref_frames
        writer.write_flag(false); // gaps_in_frame_num_value_allowed_flag
        writer.write_ue(0); // pic_width_in_mbs_minus1
        writer.write_ue(0); // pic_height_in_map_units_minus1
        writer.write_flag(true); // frame_mbs_only_flag
        writer.write_flag(true); // direct_8x8_inference_flag
        writer.write_flag(false); // frame_cropping_flag
        writer.write_flag(false); // vui_parameters_present_flag
        writer.finish_rbsp()
    }

    fn single_macroblock_pps_rbsp() -> Vec<u8> {
        let mut writer = BitWriter::default();
        writer.write_ue(0); // pic_parameter_set_id
        writer.write_ue(0); // seq_parameter_set_id
        writer.write_flag(false); // entropy_coding_mode_flag: CAVLC
        writer.write_flag(false); // bottom_field_pic_order_in_frame_present_flag
        writer.write_ue(0); // num_slice_groups_minus1
        writer.write_ue(0); // num_ref_idx_l0_default_active_minus1
        writer.write_ue(0); // num_ref_idx_l1_default_active_minus1
        writer.write_flag(false); // weighted_pred_flag
        writer.write_bits(0, 2); // weighted_bipred_idc
        writer.write_se(0); // pic_init_qp_minus26
        writer.write_se(0); // pic_init_qs_minus26
        writer.write_se(0); // chroma_qp_index_offset
        writer.write_flag(false); // deblocking_filter_control_present_flag
        writer.write_flag(false); // constrained_intra_pred_flag
        writer.write_flag(false); // redundant_pic_cnt_present_flag
        writer.finish_rbsp()
    }

    fn single_macroblock_idr_rbsp() -> Vec<u8> {
        let mut writer = BitWriter::default();
        writer.write_ue(0); // first_mb_in_slice
        writer.write_ue(2); // I slice
        writer.write_ue(0); // pic_parameter_set_id
        writer.write_bits(0, 4); // frame_num
        writer.write_ue(0); // idr_pic_id
        writer.write_bits(0, 4); // pic_order_cnt_lsb
        writer.write_flag(false); // no_output_of_prior_pics_flag
        writer.write_flag(false); // long_term_reference_flag
        writer.write_se(0); // slice_qp_delta

        writer.write_ue(0); // I_NxN
        for _ in 0..16 {
            writer.write_flag(true); // prev_intra4x4_pred_mode_flag
        }
        writer.write_ue(0); // intra_chroma_pred_mode: DC
        writer.write_ue(3); // coded_block_pattern codeNum -> zero
        writer.finish_rbsp()
    }

    fn single_macroblock_p_skip_rbsp() -> Vec<u8> {
        let mut writer = BitWriter::default();
        writer.write_ue(0); // first_mb_in_slice
        writer.write_ue(0); // P slice
        writer.write_ue(0); // pic_parameter_set_id
        writer.write_bits(1, 4); // frame_num
        writer.write_bits(2, 4); // pic_order_cnt_lsb
        writer.write_flag(false); // num_ref_idx_active_override_flag
        writer.write_flag(false); // ref_pic_list_modification_flag_l0
        writer.write_flag(false); // adaptive_ref_pic_marking_mode_flag
        writer.write_se(0); // slice_qp_delta
        writer.write_ue(1); // mb_skip_run
        writer.finish_rbsp()
    }

    fn i_slice_rbsp(first_mb: u32, frame_num: u64, pic_order_cnt_lsb: u64) -> Vec<u8> {
        let mut writer = BitWriter::default();
        writer.write_ue(first_mb);
        writer.write_ue(2);
        writer.write_ue(0);
        writer.write_bits(frame_num, 4);
        writer.write_bits(pic_order_cnt_lsb, 4);
        writer.write_se(0);
        writer.write_ue(1);
        writer.finish_rbsp()
    }

    fn reference_i_slice_type0(frame_num: u64, poc_lsb: u64, mmco5: bool) -> Vec<u8> {
        let mut writer = BitWriter::default();
        write_i_slice_prefix(&mut writer, frame_num);
        writer.write_bits(poc_lsb, 4);
        write_reference_marking(&mut writer, mmco5);
        write_i_slice_tail(&mut writer);
        writer.finish_rbsp()
    }

    fn idr_field_slice_type0(bottom_field: bool, poc_lsb: u64) -> Vec<u8> {
        let mut writer = BitWriter::default();
        write_i_slice_prefix(&mut writer, 0);
        writer.write_flag(true);
        writer.write_flag(bottom_field);
        writer.write_ue(0);
        writer.write_bits(poc_lsb, 4);
        writer.write_flag(false);
        writer.write_flag(false);
        write_i_slice_tail(&mut writer);
        writer.finish_rbsp()
    }

    fn reference_field_slice_type0(bottom_field: bool, poc_lsb: u64) -> Vec<u8> {
        let mut writer = BitWriter::default();
        write_i_slice_prefix(&mut writer, 0);
        writer.write_flag(true);
        writer.write_flag(bottom_field);
        writer.write_bits(poc_lsb, 4);
        write_reference_marking(&mut writer, false);
        write_i_slice_tail(&mut writer);
        writer.finish_rbsp()
    }

    fn i_slice_type1(frame_num: u64, delta0: i32, reference: bool, mmco5: bool) -> Vec<u8> {
        let mut writer = BitWriter::default();
        write_i_slice_prefix(&mut writer, frame_num);
        writer.write_se(delta0);
        if reference {
            write_reference_marking(&mut writer, mmco5);
        }
        write_i_slice_tail(&mut writer);
        writer.finish_rbsp()
    }

    fn i_slice_type2(frame_num: u64, reference: bool, mmco5: bool) -> Vec<u8> {
        let mut writer = BitWriter::default();
        write_i_slice_prefix(&mut writer, frame_num);
        if reference {
            write_reference_marking(&mut writer, mmco5);
        }
        write_i_slice_tail(&mut writer);
        writer.finish_rbsp()
    }

    fn write_i_slice_prefix(writer: &mut BitWriter, frame_num: u64) {
        writer.write_ue(0);
        writer.write_ue(2);
        writer.write_ue(0);
        writer.write_bits(frame_num, 4);
    }

    fn write_reference_marking(writer: &mut BitWriter, mmco5: bool) {
        writer.write_flag(mmco5);
        if mmco5 {
            writer.write_ue(5);
            writer.write_ue(0);
        }
    }

    fn write_i_slice_tail(writer: &mut BitWriter) {
        writer.write_se(0);
        writer.write_ue(1);
    }

    #[derive(Default)]
    struct BitWriter {
        bytes: Vec<u8>,
        current: u8,
        bits: u8,
    }

    impl BitWriter {
        fn write_flag(&mut self, value: bool) {
            self.write_bits(u64::from(value), 1);
        }

        fn write_bits(&mut self, value: u64, count: u8) {
            for shift in (0..count).rev() {
                self.current = (self.current << 1) | ((value >> shift) as u8 & 1);
                self.bits += 1;
                if self.bits == 8 {
                    self.bytes.push(self.current);
                    self.current = 0;
                    self.bits = 0;
                }
            }
        }

        fn write_ue(&mut self, value: u32) {
            let code_num = u64::from(value) + 1;
            let width = 64 - code_num.leading_zeros() as u8;
            self.write_bits(0, width - 1);
            self.write_bits(code_num, width);
        }

        fn write_se(&mut self, value: i32) {
            let code_num = if value <= 0 {
                u32::try_from(-i64::from(value) * 2).unwrap()
            } else {
                u32::try_from(i64::from(value) * 2 - 1).unwrap()
            };
            self.write_ue(code_num);
        }

        fn finish_rbsp(mut self) -> Vec<u8> {
            self.write_flag(true);
            if self.bits != 0 {
                self.current <<= 8 - self.bits;
                self.bytes.push(self.current);
            }
            self.bytes
        }
    }
}
