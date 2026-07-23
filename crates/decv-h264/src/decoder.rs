//! Top-level NAL dispatch, picture-boundary detection, and frame output.

use std::{borrow::Cow, collections::VecDeque, sync::Arc};

use bit_readers::BitReader;
use decv_core::{
    BitstreamFormat, DecodeInputStatus, DecodeOutput, DecodedVideoFrame, EncodedVideoPacket,
    MediaTime, PixelFormat, Size, VideoCodec, VideoDecoder, VideoDecoderConfig, VideoFormat,
};

use crate::avcc::{LengthPrefixedNalReader, parse_avcc};
use crate::intra_reconstruction::ReconstructionReferenceList;
use crate::reorder::PictureReorderBuffer;
use crate::{
    AnnexBNalUnit, AnnexBReader, DecodedPictureBuffer, DirectReference, EntropyCodingMode,
    H264Error, ImplicitWeightReference, IntraPictureReconstructor, NalHeader, NalUnit, NalUnitType,
    ParameterSetStore, ParsedSliceHeader, PictureOrderCount, PictureParameterSet, Profile,
    ReferenceKind, ReferencePictureMarking, Result, SequenceParameterSet, SliceType,
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
    max_num_reorder_frames: u32,
    log2_max_frame_num: u8,
    coded_size: Size,
}

/// Synchronous pure-Rust H.264 decoder implementing the codec-independent
/// push/pull contract.
///
/// The current reconstruction backend accepts progressive CAVLC I, P, and
/// explicit non-Direct B pictures. Unsupported H.264 coding tools return
/// explicit [`H264Error::UnsupportedFeature`] errors.
#[derive(Debug)]
pub struct H264Decoder {
    configured: bool,
    bitstream_format: BitstreamFormat,
    parser: H264StreamParser,
    current_picture: Option<PendingPicture>,
    dpb: Option<DecodedPictureBuffer>,
    dpb_configuration: Option<DpbConfiguration>,
    reorder: PictureReorderBuffer<DecodedVideoFrame>,
    outputs: VecDeque<DecodeOutput>,
    announced_format: Option<VideoFormat>,
    next_frame_id: u64,
    draining: bool,
}

impl Default for H264Decoder {
    fn default() -> Self {
        Self {
            configured: false,
            bitstream_format: BitstreamFormat::ByteStream,
            parser: H264StreamParser::new(),
            current_picture: None,
            dpb: None,
            dpb_configuration: None,
            reorder: PictureReorderBuffer::new(0),
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
        match self.bitstream_format {
            BitstreamFormat::ByteStream => {
                for unit in AnnexBReader::new(packet.data.as_ref()) {
                    let event = self.parser.push_annex_b(unit?)?;
                    self.process_parser_event(event, packet)?;
                }
            }
            BitstreamFormat::LengthPrefixed { length_size } => {
                for nal in LengthPrefixedNalReader::new(packet.data.as_ref(), length_size) {
                    let event = self.parser.push_nal(nal?)?;
                    self.process_parser_event(event, packet)?;
                }
            }
            _ => {
                return Err(H264Error::UnsupportedFeature(
                    "unknown H.264 bitstream framing",
                ));
            }
        }
        Ok(())
    }

    fn process_parser_event(
        &mut self,
        event: ParserEvent<'_>,
        packet: &EncodedVideoPacket,
    ) -> Result<()> {
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
                    self.prepare_for_idr(&parsed, nal_header)?;
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
                        reference_picture_marking: parsed.header.reference_picture_marking.clone(),
                    });
                }
                match parsed.header.slice_type {
                    SliceType::I => {
                        let picture = self
                            .current_picture
                            .as_mut()
                            .expect("the picture is initialized above");
                        match parsed.parameter_sets.picture.entropy_coding_mode {
                            EntropyCodingMode::Cavlc => {
                                picture
                                    .reconstructor
                                    .decode_cavlc_intra_slice(rbsp.as_ref(), &parsed)?;
                            }
                            EntropyCodingMode::Cabac => {
                                picture
                                    .reconstructor
                                    .decode_cabac_intra_slice(rbsp.as_ref(), &parsed)?;
                            }
                        }
                    }
                    SliceType::P => {
                        let references = self
                            .dpb
                            .as_ref()
                            .expect("the DPB is initialized with the picture")
                            .p_reference_info_list(
                                parsed.header.frame_num,
                                parsed.header.num_ref_idx_l0_active,
                                &parsed.header.ref_pic_list_modifications_l0,
                            )?;
                        let borrowed = references
                            .iter()
                            .map(|reference| {
                                reference
                                    .as_ref()
                                    .map(|reference| reference.picture.as_ref())
                            })
                            .collect::<Vec<_>>();
                        let reference_ids = references
                            .iter()
                            .map(|reference| reference.as_ref().map(|reference| reference.id))
                            .collect::<Vec<_>>();
                        let picture = self
                            .current_picture
                            .as_mut()
                            .expect("the picture is initialized above");
                        let references =
                            ReconstructionReferenceList::with_ids(&borrowed, &reference_ids);
                        match parsed.parameter_sets.picture.entropy_coding_mode {
                            EntropyCodingMode::Cavlc => {
                                picture.reconstructor.decode_cavlc_p_slice(
                                    rbsp.as_ref(),
                                    &parsed,
                                    references,
                                )?;
                            }
                            EntropyCodingMode::Cabac => {
                                picture.reconstructor.decode_cabac_p_slice(
                                    rbsp.as_ref(),
                                    &parsed,
                                    references,
                                )?;
                            }
                        }
                    }
                    SliceType::B => {
                        let current_poc = picture_order_count.stored.picture_order_count();
                        let (references_l0, references_l1) = self
                            .dpb
                            .as_ref()
                            .expect("the DPB is initialized with the picture")
                            .b_reference_info_lists(
                                parsed.header.frame_num,
                                current_poc,
                                parsed.header.num_ref_idx_l0_active,
                                &parsed.header.ref_pic_list_modifications_l0,
                                parsed.header.num_ref_idx_l1_active,
                                &parsed.header.ref_pic_list_modifications_l1,
                            )?;
                        let borrowed_l0 = references_l0
                            .iter()
                            .map(|reference| {
                                reference
                                    .as_ref()
                                    .map(|reference| reference.picture.as_ref())
                            })
                            .collect::<Vec<_>>();
                        let borrowed_l1 = references_l1
                            .iter()
                            .map(|reference| {
                                reference
                                    .as_ref()
                                    .map(|reference| reference.picture.as_ref())
                            })
                            .collect::<Vec<_>>();
                        let reference_ids_l0 = references_l0
                            .iter()
                            .map(|reference| reference.as_ref().map(|reference| reference.id))
                            .collect::<Vec<_>>();
                        let reference_ids_l1 = references_l1
                            .iter()
                            .map(|reference| reference.as_ref().map(|reference| reference.id))
                            .collect::<Vec<_>>();
                        let implicit_l0 = references_l0
                            .iter()
                            .map(|reference| {
                                reference.as_ref().map(|reference| ImplicitWeightReference {
                                    picture_order_count: reference.picture_order_count,
                                    long_term: matches!(
                                        reference.kind,
                                        ReferenceKind::LongTerm { .. }
                                    ),
                                })
                            })
                            .collect::<Vec<_>>();
                        let implicit_l1 = references_l1
                            .iter()
                            .map(|reference| {
                                reference.as_ref().map(|reference| ImplicitWeightReference {
                                    picture_order_count: reference.picture_order_count,
                                    long_term: matches!(
                                        reference.kind,
                                        ReferenceKind::LongTerm { .. }
                                    ),
                                })
                            })
                            .collect::<Vec<_>>();
                        let direct_l0 = references_l0
                            .iter()
                            .map(|reference| {
                                reference.as_ref().map(|reference| DirectReference {
                                    id: reference.id,
                                    picture_order_count: reference.picture_order_count,
                                    long_term: matches!(
                                        reference.kind,
                                        ReferenceKind::LongTerm { .. }
                                    ),
                                    motion: reference.motion.as_ref(),
                                })
                            })
                            .collect::<Vec<_>>();
                        let direct_l1 = references_l1
                            .iter()
                            .map(|reference| {
                                reference.as_ref().map(|reference| DirectReference {
                                    id: reference.id,
                                    picture_order_count: reference.picture_order_count,
                                    long_term: matches!(
                                        reference.kind,
                                        ReferenceKind::LongTerm { .. }
                                    ),
                                    motion: reference.motion.as_ref(),
                                })
                            })
                            .collect::<Vec<_>>();
                        let picture = self
                            .current_picture
                            .as_mut()
                            .expect("the picture is initialized above");
                        let list0 = ReconstructionReferenceList::with_metadata(
                            &borrowed_l0,
                            &reference_ids_l0,
                            &implicit_l0,
                            &direct_l0,
                        );
                        let list1 = ReconstructionReferenceList::with_metadata(
                            &borrowed_l1,
                            &reference_ids_l1,
                            &implicit_l1,
                            &direct_l1,
                        );
                        match parsed.parameter_sets.picture.entropy_coding_mode {
                            EntropyCodingMode::Cavlc => {
                                picture.reconstructor.decode_cavlc_b_slice(
                                    rbsp.as_ref(),
                                    &parsed,
                                    list0,
                                    list1,
                                    current_poc,
                                )?;
                            }
                            EntropyCodingMode::Cabac => {
                                picture.reconstructor.decode_cabac_b_slice(
                                    rbsp.as_ref(),
                                    &parsed,
                                    list0,
                                    list1,
                                    current_poc,
                                )?;
                            }
                        }
                    }
                    SliceType::Sp | SliceType::Si => {
                        return Err(H264Error::UnsupportedFeature(
                            "top-level reconstruction of SP and SI slices",
                        ));
                    }
                }
            }
            ParserEvent::AccessUnitDelimiter { .. } => {
                self.finish_current_picture()?;
            }
            ParserEvent::EndOfSequence => {
                self.finish_current_picture()?;
                self.drain_reorder();
                self.clear_dpb();
            }
            ParserEvent::EndOfStream => {
                self.finish_current_picture()?;
                self.drain_reorder();
                self.draining = true;
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
        let (decoded, motion) = picture.reconstructor.into_deblocked_reference_picture()?;
        let decoded = Arc::new(decoded);
        let motion = Arc::new(motion);
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
                } => dpb.store_idr_with_motion(
                    picture_order_count,
                    decoded.clone(),
                    motion.clone(),
                    *long_term_reference,
                )?,
                ReferencePictureMarking::SlidingWindow => dpb.store_short_term_with_motion(
                    picture.frame_num,
                    picture_order_count,
                    decoded.clone(),
                    motion.clone(),
                )?,
                ReferencePictureMarking::Adaptive(operations) => dpb.store_adaptive_with_motion(
                    picture.frame_num,
                    picture_order_count,
                    decoded.clone(),
                    motion.clone(),
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
        let picture_order_count = picture.picture_order_count.stored.picture_order_count();
        if let Some(frame) = self.reorder.push(picture_order_count, frame)? {
            self.outputs.push_back(DecodeOutput::Frame(frame));
        }
        Ok(())
    }

    fn prepare_for_idr(&mut self, parsed: &ParsedSliceHeader, nal_header: NalHeader) -> Result<()> {
        if nal_header.unit_type != NalUnitType::IdrSlice {
            return Ok(());
        }
        match parsed.header.reference_picture_marking {
            ReferencePictureMarking::Idr {
                no_output_of_prior_pictures: true,
                ..
            } => self.reorder.clear(),
            ReferencePictureMarking::Idr { .. } => self.drain_reorder(),
            _ => {
                return Err(H264Error::InvalidSyntax(
                    "IDR slice is missing IDR reference-picture marking",
                ));
            }
        }
        Ok(())
    }

    fn drain_reorder(&mut self) {
        self.outputs
            .extend(self.reorder.drain().into_iter().map(DecodeOutput::Frame));
    }

    fn reset_all_state(&mut self) {
        self.parser.reset();
        self.current_picture = None;
        self.clear_dpb();
        self.outputs.clear();
        self.reorder.clear();
        self.announced_format = None;
        self.next_frame_id = 0;
        self.draining = false;
    }

    fn flush_timeline(&mut self) {
        self.parser.reset_picture_history();
        self.current_picture = None;
        self.clear_dpb();
        self.outputs.clear();
        self.reorder.clear();
        self.draining = false;
    }

    fn ensure_dpb(&mut self, parsed: &ParsedSliceHeader, nal_header: NalHeader) -> Result<()> {
        let sps = &parsed.parameter_sets.sequence;
        let configuration = DpbConfiguration {
            max_num_ref_frames: sps.max_num_ref_frames,
            max_num_reorder_frames: inferred_max_num_reorder_frames(sps),
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
        self.reorder = PictureReorderBuffer::new(
            usize::try_from(configuration.max_num_reorder_frames)
                .map_err(|_| H264Error::IntegerOverflow)?,
        );
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
        self.configured = false;
        if !matches!(config.codec, VideoCodec::H264) {
            return Err(H264Error::UnsupportedFeature(
                "decoder configuration for a non-H.264 codec",
            ));
        }
        self.reset_all_state();
        self.bitstream_format = config.bitstream_format;
        if let Some(codec_data) = config.codec_data {
            let avcc = parse_avcc(codec_data.as_ref())?;
            if let BitstreamFormat::LengthPrefixed { length_size } = self.bitstream_format
                && length_size != avcc.length_size
            {
                return Err(H264Error::InvalidSyntax(
                    "configured NAL length size does not match avcC",
                ));
            }
            for bytes in &avcc.parameter_sets {
                let (&header, ebsp) = bytes.split_first().ok_or(H264Error::InvalidNalHeader)?;
                match self.parser.push_nal(NalUnit {
                    header: NalHeader::parse(header)?,
                    ebsp,
                    stream_offset: 0,
                })? {
                    ParserEvent::SequenceParameterSet(_) | ParserEvent::PictureParameterSet(_) => {}
                    _ => {
                        return Err(H264Error::InvalidSyntax(
                            "avcC contains a non-parameter-set NAL",
                        ));
                    }
                }
            }
        }
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
        self.drain_reorder();
        self.draining = true;
        Ok(())
    }
}

fn inferred_max_num_reorder_frames(sps: &SequenceParameterSet) -> u32 {
    sps.vui
        .as_ref()
        .and_then(|vui| vui.bitstream_restrictions)
        .map_or_else(
            || match sps.profile {
                Profile::Baseline => 0,
                Profile::Main | Profile::High => sps.max_num_ref_frames,
            },
            |restrictions| restrictions.max_num_reorder_frames,
        )
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
    use crate::{
        AnnexBReader, H264Error, IntraPictureReconstructor, MotionVector, NalHeader, NalUnit,
    };

    fn exercise_decoder(bytes: Vec<u8>) {
        let mut decoder = H264Decoder::new();
        decoder.configure(byte_stream_config()).unwrap();

        let Ok(DecodeInputStatus::Accepted) = decoder.send_packet(EncodedVideoPacket::new(bytes))
        else {
            return;
        };
        if decoder.drain().is_err() {
            return;
        }
        for _ in 0..16 {
            match decoder.receive_frame() {
                Ok(DecodeOutput::FormatChanged(_) | DecodeOutput::Frame(_)) => {}
                Ok(DecodeOutput::NeedInput | DecodeOutput::EndOfStream) | Err(_) => break,
            }
        }
    }

    #[test]
    fn truncated_or_single_byte_corrupted_streams_do_not_panic() {
        let valid = annex_b_stream(&[
            (0x67, single_macroblock_sps_rbsp()),
            (0x68, single_macroblock_pps_rbsp()),
            (0x65, single_macroblock_idr_rbsp()),
        ]);

        for end in 0..=valid.len() {
            exercise_decoder(valid[..end].to_vec());
        }

        for index in 0..valid.len() {
            for mask in [0x01, 0x80, 0xff] {
                let mut corrupted = valid.clone();
                corrupted[index] ^= mask;
                exercise_decoder(corrupted);
            }
        }
    }

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
    fn decodes_a_real_x264_cabac_idr_picture() {
        // 16x16 testsrc2, encoded as one High-profile CABAC IDR by x264 with
        // the informational SEI removed. Keeping the fixture inline makes the
        // regression independent of an external encoder.
        let stream = [
            0x00, 0x00, 0x00, 0x01, 0x67, 0x64, 0x10, 0x0a, 0xac, 0xbb, 0xd8, 0x08, 0x80, 0x00,
            0x00, 0x03, 0x00, 0x80, 0x00, 0x00, 0x03, 0x01, 0x02, 0x00, 0x00, 0x00, 0x01, 0x68,
            0xee, 0x0f, 0x2c, 0x8b, 0x00, 0x00, 0x01, 0x65, 0x88, 0x84, 0x37, 0xd4, 0xda, 0x29,
            0x82, 0xac, 0x9e, 0xa8, 0x19, 0x64, 0x16, 0xcd, 0x88, 0xb1, 0xfe, 0x2f, 0x49, 0x0a,
            0x9d, 0xe5, 0xc1, 0xd8, 0xc0, 0xb2, 0x37, 0x38, 0x08, 0xb2, 0x12, 0xe4, 0x9e, 0x56,
            0x4a, 0xba, 0xaf, 0x42, 0x19, 0x87, 0x13, 0xfd, 0xb7, 0x63, 0xf0, 0x80, 0x05, 0x0d,
            0x4c, 0x17, 0x9d, 0x20, 0x3e, 0x05, 0xcc, 0x84, 0x70, 0x23, 0x25, 0x5a, 0xa0, 0x35,
            0x9b, 0x65, 0x74, 0xfd, 0xa9, 0xa0, 0x4d, 0x17, 0xeb, 0x33, 0x7b, 0x77, 0x8b, 0x2c,
            0xa7, 0x84, 0xf8, 0x55, 0xcf, 0x2a, 0x68, 0x25, 0xb9, 0xeb, 0x0d, 0x3e, 0x7b, 0x20,
            0x4e, 0x5d, 0xac, 0x7f, 0xf8, 0x37, 0x17, 0xe7, 0xc2, 0x44, 0x04, 0x84, 0xf1, 0x8e,
            0x45, 0xd1, 0xa6, 0xaf, 0xed, 0xc6, 0x3d, 0x23, 0xbd, 0xc2, 0x7a, 0xbe, 0x24, 0x3a,
            0x59, 0x55, 0xa9, 0xa9, 0xad, 0x3c, 0x4d, 0x97, 0xa3, 0xc3, 0x32, 0x43, 0x5c, 0x89,
            0x53, 0xef, 0x73, 0x32, 0x11, 0xb3, 0x85, 0x5a, 0x18, 0x9c, 0xf7, 0x6f, 0xb5, 0x6e,
            0x4d, 0xb2, 0xc2, 0x91, 0x4c, 0x68, 0xa3, 0x50, 0x87, 0x9b, 0x82, 0x51, 0xf7, 0xeb,
            0xae, 0xb9, 0x9c, 0x68, 0xe2, 0xa4, 0xef, 0xc2, 0x56, 0x11, 0xbe, 0xbd, 0x28, 0x13,
            0xf9, 0xdb, 0x93, 0xbf, 0xf5, 0x74, 0xd9, 0xd3, 0x8d,
        ];
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
        let frame = match decoder.receive_frame().unwrap() {
            DecodeOutput::Frame(frame) => frame,
            output => panic!("expected CABAC frame, got {output:?}"),
        };
        let FrameStorage::Cpu(cpu) = frame.storage else {
            panic!("expected CPU frame");
        };
        assert_eq!(cpu.planes[0].bytes.len(), 384);
        assert_eq!(crc32(cpu.planes[0].bytes.as_ref()), 2_320_103_694);
        assert!(matches!(
            decoder.receive_frame().unwrap(),
            DecodeOutput::EndOfStream
        ));
    }

    #[test]
    fn decodes_real_x264_cabac_p_pictures_byte_exactly() {
        // Two 32x16 gradient pictures encoded by x264 as one High-profile
        // CABAC IDR followed by a CABAC P picture containing an inter and an
        // embedded-intra macroblock. The SEI was removed. CRCs cover the
        // complete tightly packed NV12 frame and were checked against FFmpeg.
        let stream = [
            0x00, 0x00, 0x00, 0x01, 0x67, 0x64, 0x00, 0x0a, 0xac, 0xb4, 0x5d, 0x80, 0x88, 0x00,
            0x00, 0x03, 0x00, 0x08, 0x00, 0x00, 0x03, 0x00, 0x10, 0x78, 0x91, 0x35, 0x00, 0x00,
            0x00, 0x01, 0x68, 0xee, 0x0f, 0x2c, 0x8b, 0x00, 0x00, 0x01, 0x65, 0x88, 0x84, 0x08,
            0xff, 0xf6, 0x1d, 0x64, 0x4f, 0xf9, 0x3c, 0x4b, 0xf8, 0x47, 0x3a, 0x19, 0xd8, 0xaa,
            0xdf, 0x12, 0x0c, 0x06, 0x28, 0xa0, 0x38, 0x83, 0x9e, 0x63, 0xbd, 0x00, 0x00, 0x00,
            0x01, 0x41, 0x9a, 0x22, 0x11, 0xff, 0x73, 0x44, 0xc3, 0x02, 0x47, 0x78, 0x2d, 0xff,
            0x91, 0xe9, 0x6b, 0xed, 0x82, 0xfe, 0xc0,
        ];
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

        for expected_crc in [3_812_764_094, 1_790_393_901] {
            let frame = match decoder.receive_frame().unwrap() {
                DecodeOutput::Frame(frame) => frame,
                output => panic!("expected CABAC P frame, got {output:?}"),
            };
            let FrameStorage::Cpu(cpu) = frame.storage else {
                panic!("expected CPU frame");
            };
            assert_eq!(cpu.planes[0].bytes.len(), 768);
            assert_eq!(crc32(cpu.planes[0].bytes.as_ref()), expected_crc);
        }
        assert!(matches!(
            decoder.receive_frame().unwrap(),
            DecodeOutput::EndOfStream
        ));
    }

    #[test]
    fn decodes_real_x264_cabac_b_pictures_byte_exactly() {
        // Six 32x16 gradient pictures encoded by x264 with CABAC, two B
        // pictures per reference interval, two references, and spatial Direct.
        // The SEI was removed and every NV12 CRC was checked against FFmpeg.
        let stream = [
            0x00, 0x00, 0x00, 0x01, 0x67, 0x64, 0x00, 0x0a, 0xac, 0xd9, 0x4b, 0xb0, 0x11, 0x00,
            0x00, 0x03, 0x00, 0x01, 0x00, 0x00, 0x03, 0x00, 0x04, 0x0f, 0x12, 0x25, 0x96, 0x00,
            0x00, 0x00, 0x01, 0x68, 0xea, 0x83, 0xcb, 0x22, 0xc0, 0x00, 0x00, 0x01, 0x65, 0x88,
            0x84, 0x00, 0x5f, 0xf4, 0x5e, 0x28, 0xef, 0xde, 0x02, 0x2f, 0x2c, 0x4e, 0xe9, 0x77,
            0x4c, 0x7e, 0x55, 0xb1, 0xd4, 0xe5, 0xc9, 0x7f, 0x47, 0x65, 0xd9, 0x00, 0x00, 0x00,
            0x01, 0x41, 0x9a, 0x23, 0x64, 0x6f, 0x64, 0x7a, 0xce, 0xb4, 0xe1, 0xfd, 0x36, 0x98,
            0x7d, 0x6a, 0xfe, 0x58, 0x4b, 0x4b, 0xfb, 0xd2, 0x7f, 0x3a, 0x48, 0x7e, 0xdc, 0xf7,
            0x80, 0x00, 0x00, 0x00, 0x01, 0x41, 0x9e, 0x41, 0x78, 0x8d, 0xff, 0xf7, 0xf3, 0x01,
            0xcd, 0xad, 0x3f, 0x85, 0xea, 0x33, 0x47, 0xff, 0xf9, 0x00, 0x00, 0x00, 0x01, 0x01,
            0x9e, 0x62, 0x44, 0x5f, 0xb0, 0x80, 0x00, 0x00, 0x00, 0x01, 0x41, 0x9a, 0x65, 0x34,
            0xa4, 0xa7, 0x8b, 0xff, 0x60, 0xf4, 0xe1, 0x39, 0xfd, 0x00, 0xa3, 0xe5, 0xe0, 0x42,
            0xc9, 0x9a, 0x73, 0x35, 0x56, 0x66, 0x5f, 0xe1, 0x00, 0x00, 0x00, 0x01, 0x01, 0x9e,
            0x84, 0x44, 0x5f, 0xb0, 0x81,
        ];
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

        for expected_crc in [
            2_233_814_414,
            1_801_912_313,
            3_452_851_269,
            2_181_253_705,
            1_436_501_184,
            1_038_502_273,
        ] {
            let frame = match decoder.receive_frame().unwrap() {
                DecodeOutput::Frame(frame) => frame,
                output => panic!("expected CABAC B frame, got {output:?}"),
            };
            let FrameStorage::Cpu(cpu) = frame.storage else {
                panic!("expected CPU frame");
            };
            assert_eq!(cpu.planes[0].bytes.len(), 768);
            assert_eq!(crc32(cpu.planes[0].bytes.as_ref()), expected_crc);
        }
        assert!(matches!(
            decoder.receive_frame().unwrap(),
            DecodeOutput::EndOfStream
        ));
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

        let references = decoder
            .dpb
            .as_ref()
            .unwrap()
            .p_reference_info_list(2, 1, &[])
            .unwrap();
        let p_reference = references[0].as_ref().unwrap();
        let p_motion = p_reference.motion.cell(0, 0).unwrap();
        assert_eq!(p_motion.list0.unwrap().reference_id.unwrap().get(), 1);
        assert_eq!(p_motion.list0.unwrap().vector, MotionVector::default());

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
    fn decodes_and_reorders_an_explicit_bidirectional_b_picture() {
        let stream = annex_b_stream(&[
            (0x67, single_macroblock_main_sps_rbsp()),
            (0x68, single_macroblock_pps_rbsp()),
            (0x65, single_macroblock_idr_rbsp()),
            (0x41, single_macroblock_p_skip_at_poc_rbsp(4)),
            (0x01, single_macroblock_explicit_b_rbsp()),
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
        // Decode order is I(1), P(2), B(3); POC display order is I, B, P.
        for expected_id in [1, 3, 2] {
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
    fn decodes_a_spatial_direct_b_picture() {
        let stream = annex_b_stream(&[
            (0x67, single_macroblock_main_sps_rbsp()),
            (0x68, single_macroblock_pps_rbsp()),
            (0x65, single_macroblock_idr_rbsp()),
            (0x41, single_macroblock_p_skip_at_poc_rbsp(4)),
            (0x01, single_macroblock_spatial_direct_b_rbsp()),
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
        for expected_id in [1, 3, 2] {
            let DecodeOutput::Frame(frame) = decoder.receive_frame().unwrap() else {
                panic!("expected a decoded frame");
            };
            assert_eq!(frame.id, expected_id);
            let FrameStorage::Cpu(cpu) = frame.storage else {
                panic!("expected CPU frame");
            };
            assert!(cpu.planes[0].bytes.iter().all(|&sample| sample == 128));
        }
    }

    #[test]
    fn decodes_a_spatially_skipped_b_picture() {
        let stream = annex_b_stream(&[
            (0x67, single_macroblock_main_sps_rbsp()),
            (0x68, single_macroblock_pps_rbsp()),
            (0x65, single_macroblock_idr_rbsp()),
            (0x41, single_macroblock_p_skip_at_poc_rbsp(4)),
            (0x01, single_macroblock_spatial_skip_b_rbsp()),
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
        for expected_id in [1, 3, 2] {
            let DecodeOutput::Frame(frame) = decoder.receive_frame().unwrap() else {
                panic!("expected a decoded frame");
            };
            assert_eq!(frame.id, expected_id);
            let FrameStorage::Cpu(cpu) = frame.storage else {
                panic!("expected CPU frame");
            };
            assert!(cpu.planes[0].bytes.iter().all(|&sample| sample == 128));
        }
    }

    #[test]
    fn decodes_a_temporally_skipped_b_picture() {
        let stream = annex_b_stream(&[
            (0x67, single_macroblock_main_sps_rbsp()),
            (0x68, single_macroblock_pps_rbsp()),
            (0x65, single_macroblock_idr_rbsp()),
            (0x41, single_macroblock_p_skip_at_poc_rbsp(4)),
            (0x01, single_macroblock_temporal_skip_b_rbsp()),
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
        for expected_id in [1, 3, 2] {
            let DecodeOutput::Frame(frame) = decoder.receive_frame().unwrap() else {
                panic!("expected a decoded frame");
            };
            assert_eq!(frame.id, expected_id);
            let FrameStorage::Cpu(cpu) = frame.storage else {
                panic!("expected CPU frame");
            };
            assert!(cpu.planes[0].bytes.iter().all(|&sample| sample == 128));
        }
    }

    #[test]
    fn decodes_an_all_direct_b_eight_by_eight_picture() {
        let stream = annex_b_stream(&[
            (0x67, single_macroblock_main_sps_rbsp()),
            (0x68, single_macroblock_pps_rbsp()),
            (0x65, single_macroblock_idr_rbsp()),
            (0x41, single_macroblock_p_skip_at_poc_rbsp(4)),
            (0x01, single_macroblock_all_direct_b_8x8_rbsp()),
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
        for expected_id in [1, 3, 2] {
            let DecodeOutput::Frame(frame) = decoder.receive_frame().unwrap() else {
                panic!("expected a decoded frame");
            };
            assert_eq!(frame.id, expected_id);
        }
    }

    #[test]
    fn decodes_mixed_direct_and_explicit_b_eight_by_eight_partitions() {
        let stream = annex_b_stream(&[
            (0x67, single_macroblock_main_sps_rbsp()),
            (0x68, single_macroblock_pps_rbsp()),
            (0x65, single_macroblock_idr_rbsp()),
            (0x41, single_macroblock_p_skip_at_poc_rbsp(4)),
            (0x01, single_macroblock_mixed_direct_b_8x8_rbsp()),
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
        for expected_id in [1, 3, 2] {
            let DecodeOutput::Frame(frame) = decoder.receive_frame().unwrap() else {
                panic!("expected a decoded frame");
            };
            assert_eq!(frame.id, expected_id);
            let FrameStorage::Cpu(cpu) = frame.storage else {
                panic!("expected CPU frame");
            };
            assert!(cpu.planes[0].bytes.iter().all(|&sample| sample == 128));
        }
    }

    #[test]
    fn decodes_an_explicitly_weighted_bidirectional_b_picture() {
        let stream = annex_b_stream(&[
            (0x67, single_macroblock_main_sps_rbsp()),
            (0x68, single_macroblock_explicit_b_pps_rbsp()),
            (0x65, single_macroblock_idr_rbsp()),
            (0x41, single_macroblock_p_skip_at_poc_rbsp(4)),
            (0x01, single_macroblock_explicit_b_weighted_rbsp()),
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
        let mut frames = Vec::new();
        for _ in 0..3 {
            let DecodeOutput::Frame(frame) = decoder.receive_frame().unwrap() else {
                panic!("expected a decoded frame");
            };
            frames.push(frame);
        }
        assert_eq!(
            frames.iter().map(|frame| frame.id).collect::<Vec<_>>(),
            [1, 3, 2]
        );
        let FrameStorage::Cpu(weighted_b) = &frames[1].storage else {
            panic!("expected CPU frame");
        };
        let luma = &weighted_b.planes[0];
        let luma_bytes = &luma.bytes[luma.offset..luma.offset + luma.stride * luma.rows];
        assert!(luma_bytes.iter().all(|&sample| sample == 138));
        let chroma = &weighted_b.planes[1];
        let chroma_bytes =
            &chroma.bytes[chroma.offset..chroma.offset + chroma.stride * chroma.rows];
        assert!(chroma_bytes.iter().all(|&sample| sample == 128));
    }

    #[test]
    fn decodes_an_implicitly_weighted_bidirectional_b_picture() {
        let stream = annex_b_stream(&[
            (0x67, single_macroblock_main_sps_rbsp()),
            (0x68, single_macroblock_weighted_implicit_b_pps_rbsp()),
            (0x65, single_macroblock_idr_rbsp()),
            (0x41, single_macroblock_weighted_p_skip_at_poc_rbsp(4)),
            (0x01, single_macroblock_explicit_b_at_poc_rbsp(1)),
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
        let mut frames = Vec::new();
        for _ in 0..3 {
            let DecodeOutput::Frame(frame) = decoder.receive_frame().unwrap() else {
                panic!("expected a decoded frame");
            };
            frames.push(frame);
        }
        assert_eq!(
            frames.iter().map(|frame| frame.id).collect::<Vec<_>>(),
            [1, 3, 2]
        );
        let FrameStorage::Cpu(weighted_b) = &frames[1].storage else {
            panic!("expected CPU frame");
        };
        let luma = &weighted_b.planes[0];
        let luma_bytes = &luma.bytes[luma.offset..luma.offset + luma.stride * luma.rows];
        assert!(luma_bytes.iter().all(|&sample| sample == 143));
        let chroma = &weighted_b.planes[1];
        let chroma_bytes =
            &chroma.bytes[chroma.offset..chroma.offset + chroma.stride * chroma.rows];
        assert!(
            chroma_bytes
                .chunks_exact(2)
                .all(|samples| samples == [126, 117])
        );
    }

    #[test]
    fn decodes_explicitly_weighted_reference_p_picture() {
        let stream = annex_b_stream(&[
            (0x67, single_macroblock_sps_rbsp()),
            (0x68, single_macroblock_weighted_pps_rbsp()),
            (0x65, single_macroblock_idr_rbsp()),
            (0x41, single_macroblock_weighted_p_skip_rbsp()),
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
        assert!(matches!(
            decoder.receive_frame().unwrap(),
            DecodeOutput::Frame(_)
        ));
        let frame = match decoder.receive_frame().unwrap() {
            DecodeOutput::Frame(frame) => frame,
            output => panic!("expected weighted P frame, got {output:?}"),
        };
        let FrameStorage::Cpu(cpu) = frame.storage else {
            panic!("expected CPU frame");
        };
        assert!(
            cpu.planes[0].bytes[..256]
                .iter()
                .all(|&sample| sample == 187)
        );
        assert!(
            cpu.planes[1].bytes[256..]
                .chunks_exact(2)
                .all(|samples| samples == [118, 84])
        );
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
    fn decodes_length_prefixed_input_with_avcc_parameter_sets() {
        let sps = single_macroblock_sps_rbsp();
        let pps = single_macroblock_pps_rbsp();
        let codec_data = avcc(&sps, &pps, 4);
        let packet = length_prefixed_stream(&[(0x65, single_macroblock_idr_rbsp())], 4);
        let mut decoder = H264Decoder::new();
        decoder
            .configure(VideoDecoderConfig {
                codec: VideoCodec::H264,
                bitstream_format: BitstreamFormat::LengthPrefixed { length_size: 4 },
                codec_data: Some(codec_data.into()),
            })
            .unwrap();
        assert!(matches!(
            decoder
                .send_packet(EncodedVideoPacket::new(packet))
                .unwrap(),
            DecodeInputStatus::Accepted
        ));
        decoder.drain().unwrap();
        assert!(matches!(
            decoder.receive_frame().unwrap(),
            DecodeOutput::FormatChanged(_)
        ));
        assert!(matches!(
            decoder.receive_frame().unwrap(),
            DecodeOutput::Frame(_)
        ));

        assert!(
            decoder
                .configure(VideoDecoderConfig {
                    codec: VideoCodec::H264,
                    bitstream_format: BitstreamFormat::LengthPrefixed { length_size: 2 },
                    codec_data: Some(avcc(&sps, &pps, 4).into()),
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

    fn length_prefixed_stream(nals: &[(u8, Vec<u8>)], length_size: usize) -> Vec<u8> {
        let mut stream = Vec::new();
        for (header, rbsp) in nals {
            let ebsp = encode_ebsp(rbsp);
            let length = ebsp.len() + 1;
            let length_bytes = (length as u32).to_be_bytes();
            stream.extend_from_slice(&length_bytes[4 - length_size..]);
            stream.push(*header);
            stream.extend_from_slice(&ebsp);
        }
        stream
    }

    fn avcc(sps_rbsp: &[u8], pps_rbsp: &[u8], length_size: u8) -> Vec<u8> {
        let sps = encode_ebsp(sps_rbsp);
        let pps = encode_ebsp(pps_rbsp);
        let mut data = vec![1, sps_rbsp[0], 0, 10, 0xfc | (length_size - 1), 0xe1];
        data.extend_from_slice(&u16::try_from(sps.len() + 1).unwrap().to_be_bytes());
        data.push(0x67);
        data.extend_from_slice(&sps);
        data.push(1);
        data.extend_from_slice(&u16::try_from(pps.len() + 1).unwrap().to_be_bytes());
        data.push(0x68);
        data.extend_from_slice(&pps);
        data
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

    fn crc32(bytes: &[u8]) -> u32 {
        let mut crc = u32::MAX;
        for &byte in bytes {
            crc ^= u32::from(byte);
            for _ in 0..8 {
                crc = (crc >> 1) ^ (0xedb8_8320 & 0u32.wrapping_sub(crc & 1));
            }
        }
        !crc
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

    fn single_macroblock_main_sps_rbsp() -> Vec<u8> {
        let mut writer = BitWriter::default();
        writer.write_bits(77, 8); // Main profile
        writer.write_bits(0, 8); // constraints + reserved_zero_2bits
        writer.write_bits(10, 8); // level_idc
        writer.write_ue(0); // seq_parameter_set_id
        writer.write_ue(0); // log2_max_frame_num_minus4
        writer.write_ue(0); // pic_order_cnt_type
        writer.write_ue(0); // log2_max_pic_order_cnt_lsb_minus4
        writer.write_ue(2); // max_num_ref_frames
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

    fn single_macroblock_weighted_pps_rbsp() -> Vec<u8> {
        let mut writer = BitWriter::default();
        writer.write_ue(0); // pic_parameter_set_id
        writer.write_ue(0); // seq_parameter_set_id
        writer.write_flag(false); // entropy_coding_mode_flag: CAVLC
        writer.write_flag(false); // bottom_field_pic_order_in_frame_present_flag
        writer.write_ue(0); // num_slice_groups_minus1
        writer.write_ue(0); // num_ref_idx_l0_default_active_minus1
        writer.write_ue(0); // num_ref_idx_l1_default_active_minus1
        writer.write_flag(true); // weighted_pred_flag
        writer.write_bits(0, 2); // weighted_bipred_idc
        writer.write_se(0); // pic_init_qp_minus26
        writer.write_se(0); // pic_init_qs_minus26
        writer.write_se(0); // chroma_qp_index_offset
        writer.write_flag(false); // deblocking_filter_control_present_flag
        writer.write_flag(false); // constrained_intra_pred_flag
        writer.write_flag(false); // redundant_pic_cnt_present_flag
        writer.finish_rbsp()
    }

    fn single_macroblock_explicit_b_pps_rbsp() -> Vec<u8> {
        let mut writer = BitWriter::default();
        writer.write_ue(0); // pic_parameter_set_id
        writer.write_ue(0); // seq_parameter_set_id
        writer.write_flag(false); // entropy_coding_mode_flag: CAVLC
        writer.write_flag(false); // bottom_field_pic_order_in_frame_present_flag
        writer.write_ue(0); // num_slice_groups_minus1
        writer.write_ue(0); // num_ref_idx_l0_default_active_minus1
        writer.write_ue(0); // num_ref_idx_l1_default_active_minus1
        writer.write_flag(false); // weighted_pred_flag
        writer.write_bits(1, 2); // explicit weighted_bipred_idc
        writer.write_se(0); // pic_init_qp_minus26
        writer.write_se(0); // pic_init_qs_minus26
        writer.write_se(0); // chroma_qp_index_offset
        writer.write_flag(false); // deblocking_filter_control_present_flag
        writer.write_flag(false); // constrained_intra_pred_flag
        writer.write_flag(false); // redundant_pic_cnt_present_flag
        writer.finish_rbsp()
    }

    fn single_macroblock_weighted_implicit_b_pps_rbsp() -> Vec<u8> {
        let mut writer = BitWriter::default();
        writer.write_ue(0); // pic_parameter_set_id
        writer.write_ue(0); // seq_parameter_set_id
        writer.write_flag(false); // entropy_coding_mode_flag: CAVLC
        writer.write_flag(false); // bottom_field_pic_order_in_frame_present_flag
        writer.write_ue(0); // num_slice_groups_minus1
        writer.write_ue(0); // num_ref_idx_l0_default_active_minus1
        writer.write_ue(0); // num_ref_idx_l1_default_active_minus1
        writer.write_flag(true); // weighted_pred_flag
        writer.write_bits(2, 2); // implicit weighted_bipred_idc
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

    fn single_macroblock_p_skip_at_poc_rbsp(poc_lsb: u64) -> Vec<u8> {
        let mut writer = BitWriter::default();
        writer.write_ue(0); // first_mb_in_slice
        writer.write_ue(0); // P slice
        writer.write_ue(0); // pic_parameter_set_id
        writer.write_bits(1, 4); // frame_num
        writer.write_bits(poc_lsb, 4); // pic_order_cnt_lsb
        writer.write_flag(false); // num_ref_idx_active_override_flag
        writer.write_flag(false); // ref_pic_list_modification_flag_l0
        writer.write_flag(false); // adaptive_ref_pic_marking_mode_flag
        writer.write_se(0); // slice_qp_delta
        writer.write_ue(1); // mb_skip_run
        writer.finish_rbsp()
    }

    fn single_macroblock_explicit_b_rbsp() -> Vec<u8> {
        single_macroblock_explicit_b_at_poc_rbsp(2)
    }

    fn single_macroblock_explicit_b_at_poc_rbsp(poc_lsb: u64) -> Vec<u8> {
        let mut writer = BitWriter::default();
        writer.write_ue(0); // first_mb_in_slice
        writer.write_ue(1); // B slice
        writer.write_ue(0); // pic_parameter_set_id
        writer.write_bits(1, 4); // frame_num (non-reference pictures do not advance it)
        writer.write_bits(poc_lsb, 4); // pic_order_cnt_lsb
        writer.write_flag(true); // direct_spatial_mv_pred_flag
        writer.write_flag(false); // num_ref_idx_active_override_flag
        writer.write_flag(false); // ref_pic_list_modification_flag_l0
        writer.write_flag(false); // ref_pic_list_modification_flag_l1
        writer.write_se(0); // slice_qp_delta

        writer.write_ue(0); // mb_skip_run
        writer.write_ue(3); // B_Bi_16x16
        writer.write_se(0); // mvd_l0.x
        writer.write_se(0); // mvd_l0.y
        writer.write_se(0); // mvd_l1.x
        writer.write_se(0); // mvd_l1.y
        writer.write_ue(0); // coded_block_pattern -> zero
        writer.finish_rbsp()
    }

    fn single_macroblock_spatial_direct_b_rbsp() -> Vec<u8> {
        let mut writer = BitWriter::default();
        writer.write_ue(0); // first_mb_in_slice
        writer.write_ue(1); // B slice
        writer.write_ue(0); // pic_parameter_set_id
        writer.write_bits(1, 4); // frame_num
        writer.write_bits(2, 4); // pic_order_cnt_lsb
        writer.write_flag(true); // direct_spatial_mv_pred_flag
        writer.write_flag(false); // num_ref_idx_active_override_flag
        writer.write_flag(false); // ref_pic_list_modification_flag_l0
        writer.write_flag(false); // ref_pic_list_modification_flag_l1
        writer.write_se(0); // slice_qp_delta
        writer.write_ue(0); // mb_skip_run
        writer.write_ue(0); // B_Direct_16x16
        writer.write_ue(0); // coded_block_pattern -> zero
        writer.finish_rbsp()
    }

    fn single_macroblock_spatial_skip_b_rbsp() -> Vec<u8> {
        let mut writer = BitWriter::default();
        writer.write_ue(0); // first_mb_in_slice
        writer.write_ue(1); // B slice
        writer.write_ue(0); // pic_parameter_set_id
        writer.write_bits(1, 4); // frame_num
        writer.write_bits(2, 4); // pic_order_cnt_lsb
        writer.write_flag(true); // direct_spatial_mv_pred_flag
        writer.write_flag(false); // num_ref_idx_active_override_flag
        writer.write_flag(false); // ref_pic_list_modification_flag_l0
        writer.write_flag(false); // ref_pic_list_modification_flag_l1
        writer.write_se(0); // slice_qp_delta
        writer.write_ue(1); // mb_skip_run
        writer.finish_rbsp()
    }

    fn single_macroblock_temporal_skip_b_rbsp() -> Vec<u8> {
        let mut writer = BitWriter::default();
        writer.write_ue(0); // first_mb_in_slice
        writer.write_ue(1); // B slice
        writer.write_ue(0); // pic_parameter_set_id
        writer.write_bits(1, 4); // frame_num
        writer.write_bits(2, 4); // pic_order_cnt_lsb
        writer.write_flag(false); // direct_spatial_mv_pred_flag
        writer.write_flag(false); // num_ref_idx_active_override_flag
        writer.write_flag(false); // ref_pic_list_modification_flag_l0
        writer.write_flag(false); // ref_pic_list_modification_flag_l1
        writer.write_se(0); // slice_qp_delta
        writer.write_ue(1); // mb_skip_run
        writer.finish_rbsp()
    }

    fn single_macroblock_all_direct_b_8x8_rbsp() -> Vec<u8> {
        let mut writer = BitWriter::default();
        writer.write_ue(0); // first_mb_in_slice
        writer.write_ue(1); // B slice
        writer.write_ue(0); // pic_parameter_set_id
        writer.write_bits(1, 4); // frame_num
        writer.write_bits(2, 4); // pic_order_cnt_lsb
        writer.write_flag(true); // direct_spatial_mv_pred_flag
        writer.write_flag(false); // num_ref_idx_active_override_flag
        writer.write_flag(false); // ref_pic_list_modification_flag_l0
        writer.write_flag(false); // ref_pic_list_modification_flag_l1
        writer.write_se(0); // slice_qp_delta
        writer.write_ue(0); // mb_skip_run
        writer.write_ue(22); // B_8x8
        for _ in 0..4 {
            writer.write_ue(0); // B_Direct_8x8
        }
        writer.write_ue(0); // coded_block_pattern -> zero
        writer.finish_rbsp()
    }

    fn single_macroblock_mixed_direct_b_8x8_rbsp() -> Vec<u8> {
        let mut writer = BitWriter::default();
        writer.write_ue(0); // first_mb_in_slice
        writer.write_ue(1); // B slice
        writer.write_ue(0); // pic_parameter_set_id
        writer.write_bits(1, 4); // frame_num
        writer.write_bits(2, 4); // pic_order_cnt_lsb
        writer.write_flag(true); // direct_spatial_mv_pred_flag
        writer.write_flag(false); // num_ref_idx_active_override_flag
        writer.write_flag(false); // ref_pic_list_modification_flag_l0
        writer.write_flag(false); // ref_pic_list_modification_flag_l1
        writer.write_se(0); // slice_qp_delta
        writer.write_ue(0); // mb_skip_run
        writer.write_ue(22); // B_8x8
        writer.write_ue(0); // B_Direct_8x8
        writer.write_ue(1); // B_L0_8x8
        writer.write_ue(2); // B_L1_8x8
        writer.write_ue(3); // B_Bi_8x8
        for _ in 0..2 {
            writer.write_se(0); // mvd_l0.x
            writer.write_se(0); // mvd_l0.y
        }
        for _ in 0..2 {
            writer.write_se(0); // mvd_l1.x
            writer.write_se(0); // mvd_l1.y
        }
        writer.write_ue(0); // coded_block_pattern -> zero
        writer.finish_rbsp()
    }

    fn single_macroblock_explicit_b_weighted_rbsp() -> Vec<u8> {
        let mut writer = BitWriter::default();
        writer.write_ue(0); // first_mb_in_slice
        writer.write_ue(1); // B slice
        writer.write_ue(0); // pic_parameter_set_id
        writer.write_bits(1, 4); // frame_num
        writer.write_bits(2, 4); // pic_order_cnt_lsb
        writer.write_flag(true); // direct_spatial_mv_pred_flag
        writer.write_flag(false); // num_ref_idx_active_override_flag
        writer.write_flag(false); // ref_pic_list_modification_flag_l0
        writer.write_flag(false); // ref_pic_list_modification_flag_l1

        writer.write_ue(0); // luma_log2_weight_denom
        writer.write_ue(0); // chroma_log2_weight_denom
        writer.write_flag(true); // luma_weight_l0_flag
        writer.write_se(1); // luma_weight_l0
        writer.write_se(20); // luma_offset_l0
        writer.write_flag(false); // chroma_weight_l0_flag
        writer.write_flag(true); // luma_weight_l1_flag
        writer.write_se(1); // luma_weight_l1
        writer.write_se(0); // luma_offset_l1
        writer.write_flag(false); // chroma_weight_l1_flag

        writer.write_se(0); // slice_qp_delta
        writer.write_ue(0); // mb_skip_run
        writer.write_ue(3); // B_Bi_16x16
        writer.write_se(0); // mvd_l0.x
        writer.write_se(0); // mvd_l0.y
        writer.write_se(0); // mvd_l1.x
        writer.write_se(0); // mvd_l1.y
        writer.write_ue(0); // coded_block_pattern -> zero
        writer.finish_rbsp()
    }

    fn single_macroblock_weighted_p_skip_rbsp() -> Vec<u8> {
        single_macroblock_weighted_p_skip_at_poc_rbsp(2)
    }

    fn single_macroblock_weighted_p_skip_at_poc_rbsp(poc_lsb: u64) -> Vec<u8> {
        let mut writer = BitWriter::default();
        writer.write_ue(0); // first_mb_in_slice
        writer.write_ue(0); // P slice
        writer.write_ue(0); // pic_parameter_set_id
        writer.write_bits(1, 4); // frame_num
        writer.write_bits(poc_lsb, 4); // pic_order_cnt_lsb
        writer.write_flag(false); // num_ref_idx_active_override_flag
        writer.write_flag(false); // ref_pic_list_modification_flag_l0
        writer.write_ue(1); // luma_log2_weight_denom
        writer.write_ue(1); // chroma_log2_weight_denom
        writer.write_flag(true); // luma_weight_l0_flag
        writer.write_se(3); // luma_weight_l0
        writer.write_se(-5); // luma_offset_l0
        writer.write_flag(true); // chroma_weight_l0_flag
        writer.write_se(2); // chroma_weight_l0[Cb]
        writer.write_se(-10); // chroma_offset_l0[Cb]
        writer.write_se(1); // chroma_weight_l0[Cr]
        writer.write_se(20); // chroma_offset_l0[Cr]
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
