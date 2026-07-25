use std::{num::NonZeroUsize, sync::Arc};

use decv::{
    AacDecoder, AudioCodec, AudioDecodeInputStatus, AudioDecodeOutput, AudioDecoder,
    AudioDecoderConfig, BitstreamFormat, ChannelLayout, ColorInfo, ColorMatrix, ColorPrimaries,
    ColorRange, CpuFrame, CpuPlane, DecodeInputStatus, DecodeOutput, DecodedVideoFrame,
    EncodedAudioPacket, EncodedVideoPacket, FourCc, FrameStorage, H264Decoder, H264Error,
    H264Mp4SeekController, H264Mp4SeekOutcome, H264Parallelism, H264SeekCheckpointCache, MediaTime,
    Mp4Demuxer, PcmCodec, PixelFormat, Rect, Size, SoftwareAudioDecoder, TransferFunction,
    VideoCodec, VideoDecoder, VideoDecoderConfig, VideoFormat,
};

fn accepts_consumer_decoder<D>(decoder: &mut D)
where
    D: VideoDecoder<Error = H264Error>,
{
    decoder
        .configure(VideoDecoderConfig::new(
            VideoCodec::H264,
            BitstreamFormat::ByteStream,
        ))
        .unwrap();

    let packet = EncodedVideoPacket::new(Arc::<[u8]>::from([]));
    match decoder.send_packet(packet).unwrap() {
        DecodeInputStatus::Accepted => {}
        DecodeInputStatus::NeedOutput(_) => panic!("an empty fresh decoder has no pending output"),
        _ => panic!("unexpected decoder input status"),
    }
    assert!(matches!(
        decoder.receive_frame().unwrap(),
        DecodeOutput::NeedInput
    ));
}

#[test]
fn facade_exposes_the_complete_decoder_contract() {
    let mut decoder = H264Decoder::new();
    decoder
        .set_parallelism(H264Parallelism::Threads(NonZeroUsize::new(1).unwrap()))
        .unwrap();
    accepts_consumer_decoder(&mut decoder);
    decoder.flush_for_seek(MediaTime::from_parts(0, 1).unwrap());
}

#[test]
fn facade_exposes_random_access_mp4_input() {
    let error = Mp4Demuxer::open(Vec::<u8>::new()).unwrap_err();
    assert!(!error.to_string().is_empty());
}

#[test]
fn facade_decodes_a_real_aac_lc_access_unit() {
    let access_unit = [
        0xde, 0x02, 0x00, 0x4c, 0x61, 0x76, 0x63, 0x35, 0x39, 0x2e, 0x33, 0x37, 0x2e, 0x31, 0x30,
        0x30, 0x00, 0x42, 0x20, 0x08, 0xc1, 0x18, 0x38,
    ];
    let pts = MediaTime::from_parts(-1_024, 44_100);
    let mut packet = EncodedAudioPacket::new(access_unit);
    packet.pts = pts;
    packet.duration = MediaTime::from_parts(1_024, 44_100);

    let config = AudioDecoderConfig::new(AudioCodec::Aac, 44_100, ChannelLayout::Stereo)
        .with_codec_data([0x12, 0x10, 0x56, 0xe5, 0x00]);
    let mut decoder = AacDecoder::new();
    decoder.configure(config).unwrap();
    assert!(matches!(
        decoder.send_packet(packet).unwrap(),
        AudioDecodeInputStatus::Accepted
    ));
    assert!(matches!(
        decoder.receive_frame().unwrap(),
        AudioDecodeOutput::FormatChanged(_)
    ));
    let AudioDecodeOutput::Frame(frame) = decoder.receive_frame().unwrap() else {
        panic!("expected decoded AAC frame");
    };
    frame.validate().unwrap();
    assert_eq!(frame.pts, pts);
    assert_eq!(frame.duration, MediaTime::from_parts(1_024, 44_100));
    assert_eq!(frame.channels(), 2);
    assert_eq!(frame.sample_frames(), 1_024);
    assert_eq!(frame.samples.len(), 2_048);
}

#[test]
fn facade_exposes_the_multi_codec_audio_decoder() {
    let config = AudioDecoderConfig::new(
        AudioCodec::Pcm(PcmCodec::Signed16Le),
        48_000,
        ChannelLayout::Mono,
    )
    .with_bits_per_sample(16);
    let mut decoder = SoftwareAudioDecoder::new();
    decoder.configure(config).unwrap();

    let mut packet = EncodedAudioPacket::new([0x00, 0x80, 0xff, 0x7f]);
    packet.duration = MediaTime::from_parts(2, 48_000);
    assert!(matches!(
        decoder.send_packet(packet).unwrap(),
        AudioDecodeInputStatus::Accepted
    ));
    assert!(matches!(
        decoder.receive_frame().unwrap(),
        AudioDecodeOutput::FormatChanged(_)
    ));
    let AudioDecodeOutput::Frame(frame) = decoder.receive_frame().unwrap() else {
        panic!("expected decoded PCM frame");
    };
    assert_eq!(frame.sample_frames(), 2);
    assert_eq!(frame.samples.as_ref(), [-1.0, 0.999_969_5]);
}

#[test]
fn facade_decodes_a_real_h264_access_unit_end_to_end() {
    // 16x16 testsrc2 encoded as one High-profile CABAC IDR by x264, with
    // informational SEI removed. This fixture is intentionally consumed only
    // through the public `decv` facade.
    let stream = [
        0x00, 0x00, 0x00, 0x01, 0x67, 0x64, 0x10, 0x0a, 0xac, 0xbb, 0xd8, 0x08, 0x80, 0x00, 0x00,
        0x03, 0x00, 0x80, 0x00, 0x00, 0x03, 0x01, 0x02, 0x00, 0x00, 0x00, 0x01, 0x68, 0xee, 0x0f,
        0x2c, 0x8b, 0x00, 0x00, 0x01, 0x65, 0x88, 0x84, 0x37, 0xd4, 0xda, 0x29, 0x82, 0xac, 0x9e,
        0xa8, 0x19, 0x64, 0x16, 0xcd, 0x88, 0xb1, 0xfe, 0x2f, 0x49, 0x0a, 0x9d, 0xe5, 0xc1, 0xd8,
        0xc0, 0xb2, 0x37, 0x38, 0x08, 0xb2, 0x12, 0xe4, 0x9e, 0x56, 0x4a, 0xba, 0xaf, 0x42, 0x19,
        0x87, 0x13, 0xfd, 0xb7, 0x63, 0xf0, 0x80, 0x05, 0x0d, 0x4c, 0x17, 0x9d, 0x20, 0x3e, 0x05,
        0xcc, 0x84, 0x70, 0x23, 0x25, 0x5a, 0xa0, 0x35, 0x9b, 0x65, 0x74, 0xfd, 0xa9, 0xa0, 0x4d,
        0x17, 0xeb, 0x33, 0x7b, 0x77, 0x8b, 0x2c, 0xa7, 0x84, 0xf8, 0x55, 0xcf, 0x2a, 0x68, 0x25,
        0xb9, 0xeb, 0x0d, 0x3e, 0x7b, 0x20, 0x4e, 0x5d, 0xac, 0x7f, 0xf8, 0x37, 0x17, 0xe7, 0xc2,
        0x44, 0x04, 0x84, 0xf1, 0x8e, 0x45, 0xd1, 0xa6, 0xaf, 0xed, 0xc6, 0x3d, 0x23, 0xbd, 0xc2,
        0x7a, 0xbe, 0x24, 0x3a, 0x59, 0x55, 0xa9, 0xa9, 0xad, 0x3c, 0x4d, 0x97, 0xa3, 0xc3, 0x32,
        0x43, 0x5c, 0x89, 0x53, 0xef, 0x73, 0x32, 0x11, 0xb3, 0x85, 0x5a, 0x18, 0x9c, 0xf7, 0x6f,
        0xb5, 0x6e, 0x4d, 0xb2, 0xc2, 0x91, 0x4c, 0x68, 0xa3, 0x50, 0x87, 0x9b, 0x82, 0x51, 0xf7,
        0xeb, 0xae, 0xb9, 0x9c, 0x68, 0xe2, 0xa4, 0xef, 0xc2, 0x56, 0x11, 0xbe, 0xbd, 0x28, 0x13,
        0xf9, 0xdb, 0x93, 0xbf, 0xf5, 0x74, 0xd9, 0xd3, 0x8d,
    ];
    let pts = MediaTime::from_parts(9000, 90_000);
    let duration = MediaTime::from_parts(3000, 90_000);
    let mut packet = EncodedVideoPacket::new(stream);
    packet.pts = pts;
    packet.duration = duration;
    packet.keyframe = true;

    let mut decoder = H264Decoder::new();
    decoder
        .configure(VideoDecoderConfig::new(
            VideoCodec::H264,
            BitstreamFormat::ByteStream,
        ))
        .unwrap();
    assert!(matches!(
        decoder.send_packet(packet).unwrap(),
        DecodeInputStatus::Accepted
    ));
    assert!(matches!(
        decoder.receive_frame().unwrap(),
        DecodeOutput::NeedInput
    ));

    decoder.drain().unwrap();
    let format = match decoder.receive_frame().unwrap() {
        DecodeOutput::FormatChanged(format) => format,
        output => panic!("expected format change, got {output:?}"),
    };
    assert_eq!(format.coded_size, Size::new(16, 16));
    assert_eq!(format.visible_rect, Rect::new(0, 0, 16, 16));
    assert_eq!(format.pixel_format, PixelFormat::Nv12);

    let frame = match decoder.receive_frame().unwrap() {
        DecodeOutput::Frame(frame) => frame,
        output => panic!("expected decoded frame, got {output:?}"),
    };
    frame.validate().unwrap();
    assert_eq!(frame.pts, pts);
    assert_eq!(frame.duration, duration);
    assert_eq!(frame.format, format);

    let cpu = match &frame.storage {
        FrameStorage::Cpu(cpu) => cpu,
        _ => panic!("expected CPU-backed frame"),
    };
    assert_eq!(cpu.planes.len(), 2);
    assert_eq!((cpu.planes[0].stride, cpu.planes[0].rows), (16, 16));
    assert_eq!((cpu.planes[1].stride, cpu.planes[1].rows), (16, 8));
    assert_eq!(crc32(&tightly_packed_bytes(cpu)), 2_320_103_694);
    assert!(matches!(
        decoder.receive_frame().unwrap(),
        DecodeOutput::EndOfStream
    ));
}

#[test]
fn facade_demuxes_and_decodes_a_real_mp4_in_presentation_order() {
    // Three 16x16 High-profile CABAC pictures in MP4 decode order I, P, B.
    // FFmpeg independently produced the expected presentation order and NV12
    // CRC below. The test intentionally uses no implementation-crate APIs.
    let mp4 = decode_hex(include_str!("fixtures/three-frame-high-b.mp4.hex"));
    let demuxer = Mp4Demuxer::open(mp4).unwrap();
    let track_index = demuxer
        .movie()
        .tracks()
        .iter()
        .position(|track| track.handler() == FourCc::new(*b"vide"))
        .unwrap();
    let mut cursor = demuxer.packet_cursor(track_index).unwrap();
    assert_eq!(cursor.track().samples().len(), 3);
    assert_eq!(
        cursor
            .seek_to_nearest_keyframe(MediaTime::from_parts(512, 15_360).unwrap())
            .unwrap(),
        Some(0)
    );
    cursor.rewind();

    let config = cursor.decoder_config().unwrap().unwrap();
    assert_eq!(
        config.bitstream_format,
        BitstreamFormat::LengthPrefixed { length_size: 4 }
    );
    assert!(config.codec_data.is_some());

    let mut decoder = H264Decoder::new();
    decoder.configure(config).unwrap();
    let mut format = None;
    let mut frames = Vec::new();
    while let Some(mut packet) = cursor.next_packet().unwrap() {
        loop {
            match decoder.send_packet(packet).unwrap() {
                DecodeInputStatus::Accepted => break,
                DecodeInputStatus::NeedOutput(unconsumed) => {
                    packet = unconsumed;
                    assert!(!pull_available(&mut decoder, &mut format, &mut frames));
                }
                _ => panic!("unexpected decoder input status"),
            }
        }
        assert!(!pull_available(&mut decoder, &mut format, &mut frames));
    }

    decoder.drain().unwrap();
    assert!(pull_available(&mut decoder, &mut format, &mut frames));
    let format = format.unwrap();
    assert_eq!(format.coded_size, Size::new(16, 16));
    assert_eq!(format.pixel_format, PixelFormat::Nv12);
    assert_eq!(frames.len(), 3);

    let expected_pts = [0, 512, 1024];
    for ((index, frame), expected_pts) in frames.iter().enumerate().zip(expected_pts) {
        frame.validate().unwrap();
        assert_eq!(frame.id, u64::try_from(index + 1).unwrap());
        assert_eq!(frame.pts.unwrap().value, expected_pts);
        assert_eq!(frame.pts.unwrap().timescale.get(), 15_360);
        assert_eq!(frame.duration.unwrap().value, 512);
        assert_eq!(frame.format, format);
        let cpu = match &frame.storage {
            FrameStorage::Cpu(cpu) => cpu,
            _ => panic!("expected CPU-backed frame"),
        };
        assert_eq!(crc32(&tightly_packed_bytes(cpu)), 3_859_821_206);
    }
}

#[test]
fn facade_retargets_an_active_exact_seek_without_rewinding_mp4() {
    let mp4 = decode_hex(include_str!("fixtures/three-frame-high-b.mp4.hex"));
    let demuxer = Mp4Demuxer::open(mp4).unwrap();
    let track_index = demuxer
        .movie()
        .tracks()
        .iter()
        .position(|track| track.handler() == FourCc::new(*b"vide"))
        .unwrap();
    let mut cursor = demuxer.packet_cursor(track_index).unwrap();
    let zero = MediaTime::from_parts(0, 15_360).unwrap();
    assert_eq!(cursor.seek_to_keyframe(zero).unwrap(), Some(0));

    let mut decoder = H264Decoder::new();
    decoder.configure(cursor.decoder_config().unwrap().unwrap()).unwrap();
    decoder.flush_for_seek(zero);

    let mut first = cursor.next_packet().unwrap().unwrap();
    first.discontinuity = true;
    assert!(matches!(
        decoder.send_packet(first).unwrap(),
        DecodeInputStatus::Accepted
    ));
    assert!(matches!(
        decoder.receive_frame().unwrap(),
        DecodeOutput::NeedInput
    ));

    let final_target = MediaTime::from_parts(1024, 15_360).unwrap();
    decoder.retarget_seek_forward(final_target).unwrap();
    let mut format = None;
    let mut frames = Vec::new();
    while let Some(mut packet) = cursor.next_packet().unwrap() {
        loop {
            match decoder.send_packet(packet).unwrap() {
                DecodeInputStatus::Accepted => break,
                DecodeInputStatus::NeedOutput(unconsumed) => {
                    packet = unconsumed;
                    assert!(!pull_available(&mut decoder, &mut format, &mut frames));
                }
                _ => panic!("unexpected decoder input status"),
            }
        }
        assert!(!pull_available(&mut decoder, &mut format, &mut frames));
    }

    decoder.drain().unwrap();
    assert!(pull_available(&mut decoder, &mut format, &mut frames));
    assert_eq!(frames.len(), 1);
    assert_eq!(frames[0].pts, Some(final_target));
    frames[0].validate().unwrap();
}

#[test]
fn facade_restores_an_mp4_seek_checkpoint_at_an_earlier_target() {
    let mp4 = decode_hex(include_str!("fixtures/three-frame-high-b.mp4.hex"));
    let demuxer = Mp4Demuxer::open(mp4).unwrap();
    let track_index = demuxer
        .movie()
        .tracks()
        .iter()
        .position(|track| track.handler() == FourCc::new(*b"vide"))
        .unwrap();
    let mut cursor = demuxer.packet_cursor(track_index).unwrap();
    let anchor_time = MediaTime::from_parts(0, 15_360).unwrap();
    assert_eq!(cursor.seek_to_keyframe(anchor_time).unwrap(), Some(0));

    let mut decoder = H264Decoder::new();
    decoder.configure(cursor.decoder_config().unwrap().unwrap()).unwrap();
    let final_target = MediaTime::from_parts(1024, 15_360).unwrap();
    decoder.flush_for_seek(final_target);
    let mut anchor = cursor.next_packet().unwrap().unwrap();
    anchor.discontinuity = true;
    assert!(matches!(
        decoder.send_packet(anchor).unwrap(),
        DecodeInputStatus::Accepted
    ));
    let resume_sample = cursor.next_sample_index();
    let mut checkpoints = H264SeekCheckpointCache::new(4, 128 * 1024 * 1024);
    assert!(checkpoints.capture(&mut decoder, resume_sample).unwrap());
    let cached = checkpoints.latest_before(final_target).unwrap();
    assert_eq!(cached.checkpoint().retained_reference_count(), 1);
    assert!(cached.estimated_retained_reference_bytes() > 0);
    let remaining_packets = std::iter::from_fn(|| cursor.next_packet().transpose())
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    let mut format = None;
    let mut frames = Vec::new();
    send_packets(&mut decoder, remaining_packets, &mut format, &mut frames);
    decoder.drain().unwrap();
    assert!(pull_available(&mut decoder, &mut format, &mut frames));
    assert_eq!(frames.len(), 1);
    assert_eq!(frames[0].pts, Some(final_target));

    let earlier_target = MediaTime::from_parts(512, 15_360).unwrap();
    let resume_sample = *checkpoints
        .restore_latest_before(&mut decoder, earlier_target)
        .unwrap()
        .unwrap();
    cursor.seek_to_sample(resume_sample).unwrap();
    let replay_packets = std::iter::from_fn(|| cursor.next_packet().transpose())
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    frames.clear();
    send_packets(&mut decoder, replay_packets, &mut format, &mut frames);
    decoder.drain().unwrap();
    assert!(pull_available(&mut decoder, &mut format, &mut frames));
    assert_eq!(
        frames.iter().map(|frame| frame.pts).collect::<Vec<_>>(),
        [Some(earlier_target), Some(final_target)]
    );
    assert!(frames.iter().all(|frame| frame.validate().is_ok()));
}

#[test]
fn facade_controller_selects_keyframe_checkpoint_and_forward_retarget() {
    let mp4 = decode_hex(include_str!("fixtures/three-frame-high-b.mp4.hex"));
    let demuxer = Mp4Demuxer::open(mp4).unwrap();
    let track_index = demuxer
        .movie()
        .tracks()
        .iter()
        .position(|track| track.handler() == FourCc::new(*b"vide"))
        .unwrap();
    let mut cursor = demuxer.packet_cursor(track_index).unwrap();
    let mut decoder = H264Decoder::new();
    decoder
        .configure(cursor.decoder_config().unwrap().unwrap())
        .unwrap();
    let mut seeks = H264Mp4SeekController::new(track_index, 4, 128 * 1024 * 1024);

    let final_target = MediaTime::from_parts(1024, 15_360).unwrap();
    let cold = seeks
        .begin_exact_seek(&mut decoder, &mut cursor, final_target, false)
        .unwrap();
    assert_eq!(cold, H264Mp4SeekOutcome::Keyframe { sample_index: 0 });
    assert!(cold.requires_discontinuity());

    let mut anchor = cursor.next_packet().unwrap().unwrap();
    anchor.discontinuity = cold.requires_discontinuity();
    assert!(matches!(
        decoder.send_packet(anchor).unwrap(),
        DecodeInputStatus::Accepted
    ));
    assert!(seeks.capture_checkpoint(&mut decoder, &cursor).unwrap());
    assert_eq!(seeks.checkpoint_count(), 1);
    assert!(!seeks.capture_checkpoint(&mut decoder, &cursor).unwrap());
    assert_eq!(
        seeks.minimum_checkpoint_sample_distance(),
        H264Mp4SeekController::DEFAULT_MINIMUM_CHECKPOINT_SAMPLE_DISTANCE
    );

    let earlier_target = MediaTime::from_parts(512, 15_360).unwrap();
    let restored = seeks
        .begin_exact_seek(&mut decoder, &mut cursor, earlier_target, false)
        .unwrap();
    assert_eq!(restored, H264Mp4SeekOutcome::Checkpoint { sample_index: 1 });
    assert!(!restored.requires_discontinuity());
    assert_eq!(cursor.next_sample_index(), 1);

    let retargeted = seeks
        .begin_exact_seek(&mut decoder, &mut cursor, final_target, true)
        .unwrap();
    assert_eq!(retargeted, H264Mp4SeekOutcome::ForwardRetarget);
    assert!(!retargeted.requires_discontinuity());
    assert_eq!(cursor.next_sample_index(), 1);
}

#[test]
fn facade_controller_uses_a_discontinuous_keyframe_for_preview() {
    let mp4 = decode_hex(include_str!("fixtures/three-frame-high-b.mp4.hex"));
    let demuxer = Mp4Demuxer::open(mp4).unwrap();
    let track_index = demuxer
        .movie()
        .tracks()
        .iter()
        .position(|track| track.handler() == FourCc::new(*b"vide"))
        .unwrap();
    let mut cursor = demuxer.packet_cursor(track_index).unwrap();
    let mut decoder = H264Decoder::new();
    decoder
        .configure(cursor.decoder_config().unwrap().unwrap())
        .unwrap();
    let mut seeks = H264Mp4SeekController::new(track_index, 2, 64 * 1024 * 1024);

    let target = MediaTime::from_parts(512, 15_360).unwrap();
    assert_eq!(
        seeks
            .begin_nearest_preview(&mut decoder, &mut cursor, target)
            .unwrap(),
        0
    );
    assert_eq!(cursor.next_sample_index(), 0);
    assert_eq!(seeks.active_exact_target(), None);
}

#[test]
fn frame_storage_requires_forward_compatible_matching() {
    fn storage_kind(storage: &FrameStorage) -> &'static str {
        match storage {
            FrameStorage::Cpu(_) => "cpu",
            _ => "future",
        }
    }

    let format = VideoFormat::new(
        Size::new(2, 2),
        Rect::new(0, 0, 2, 2),
        Size::new(2, 2),
        PixelFormat::Nv12,
        ColorInfo::new(
            ColorRange::Limited,
            ColorMatrix::Bt709,
            ColorPrimaries::Bt709,
            TransferFunction::Bt709,
        ),
    );
    let frame = DecodedVideoFrame::new(
        1,
        None,
        None,
        format,
        FrameStorage::Cpu(CpuFrame::new(vec![
            CpuPlane::new(Arc::<[u8]>::from([0; 4]), 0, 2, 2),
            CpuPlane::new(Arc::<[u8]>::from([128; 2]), 0, 2, 1),
        ])),
    );

    assert_eq!(storage_kind(&frame.storage), "cpu");
    frame.validate().unwrap();
}

fn pull_available(
    decoder: &mut H264Decoder,
    format: &mut Option<VideoFormat>,
    frames: &mut Vec<DecodedVideoFrame>,
) -> bool {
    loop {
        match decoder.receive_frame().unwrap() {
            DecodeOutput::FormatChanged(changed) => {
                assert!(format.replace(changed).is_none());
            }
            DecodeOutput::Frame(frame) => frames.push(frame),
            DecodeOutput::NeedInput => return false,
            DecodeOutput::EndOfStream => return true,
            _ => panic!("unexpected decoder output"),
        }
    }
}

fn send_packets(
    decoder: &mut H264Decoder,
    packets: impl IntoIterator<Item = EncodedVideoPacket>,
    format: &mut Option<VideoFormat>,
    frames: &mut Vec<DecodedVideoFrame>,
) {
    for mut packet in packets {
        loop {
            match decoder.send_packet(packet).unwrap() {
                DecodeInputStatus::Accepted => break,
                DecodeInputStatus::NeedOutput(unconsumed) => {
                    packet = unconsumed;
                    assert!(!pull_available(decoder, format, frames));
                }
                _ => panic!("unexpected decoder input status"),
            }
        }
        assert!(!pull_available(decoder, format, frames));
    }
}

fn tightly_packed_bytes(frame: &CpuFrame) -> Vec<u8> {
    let mut bytes = Vec::new();
    for plane in &frame.planes {
        for row in 0..plane.rows {
            let start = plane.offset + row * plane.stride;
            bytes.extend_from_slice(&plane.bytes[start..start + plane.stride]);
        }
    }
    bytes
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

fn decode_hex(text: &str) -> Vec<u8> {
    let digits = text
        .bytes()
        .filter(|byte| !byte.is_ascii_whitespace())
        .collect::<Vec<_>>();
    assert!(digits.len().is_multiple_of(2));
    digits
        .chunks_exact(2)
        .map(|pair| hex_digit(pair[0]) << 4 | hex_digit(pair[1]))
        .collect()
}

fn hex_digit(byte: u8) -> u8 {
    match byte {
        b'0'..=b'9' => byte - b'0',
        b'a'..=b'f' => byte - b'a' + 10,
        b'A'..=b'F' => byte - b'A' + 10,
        _ => panic!("fixture contains a non-hex byte"),
    }
}
