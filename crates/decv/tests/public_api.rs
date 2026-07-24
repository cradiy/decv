use std::{num::NonZeroUsize, sync::Arc};

use decv::{
    BitstreamFormat, ColorInfo, ColorMatrix, ColorPrimaries, ColorRange, CpuFrame, CpuPlane,
    DecodeInputStatus, DecodeOutput, DecodedVideoFrame, EncodedVideoPacket, FrameStorage,
    H264Decoder, H264Error, H264Parallelism, MediaTime, Mp4Demuxer, PixelFormat, Rect, Size,
    TransferFunction, VideoCodec, VideoDecoder, VideoDecoderConfig, VideoFormat,
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
}

#[test]
fn facade_exposes_random_access_mp4_input() {
    let error = Mp4Demuxer::open(Vec::<u8>::new()).unwrap_err();
    assert!(!error.to_string().is_empty());
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
