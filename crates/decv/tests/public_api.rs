use std::{num::NonZeroUsize, sync::Arc};

use decv::{
    BitstreamFormat, ColorInfo, ColorMatrix, ColorPrimaries, ColorRange, CpuFrame, CpuPlane,
    DecodeInputStatus, DecodeOutput, DecodedVideoFrame, EncodedVideoPacket, FrameStorage,
    H264Decoder, H264Error, H264Parallelism, Mp4Demuxer, PixelFormat, Rect, Size, TransferFunction,
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
}

#[test]
fn facade_exposes_random_access_mp4_input() {
    let error = Mp4Demuxer::open(Vec::<u8>::new()).unwrap_err();
    assert!(!error.to_string().is_empty());
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
