use std::{num::NonZeroUsize, sync::Arc};

use decv::{
    BitstreamFormat, DecodeInputStatus, DecodeOutput, EncodedVideoPacket, FrameStorage,
    H264Decoder, H264Error, H264Parallelism, Mp4Demuxer, VideoCodec, VideoDecoder,
    VideoDecoderConfig,
};

fn accepts_consumer_decoder<D>(decoder: &mut D)
where
    D: VideoDecoder<Error = H264Error>,
{
    decoder
        .configure(VideoDecoderConfig {
            codec: VideoCodec::H264,
            bitstream_format: BitstreamFormat::ByteStream,
            codec_data: None,
        })
        .unwrap();

    let packet = EncodedVideoPacket::new(Arc::<[u8]>::from([]));
    match decoder.send_packet(packet).unwrap() {
        DecodeInputStatus::Accepted => {}
        DecodeInputStatus::NeedOutput(_) => panic!("an empty fresh decoder has no pending output"),
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

    let _ = storage_kind;
}
