//! Stable-candidate consumer facade for the decv workspace.
//!
//! Applications should prefer this crate over depending on codec syntax and
//! reconstruction internals directly. The facade contains the synchronous
//! decoder contract, immutable frame types, H.264 decoder configuration, and
//! ordinary MP4 packet access needed by a playback or transcoding pipeline.

#![forbid(unsafe_code)]

pub use decv_core::{
    BitstreamFormat, ColorInfo, ColorMatrix, ColorPrimaries, ColorRange, CpuFrame, CpuPlane,
    DecodeInputStatus, DecodeOutput, DecodedVideoFrame, EncodedVideoPacket, FrameStorage,
    MediaError, MediaInput, MediaTime, PixelFormat, Rect, Size, TransferFunction, VideoCodec,
    VideoDecoder, VideoDecoderConfig, VideoFormat,
};
pub use decv_h264::{H264Decoder, H264Error, H264Parallelism};
pub use decv_mp4::{FourCc, Movie, Mp4Demuxer, Mp4Error, PacketCursor, Sample, Track};
