//! Stable-candidate consumer facade for the decv workspace.
//!
//! Applications should prefer this crate over depending on codec syntax and
//! reconstruction internals directly. The facade contains the synchronous
//! decoder contract, immutable frame types, H.264 decoder configuration, and
//! ordinary MP4 packet access needed by a playback or transcoding pipeline.

#![forbid(unsafe_code)]

mod seek;

pub use decv_audio::{AacDecoder, AacError, AudioError, SoftwareAudioDecoder};
pub use decv_core::{
    AdpcmCodec, AlignedBytes, AudioCodec, AudioDecodeInputStatus, AudioDecodeOutput, AudioDecoder,
    AudioDecoderConfig, AudioFormat, AudioSampleFormat, BitstreamFormat, CPU_BUFFER_ALIGNMENT,
    ChannelLayout, ColorInfo, ColorMatrix, ColorPrimaries, ColorRange, CpuBuffer, CpuFrame,
    CpuPlane, DecodeInputStatus, DecodeOutput, DecodedAudioFrame, DecodedVideoFrame,
    EncodedAudioPacket, EncodedVideoPacket, FrameStorage, MediaError, MediaInput, MediaTime,
    PcmCodec, PixelFormat, Rect, Size, TransferFunction, VideoCodec, VideoDecoder,
    VideoDecoderConfig, VideoFormat,
};
pub use decv_h264::{
    H264Decoder, H264Error, H264Parallelism, H264SeekCheckpoint, H264SeekCheckpointCache,
    H264SeekCheckpointEntry,
};
pub use decv_mp4::{
    AacSampleEntry, AudioPacketCursor, FourCc, Movie, Mp4Demuxer, Mp4Error, PacketCursor, Sample,
    SampleDescription, Track, TrackKind,
};
#[cfg(feature = "network")]
pub use decv_network::{
    CachedRangeInput, HttpRangeInput, HttpRangeInputBuilder, RangeCacheConfig, RangeCacheStats,
    RangeFetcher, RangeInputStats,
};
pub use seek::{
    H264Mp4InteractiveSeekOutcome, H264Mp4SeekController, H264Mp4SeekError, H264Mp4SeekOutcome,
    H264Mp4SeekPlan, H264Mp4SeekSource,
};
