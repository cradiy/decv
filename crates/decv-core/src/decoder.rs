use std::sync::Arc;

use crate::{DecodedVideoFrame, EncodedVideoPacket, VideoFormat};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum VideoCodec {
    H264,
    Vp9,
}

/// Framing of compressed packets supplied to a decoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum BitstreamFormat {
    /// Codec-defined byte stream, such as H.264 Annex-B.
    ByteStream,
    /// NAL-like units prefixed by a one-to-four-byte big-endian length.
    LengthPrefixed { length_size: u8 },
    /// One codec packet containing one complete frame or a codec-defined
    /// collection of frames, such as a VP9 superframe.
    Frame,
}

/// Codec selection, packet framing, and optional out-of-band configuration.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct VideoDecoderConfig {
    pub codec: VideoCodec,
    pub bitstream_format: BitstreamFormat,
    /// Codec-private configuration, such as an H.264 `avcC` box payload.
    pub codec_data: Option<Arc<[u8]>>,
}

impl VideoDecoderConfig {
    #[inline]
    pub const fn new(codec: VideoCodec, bitstream_format: BitstreamFormat) -> Self {
        Self {
            codec,
            bitstream_format,
            codec_data: None,
        }
    }

    #[inline]
    pub fn with_codec_data(mut self, codec_data: impl Into<Arc<[u8]>>) -> Self {
        self.codec_data = Some(codec_data.into());
        self
    }

    pub fn validate(&self) -> crate::Result<()> {
        if let BitstreamFormat::LengthPrefixed { length_size } = self.bitstream_format
            && !(1..=4).contains(&length_size)
        {
            return Err(crate::MediaError::InvalidDecoderConfig(
                "length prefix must contain one to four bytes",
            ));
        }
        Ok(())
    }
}

/// Result of attempting to transfer packet ownership into a decoder.
#[derive(Debug, Clone)]
#[must_use]
#[non_exhaustive]
pub enum DecodeInputStatus {
    Accepted,
    /// The decoder must be drained before this unconsumed packet is retried.
    NeedOutput(EncodedVideoPacket),
}

/// One event produced by the decoder's pull side.
#[derive(Debug, Clone)]
#[must_use]
#[non_exhaustive]
pub enum DecodeOutput {
    Frame(DecodedVideoFrame),
    /// Always emitted before the first frame using this new format.
    FormatChanged(VideoFormat),
    NeedInput,
    EndOfStream,
}

/// A synchronous, runtime-independent compressed-video decoder.
///
/// One input packet may produce zero, one, or multiple output frames. When
/// `send_packet` returns `NeedOutput`, it returns ownership of the packet so
/// the caller can drain output and retry without cloning compressed data.
pub trait VideoDecoder: Send {
    type Error: std::error::Error + Send + Sync + 'static;

    fn configure(&mut self, config: VideoDecoderConfig) -> std::result::Result<(), Self::Error>;

    fn send_packet(
        &mut self,
        packet: EncodedVideoPacket,
    ) -> std::result::Result<DecodeInputStatus, Self::Error>;

    fn receive_frame(&mut self) -> std::result::Result<DecodeOutput, Self::Error>;

    /// Clears the DPB, delayed output, and all state tied to the old timeline.
    fn flush(&mut self);

    /// Marks the input as complete so delayed frames can be received.
    fn drain(&mut self) -> std::result::Result<(), Self::Error>;
}

#[cfg(test)]
mod tests {
    use super::{BitstreamFormat, VideoCodec, VideoDecoderConfig};
    use crate::MediaError;

    #[test]
    fn validates_length_prefix_size() {
        for length_size in 1..=4 {
            let config = VideoDecoderConfig {
                codec: VideoCodec::H264,
                bitstream_format: BitstreamFormat::LengthPrefixed { length_size },
                codec_data: None,
            };
            assert_eq!(config.validate(), Ok(()));
        }

        let invalid = VideoDecoderConfig {
            codec: VideoCodec::H264,
            bitstream_format: BitstreamFormat::LengthPrefixed { length_size: 0 },
            codec_data: None,
        };
        assert_eq!(
            invalid.validate(),
            Err(MediaError::InvalidDecoderConfig(
                "length prefix must contain one to four bytes"
            ))
        );
    }

    #[test]
    fn accepts_complete_frame_packets() {
        assert_eq!(
            VideoDecoderConfig::new(VideoCodec::Vp9, BitstreamFormat::Frame).validate(),
            Ok(())
        );
    }
}
