use std::fmt;

use decv_core::{
    DecodeInputStatus, DecodeOutput, EncodedVideoPacket, VideoCodec, VideoDecoder,
    VideoDecoderConfig,
};
use decv_h264::{H264Decoder, H264Error};
use decv_vp9::{Vp9Decoder, Vp9Error};

#[derive(Debug)]
#[non_exhaustive]
pub enum SoftwareVideoError {
    H264(H264Error),
    Vp9(Vp9Error),
    UnsupportedCodec,
    NotConfigured,
}

impl fmt::Display for SoftwareVideoError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::H264(error) => error.fmt(formatter),
            Self::Vp9(error) => error.fmt(formatter),
            Self::UnsupportedCodec => formatter.write_str("unsupported software video codec"),
            Self::NotConfigured => formatter.write_str("software video decoder is not configured"),
        }
    }
}

impl std::error::Error for SoftwareVideoError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::H264(error) => Some(error),
            Self::Vp9(error) => Some(error),
            Self::UnsupportedCodec | Self::NotConfigured => None,
        }
    }
}

#[derive(Debug)]
enum Backend {
    H264(Box<H264Decoder>),
    Vp9(Box<Vp9Decoder>),
}

/// Codec-selecting software video decoder for the public `decv` facade.
///
/// Calling [`VideoDecoder::configure`] selects the native H.264 or VP9
/// implementation from the codec carried by [`VideoDecoderConfig`].
#[derive(Debug, Default)]
pub struct SoftwareVideoDecoder {
    backend: Option<Backend>,
}

impl SoftwareVideoDecoder {
    #[inline]
    pub const fn new() -> Self {
        Self { backend: None }
    }
}

impl VideoDecoder for SoftwareVideoDecoder {
    type Error = SoftwareVideoError;

    fn configure(&mut self, config: VideoDecoderConfig) -> Result<(), SoftwareVideoError> {
        self.backend = Some(match config.codec {
            VideoCodec::H264 => {
                let mut decoder = H264Decoder::new();
                decoder
                    .configure(config)
                    .map_err(SoftwareVideoError::H264)?;
                Backend::H264(Box::new(decoder))
            }
            VideoCodec::Vp9 => {
                let mut decoder = Vp9Decoder::new();
                decoder.configure(config).map_err(SoftwareVideoError::Vp9)?;
                Backend::Vp9(Box::new(decoder))
            }
            _ => return Err(SoftwareVideoError::UnsupportedCodec),
        });
        Ok(())
    }

    fn send_packet(
        &mut self,
        packet: EncodedVideoPacket,
    ) -> Result<DecodeInputStatus, SoftwareVideoError> {
        match self.backend.as_mut() {
            Some(Backend::H264(decoder)) => decoder
                .send_packet(packet)
                .map_err(SoftwareVideoError::H264),
            Some(Backend::Vp9(decoder)) => {
                decoder.send_packet(packet).map_err(SoftwareVideoError::Vp9)
            }
            None => Err(SoftwareVideoError::NotConfigured),
        }
    }

    fn receive_frame(&mut self) -> Result<DecodeOutput, SoftwareVideoError> {
        match self.backend.as_mut() {
            Some(Backend::H264(decoder)) => {
                decoder.receive_frame().map_err(SoftwareVideoError::H264)
            }
            Some(Backend::Vp9(decoder)) => decoder.receive_frame().map_err(SoftwareVideoError::Vp9),
            None => Err(SoftwareVideoError::NotConfigured),
        }
    }

    fn flush(&mut self) {
        match self.backend.as_mut() {
            Some(Backend::H264(decoder)) => decoder.flush(),
            Some(Backend::Vp9(decoder)) => decoder.flush(),
            None => {}
        }
    }

    fn drain(&mut self) -> Result<(), SoftwareVideoError> {
        match self.backend.as_mut() {
            Some(Backend::H264(decoder)) => decoder.drain().map_err(SoftwareVideoError::H264),
            Some(Backend::Vp9(decoder)) => decoder.drain().map_err(SoftwareVideoError::Vp9),
            None => Err(SoftwareVideoError::NotConfigured),
        }
    }
}

#[cfg(test)]
mod tests {
    use decv_core::{BitstreamFormat, DecodeOutput, VideoCodec, VideoDecoder, VideoDecoderConfig};

    use super::SoftwareVideoDecoder;

    #[test]
    fn selects_vp9_from_the_public_config() {
        let mut decoder = SoftwareVideoDecoder::new();
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
    }
}
