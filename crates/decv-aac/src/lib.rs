//! Pure-Rust AAC-LC decoding behind the codec-independent decv audio contract.
//!
//! The public surface deliberately exposes no syntax, Huffman, transform, or
//! filter-bank implementation details.

#![forbid(unsafe_code)]

use std::{collections::VecDeque, fmt, sync::Arc};

use decv_core::{
    AudioCodec, AudioDecodeInputStatus, AudioDecodeOutput, AudioDecoder, AudioDecoderConfig,
    AudioFormat, ChannelLayout, DecodedAudioFrame, EncodedAudioPacket,
};
use symphonia_codec_aac::AacDecoder as SymphoniaAacDecoder;
use symphonia_core::{
    audio::{Audio, GenericAudioBufferRef},
    codecs::audio::{
        AudioCodecParameters, AudioDecoder as SymphoniaAudioDecoder, AudioDecoderOptions,
        FinalizeResult, well_known::CODEC_ID_AAC,
    },
    packet::PacketRef,
    units::{Duration, Timestamp},
};

#[derive(Debug)]
#[non_exhaustive]
pub enum AacError {
    InvalidConfiguration(&'static str),
    BackendConfiguration(String),
    InvalidState(&'static str),
    Decode(String),
    IntegerOverflow,
}

impl fmt::Display for AacError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfiguration(message) => {
                write!(formatter, "invalid AAC configuration: {message}")
            }
            Self::BackendConfiguration(message) => {
                write!(formatter, "AAC backend rejected configuration: {message}")
            }
            Self::InvalidState(message) => {
                write!(formatter, "invalid AAC decoder state: {message}")
            }
            Self::Decode(message) => write!(formatter, "AAC decode error: {message}"),
            Self::IntegerOverflow => formatter.write_str("AAC integer overflow"),
        }
    }
}

impl std::error::Error for AacError {}

pub struct AacDecoder {
    decoder: Option<SymphoniaAacDecoder>,
    format: Option<AudioFormat>,
    outputs: VecDeque<AudioDecodeOutput>,
    next_frame_id: u64,
    format_announced: bool,
    draining: bool,
    ended: bool,
}

impl fmt::Debug for AacDecoder {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AacDecoder")
            .field("configured", &self.decoder.is_some())
            .field("format", &self.format)
            .field("queued_outputs", &self.outputs.len())
            .field("next_frame_id", &self.next_frame_id)
            .field("format_announced", &self.format_announced)
            .field("draining", &self.draining)
            .field("ended", &self.ended)
            .finish()
    }
}

impl Default for AacDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl AacDecoder {
    pub const fn new() -> Self {
        Self {
            decoder: None,
            format: None,
            outputs: VecDeque::new(),
            next_frame_id: 0,
            format_announced: false,
            draining: false,
            ended: false,
        }
    }

    fn reset_timeline(&mut self) {
        self.outputs.clear();
        self.next_frame_id = 0;
        self.format_announced = false;
        self.draining = false;
        self.ended = false;
    }

    fn configured_decoder(&mut self) -> Result<&mut SymphoniaAacDecoder, AacError> {
        self.decoder
            .as_mut()
            .ok_or(AacError::InvalidState("decoder is not configured"))
    }

    fn decode_packet(&mut self, packet: EncodedAudioPacket) -> Result<(), AacError> {
        if packet.data.is_empty() {
            return Err(AacError::Decode("empty AAC access unit".into()));
        }
        let codec_packet = PacketRef::new(0, Timestamp::ZERO, Duration::ZERO, packet.data.as_ref());
        let decoded = self
            .configured_decoder()?
            .decode_ref(&codec_packet)
            .map_err(|error| AacError::Decode(error.to_string()))?;
        let (sample_rate, channels, sample_frames, samples) = copy_interleaved_f32(decoded)?;
        let format = self
            .format
            .ok_or(AacError::InvalidState("decoder format is unavailable"))?;
        if sample_rate != format.sample_rate {
            return Err(AacError::Decode(
                "decoded sample rate differs from configuration".into(),
            ));
        }
        if channels != usize::from(format.channel_layout.channels()) {
            return Err(AacError::Decode(
                "decoded channel count differs from configuration".into(),
            ));
        }
        let duration = format
            .duration_for_sample_frames(sample_frames)
            .map_err(|_| AacError::IntegerOverflow)?;
        let frame = DecodedAudioFrame::new(
            self.next_frame_id,
            packet.pts,
            Some(duration),
            format,
            Arc::<[f32]>::from(samples),
        );
        frame
            .validate()
            .map_err(|error| AacError::Decode(error.to_string()))?;
        self.next_frame_id = self
            .next_frame_id
            .checked_add(1)
            .ok_or(AacError::IntegerOverflow)?;
        if !self.format_announced {
            self.outputs
                .push_back(AudioDecodeOutput::FormatChanged(format));
            self.format_announced = true;
        }
        self.outputs.push_back(AudioDecodeOutput::Frame(frame));
        Ok(())
    }
}

impl AudioDecoder for AacDecoder {
    type Error = AacError;

    fn configure(&mut self, config: AudioDecoderConfig) -> Result<(), Self::Error> {
        config
            .validate()
            .map_err(|_| AacError::InvalidConfiguration("invalid core audio configuration"))?;
        if config.codec != AudioCodec::Aac {
            return Err(AacError::InvalidConfiguration("codec is not AAC"));
        }
        if !matches!(
            config.channel_layout,
            ChannelLayout::Mono | ChannelLayout::Stereo
        ) {
            return Err(AacError::InvalidConfiguration(
                "only mono and stereo AAC-LC are supported",
            ));
        }
        let codec_data = config
            .codec_data
            .as_ref()
            .ok_or(AacError::InvalidConfiguration(
                "AAC AudioSpecificConfig is missing",
            ))?;
        let mut parameters = AudioCodecParameters::new();
        parameters
            .for_codec(CODEC_ID_AAC)
            .with_sample_rate(config.sample_rate)
            .with_extra_data(codec_data.as_ref().into());
        let decoder = SymphoniaAacDecoder::try_new(&parameters, &AudioDecoderOptions::default())
            .map_err(|error| AacError::BackendConfiguration(error.to_string()))?;
        self.decoder = Some(decoder);
        self.format = Some(AudioFormat::new(config.sample_rate, config.channel_layout));
        self.reset_timeline();
        Ok(())
    }

    fn send_packet(
        &mut self,
        packet: EncodedAudioPacket,
    ) -> Result<AudioDecodeInputStatus, Self::Error> {
        if self.decoder.is_none() {
            return Err(AacError::InvalidState("decoder is not configured"));
        }
        if self.draining || self.ended {
            return Err(AacError::InvalidState("cannot accept packets after drain"));
        }
        if !self.outputs.is_empty() {
            return Ok(AudioDecodeInputStatus::NeedOutput(packet));
        }
        self.decode_packet(packet)?;
        Ok(AudioDecodeInputStatus::Accepted)
    }

    fn receive_frame(&mut self) -> Result<AudioDecodeOutput, Self::Error> {
        if self.decoder.is_none() {
            return Err(AacError::InvalidState("decoder is not configured"));
        }
        if let Some(output) = self.outputs.pop_front() {
            return Ok(output);
        }
        if self.draining {
            self.ended = true;
            return Ok(AudioDecodeOutput::EndOfStream);
        }
        Ok(AudioDecodeOutput::NeedInput)
    }

    fn flush(&mut self) {
        if let Some(decoder) = self.decoder.as_mut() {
            decoder.reset();
        }
        self.reset_timeline();
    }

    fn drain(&mut self) -> Result<(), Self::Error> {
        let decoder = self.configured_decoder()?;
        let FinalizeResult { verify_ok: _ } = decoder.finalize();
        self.draining = true;
        Ok(())
    }
}

fn copy_interleaved_f32(
    decoded: GenericAudioBufferRef<'_>,
) -> Result<(u32, usize, usize, Vec<f32>), AacError> {
    let GenericAudioBufferRef::F32(buffer) = decoded else {
        return Err(AacError::Decode(
            "AAC backend produced a non-f32 sample format".into(),
        ));
    };
    let sample_rate = buffer.spec().rate();
    let channels = buffer.spec().channels().count();
    let sample_frames = buffer.frames();
    let sample_count = sample_frames
        .checked_mul(channels)
        .ok_or(AacError::IntegerOverflow)?;
    let mut samples = vec![0.0; sample_count];
    buffer.copy_to_slice_interleaved::<f32, _>(&mut samples);
    Ok((sample_rate, channels, sample_frames, samples))
}

#[cfg(test)]
mod tests {
    use super::*;
    use decv_core::{AudioSampleFormat, MediaTime};

    const FIRST_AAC_ACCESS_UNIT: [u8; 23] = [
        0xde, 0x02, 0x00, 0x4c, 0x61, 0x76, 0x63, 0x35, 0x39, 0x2e, 0x33, 0x37, 0x2e, 0x31, 0x30,
        0x30, 0x00, 0x42, 0x20, 0x08, 0xc1, 0x18, 0x38,
    ];
    const SECOND_AAC_ACCESS_UNIT: [u8; 6] = [0x21, 0x20, 0x04, 0x60, 0x8c, 0x1c];

    #[test]
    fn returns_unconsumed_packet_while_output_is_pending() {
        let mut decoder = AacDecoder::new();
        let config = AudioDecoderConfig::new(AudioCodec::Aac, 44_100, ChannelLayout::Stereo)
            .with_codec_data([0x12, 0x10, 0x56, 0xe5, 0x00]);
        decoder.configure(config).unwrap();
        let mut first = EncodedAudioPacket::new(FIRST_AAC_ACCESS_UNIT);
        first.pts = MediaTime::from_parts(-1_024, 44_100);
        first.duration = MediaTime::from_parts(1_024, 44_100);
        assert!(matches!(
            decoder.send_packet(first).unwrap(),
            AudioDecodeInputStatus::Accepted
        ));

        let mut second = EncodedAudioPacket::new(SECOND_AAC_ACCESS_UNIT);
        second.pts = MediaTime::from_parts(0, 44_100);
        second.duration = MediaTime::from_parts(1_024, 44_100);
        let AudioDecodeInputStatus::NeedOutput(second) = decoder.send_packet(second).unwrap()
        else {
            panic!("pending format and frame must apply backpressure");
        };
        assert_eq!(second.data.as_ref(), SECOND_AAC_ACCESS_UNIT);
        assert!(matches!(
            decoder.receive_frame().unwrap(),
            AudioDecodeOutput::FormatChanged(AudioFormat {
                sample_rate: 44_100,
                channel_layout: ChannelLayout::Stereo,
                ..
            })
        ));
        let AudioDecodeInputStatus::NeedOutput(second) = decoder.send_packet(second).unwrap()
        else {
            panic!("pending PCM frame must apply backpressure");
        };
        let AudioDecodeOutput::Frame(first) = decoder.receive_frame().unwrap() else {
            panic!("expected first decoded AAC frame");
        };
        assert_eq!(first.id, 0);
        assert_eq!(first.pts, MediaTime::from_parts(-1_024, 44_100));
        assert_eq!(first.duration, MediaTime::from_parts(1_024, 44_100));
        assert_eq!(first.sample_frames(), 1_024);
        assert_eq!(first.samples.len(), 2_048);
        assert!(first.samples.iter().all(|sample| sample.is_finite()));

        assert!(matches!(
            decoder.send_packet(second).unwrap(),
            AudioDecodeInputStatus::Accepted
        ));
        let AudioDecodeOutput::Frame(second) = decoder.receive_frame().unwrap() else {
            panic!("expected second decoded AAC frame");
        };
        assert_eq!(second.id, 1);
        assert_eq!(second.pts, MediaTime::from_parts(0, 44_100));
        assert_eq!(second.sample_frames(), 1_024);
        assert!(matches!(
            decoder.receive_frame().unwrap(),
            AudioDecodeOutput::NeedInput
        ));
    }

    #[test]
    fn rejects_a_truncated_access_unit_without_panicking() {
        let mut decoder = AacDecoder::new();
        let config = AudioDecoderConfig::new(AudioCodec::Aac, 44_100, ChannelLayout::Stereo)
            .with_codec_data([0x12, 0x10]);
        decoder.configure(config).unwrap();
        assert!(
            decoder
                .send_packet(EncodedAudioPacket::new([0x00]))
                .is_err()
        );
    }

    #[test]
    fn drain_and_flush_obey_the_audio_contract() {
        let mut decoder = AacDecoder::new();
        let config = AudioDecoderConfig::new(AudioCodec::Aac, 48_000, ChannelLayout::Mono)
            .with_codec_data([0x11, 0x88]);
        decoder.configure(config).unwrap();
        decoder.drain().unwrap();
        assert!(matches!(
            decoder.receive_frame().unwrap(),
            AudioDecodeOutput::EndOfStream
        ));
        decoder.flush();
        assert!(matches!(
            decoder.receive_frame().unwrap(),
            AudioDecodeOutput::NeedInput
        ));
    }

    #[test]
    fn public_audio_types_keep_expected_shapes() {
        let format = AudioFormat::new(44_100, ChannelLayout::Stereo);
        assert_eq!(format.sample_format, AudioSampleFormat::F32Interleaved);
        assert_eq!(
            format.duration_for_sample_frames(1_024).unwrap(),
            MediaTime::from_parts(1_024, 44_100).unwrap()
        );
    }
}
