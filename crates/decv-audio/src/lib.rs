//! Pure-Rust multi-codec audio decoding behind the codec-independent decv contract.
//!
//! Symphonia remains an implementation detail. All decoded audio is copied to
//! owned interleaved `f32` storage before it crosses the crate boundary.

#![forbid(unsafe_code)]

use std::{collections::VecDeque, fmt, sync::Arc};

use decv_core::{
    AdpcmCodec, AudioCodec, AudioDecodeInputStatus, AudioDecodeOutput, AudioDecoder,
    AudioDecoderConfig, AudioFormat, ChannelLayout, DecodedAudioFrame, EncodedAudioPacket,
    PcmCodec,
};
use symphonia::{
    core::{
        audio::{Channels, GenericAudioBufferRef, layouts},
        codecs::audio::{
            AudioCodecId, AudioCodecParameters, AudioDecoder as BackendAudioDecoder,
            AudioDecoderOptions, FinalizeResult,
            well_known::{
                CODEC_ID_AAC, CODEC_ID_ADPCM_IMA_QT, CODEC_ID_ADPCM_IMA_WAV, CODEC_ID_ADPCM_MS,
                CODEC_ID_ALAC, CODEC_ID_FLAC, CODEC_ID_MP1, CODEC_ID_MP2, CODEC_ID_MP3,
                CODEC_ID_PCM_ALAW, CODEC_ID_PCM_F32BE, CODEC_ID_PCM_F32LE, CODEC_ID_PCM_F64BE,
                CODEC_ID_PCM_F64LE, CODEC_ID_PCM_MULAW, CODEC_ID_PCM_S8, CODEC_ID_PCM_S16BE,
                CODEC_ID_PCM_S16LE, CODEC_ID_PCM_S24BE, CODEC_ID_PCM_S24LE, CODEC_ID_PCM_S32BE,
                CODEC_ID_PCM_S32LE, CODEC_ID_PCM_U8, CODEC_ID_PCM_U16BE, CODEC_ID_PCM_U16LE,
                CODEC_ID_PCM_U24BE, CODEC_ID_PCM_U24LE, CODEC_ID_PCM_U32BE, CODEC_ID_PCM_U32LE,
                CODEC_ID_VORBIS,
            },
        },
        packet::PacketRef,
        units::{Duration, Timestamp},
    },
    default::get_codecs,
};

#[derive(Debug)]
#[non_exhaustive]
pub enum AudioError {
    InvalidConfiguration(&'static str),
    BackendConfiguration(String),
    InvalidState(&'static str),
    Decode(String),
    IntegerOverflow,
}

impl fmt::Display for AudioError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfiguration(message) => {
                write!(formatter, "invalid audio configuration: {message}")
            }
            Self::BackendConfiguration(message) => {
                write!(formatter, "audio backend rejected configuration: {message}")
            }
            Self::InvalidState(message) => {
                write!(formatter, "invalid audio decoder state: {message}")
            }
            Self::Decode(message) => write!(formatter, "audio decode error: {message}"),
            Self::IntegerOverflow => formatter.write_str("audio integer overflow"),
        }
    }
}

impl std::error::Error for AudioError {}

pub struct SoftwareAudioDecoder {
    decoder: Option<Box<dyn BackendAudioDecoder>>,
    codec: Option<AudioCodec>,
    configured_format: Option<AudioFormat>,
    active_format: Option<AudioFormat>,
    default_packet_frames: Option<u64>,
    outputs: VecDeque<AudioDecodeOutput>,
    next_frame_id: u64,
    draining: bool,
    ended: bool,
}

impl fmt::Debug for SoftwareAudioDecoder {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SoftwareAudioDecoder")
            .field("configured", &self.decoder.is_some())
            .field("codec", &self.codec)
            .field("configured_format", &self.configured_format)
            .field("active_format", &self.active_format)
            .field("queued_outputs", &self.outputs.len())
            .field("next_frame_id", &self.next_frame_id)
            .field("draining", &self.draining)
            .field("ended", &self.ended)
            .finish()
    }
}

impl Default for SoftwareAudioDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl SoftwareAudioDecoder {
    pub const fn new() -> Self {
        Self {
            decoder: None,
            codec: None,
            configured_format: None,
            active_format: None,
            default_packet_frames: None,
            outputs: VecDeque::new(),
            next_frame_id: 0,
            draining: false,
            ended: false,
        }
    }

    fn reset_timeline(&mut self) {
        self.outputs.clear();
        self.next_frame_id = 0;
        self.active_format = None;
        self.draining = false;
        self.ended = false;
    }

    fn configured_decoder(&mut self) -> Result<&mut Box<dyn BackendAudioDecoder>, AudioError> {
        self.decoder
            .as_mut()
            .ok_or(AudioError::InvalidState("decoder is not configured"))
    }

    fn decode_packet(&mut self, packet: EncodedAudioPacket) -> Result<(), AudioError> {
        if packet.data.is_empty() {
            return Err(AudioError::Decode("empty audio packet".into()));
        }
        let packet_frames = packet
            .duration
            .map(|duration| {
                let configured = self
                    .configured_format
                    .ok_or(AudioError::InvalidState("decoder format is unavailable"))?;
                media_duration_to_frames(duration, configured.sample_rate)
            })
            .transpose()?
            .or(self.default_packet_frames)
            .unwrap_or(0);
        let codec_packet = PacketRef::new(
            0,
            Timestamp::ZERO,
            Duration::new(packet_frames),
            packet.data.as_ref(),
        );
        let decoded = self
            .configured_decoder()?
            .decode_ref(&codec_packet)
            .map_err(|error| AudioError::Decode(error.to_string()))?;
        let (sample_rate, channels, sample_frames, samples) = copy_interleaved_f32(decoded)?;
        if sample_frames == 0 {
            return Ok(());
        }
        let channel_layout = match channels {
            1 => ChannelLayout::Mono,
            2 => ChannelLayout::Stereo,
            value => ChannelLayout::Discrete(
                u16::try_from(value).map_err(|_| AudioError::IntegerOverflow)?,
            ),
        };
        let format = AudioFormat::new(sample_rate, channel_layout);
        let duration = format
            .duration_for_sample_frames(sample_frames)
            .map_err(|_| AudioError::IntegerOverflow)?;
        let frame = DecodedAudioFrame::new(
            self.next_frame_id,
            packet.pts,
            Some(duration),
            format,
            Arc::<[f32]>::from(samples),
        );
        frame
            .validate()
            .map_err(|error| AudioError::Decode(error.to_string()))?;
        self.next_frame_id = self
            .next_frame_id
            .checked_add(1)
            .ok_or(AudioError::IntegerOverflow)?;
        if self.active_format != Some(format) {
            self.outputs
                .push_back(AudioDecodeOutput::FormatChanged(format));
            self.active_format = Some(format);
        }
        self.outputs.push_back(AudioDecodeOutput::Frame(frame));
        Ok(())
    }
}

impl AudioDecoder for SoftwareAudioDecoder {
    type Error = AudioError;

    fn configure(&mut self, config: AudioDecoderConfig) -> Result<(), Self::Error> {
        config
            .validate()
            .map_err(|_| AudioError::InvalidConfiguration("invalid core audio configuration"))?;
        let mut parameters = AudioCodecParameters::new();
        parameters
            .for_codec(audio_codec_id(config.codec)?)
            .with_sample_rate(config.sample_rate)
            .with_channels(channels(config.channel_layout));
        if let Some(bits) = config.bits_per_sample {
            parameters.with_bits_per_sample(bits);
        }
        if let Some(bits) = config.bits_per_coded_sample {
            parameters.with_bits_per_coded_sample(bits);
        }
        if let Some(frames) = config.max_frames_per_packet {
            parameters.with_max_frames_per_packet(frames);
        }
        if let Some(frames) = config.frames_per_block {
            parameters.with_frames_per_block(frames);
        }
        if let Some(codec_data) = &config.codec_data {
            parameters.with_extra_data(codec_data.as_ref().into());
        }
        let options = AudioDecoderOptions::default().gapless(false);
        let decoder = get_codecs()
            .make_audio_decoder(&parameters, &options)
            .map_err(|error| AudioError::BackendConfiguration(error.to_string()))?;
        self.decoder = Some(decoder);
        self.codec = Some(config.codec);
        self.configured_format = Some(AudioFormat::new(config.sample_rate, config.channel_layout));
        self.default_packet_frames = config.max_frames_per_packet;
        self.reset_timeline();
        Ok(())
    }

    fn send_packet(
        &mut self,
        packet: EncodedAudioPacket,
    ) -> Result<AudioDecodeInputStatus, Self::Error> {
        if self.decoder.is_none() {
            return Err(AudioError::InvalidState("decoder is not configured"));
        }
        if self.draining || self.ended {
            return Err(AudioError::InvalidState(
                "cannot accept packets after drain",
            ));
        }
        if !self.outputs.is_empty() {
            return Ok(AudioDecodeInputStatus::NeedOutput(packet));
        }
        self.decode_packet(packet)?;
        Ok(AudioDecodeInputStatus::Accepted)
    }

    fn receive_frame(&mut self) -> Result<AudioDecodeOutput, Self::Error> {
        if self.decoder.is_none() {
            return Err(AudioError::InvalidState("decoder is not configured"));
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
) -> Result<(u32, usize, usize, Vec<f32>), AudioError> {
    let sample_rate = decoded.spec().rate();
    let channels = decoded.spec().channels().count();
    let sample_frames = decoded.frames();
    let sample_count = sample_frames
        .checked_mul(channels)
        .ok_or(AudioError::IntegerOverflow)?;
    let mut samples = vec![0.0; sample_count];
    decoded.copy_to_slice_interleaved::<f32, _>(&mut samples);
    Ok((sample_rate, channels, sample_frames, samples))
}

fn media_duration_to_frames(
    duration: decv_core::MediaTime,
    sample_rate: u32,
) -> Result<u64, AudioError> {
    if duration.value < 0 {
        return Err(AudioError::Decode(
            "audio packet duration is negative".into(),
        ));
    }
    let frames = i128::from(duration.value)
        .checked_mul(i128::from(sample_rate))
        .ok_or(AudioError::IntegerOverflow)?
        / i128::from(duration.timescale.get());
    u64::try_from(frames).map_err(|_| AudioError::IntegerOverflow)
}

fn channels(layout: ChannelLayout) -> Channels {
    match layout {
        ChannelLayout::Mono => layouts::CHANNEL_LAYOUT_MONO.clone(),
        ChannelLayout::Stereo => layouts::CHANNEL_LAYOUT_STEREO.clone(),
        ChannelLayout::Discrete(channels) => Channels::Discrete(channels),
        _ => Channels::Discrete(layout.channels()),
    }
}

fn audio_codec_id(codec: AudioCodec) -> Result<AudioCodecId, AudioError> {
    let id = match codec {
        AudioCodec::Aac => CODEC_ID_AAC,
        AudioCodec::Adpcm(AdpcmCodec::Microsoft) => CODEC_ID_ADPCM_MS,
        AudioCodec::Adpcm(AdpcmCodec::ImaWav) => CODEC_ID_ADPCM_IMA_WAV,
        AudioCodec::Adpcm(AdpcmCodec::ImaQuickTime) => CODEC_ID_ADPCM_IMA_QT,
        AudioCodec::Alac => CODEC_ID_ALAC,
        AudioCodec::Flac => CODEC_ID_FLAC,
        AudioCodec::Mp1 => CODEC_ID_MP1,
        AudioCodec::Mp2 => CODEC_ID_MP2,
        AudioCodec::Mp3 => CODEC_ID_MP3,
        AudioCodec::Pcm(PcmCodec::Signed8) => CODEC_ID_PCM_S8,
        AudioCodec::Pcm(PcmCodec::Signed16Le) => CODEC_ID_PCM_S16LE,
        AudioCodec::Pcm(PcmCodec::Signed16Be) => CODEC_ID_PCM_S16BE,
        AudioCodec::Pcm(PcmCodec::Signed24Le) => CODEC_ID_PCM_S24LE,
        AudioCodec::Pcm(PcmCodec::Signed24Be) => CODEC_ID_PCM_S24BE,
        AudioCodec::Pcm(PcmCodec::Signed32Le) => CODEC_ID_PCM_S32LE,
        AudioCodec::Pcm(PcmCodec::Signed32Be) => CODEC_ID_PCM_S32BE,
        AudioCodec::Pcm(PcmCodec::Unsigned8) => CODEC_ID_PCM_U8,
        AudioCodec::Pcm(PcmCodec::Unsigned16Le) => CODEC_ID_PCM_U16LE,
        AudioCodec::Pcm(PcmCodec::Unsigned16Be) => CODEC_ID_PCM_U16BE,
        AudioCodec::Pcm(PcmCodec::Unsigned24Le) => CODEC_ID_PCM_U24LE,
        AudioCodec::Pcm(PcmCodec::Unsigned24Be) => CODEC_ID_PCM_U24BE,
        AudioCodec::Pcm(PcmCodec::Unsigned32Le) => CODEC_ID_PCM_U32LE,
        AudioCodec::Pcm(PcmCodec::Unsigned32Be) => CODEC_ID_PCM_U32BE,
        AudioCodec::Pcm(PcmCodec::Float32Le) => CODEC_ID_PCM_F32LE,
        AudioCodec::Pcm(PcmCodec::Float32Be) => CODEC_ID_PCM_F32BE,
        AudioCodec::Pcm(PcmCodec::Float64Le) => CODEC_ID_PCM_F64LE,
        AudioCodec::Pcm(PcmCodec::Float64Be) => CODEC_ID_PCM_F64BE,
        AudioCodec::Pcm(PcmCodec::ALaw) => CODEC_ID_PCM_ALAW,
        AudioCodec::Pcm(PcmCodec::MuLaw) => CODEC_ID_PCM_MULAW,
        AudioCodec::Vorbis => CODEC_ID_VORBIS,
        _ => {
            return Err(AudioError::InvalidConfiguration(
                "audio codec is not supported",
            ));
        }
    };
    Ok(id)
}

/// Compatibility alias for callers that previously constructed the AAC-only decoder.
pub type AacDecoder = SoftwareAudioDecoder;
/// Compatibility alias for the former AAC-specific error name.
pub type AacError = AudioError;

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
    fn decodes_integer_pcm_to_interleaved_f32() {
        let mut decoder = SoftwareAudioDecoder::new();
        let config = AudioDecoderConfig::new(
            AudioCodec::Pcm(PcmCodec::Signed16Le),
            48_000,
            ChannelLayout::Stereo,
        )
        .with_bits_per_sample(16);
        decoder.configure(config).unwrap();

        let mut packet = EncodedAudioPacket::new([0x00, 0x80, 0x00, 0x00, 0xff, 0x7f, 0x00, 0x40]);
        packet.duration = MediaTime::from_parts(2, 48_000);
        assert!(matches!(
            decoder.send_packet(packet).unwrap(),
            AudioDecodeInputStatus::Accepted
        ));
        assert!(matches!(
            decoder.receive_frame().unwrap(),
            AudioDecodeOutput::FormatChanged(AudioFormat {
                sample_rate: 48_000,
                channel_layout: ChannelLayout::Stereo,
                ..
            })
        ));
        let AudioDecodeOutput::Frame(frame) = decoder.receive_frame().unwrap() else {
            panic!("expected decoded PCM frame");
        };
        assert_eq!(frame.sample_frames(), 2);
        assert_eq!(frame.samples.len(), 4);
        assert_eq!(frame.samples[0], -1.0);
        assert_eq!(frame.samples[1], 0.0);
        assert!((frame.samples[2] - 0.999_969_5).abs() < 1e-6);
        assert_eq!(frame.samples[3], 0.5);
    }

    #[test]
    fn registers_every_stable_symphonia_audio_codec() {
        let codecs = [
            AudioCodec::Aac,
            AudioCodec::Adpcm(AdpcmCodec::Microsoft),
            AudioCodec::Adpcm(AdpcmCodec::ImaWav),
            AudioCodec::Adpcm(AdpcmCodec::ImaQuickTime),
            AudioCodec::Alac,
            AudioCodec::Flac,
            AudioCodec::Mp1,
            AudioCodec::Mp2,
            AudioCodec::Mp3,
            AudioCodec::Pcm(PcmCodec::Signed8),
            AudioCodec::Pcm(PcmCodec::Signed16Le),
            AudioCodec::Pcm(PcmCodec::Signed16Be),
            AudioCodec::Pcm(PcmCodec::Signed24Le),
            AudioCodec::Pcm(PcmCodec::Signed24Be),
            AudioCodec::Pcm(PcmCodec::Signed32Le),
            AudioCodec::Pcm(PcmCodec::Signed32Be),
            AudioCodec::Pcm(PcmCodec::Unsigned8),
            AudioCodec::Pcm(PcmCodec::Unsigned16Le),
            AudioCodec::Pcm(PcmCodec::Unsigned16Be),
            AudioCodec::Pcm(PcmCodec::Unsigned24Le),
            AudioCodec::Pcm(PcmCodec::Unsigned24Be),
            AudioCodec::Pcm(PcmCodec::Unsigned32Le),
            AudioCodec::Pcm(PcmCodec::Unsigned32Be),
            AudioCodec::Pcm(PcmCodec::Float32Le),
            AudioCodec::Pcm(PcmCodec::Float32Be),
            AudioCodec::Pcm(PcmCodec::Float64Le),
            AudioCodec::Pcm(PcmCodec::Float64Be),
            AudioCodec::Pcm(PcmCodec::ALaw),
            AudioCodec::Pcm(PcmCodec::MuLaw),
            AudioCodec::Vorbis,
        ];
        for codec in codecs {
            let codec_id = audio_codec_id(codec).unwrap();
            assert!(
                get_codecs().get_audio_decoder(codec_id).is_some(),
                "codec {codec:?} is not registered"
            );
        }
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
