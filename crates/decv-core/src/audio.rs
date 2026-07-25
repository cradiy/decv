use std::sync::Arc;

use crate::{MediaError, MediaTime, Result};

/// A compressed audio format understood by an audio decoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum AudioCodec {
    Aac,
    Adpcm(AdpcmCodec),
    Alac,
    Flac,
    Mp1,
    Mp2,
    Mp3,
    Pcm(PcmCodec),
    Vorbis,
}

/// An ADPCM bitstream layout supported by the software audio decoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum AdpcmCodec {
    Microsoft,
    ImaWav,
    ImaQuickTime,
}

/// An interleaved PCM encoding supported by the software audio decoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum PcmCodec {
    Signed8,
    Signed16Le,
    Signed16Be,
    Signed24Le,
    Signed24Be,
    Signed32Le,
    Signed32Be,
    Unsigned8,
    Unsigned16Le,
    Unsigned16Be,
    Unsigned24Le,
    Unsigned24Be,
    Unsigned32Le,
    Unsigned32Be,
    Float32Le,
    Float32Be,
    Float64Le,
    Float64Be,
    ALaw,
    MuLaw,
}

/// The channel order carried by interleaved PCM samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ChannelLayout {
    Mono,
    Stereo,
    /// An unspecified channel order with a known channel count.
    Discrete(u16),
}

impl ChannelLayout {
    #[inline]
    pub const fn channels(self) -> u16 {
        match self {
            Self::Mono => 1,
            Self::Stereo => 2,
            Self::Discrete(channels) => channels,
        }
    }
}

/// Codec selection and out-of-band configuration for an audio decoder.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct AudioDecoderConfig {
    pub codec: AudioCodec,
    pub sample_rate: u32,
    pub channel_layout: ChannelLayout,
    /// Codec-private initialization bytes, such as AAC AudioSpecificConfig,
    /// ALAC magic cookie, FLAC STREAMINFO, or Vorbis identification/setup data.
    pub codec_data: Option<Arc<[u8]>>,
    pub bits_per_sample: Option<u32>,
    pub bits_per_coded_sample: Option<u32>,
    pub max_frames_per_packet: Option<u64>,
    pub frames_per_block: Option<u64>,
}

impl AudioDecoderConfig {
    #[inline]
    pub const fn new(codec: AudioCodec, sample_rate: u32, channel_layout: ChannelLayout) -> Self {
        Self {
            codec,
            sample_rate,
            channel_layout,
            codec_data: None,
            bits_per_sample: None,
            bits_per_coded_sample: None,
            max_frames_per_packet: None,
            frames_per_block: None,
        }
    }

    #[inline]
    pub fn with_codec_data(mut self, codec_data: impl Into<Arc<[u8]>>) -> Self {
        self.codec_data = Some(codec_data.into());
        self
    }

    #[inline]
    pub const fn with_bits_per_sample(mut self, bits: u32) -> Self {
        self.bits_per_sample = Some(bits);
        self
    }

    #[inline]
    pub const fn with_bits_per_coded_sample(mut self, bits: u32) -> Self {
        self.bits_per_coded_sample = Some(bits);
        self
    }

    #[inline]
    pub const fn with_max_frames_per_packet(mut self, frames: u64) -> Self {
        self.max_frames_per_packet = Some(frames);
        self
    }

    #[inline]
    pub const fn with_frames_per_block(mut self, frames: u64) -> Self {
        self.frames_per_block = Some(frames);
        self
    }

    pub fn validate(&self) -> Result<()> {
        if self.sample_rate == 0 {
            return Err(MediaError::InvalidDecoderConfig(
                "audio sample rate must be non-zero",
            ));
        }
        if self.channel_layout.channels() == 0 {
            return Err(MediaError::InvalidDecoderConfig(
                "audio channel count must be non-zero",
            ));
        }
        if self.codec_data.as_ref().is_some_and(|data| data.is_empty()) {
            return Err(MediaError::InvalidDecoderConfig(
                "audio codec data must not be empty",
            ));
        }
        if matches!(
            self.codec,
            AudioCodec::Aac | AudioCodec::Alac | AudioCodec::Flac | AudioCodec::Vorbis
        ) && self.codec_data.is_none()
        {
            return Err(MediaError::InvalidDecoderConfig(
                "audio codec data is required for this codec",
            ));
        }
        if self.bits_per_sample == Some(0) || self.bits_per_coded_sample == Some(0) {
            return Err(MediaError::InvalidDecoderConfig(
                "audio sample bit widths must be non-zero",
            ));
        }
        if let (Some(decoded), Some(coded)) = (self.bits_per_sample, self.bits_per_coded_sample)
            && coded > decoded
        {
            return Err(MediaError::InvalidDecoderConfig(
                "coded sample width exceeds decoded sample width",
            ));
        }
        if self.max_frames_per_packet == Some(0) || self.frames_per_block == Some(0) {
            return Err(MediaError::InvalidDecoderConfig(
                "audio packet frame counts must be non-zero",
            ));
        }
        if matches!(self.codec, AudioCodec::Adpcm(_))
            && (self.max_frames_per_packet.is_none() || self.frames_per_block.is_none())
        {
            return Err(MediaError::InvalidDecoderConfig(
                "ADPCM requires packet and block frame counts",
            ));
        }
        Ok(())
    }
}

/// One owned compressed audio access unit.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct EncodedAudioPacket {
    pub data: Arc<[u8]>,
    pub pts: Option<MediaTime>,
    pub dts: Option<MediaTime>,
    pub duration: Option<MediaTime>,
}

impl EncodedAudioPacket {
    pub fn new(data: impl Into<Arc<[u8]>>) -> Self {
        Self {
            data: data.into(),
            pts: None,
            dts: None,
            duration: None,
        }
    }
}

/// PCM storage layout produced by an audio decoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum AudioSampleFormat {
    F32Interleaved,
}

/// The sample rate, channel order, and storage layout of decoded PCM.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct AudioFormat {
    pub sample_rate: u32,
    pub channel_layout: ChannelLayout,
    pub sample_format: AudioSampleFormat,
}

impl AudioFormat {
    #[inline]
    pub const fn new(sample_rate: u32, channel_layout: ChannelLayout) -> Self {
        Self {
            sample_rate,
            channel_layout,
            sample_format: AudioSampleFormat::F32Interleaved,
        }
    }

    pub fn validate(self) -> Result<()> {
        if self.sample_rate == 0 {
            return Err(MediaError::InvalidAudioFormat(
                "sample rate must be non-zero",
            ));
        }
        if self.channel_layout.channels() == 0 {
            return Err(MediaError::InvalidAudioFormat(
                "channel count must be non-zero",
            ));
        }
        Ok(())
    }

    /// Converts a number of per-channel PCM sample frames into exact media time.
    pub fn duration_for_sample_frames(self, sample_frames: usize) -> Result<MediaTime> {
        self.validate()?;
        let value = i64::try_from(sample_frames).map_err(|_| MediaError::IntegerOverflow)?;
        MediaTime::from_parts(value, self.sample_rate).ok_or(MediaError::IntegerOverflow)
    }
}

/// One immutable block of interleaved decoded PCM.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct DecodedAudioFrame {
    pub id: u64,
    pub pts: Option<MediaTime>,
    pub duration: Option<MediaTime>,
    pub format: AudioFormat,
    pub samples: Arc<[f32]>,
}

impl DecodedAudioFrame {
    #[inline]
    pub fn new(
        id: u64,
        pts: Option<MediaTime>,
        duration: Option<MediaTime>,
        format: AudioFormat,
        samples: impl Into<Arc<[f32]>>,
    ) -> Self {
        Self {
            id,
            pts,
            duration,
            format,
            samples: samples.into(),
        }
    }

    #[inline]
    pub const fn channels(&self) -> u16 {
        self.format.channel_layout.channels()
    }

    #[inline]
    pub fn sample_frames(&self) -> usize {
        let channels = usize::from(self.channels());
        self.samples.len().checked_div(channels).unwrap_or(0)
    }

    pub fn validate(&self) -> Result<()> {
        self.format.validate()?;
        let channels = usize::from(self.channels());
        if !self.samples.len().is_multiple_of(channels) {
            return Err(MediaError::InvalidAudioFrame(
                "PCM sample count is not divisible by the channel count",
            ));
        }
        if self.duration.is_some_and(|duration| duration.value < 0) {
            return Err(MediaError::InvalidAudioFrame(
                "audio frame duration must not be negative",
            ));
        }
        if let Some(duration) = self.duration {
            let actual_scaled = i128::from(duration.value) * i128::from(self.format.sample_rate);
            let expected_scaled = i128::try_from(self.sample_frames())
                .map_err(|_| MediaError::IntegerOverflow)?
                * i128::from(duration.timescale.get());
            let error = actual_scaled
                .checked_sub(expected_scaled)
                .ok_or(MediaError::IntegerOverflow)?
                .abs();
            if error > i128::from(duration.timescale.get()) {
                return Err(MediaError::InvalidAudioFrame(
                    "audio frame duration differs from its PCM sample count",
                ));
            }
        }
        Ok(())
    }
}

/// Result of attempting to transfer packet ownership into an audio decoder.
#[derive(Debug, Clone)]
#[must_use]
#[non_exhaustive]
pub enum AudioDecodeInputStatus {
    Accepted,
    /// The decoder must be drained before this unconsumed packet is retried.
    NeedOutput(EncodedAudioPacket),
}

/// One event produced by an audio decoder's pull side.
#[derive(Debug, Clone)]
#[must_use]
#[non_exhaustive]
pub enum AudioDecodeOutput {
    Frame(DecodedAudioFrame),
    /// Always emitted before the first frame using this new format.
    FormatChanged(AudioFormat),
    NeedInput,
    EndOfStream,
}

/// A synchronous, runtime-independent compressed-audio decoder.
pub trait AudioDecoder: Send {
    type Error: std::error::Error + Send + Sync + 'static;

    fn configure(&mut self, config: AudioDecoderConfig) -> std::result::Result<(), Self::Error>;

    fn send_packet(
        &mut self,
        packet: EncodedAudioPacket,
    ) -> std::result::Result<AudioDecodeInputStatus, Self::Error>;

    fn receive_frame(&mut self) -> std::result::Result<AudioDecodeOutput, Self::Error>;

    /// Clears delayed output and all state tied to the old timeline.
    fn flush(&mut self);

    /// Marks input complete so delayed PCM can be received.
    fn drain(&mut self) -> std::result::Result<(), Self::Error>;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stereo_format() -> AudioFormat {
        AudioFormat::new(48_000, ChannelLayout::Stereo)
    }

    #[test]
    fn validates_audio_decoder_configuration() {
        let valid = AudioDecoderConfig::new(AudioCodec::Aac, 44_100, ChannelLayout::Stereo)
            .with_codec_data([0x12, 0x10]);
        assert_eq!(valid.validate(), Ok(()));

        let zero_rate = AudioDecoderConfig::new(AudioCodec::Aac, 0, ChannelLayout::Mono)
            .with_codec_data([0x12, 0x08]);
        assert_eq!(
            zero_rate.validate(),
            Err(MediaError::InvalidDecoderConfig(
                "audio sample rate must be non-zero"
            ))
        );

        let zero_channels =
            AudioDecoderConfig::new(AudioCodec::Aac, 48_000, ChannelLayout::Discrete(0))
                .with_codec_data([0x11, 0x88]);
        assert_eq!(
            zero_channels.validate(),
            Err(MediaError::InvalidDecoderConfig(
                "audio channel count must be non-zero"
            ))
        );

        let no_codec_data = AudioDecoderConfig::new(AudioCodec::Aac, 48_000, ChannelLayout::Stereo);
        assert_eq!(
            no_codec_data.validate(),
            Err(MediaError::InvalidDecoderConfig(
                "audio codec data is required for this codec"
            ))
        );

        let pcm = AudioDecoderConfig::new(
            AudioCodec::Pcm(PcmCodec::Signed16Le),
            48_000,
            ChannelLayout::Stereo,
        )
        .with_bits_per_sample(16);
        assert_eq!(pcm.validate(), Ok(()));

        let incomplete_adpcm = AudioDecoderConfig::new(
            AudioCodec::Adpcm(AdpcmCodec::ImaWav),
            48_000,
            ChannelLayout::Mono,
        );
        assert_eq!(
            incomplete_adpcm.validate(),
            Err(MediaError::InvalidDecoderConfig(
                "ADPCM requires packet and block frame counts"
            ))
        );
    }

    #[test]
    fn computes_exact_pcm_durations_without_floating_point() {
        let format = stereo_format();
        assert_eq!(
            format.duration_for_sample_frames(1_024).unwrap(),
            MediaTime::from_parts(1_024, 48_000).unwrap()
        );
        assert_eq!(
            format.duration_for_sample_frames(usize::MAX),
            Err(MediaError::IntegerOverflow)
        );
    }

    #[test]
    fn validates_interleaved_pcm_and_duration() {
        let format = stereo_format();
        let duration = format.duration_for_sample_frames(1_024).unwrap();
        let frame = DecodedAudioFrame::new(
            7,
            MediaTime::from_parts(0, 48_000),
            Some(duration),
            format,
            vec![0.0; 2_048],
        );
        assert_eq!(frame.channels(), 2);
        assert_eq!(frame.sample_frames(), 1_024);
        assert_eq!(frame.validate(), Ok(()));

        let malformed = DecodedAudioFrame::new(8, None, None, format, vec![0.0; 3]);
        assert_eq!(
            malformed.validate(),
            Err(MediaError::InvalidAudioFrame(
                "PCM sample count is not divisible by the channel count"
            ))
        );

        let negative = DecodedAudioFrame::new(
            9,
            None,
            MediaTime::from_parts(-1, 48_000),
            format,
            Vec::<f32>::new(),
        );
        assert_eq!(
            negative.validate(),
            Err(MediaError::InvalidAudioFrame(
                "audio frame duration must not be negative"
            ))
        );

        let inaccurate = DecodedAudioFrame::new(
            10,
            None,
            MediaTime::from_parts(1_100, 48_000),
            format,
            vec![0.0; 2_048],
        );
        assert_eq!(
            inaccurate.validate(),
            Err(MediaError::InvalidAudioFrame(
                "audio frame duration differs from its PCM sample count"
            ))
        );
    }
}
