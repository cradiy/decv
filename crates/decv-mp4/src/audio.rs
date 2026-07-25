use std::sync::Arc;

use decv_core::{AudioCodec, AudioDecoderConfig, ChannelLayout};

use crate::{
    BoxHeader, FourCc, Mp4Error, Mp4File, Result,
    reader::{BoundedReader, read_full_box},
};

pub(crate) const MP4A: FourCc = FourCc::new(*b"mp4a");
const ESDS: FourCc = FourCc::new(*b"esds");
const AUDIO_SAMPLE_ENTRY_FIELDS_SIZE: u64 = 28;
const MAX_CODEC_CONFIGURATION_SIZE: usize = 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AacSampleEntry {
    format: FourCc,
    data_reference_index: u16,
    channel_count: u16,
    sample_size: u16,
    sample_rate: u32,
    audio_specific_config: Arc<[u8]>,
}

impl AacSampleEntry {
    #[inline]
    pub const fn format(&self) -> FourCc {
        self.format
    }

    #[inline]
    pub const fn data_reference_index(&self) -> u16 {
        self.data_reference_index
    }

    #[inline]
    pub const fn channel_count(&self) -> u16 {
        self.channel_count
    }

    #[inline]
    pub const fn sample_size(&self) -> u16 {
        self.sample_size
    }

    #[inline]
    pub const fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    #[inline]
    pub fn audio_specific_config(&self) -> &Arc<[u8]> {
        &self.audio_specific_config
    }

    pub fn decoder_config(&self) -> Result<AudioDecoderConfig> {
        let parsed = parse_audio_specific_config(&self.audio_specific_config)?;
        if parsed.sample_rate != self.sample_rate {
            return Err(Mp4Error::InvalidData(
                "AAC sample rate differs between mp4a and AudioSpecificConfig",
            ));
        }
        if parsed.channel_layout.channels() != self.channel_count {
            return Err(Mp4Error::InvalidData(
                "AAC channel count differs between mp4a and AudioSpecificConfig",
            ));
        }
        let config =
            AudioDecoderConfig::new(AudioCodec::Aac, parsed.sample_rate, parsed.channel_layout)
                .with_codec_data(Arc::clone(&self.audio_specific_config));
        config
            .validate()
            .map_err(|_| Mp4Error::InvalidData("invalid AAC decoder configuration"))?;
        Ok(config)
    }
}

pub(crate) fn parse_aac_sample_entry(
    file: Mp4File<'_>,
    entry: BoxHeader,
) -> Result<AacSampleEntry> {
    if entry.payload_size() < AUDIO_SAMPLE_ENTRY_FIELDS_SIZE {
        return Err(Mp4Error::InvalidData("audio sample entry is truncated"));
    }
    let payload = entry.payload_range()?;
    let mut reader = BoundedReader::new(file.input(), payload.start, payload.end)?;
    reader.skip(6)?;
    let data_reference_index = reader.read_u16()?;
    let version = reader.read_u16()?;
    reader.skip(2)?;
    reader.skip(4)?;
    let channel_count = reader.read_u16()?;
    let sample_size = reader.read_u16()?;
    reader.skip(4)?;
    let sample_rate_fixed = reader.read_u32()?;
    if version != 0 {
        return Err(Mp4Error::UnsupportedFeature(
            "versioned mp4a sample entries",
        ));
    }
    if channel_count == 0 {
        return Err(Mp4Error::InvalidData("mp4a channel count is zero"));
    }
    if sample_rate_fixed & 0xffff != 0 {
        return Err(Mp4Error::UnsupportedFeature("fractional mp4a sample rates"));
    }
    let sample_rate = sample_rate_fixed >> 16;
    if sample_rate == 0 {
        return Err(Mp4Error::InvalidData("mp4a sample rate is zero"));
    }

    let mut audio_specific_config = None;
    for child in file.boxes_in(reader.position()..payload.end)? {
        let child = child?;
        if child.kind() == ESDS {
            if audio_specific_config.is_some() {
                return Err(Mp4Error::InvalidData(
                    "mp4a sample entry has multiple esds boxes",
                ));
            }
            audio_specific_config = Some(parse_esds(file, child)?);
        }
    }
    let audio_specific_config =
        audio_specific_config.ok_or(Mp4Error::InvalidData("mp4a sample entry is missing esds"))?;
    let value = AacSampleEntry {
        format: entry.kind(),
        data_reference_index,
        channel_count,
        sample_size,
        sample_rate,
        audio_specific_config,
    };
    value.decoder_config()?;
    Ok(value)
}

fn parse_esds(file: Mp4File<'_>, header: BoxHeader) -> Result<Arc<[u8]>> {
    let range = header.payload_range()?;
    let mut reader = BoundedReader::new(file.input(), range.start, range.end)?;
    let (version, flags) = read_full_box(&mut reader)?;
    if version != 0 || flags != 0 {
        return Err(Mp4Error::InvalidData("unsupported esds version or flags"));
    }
    let data = reader.read_vec(reader.remaining()?, MAX_CODEC_CONFIGURATION_SIZE)?;
    let root = Descriptor::parse_at(&data, 0)?;
    if root.tag != 0x03 || root.end != data.len() {
        return Err(Mp4Error::InvalidData(
            "esds does not contain one complete ES descriptor",
        ));
    }
    let decoder_config_offset = skip_es_descriptor_header(root.body)?;
    let decoder_config = find_descriptor(root.body, decoder_config_offset, 0x04)?
        .ok_or(Mp4Error::InvalidData("esds has no DecoderConfigDescriptor"))?;
    if decoder_config.body.len() < 13 {
        return Err(Mp4Error::InvalidData(
            "esds DecoderConfigDescriptor is truncated",
        ));
    }
    if decoder_config.body[0] != 0x40 {
        return Err(Mp4Error::UnsupportedFeature(
            "esds object type is not MPEG-4 Audio",
        ));
    }
    let stream_type = decoder_config.body[1];
    if stream_type >> 2 != 5 || stream_type & 1 == 0 {
        return Err(Mp4Error::InvalidData(
            "esds decoder configuration is not an audio stream",
        ));
    }
    let decoder_specific = find_descriptor(decoder_config.body, 13, 0x05)?
        .ok_or(Mp4Error::InvalidData("esds has no DecoderSpecificInfo"))?;
    if decoder_specific.body.is_empty() {
        return Err(Mp4Error::InvalidData("esds DecoderSpecificInfo is empty"));
    }
    Ok(Arc::from(decoder_specific.body))
}

fn skip_es_descriptor_header(body: &[u8]) -> Result<usize> {
    if body.len() < 3 {
        return Err(Mp4Error::InvalidData("esds ES descriptor is truncated"));
    }
    let flags = body[2];
    let mut position = 3usize;
    if flags & 0x80 != 0 {
        position = position.checked_add(2).ok_or(Mp4Error::IntegerOverflow)?;
    }
    if flags & 0x40 != 0 {
        let length = usize::from(
            *body
                .get(position)
                .ok_or(Mp4Error::InvalidData("esds URL descriptor is truncated"))?,
        );
        position = position
            .checked_add(1)
            .and_then(|value| value.checked_add(length))
            .ok_or(Mp4Error::IntegerOverflow)?;
    }
    if flags & 0x20 != 0 {
        position = position.checked_add(2).ok_or(Mp4Error::IntegerOverflow)?;
    }
    if position > body.len() {
        return Err(Mp4Error::InvalidData(
            "esds ES descriptor flags exceed its body",
        ));
    }
    Ok(position)
}

fn find_descriptor<'a>(
    data: &'a [u8],
    mut position: usize,
    wanted_tag: u8,
) -> Result<Option<Descriptor<'a>>> {
    while position < data.len() {
        let descriptor = Descriptor::parse_at(data, position)?;
        if descriptor.tag == wanted_tag {
            return Ok(Some(descriptor));
        }
        position = descriptor.end;
    }
    Ok(None)
}

#[derive(Clone, Copy)]
struct Descriptor<'a> {
    tag: u8,
    body: &'a [u8],
    end: usize,
}

impl<'a> Descriptor<'a> {
    fn parse_at(data: &'a [u8], position: usize) -> Result<Self> {
        let tag = *data
            .get(position)
            .ok_or(Mp4Error::InvalidData("esds descriptor is truncated"))?;
        let mut cursor = position.checked_add(1).ok_or(Mp4Error::IntegerOverflow)?;
        let mut length = 0usize;
        let mut terminated = false;
        for _ in 0..4 {
            let byte = *data
                .get(cursor)
                .ok_or(Mp4Error::InvalidData("esds descriptor length is truncated"))?;
            cursor = cursor.checked_add(1).ok_or(Mp4Error::IntegerOverflow)?;
            length = length
                .checked_mul(128)
                .and_then(|value| value.checked_add(usize::from(byte & 0x7f)))
                .ok_or(Mp4Error::IntegerOverflow)?;
            if byte & 0x80 == 0 {
                terminated = true;
                break;
            }
        }
        if !terminated {
            return Err(Mp4Error::InvalidData(
                "esds descriptor length exceeds four bytes",
            ));
        }
        let end = cursor
            .checked_add(length)
            .ok_or(Mp4Error::IntegerOverflow)?;
        let body = data
            .get(cursor..end)
            .ok_or(Mp4Error::InvalidData("esds descriptor body is truncated"))?;
        Ok(Self { tag, body, end })
    }
}

#[derive(Clone, Copy)]
struct ParsedAudioSpecificConfig {
    sample_rate: u32,
    channel_layout: ChannelLayout,
}

fn parse_audio_specific_config(data: &[u8]) -> Result<ParsedAudioSpecificConfig> {
    let mut bits = AudioBits::new(data);
    let object_type = read_audio_object_type(&mut bits)?;
    if object_type != 2 {
        return Err(Mp4Error::UnsupportedFeature(
            "AAC object type is not AAC-LC",
        ));
    }
    let frequency_index = bits.read(4)?;
    let sample_rate = if frequency_index == 15 {
        bits.read(24)?
    } else {
        const SAMPLE_RATES: [u32; 13] = [
            96_000, 88_200, 64_000, 48_000, 44_100, 32_000, 24_000, 22_050, 16_000, 12_000, 11_025,
            8_000, 7_350,
        ];
        *SAMPLE_RATES
            .get(usize::try_from(frequency_index).map_err(|_| Mp4Error::IntegerOverflow)?)
            .ok_or(Mp4Error::InvalidData(
                "AAC sampling-frequency index is reserved",
            ))?
    };
    if sample_rate == 0 {
        return Err(Mp4Error::InvalidData("AAC sample rate is zero"));
    }
    let channel_layout = match bits.read(4)? {
        1 => ChannelLayout::Mono,
        2 => ChannelLayout::Stereo,
        0 => {
            return Err(Mp4Error::UnsupportedFeature(
                "AAC program-config-element channel layout",
            ));
        }
        _ => {
            return Err(Mp4Error::UnsupportedFeature(
                "AAC channel configuration is not mono or stereo",
            ));
        }
    };
    Ok(ParsedAudioSpecificConfig {
        sample_rate,
        channel_layout,
    })
}

fn read_audio_object_type(bits: &mut AudioBits<'_>) -> Result<u32> {
    let value = bits.read(5)?;
    if value == 31 {
        bits.read(6)?
            .checked_add(32)
            .ok_or(Mp4Error::IntegerOverflow)
    } else {
        Ok(value)
    }
}

struct AudioBits<'a> {
    data: &'a [u8],
    position: usize,
}

impl<'a> AudioBits<'a> {
    const fn new(data: &'a [u8]) -> Self {
        Self { data, position: 0 }
    }

    fn read(&mut self, width: usize) -> Result<u32> {
        let end = self
            .position
            .checked_add(width)
            .ok_or(Mp4Error::IntegerOverflow)?;
        if width > 32 || end > self.data.len().saturating_mul(8) {
            return Err(Mp4Error::InvalidData(
                "AAC AudioSpecificConfig is truncated",
            ));
        }
        let mut value = 0u32;
        while self.position < end {
            let byte = self.data[self.position / 8];
            value = value << 1 | u32::from(byte >> (7 - self.position % 8) & 1);
            self.position += 1;
        }
        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_common_aac_lc_audio_specific_configs() {
        let stereo = parse_audio_specific_config(&[0x12, 0x10, 0x56, 0xe5, 0x00]).unwrap();
        assert_eq!(stereo.sample_rate, 44_100);
        assert_eq!(stereo.channel_layout, ChannelLayout::Stereo);

        let mono = parse_audio_specific_config(&[0x11, 0x88]).unwrap();
        assert_eq!(mono.sample_rate, 48_000);
        assert_eq!(mono.channel_layout, ChannelLayout::Mono);
    }

    #[test]
    fn rejects_unsupported_or_truncated_audio_specific_configs() {
        assert!(matches!(
            parse_audio_specific_config(&[0x2a, 0x10]),
            Err(Mp4Error::UnsupportedFeature(
                "AAC object type is not AAC-LC"
            ))
        ));
        assert!(matches!(
            parse_audio_specific_config(&[0x12]),
            Err(Mp4Error::InvalidData(
                "AAC AudioSpecificConfig is truncated"
            ))
        ));
    }
}
