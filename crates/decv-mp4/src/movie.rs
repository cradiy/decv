use std::{num::NonZeroU32, sync::Arc};

use decv_core::{BitstreamFormat, MediaTime, VideoCodec, VideoDecoderConfig};

use crate::{
    BoxHeader, FourCc, Mp4Error, Mp4File, Result,
    edit::{EDTS, Edit, linear_timeline_offset, parse_edit_container},
    reader::{BoundedReader, read_full_box},
    sample_table::{Sample, parse_sample_table},
};

const MOOV: FourCc = FourCc::new(*b"moov");
const MVHD: FourCc = FourCc::new(*b"mvhd");
const TRAK: FourCc = FourCc::new(*b"trak");
const TKHD: FourCc = FourCc::new(*b"tkhd");
const MDIA: FourCc = FourCc::new(*b"mdia");
const MDHD: FourCc = FourCc::new(*b"mdhd");
const HDLR: FourCc = FourCc::new(*b"hdlr");
const MINF: FourCc = FourCc::new(*b"minf");
const STBL: FourCc = FourCc::new(*b"stbl");
const STSD: FourCc = FourCc::new(*b"stsd");
const VIDE: FourCc = FourCc::new(*b"vide");
const AVC1: FourCc = FourCc::new(*b"avc1");
const AVC3: FourCc = FourCc::new(*b"avc3");
const AVCC: FourCc = FourCc::new(*b"avcC");

const VISUAL_SAMPLE_ENTRY_FIELDS_SIZE: u64 = 78;
const MAX_CODEC_CONFIGURATION_SIZE: usize = 1024 * 1024;
const MAX_TRACK_COUNT: usize = 1024;
const MAX_SAMPLE_DESCRIPTION_COUNT: usize = 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Movie {
    timescale: NonZeroU32,
    duration: Option<u64>,
    tracks: Vec<Track>,
}

impl Movie {
    pub fn parse(file: Mp4File<'_>) -> Result<Self> {
        let mut movie_box = None;
        for header in file.boxes() {
            let header = header?;
            if header.kind() == MOOV && movie_box.replace(header).is_some() {
                return Err(Mp4Error::InvalidData("MP4 contains multiple moov boxes"));
            }
        }
        let movie_box = movie_box.ok_or(Mp4Error::InvalidData("MP4 has no moov box"))?;

        let mut movie_header = None;
        let mut track_boxes = Vec::new();
        for child in file.children(movie_box)? {
            let child = child?;
            match child.kind() {
                MVHD => set_once(&mut movie_header, child, "duplicate mvhd box")?,
                TRAK => {
                    if track_boxes.len() == MAX_TRACK_COUNT {
                        return Err(Mp4Error::InvalidData("MP4 track count exceeds its limit"));
                    }
                    track_boxes.push(child);
                }
                _ => {}
            }
        }
        let (timescale, duration) = parse_duration_header(
            file,
            movie_header.ok_or(Mp4Error::InvalidData("moov has no mvhd box"))?,
        )?;
        let tracks = track_boxes
            .into_iter()
            .map(|header| Track::parse(file, header, timescale))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            timescale,
            duration,
            tracks,
        })
    }

    #[inline]
    pub const fn timescale(&self) -> NonZeroU32 {
        self.timescale
    }

    #[inline]
    pub const fn duration(&self) -> Option<u64> {
        self.duration
    }

    #[inline]
    pub fn tracks(&self) -> &[Track] {
        &self.tracks
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Track {
    id: u32,
    flags: u32,
    movie_timescale: NonZeroU32,
    media_timescale: NonZeroU32,
    media_duration: Option<u64>,
    handler: FourCc,
    display_width_16_16: u32,
    display_height_16_16: u32,
    sample_descriptions: Vec<SampleDescription>,
    samples: Vec<Sample>,
    sync_sample_indices: Vec<usize>,
    sync_sample_indices_by_presentation: Vec<usize>,
    edits: Vec<Edit>,
}

impl Track {
    fn parse(file: Mp4File<'_>, track_box: BoxHeader, movie_timescale: NonZeroU32) -> Result<Self> {
        let mut track_header = None;
        let mut media_box = None;
        let mut edit_box = None;
        for child in file.children(track_box)? {
            let child = child?;
            match child.kind() {
                TKHD => set_once(&mut track_header, child, "duplicate tkhd box")?,
                MDIA => set_once(&mut media_box, child, "duplicate mdia box")?,
                EDTS => set_once(&mut edit_box, child, "duplicate edts box")?,
                _ => {}
            }
        }
        let (id, flags, display_width_16_16, display_height_16_16) = parse_track_header(
            file,
            track_header.ok_or(Mp4Error::InvalidData("trak has no tkhd box"))?,
        )?;
        let media_box = media_box.ok_or(Mp4Error::InvalidData("trak has no mdia box"))?;
        let edits = edit_box
            .map(|header| parse_edit_container(file, header))
            .transpose()?
            .unwrap_or_default();

        let mut media_header = None;
        let mut handler_box = None;
        let mut media_information = None;
        for child in file.children(media_box)? {
            let child = child?;
            match child.kind() {
                MDHD => set_once(&mut media_header, child, "duplicate mdhd box")?,
                HDLR => set_once(&mut handler_box, child, "duplicate hdlr box")?,
                MINF => set_once(&mut media_information, child, "duplicate minf box")?,
                _ => {}
            }
        }
        let (media_timescale, media_duration) = parse_duration_header(
            file,
            media_header.ok_or(Mp4Error::InvalidData("mdia has no mdhd box"))?,
        )?;
        let handler = parse_handler(
            file,
            handler_box.ok_or(Mp4Error::InvalidData("mdia has no hdlr box"))?,
        )?;
        let (sample_descriptions, samples) = if handler == VIDE {
            parse_video_sample_table(
                file,
                media_information.ok_or(Mp4Error::InvalidData("video track has no minf box"))?,
            )?
        } else {
            (Vec::new(), Vec::new())
        };
        let sync_sample_indices = samples
            .iter()
            .enumerate()
            .filter_map(|(index, sample)| sample.is_sync().then_some(index))
            .collect::<Vec<_>>();
        let mut sync_sample_indices_by_presentation = sync_sample_indices.clone();
        sync_sample_indices_by_presentation
            .sort_unstable_by_key(|&index| (samples[index].presentation_time(), index));

        Ok(Self {
            id,
            flags,
            movie_timescale,
            media_timescale,
            media_duration,
            handler,
            display_width_16_16,
            display_height_16_16,
            sample_descriptions,
            samples,
            sync_sample_indices,
            sync_sample_indices_by_presentation,
            edits,
        })
    }

    #[inline]
    pub const fn id(&self) -> u32 {
        self.id
    }

    #[inline]
    pub const fn flags(&self) -> u32 {
        self.flags
    }

    #[inline]
    pub const fn movie_timescale(&self) -> NonZeroU32 {
        self.movie_timescale
    }

    #[inline]
    pub const fn media_timescale(&self) -> NonZeroU32 {
        self.media_timescale
    }

    #[inline]
    pub const fn media_duration(&self) -> Option<u64> {
        self.media_duration
    }

    #[inline]
    pub const fn handler(&self) -> FourCc {
        self.handler
    }

    #[inline]
    pub const fn display_width_16_16(&self) -> u32 {
        self.display_width_16_16
    }

    #[inline]
    pub const fn display_height_16_16(&self) -> u32 {
        self.display_height_16_16
    }

    #[inline]
    pub fn sample_descriptions(&self) -> &[SampleDescription] {
        &self.sample_descriptions
    }

    #[inline]
    pub fn samples(&self) -> &[Sample] {
        &self.samples
    }

    #[inline]
    pub fn sync_sample_indices(&self) -> &[usize] {
        &self.sync_sample_indices
    }

    #[inline]
    pub fn edits(&self) -> &[Edit] {
        &self.edits
    }

    pub fn decoder_config(&self, description_index: usize) -> Result<VideoDecoderConfig> {
        let description =
            self.sample_descriptions
                .get(description_index)
                .ok_or(Mp4Error::IndexOutOfRange {
                    kind: "sample-description",
                    index: description_index,
                })?;
        match description {
            SampleDescription::Avc(entry) => entry.decoder_config(),
            SampleDescription::Unsupported { .. } => Err(Mp4Error::UnsupportedFeature(
                "sample description has no supported video decoder",
            )),
        }
    }

    pub fn decoder_config_for_sample(&self, sample_index: usize) -> Result<VideoDecoderConfig> {
        let sample = self
            .samples
            .get(sample_index)
            .ok_or(Mp4Error::IndexOutOfRange {
                kind: "sample",
                index: sample_index,
            })?;
        self.decoder_config(sample.description_index())
    }

    /// Offset that maps raw sample-table times onto the movie presentation
    /// timeline. Complex edit lists return an explicit unsupported-data error.
    pub fn presentation_time_offset(&self) -> Result<MediaTime> {
        let value =
            linear_timeline_offset(&self.edits, self.movie_timescale, self.media_timescale)?;
        Ok(MediaTime::new(value, self.media_timescale))
    }

    /// Finds the sync sample with the greatest presentation time not after
    /// `target`. The returned index is in decode/sample-table order.
    pub fn keyframe_at_or_before(&self, target: MediaTime) -> Result<Option<usize>> {
        let offset = self.presentation_time_offset()?.value;
        self.validate_sync_presentation_range(offset)?;
        let position = self
            .sync_sample_indices_by_presentation
            .partition_point(|&index| {
                adjusted_presentation_time_is_not_after(
                    self.samples[index].presentation_time(),
                    offset,
                    self.media_timescale,
                    target,
                )
            });
        if position == 0 {
            return Ok(None);
        }

        let candidate = self.sync_sample_indices_by_presentation[position - 1];
        let presentation_time = self.samples[candidate].presentation_time();
        let first_equal = self.sync_sample_indices_by_presentation[..position]
            .partition_point(|&index| self.samples[index].presentation_time() < presentation_time);
        Ok(Some(self.sync_sample_indices_by_presentation[first_equal]))
    }

    /// Finds the sync sample with the least presentation time not before
    /// `target`. This is useful for low-latency seek previews that may jump
    /// forward to the next independently decodable picture.
    pub fn keyframe_at_or_after(&self, target: MediaTime) -> Result<Option<usize>> {
        let offset = self.presentation_time_offset()?.value;
        self.validate_sync_presentation_range(offset)?;
        let position = self
            .sync_sample_indices_by_presentation
            .partition_point(|&index| {
                adjusted_presentation_time_is_before(
                    self.samples[index].presentation_time(),
                    offset,
                    self.media_timescale,
                    target,
                )
            });
        Ok(self
            .sync_sample_indices_by_presentation
            .get(position)
            .copied())
    }

    /// Finds the sync sample with presentation time nearest to `target`.
    ///
    /// This is intended for low-latency interactive previews. The result may
    /// precede or follow the requested time and therefore must not be used as
    /// a substitute for [`Self::keyframe_at_or_before`] when exact seek
    /// preroll is required. Equidistant keyframes prefer the earlier one.
    pub fn keyframe_nearest(&self, target: MediaTime) -> Result<Option<usize>> {
        let offset = self.presentation_time_offset()?.value;
        self.validate_sync_presentation_range(offset)?;
        let position = self
            .sync_sample_indices_by_presentation
            .partition_point(|&index| {
                adjusted_presentation_time_is_before(
                    self.samples[index].presentation_time(),
                    offset,
                    self.media_timescale,
                    target,
                )
            });
        let after = self
            .sync_sample_indices_by_presentation
            .get(position)
            .copied();
        let Some(previous_position) = position.checked_sub(1) else {
            return Ok(after);
        };
        let previous = self.sync_sample_indices_by_presentation[previous_position];
        let previous_time = self.samples[previous].presentation_time();
        let first_equal = self.sync_sample_indices_by_presentation[..position]
            .partition_point(|&index| self.samples[index].presentation_time() < previous_time);
        let before = self.sync_sample_indices_by_presentation[first_equal];
        let Some(after) = after else {
            return Ok(Some(before));
        };

        let before_distance = adjusted_presentation_distance(
            self.samples[before].presentation_time(),
            offset,
            self.media_timescale,
            target,
        );
        let after_distance = adjusted_presentation_distance(
            self.samples[after].presentation_time(),
            offset,
            self.media_timescale,
            target,
        );
        Ok(Some(if before_distance <= after_distance {
            before
        } else {
            after
        }))
    }

    fn validate_sync_presentation_range(&self, offset: i64) -> Result<()> {
        let Some((&first, &last)) = self
            .sync_sample_indices_by_presentation
            .first()
            .zip(self.sync_sample_indices_by_presentation.last())
        else {
            return Ok(());
        };
        self.samples[first]
            .presentation_time()
            .checked_add(offset)
            .ok_or(Mp4Error::IntegerOverflow)?;
        self.samples[last]
            .presentation_time()
            .checked_add(offset)
            .ok_or(Mp4Error::IntegerOverflow)?;
        Ok(())
    }
}

#[inline]
fn adjusted_presentation_time_is_before(
    presentation_time: i64,
    offset: i64,
    timescale: NonZeroU32,
    target: MediaTime,
) -> bool {
    let adjusted = i128::from(presentation_time) + i128::from(offset);
    adjusted * i128::from(target.timescale.get())
        < i128::from(target.value) * i128::from(timescale.get())
}

#[inline]
fn adjusted_presentation_time_is_not_after(
    presentation_time: i64,
    offset: i64,
    timescale: NonZeroU32,
    target: MediaTime,
) -> bool {
    let adjusted = i128::from(presentation_time) + i128::from(offset);
    adjusted * i128::from(target.timescale.get())
        <= i128::from(target.value) * i128::from(timescale.get())
}

#[inline]
fn adjusted_presentation_distance(
    presentation_time: i64,
    offset: i64,
    timescale: NonZeroU32,
    target: MediaTime,
) -> i128 {
    let adjusted = i128::from(presentation_time) + i128::from(offset);
    let sample = adjusted * i128::from(target.timescale.get());
    let target = i128::from(target.value) * i128::from(timescale.get());
    (sample - target).abs()
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SampleDescription {
    Avc(AvcSampleEntry),
    Unsupported { format: FourCc },
}

impl SampleDescription {
    pub const fn format(&self) -> FourCc {
        match self {
            Self::Avc(entry) => entry.format,
            Self::Unsupported { format } => *format,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AvcSampleEntry {
    format: FourCc,
    data_reference_index: u16,
    width: u16,
    height: u16,
    codec_configuration: Arc<[u8]>,
}

impl AvcSampleEntry {
    #[inline]
    pub const fn format(&self) -> FourCc {
        self.format
    }

    #[inline]
    pub const fn data_reference_index(&self) -> u16 {
        self.data_reference_index
    }

    #[inline]
    pub const fn width(&self) -> u16 {
        self.width
    }

    #[inline]
    pub const fn height(&self) -> u16 {
        self.height
    }

    #[inline]
    pub fn codec_configuration(&self) -> &Arc<[u8]> {
        &self.codec_configuration
    }

    pub fn decoder_config(&self) -> Result<VideoDecoderConfig> {
        let data = self.codec_configuration.as_ref();
        if data.len() < 6 || data[0] != 1 {
            return Err(Mp4Error::InvalidData(
                "invalid AVCDecoderConfigurationRecord header",
            ));
        }
        if data[4] & 0xfc != 0xfc || data[5] & 0xe0 != 0xe0 {
            return Err(Mp4Error::InvalidData(
                "avcC reserved bits do not have their required values",
            ));
        }
        Ok(VideoDecoderConfig::new(
            VideoCodec::H264,
            BitstreamFormat::LengthPrefixed {
                length_size: (data[4] & 3) + 1,
            },
        )
        .with_codec_data(Arc::clone(&self.codec_configuration)))
    }
}

fn parse_duration_header(
    file: Mp4File<'_>,
    header: BoxHeader,
) -> Result<(NonZeroU32, Option<u64>)> {
    let range = header.payload_range()?;
    let mut reader = BoundedReader::new(file.input(), range.start, range.end)?;
    let (version, _) = read_full_box(&mut reader)?;
    let (timescale, duration, unknown_duration) = match version {
        0 => {
            reader.skip(8)?;
            let timescale = reader.read_u32()?;
            let duration = reader.read_u32()?;
            (timescale, u64::from(duration), u32::MAX.into())
        }
        1 => {
            reader.skip(16)?;
            let timescale = reader.read_u32()?;
            let duration = reader.read_u64()?;
            (timescale, duration, u64::MAX)
        }
        _ => {
            return Err(Mp4Error::InvalidData(
                "unsupported movie/media header version",
            ));
        }
    };
    let timescale =
        NonZeroU32::new(timescale).ok_or(Mp4Error::InvalidData("movie/media timescale is zero"))?;
    Ok((
        timescale,
        (duration != unknown_duration).then_some(duration),
    ))
}

fn parse_track_header(file: Mp4File<'_>, header: BoxHeader) -> Result<(u32, u32, u32, u32)> {
    let range = header.payload_range()?;
    let mut reader = BoundedReader::new(file.input(), range.start, range.end)?;
    let (version, flags) = read_full_box(&mut reader)?;
    let id = match version {
        0 => {
            reader.skip(8)?;
            let id = reader.read_u32()?;
            reader.skip(8)?;
            id
        }
        1 => {
            reader.skip(16)?;
            let id = reader.read_u32()?;
            reader.skip(12)?;
            id
        }
        _ => return Err(Mp4Error::InvalidData("unsupported track header version")),
    };
    if id == 0 {
        return Err(Mp4Error::InvalidData("track id is zero"));
    }
    if reader.remaining()? < 60 {
        return Err(Mp4Error::InvalidData("track header is truncated"));
    }
    reader.skip(reader.remaining()? - 8)?;
    let width = reader.read_u32()?;
    let height = reader.read_u32()?;
    Ok((id, flags, width, height))
}

fn parse_handler(file: Mp4File<'_>, header: BoxHeader) -> Result<FourCc> {
    let range = header.payload_range()?;
    let mut reader = BoundedReader::new(file.input(), range.start, range.end)?;
    read_full_box(&mut reader)?;
    reader.skip(4)?;
    reader.read_fourcc()
}

fn parse_video_sample_table(
    file: Mp4File<'_>,
    media_information: BoxHeader,
) -> Result<(Vec<SampleDescription>, Vec<Sample>)> {
    let mut sample_table = None;
    for child in file.children(media_information)? {
        let child = child?;
        if child.kind() == STBL {
            set_once(&mut sample_table, child, "duplicate stbl box")?;
        }
    }
    let sample_table = sample_table.ok_or(Mp4Error::InvalidData("video minf has no stbl box"))?;
    let mut description_box = None;
    for child in file.children(sample_table)? {
        let child = child?;
        if child.kind() == STSD {
            set_once(&mut description_box, child, "duplicate stsd box")?;
        }
    }
    let description_box =
        description_box.ok_or(Mp4Error::InvalidData("video stbl has no stsd box"))?;
    let descriptions = parse_sample_description_box(file, description_box)?;
    let samples = parse_sample_table(file, sample_table, descriptions.len())?;
    Ok((descriptions, samples))
}

fn parse_sample_description_box(
    file: Mp4File<'_>,
    description_box: BoxHeader,
) -> Result<Vec<SampleDescription>> {
    let range = description_box.payload_range()?;
    let mut reader = BoundedReader::new(file.input(), range.start, range.end)?;
    let (version, _) = read_full_box(&mut reader)?;
    if version != 0 {
        return Err(Mp4Error::InvalidData("unsupported stsd version"));
    }
    let entry_count = usize::try_from(reader.read_u32()?).map_err(|_| Mp4Error::IntegerOverflow)?;
    if entry_count > MAX_SAMPLE_DESCRIPTION_COUNT {
        return Err(Mp4Error::InvalidData(
            "sample-description count exceeds its limit",
        ));
    }
    let entries_start = reader.position();
    let mut entries = file.boxes_in(entries_start..range.end)?;
    let mut descriptions = Vec::with_capacity(entry_count);
    for _ in 0..entry_count {
        let entry = entries.next().ok_or(Mp4Error::InvalidData(
            "stsd has fewer entries than declared",
        ))??;
        descriptions.push(parse_video_sample_entry(file, entry)?);
    }
    if entries.next().is_some() {
        return Err(Mp4Error::InvalidData("stsd has more entries than declared"));
    }
    Ok(descriptions)
}

fn parse_video_sample_entry(file: Mp4File<'_>, entry: BoxHeader) -> Result<SampleDescription> {
    let format = entry.kind();
    if format != AVC1 && format != AVC3 {
        return Ok(SampleDescription::Unsupported { format });
    }
    if entry.payload_size() < VISUAL_SAMPLE_ENTRY_FIELDS_SIZE {
        return Err(Mp4Error::InvalidData("visual sample entry is truncated"));
    }
    let payload = entry.payload_range()?;
    let mut reader = BoundedReader::new(file.input(), payload.start, payload.end)?;
    reader.skip(6)?;
    let data_reference_index = reader.read_u16()?;
    reader.skip(16)?;
    let width = reader.read_u16()?;
    let height = reader.read_u16()?;
    reader.skip(VISUAL_SAMPLE_ENTRY_FIELDS_SIZE - 28)?;

    let mut codec_configuration = None;
    for child in file.boxes_in(reader.position()..payload.end)? {
        let child = child?;
        if child.kind() == AVCC {
            if codec_configuration.is_some() {
                return Err(Mp4Error::InvalidData(
                    "AVC sample entry has multiple avcC boxes",
                ));
            }
            let range = child.payload_range()?;
            let mut reader = BoundedReader::new(file.input(), range.start, range.end)?;
            codec_configuration = Some(
                reader
                    .read_vec(reader.remaining()?, MAX_CODEC_CONFIGURATION_SIZE)?
                    .into(),
            );
        }
    }
    let codec_configuration =
        codec_configuration.ok_or(Mp4Error::InvalidData("AVC sample entry is missing avcC"))?;
    Ok(SampleDescription::Avc(AvcSampleEntry {
        format,
        data_reference_index,
        width,
        height,
        codec_configuration,
    }))
}

fn set_once<T>(slot: &mut Option<T>, value: T, duplicate: &'static str) -> Result<()> {
    if slot.replace(value).is_some() {
        return Err(Mp4Error::InvalidData(duplicate));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::io;

    use decv_core::MediaInput;

    use super::*;

    #[derive(Debug)]
    struct MemoryInput(Vec<u8>);

    impl MediaInput for MemoryInput {
        fn len(&self) -> io::Result<Option<u64>> {
            Ok(Some(u64::try_from(self.0.len()).unwrap()))
        }

        fn read_at(&self, offset: u64, buffer: &mut [u8]) -> io::Result<usize> {
            let offset = usize::try_from(offset).unwrap_or(usize::MAX);
            let Some(source) = self.0.get(offset..) else {
                return Ok(0);
            };
            let count = source.len().min(buffer.len()).min(3);
            buffer[..count].copy_from_slice(&source[..count]);
            Ok(count)
        }
    }

    fn boxed(kind: [u8; 4], payload: &[u8]) -> Vec<u8> {
        let mut bytes = Vec::from(u32::try_from(8 + payload.len()).unwrap().to_be_bytes());
        bytes.extend_from_slice(&kind);
        bytes.extend_from_slice(payload);
        bytes
    }

    fn full(version: u8, flags: u32, body: &[u8]) -> Vec<u8> {
        let mut bytes = vec![
            version,
            u8::try_from(flags >> 16).unwrap(),
            u8::try_from(flags >> 8 & 0xff).unwrap(),
            u8::try_from(flags & 0xff).unwrap(),
        ];
        bytes.extend_from_slice(body);
        bytes
    }

    fn synthetic_movie() -> Vec<u8> {
        let mut mvhd = vec![0; 8];
        mvhd.extend_from_slice(&1_000u32.to_be_bytes());
        mvhd.extend_from_slice(&5_000u32.to_be_bytes());
        mvhd.extend_from_slice(&[0; 80]);

        let mut tkhd = vec![0; 8];
        tkhd.extend_from_slice(&7u32.to_be_bytes());
        tkhd.extend_from_slice(&0u32.to_be_bytes());
        tkhd.extend_from_slice(&5_000u32.to_be_bytes());
        tkhd.extend_from_slice(&[0; 52]);
        tkhd.extend_from_slice(&(1_920u32 << 16).to_be_bytes());
        tkhd.extend_from_slice(&(1_080u32 << 16).to_be_bytes());

        let mut mdhd = vec![0; 8];
        mdhd.extend_from_slice(&90_000u32.to_be_bytes());
        mdhd.extend_from_slice(&450_000u32.to_be_bytes());
        mdhd.extend_from_slice(&[0; 4]);

        let mut hdlr = vec![0; 4];
        hdlr.extend_from_slice(b"vide");
        hdlr.extend_from_slice(&[0; 12]);
        hdlr.extend_from_slice(b"Video\0");

        let avcc = boxed(
            *b"avcC",
            &[
                1, 100, 0, 40, 0xff, 0xe1, 0, 3, 0x67, 1, 2, 1, 0, 2, 0x68, 3,
            ],
        );
        let mut avc1 = vec![0; 6];
        avc1.extend_from_slice(&1u16.to_be_bytes());
        avc1.extend_from_slice(&[0; 16]);
        avc1.extend_from_slice(&1_920u16.to_be_bytes());
        avc1.extend_from_slice(&1_080u16.to_be_bytes());
        avc1.extend_from_slice(&[0; 50]);
        avc1.extend_from_slice(&avcc);
        let avc1 = boxed(*b"avc1", &avc1);

        let mut stsd = full(0, 0, &1u32.to_be_bytes());
        stsd.extend_from_slice(&avc1);
        let stts = boxed(
            *b"stts",
            &full(0, 0, &[1u32, 3, 3_000].map(u32::to_be_bytes).concat()),
        );
        let stsc = boxed(
            *b"stsc",
            &full(0, 0, &[1u32, 1, 3, 1].map(u32::to_be_bytes).concat()),
        );
        let stsz = boxed(
            *b"stsz",
            &full(0, 0, &[1u32, 3].map(u32::to_be_bytes).concat()),
        );
        let stco = boxed(
            *b"stco",
            &full(0, 0, &[1u32, 0].map(u32::to_be_bytes).concat()),
        );
        let stss = boxed(
            *b"stss",
            &full(0, 0, &[2u32, 1, 3].map(u32::to_be_bytes).concat()),
        );
        let mut stbl_payload = boxed(*b"stsd", &stsd);
        stbl_payload.extend_from_slice(&stts);
        stbl_payload.extend_from_slice(&stsc);
        stbl_payload.extend_from_slice(&stsz);
        stbl_payload.extend_from_slice(&stco);
        stbl_payload.extend_from_slice(&stss);
        let stbl = boxed(*b"stbl", &stbl_payload);
        let minf = boxed(*b"minf", &stbl);

        let mut mdia = boxed(*b"mdhd", &full(0, 0, &mdhd));
        mdia.extend_from_slice(&boxed(*b"hdlr", &full(0, 0, &hdlr)));
        mdia.extend_from_slice(&minf);

        let mut elst = Vec::from(1u32.to_be_bytes());
        elst.extend_from_slice(&5_000u32.to_be_bytes());
        elst.extend_from_slice(&9_000i32.to_be_bytes());
        elst.extend_from_slice(&1i16.to_be_bytes());
        elst.extend_from_slice(&0i16.to_be_bytes());
        let edts = boxed(*b"edts", &boxed(*b"elst", &full(0, 0, &elst)));

        let mut trak = boxed(*b"tkhd", &full(0, 3, &tkhd));
        trak.extend_from_slice(&edts);
        trak.extend_from_slice(&boxed(*b"mdia", &mdia));

        let mut moov = boxed(*b"mvhd", &full(0, 0, &mvhd));
        moov.extend_from_slice(&boxed(*b"trak", &trak));
        let mut file = boxed(*b"ftyp", b"isom\0\0\0\0isom");
        file.extend_from_slice(&boxed(*b"moov", &moov));
        file
    }

    #[test]
    fn parses_movie_video_track_and_avc_configuration() {
        let input = MemoryInput(synthetic_movie());
        let movie = Movie::parse(Mp4File::open(&input).unwrap()).unwrap();
        assert_eq!(movie.timescale().get(), 1_000);
        assert_eq!(movie.duration(), Some(5_000));
        assert_eq!(movie.tracks().len(), 1);

        let track = &movie.tracks()[0];
        assert_eq!(track.id(), 7);
        assert_eq!(track.flags(), 3);
        assert_eq!(track.handler(), VIDE);
        assert_eq!(track.media_timescale().get(), 90_000);
        assert_eq!(track.media_duration(), Some(450_000));
        assert_eq!(track.display_width_16_16(), 1_920 << 16);
        assert_eq!(track.display_height_16_16(), 1_080 << 16);
        assert_eq!(track.samples().len(), 3);
        assert_eq!(track.sync_sample_indices(), [0, 2]);
        assert_eq!(track.edits().len(), 1);
        assert_eq!(track.edits()[0].media_time(), Some(9_000));
        assert_eq!(
            track.presentation_time_offset().unwrap(),
            MediaTime::from_parts(-9_000, 90_000).unwrap()
        );
        assert_eq!(
            track
                .keyframe_at_or_before(MediaTime::from_parts(-9_001, 90_000).unwrap())
                .unwrap(),
            None
        );
        assert_eq!(
            track
                .keyframe_at_or_before(MediaTime::from_parts(-5_000, 90_000).unwrap())
                .unwrap(),
            Some(0)
        );
        assert_eq!(
            track
                .keyframe_at_or_before(MediaTime::from_parts(-1, 30).unwrap())
                .unwrap(),
            Some(2)
        );
        assert_eq!(
            track
                .keyframe_at_or_after(MediaTime::from_parts(-9_001, 90_000).unwrap())
                .unwrap(),
            Some(0)
        );
        assert_eq!(
            track
                .keyframe_at_or_after(MediaTime::from_parts(-5_000, 90_000).unwrap())
                .unwrap(),
            Some(2)
        );
        assert_eq!(
            track
                .keyframe_at_or_after(MediaTime::from_parts(-1, 30).unwrap())
                .unwrap(),
            Some(2)
        );
        assert_eq!(
            track
                .keyframe_at_or_after(MediaTime::from_parts(0, 1).unwrap())
                .unwrap(),
            None
        );
        assert_eq!(
            track
                .keyframe_nearest(MediaTime::from_parts(-9_001, 90_000).unwrap())
                .unwrap(),
            Some(0)
        );
        assert_eq!(
            track
                .keyframe_nearest(MediaTime::from_parts(-7_000, 90_000).unwrap())
                .unwrap(),
            Some(0)
        );
        assert_eq!(
            track
                .keyframe_nearest(MediaTime::from_parts(-1, 15).unwrap())
                .unwrap(),
            Some(0),
            "an equidistant target prefers the earlier keyframe"
        );
        assert_eq!(
            track
                .keyframe_nearest(MediaTime::from_parts(-5_000, 90_000).unwrap())
                .unwrap(),
            Some(2)
        );
        assert_eq!(
            track
                .keyframe_nearest(MediaTime::from_parts(0, 1).unwrap())
                .unwrap(),
            Some(2)
        );

        let SampleDescription::Avc(entry) = &track.sample_descriptions()[0] else {
            panic!("expected AVC sample entry");
        };
        assert_eq!(entry.format(), AVC1);
        assert_eq!(entry.data_reference_index(), 1);
        assert_eq!((entry.width(), entry.height()), (1_920, 1_080));
        assert_eq!(
            entry.codec_configuration().as_ref(),
            [
                1, 100, 0, 40, 0xff, 0xe1, 0, 3, 0x67, 1, 2, 1, 0, 2, 0x68, 3,
            ]
        );
        let config = track.decoder_config_for_sample(0).unwrap();
        assert_eq!(config.codec, VideoCodec::H264);
        assert_eq!(
            config.bitstream_format,
            BitstreamFormat::LengthPrefixed { length_size: 4 }
        );
        assert_eq!(
            config.codec_data.unwrap().as_ref(),
            entry.codec_configuration().as_ref()
        );

        let packet = track.read_packet(&input, 0).unwrap();
        assert_eq!(packet.data.as_ref(), [0]);
        assert_eq!(packet.pts, MediaTime::from_parts(-9_000, 90_000));
        assert_eq!(packet.dts, MediaTime::from_parts(-9_000, 90_000));
        assert_eq!(packet.duration, MediaTime::from_parts(3_000, 90_000));
        assert!(packet.keyframe);
    }

    #[test]
    fn rejects_missing_movie_and_zero_timescale() {
        let input = MemoryInput(boxed(*b"ftyp", b"isom"));
        assert!(matches!(
            Movie::parse(Mp4File::open(&input).unwrap()),
            Err(Mp4Error::InvalidData("MP4 has no moov box"))
        ));

        let mut mvhd = vec![0; 16];
        mvhd.extend_from_slice(&[0; 80]);
        let input = MemoryInput(boxed(*b"moov", &boxed(*b"mvhd", &full(0, 0, &mvhd))));
        assert!(matches!(
            Movie::parse(Mp4File::open(&input).unwrap()),
            Err(Mp4Error::InvalidData("movie/media timescale is zero"))
        ));
    }

    #[test]
    fn rejects_declared_sample_description_count_mismatch() {
        let mut bytes = synthetic_movie();
        let position = bytes
            .windows(4)
            .position(|window| window == b"stsd")
            .unwrap();
        let entry_count = position + 8;
        bytes[entry_count..entry_count + 4].copy_from_slice(&2u32.to_be_bytes());
        let input = MemoryInput(bytes);
        assert!(matches!(
            Movie::parse(Mp4File::open(&input).unwrap()),
            Err(Mp4Error::InvalidData(
                "stsd has fewer entries than declared"
            ))
        ));
    }

    #[test]
    fn owned_demuxer_reads_indexed_packets() {
        let demuxer = crate::Mp4Demuxer::open(MemoryInput(synthetic_movie())).unwrap();
        assert_eq!(demuxer.movie().tracks().len(), 1);
        assert_eq!(demuxer.read_packet(0, 0).unwrap().data.as_ref(), [0]);
        assert!(matches!(
            demuxer.read_packet(1, 0),
            Err(Mp4Error::IndexOutOfRange {
                kind: "track",
                index: 1
            })
        ));

        let mut cursor = demuxer.packet_cursor(0).unwrap();
        assert_eq!(cursor.next_sample_index(), 0);
        assert_eq!(
            cursor.decoder_config().unwrap().unwrap().bitstream_format,
            BitstreamFormat::LengthPrefixed { length_size: 4 }
        );
        assert!(cursor.next_packet().unwrap().is_some());
        assert_eq!(cursor.next_sample_index(), 1);
        assert_eq!(
            cursor
                .seek_to_keyframe(MediaTime::from_parts(-5_000, 90_000).unwrap())
                .unwrap(),
            Some(0)
        );
        assert_eq!(cursor.next_sample_index(), 0);
        assert_eq!(
            cursor
                .seek_to_keyframe(MediaTime::from_parts(-3_000, 90_000).unwrap())
                .unwrap(),
            Some(2)
        );
        assert_eq!(cursor.next_sample_index(), 2);
        assert_eq!(
            cursor
                .seek_to_keyframe_at_or_after(MediaTime::from_parts(-8_999, 90_000).unwrap())
                .unwrap(),
            Some(2)
        );
        assert_eq!(cursor.next_sample_index(), 2);
        assert_eq!(
            cursor
                .seek_to_nearest_keyframe(MediaTime::from_parts(-7_000, 90_000).unwrap())
                .unwrap(),
            Some(0)
        );
        assert_eq!(cursor.next_sample_index(), 0);
        assert_eq!(
            cursor
                .seek_to_nearest_keyframe(MediaTime::from_parts(-5_000, 90_000).unwrap())
                .unwrap(),
            Some(2)
        );
        assert_eq!(cursor.next_sample_index(), 2);
        assert!(cursor.next_packet().unwrap().is_some());
        assert!(cursor.next_packet().unwrap().is_none());
        assert!(cursor.decoder_config().unwrap().is_none());
        cursor.seek_to_sample(1).unwrap();
        assert_eq!(cursor.next_sample_index(), 1);
        assert!(cursor.next_packet().unwrap().is_some());
        assert_eq!(cursor.next_sample_index(), 2);
        cursor.seek_to_sample(3).unwrap();
        assert!(cursor.next_packet().unwrap().is_none());
        assert!(matches!(
            cursor.seek_to_sample(4),
            Err(Mp4Error::IndexOutOfRange {
                kind: "sample cursor",
                index: 4
            })
        ));
        cursor.rewind();
        assert_eq!(cursor.next_sample_index(), 0);
    }

    fn exercise_demuxer(bytes: Vec<u8>) {
        let Ok(demuxer) = crate::Mp4Demuxer::open(MemoryInput(bytes)) else {
            return;
        };

        for (track_index, track) in demuxer.movie().tracks().iter().enumerate() {
            for sample_index in 0..track.samples().len() {
                let _ = track.decoder_config_for_sample(sample_index);
                let _ = demuxer.read_packet(track_index, sample_index);
            }
        }
    }

    #[test]
    fn truncated_or_single_byte_corrupted_movies_do_not_panic() {
        let valid = synthetic_movie();

        for end in 0..=valid.len() {
            exercise_demuxer(valid[..end].to_vec());
        }

        for index in 0..valid.len() {
            for mask in [0x01, 0x80, 0xff] {
                let mut corrupted = valid.clone();
                corrupted[index] ^= mask;
                exercise_demuxer(corrupted);
            }
        }
    }
}
