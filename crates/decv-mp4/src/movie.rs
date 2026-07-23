use std::{num::NonZeroU32, sync::Arc};

use crate::{
    BoxHeader, FourCc, Mp4Error, Mp4File, Result,
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
            .map(|header| Track::parse(file, header))
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
    media_timescale: NonZeroU32,
    media_duration: Option<u64>,
    handler: FourCc,
    display_width_16_16: u32,
    display_height_16_16: u32,
    sample_descriptions: Vec<SampleDescription>,
    samples: Vec<Sample>,
}

impl Track {
    fn parse(file: Mp4File<'_>, track_box: BoxHeader) -> Result<Self> {
        let mut track_header = None;
        let mut media_box = None;
        for child in file.children(track_box)? {
            let child = child?;
            match child.kind() {
                TKHD => set_once(&mut track_header, child, "duplicate tkhd box")?,
                MDIA => set_once(&mut media_box, child, "duplicate mdia box")?,
                _ => {}
            }
        }
        let (id, flags, display_width_16_16, display_height_16_16) = parse_track_header(
            file,
            track_header.ok_or(Mp4Error::InvalidData("trak has no tkhd box"))?,
        )?;
        let media_box = media_box.ok_or(Mp4Error::InvalidData("trak has no mdia box"))?;

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

        Ok(Self {
            id,
            flags,
            media_timescale,
            media_duration,
            handler,
            display_width_16_16,
            display_height_16_16,
            sample_descriptions,
            samples,
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

        let avcc = boxed(*b"avcC", &[1, 100, 0, 40, 0xff, 0xe1]);
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
            &full(0, 0, &[1u32, 1, 3_000].map(u32::to_be_bytes).concat()),
        );
        let stsc = boxed(
            *b"stsc",
            &full(0, 0, &[1u32, 1, 1, 1].map(u32::to_be_bytes).concat()),
        );
        let stsz = boxed(
            *b"stsz",
            &full(0, 0, &[1u32, 1].map(u32::to_be_bytes).concat()),
        );
        let stco = boxed(
            *b"stco",
            &full(0, 0, &[1u32, 0].map(u32::to_be_bytes).concat()),
        );
        let mut stbl_payload = boxed(*b"stsd", &stsd);
        stbl_payload.extend_from_slice(&stts);
        stbl_payload.extend_from_slice(&stsc);
        stbl_payload.extend_from_slice(&stsz);
        stbl_payload.extend_from_slice(&stco);
        let stbl = boxed(*b"stbl", &stbl_payload);
        let minf = boxed(*b"minf", &stbl);

        let mut mdia = boxed(*b"mdhd", &full(0, 0, &mdhd));
        mdia.extend_from_slice(&boxed(*b"hdlr", &full(0, 0, &hdlr)));
        mdia.extend_from_slice(&minf);

        let mut trak = boxed(*b"tkhd", &full(0, 3, &tkhd));
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

        let SampleDescription::Avc(entry) = &track.sample_descriptions()[0] else {
            panic!("expected AVC sample entry");
        };
        assert_eq!(entry.format(), AVC1);
        assert_eq!(entry.data_reference_index(), 1);
        assert_eq!((entry.width(), entry.height()), (1_920, 1_080));
        assert_eq!(
            entry.codec_configuration().as_ref(),
            [1, 100, 0, 40, 0xff, 0xe1]
        );
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
}
