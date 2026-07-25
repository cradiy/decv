use crate::{
    BoxHeader, FourCc, Mp4Error, Mp4File, Result, Track,
    reader::{BoundedReader, read_full_box},
    sample_table::{MAX_SAMPLE_COUNT, Sample},
};

const MVEX: FourCc = FourCc::new(*b"mvex");
const TREX: FourCc = FourCc::new(*b"trex");
const MOOF: FourCc = FourCc::new(*b"moof");
const TRAF: FourCc = FourCc::new(*b"traf");
const TFHD: FourCc = FourCc::new(*b"tfhd");
const TFDT: FourCc = FourCc::new(*b"tfdt");
const TRUN: FourCc = FourCc::new(*b"trun");

const TFHD_BASE_DATA_OFFSET_PRESENT: u32 = 0x000001;
const TFHD_SAMPLE_DESCRIPTION_INDEX_PRESENT: u32 = 0x000002;
const TFHD_DEFAULT_SAMPLE_DURATION_PRESENT: u32 = 0x000008;
const TFHD_DEFAULT_SAMPLE_SIZE_PRESENT: u32 = 0x000010;
const TFHD_DEFAULT_SAMPLE_FLAGS_PRESENT: u32 = 0x000020;
const TFHD_DURATION_IS_EMPTY: u32 = 0x010000;
const TFHD_DEFAULT_BASE_IS_MOOF: u32 = 0x020000;
const TFHD_KNOWN_FLAGS: u32 = TFHD_BASE_DATA_OFFSET_PRESENT
    | TFHD_SAMPLE_DESCRIPTION_INDEX_PRESENT
    | TFHD_DEFAULT_SAMPLE_DURATION_PRESENT
    | TFHD_DEFAULT_SAMPLE_SIZE_PRESENT
    | TFHD_DEFAULT_SAMPLE_FLAGS_PRESENT
    | TFHD_DURATION_IS_EMPTY
    | TFHD_DEFAULT_BASE_IS_MOOF;

const TRUN_DATA_OFFSET_PRESENT: u32 = 0x000001;
const TRUN_FIRST_SAMPLE_FLAGS_PRESENT: u32 = 0x000004;
const TRUN_SAMPLE_DURATION_PRESENT: u32 = 0x000100;
const TRUN_SAMPLE_SIZE_PRESENT: u32 = 0x000200;
const TRUN_SAMPLE_FLAGS_PRESENT: u32 = 0x000400;
const TRUN_SAMPLE_COMPOSITION_TIME_OFFSET_PRESENT: u32 = 0x000800;
const TRUN_KNOWN_FLAGS: u32 = TRUN_DATA_OFFSET_PRESENT
    | TRUN_FIRST_SAMPLE_FLAGS_PRESENT
    | TRUN_SAMPLE_DURATION_PRESENT
    | TRUN_SAMPLE_SIZE_PRESENT
    | TRUN_SAMPLE_FLAGS_PRESENT
    | TRUN_SAMPLE_COMPOSITION_TIME_OFFSET_PRESENT;

const SAMPLE_IS_NON_SYNC: u32 = 0x0001_0000;

#[derive(Debug, Clone, Copy)]
struct TrackExtends {
    track_id: u32,
    description_index: u32,
    duration: u32,
    size: u32,
    flags: u32,
}

#[derive(Debug, Clone, Copy)]
struct TrackFragmentHeader {
    track_id: u32,
    base_data_offset: Option<u64>,
    description_index: Option<u32>,
    default_duration: Option<u32>,
    default_size: Option<u32>,
    default_flags: Option<u32>,
    duration_is_empty: bool,
    default_base_is_moof: bool,
}

pub(crate) fn find_movie_extends(
    file: Mp4File<'_>,
    movie_box: BoxHeader,
) -> Result<Option<BoxHeader>> {
    let mut movie_extends = None;
    for child in file.children(movie_box)? {
        let child = child?;
        if child.kind() == MVEX {
            set_once(&mut movie_extends, child, "duplicate mvex box")?;
        }
    }
    Ok(movie_extends)
}

pub(crate) fn append_fragmented_samples(
    file: Mp4File<'_>,
    movie_extends: Option<BoxHeader>,
    tracks: &mut [Track],
) -> Result<()> {
    let Some(movie_extends) = movie_extends else {
        return Ok(());
    };
    let defaults = parse_track_extends(file, movie_extends)?;
    let mut next_decode_times = tracks
        .iter()
        .map(|track| {
            let next = track
                .samples()
                .last()
                .map(|sample| {
                    sample
                        .decode_time()
                        .checked_add(i64::from(sample.duration()))
                        .ok_or(Mp4Error::IntegerOverflow)
                })
                .transpose()?
                .unwrap_or(0);
            Ok((track.id(), next))
        })
        .collect::<Result<Vec<_>>>()?;

    for header in file.boxes() {
        let header = header?;
        if header.kind() != MOOF {
            continue;
        }
        parse_movie_fragment(file, header, &defaults, tracks, &mut next_decode_times)?;
    }
    for track in tracks {
        track.rebuild_sync_sample_indices();
    }
    Ok(())
}

fn parse_track_extends(file: Mp4File<'_>, movie_extends: BoxHeader) -> Result<Vec<TrackExtends>> {
    let mut defaults = Vec::new();
    for child in file.children(movie_extends)? {
        let child = child?;
        if child.kind() != TREX {
            continue;
        }
        let mut reader = payload_reader(file, child)?;
        let (version, flags) = read_full_box(&mut reader)?;
        if version != 0 || flags != 0 {
            return Err(Mp4Error::InvalidData("unsupported trex version or flags"));
        }
        let entry = TrackExtends {
            track_id: reader.read_u32()?,
            description_index: reader.read_u32()?,
            duration: reader.read_u32()?,
            size: reader.read_u32()?,
            flags: reader.read_u32()?,
        };
        require_finished(&reader, "trex has trailing bytes")?;
        if entry.track_id == 0 {
            return Err(Mp4Error::InvalidData("trex track id is zero"));
        }
        if entry.description_index == 0 {
            return Err(Mp4Error::InvalidData(
                "trex sample-description index is zero",
            ));
        }
        if defaults
            .iter()
            .any(|previous: &TrackExtends| previous.track_id == entry.track_id)
        {
            return Err(Mp4Error::InvalidData("duplicate trex track id"));
        }
        defaults.push(entry);
    }
    Ok(defaults)
}

fn parse_movie_fragment(
    file: Mp4File<'_>,
    movie_fragment: BoxHeader,
    defaults: &[TrackExtends],
    tracks: &mut [Track],
    next_decode_times: &mut [(u32, i64)],
) -> Result<()> {
    for child in file.children(movie_fragment)? {
        let child = child?;
        if child.kind() != TRAF {
            continue;
        }
        parse_track_fragment(
            file,
            movie_fragment,
            child,
            defaults,
            tracks,
            next_decode_times,
        )?;
    }
    Ok(())
}

fn parse_track_fragment(
    file: Mp4File<'_>,
    movie_fragment: BoxHeader,
    track_fragment: BoxHeader,
    defaults: &[TrackExtends],
    tracks: &mut [Track],
    next_decode_times: &mut [(u32, i64)],
) -> Result<()> {
    let mut track_header_box = None;
    let mut decode_time_box = None;
    let mut run_boxes = Vec::new();
    for child in file.children(track_fragment)? {
        let child = child?;
        match child.kind() {
            TFHD => set_once(&mut track_header_box, child, "duplicate tfhd box")?,
            TFDT => set_once(&mut decode_time_box, child, "duplicate tfdt box")?,
            TRUN => run_boxes.push(child),
            _ => {}
        }
    }
    let track_header = parse_track_fragment_header(
        file,
        track_header_box.ok_or(Mp4Error::InvalidData("traf has no tfhd box"))?,
    )?;
    let track_index = tracks
        .iter()
        .position(|track| track.id() == track_header.track_id)
        .ok_or(Mp4Error::InvalidData(
            "fragment references an unknown track",
        ))?;

    // Tracks whose media type is not indexed yet must not prevent a supported
    // video track in the same movie from being used.
    if tracks[track_index].sample_descriptions().is_empty() {
        return Ok(());
    }
    if run_boxes.is_empty() {
        if track_header.duration_is_empty {
            return Ok(());
        }
        return Err(Mp4Error::InvalidData("non-empty traf has no trun box"));
    }
    if track_header.duration_is_empty {
        return Err(Mp4Error::InvalidData(
            "duration-is-empty traf contains sample runs",
        ));
    }

    let track_defaults = defaults
        .iter()
        .find(|entry| entry.track_id == track_header.track_id)
        .copied();
    let description_index = track_header
        .description_index
        .or(track_defaults.map(|entry| entry.description_index))
        .ok_or(Mp4Error::InvalidData(
            "fragment has no sample-description index",
        ))?;
    let description_index =
        usize::try_from(description_index - 1).map_err(|_| Mp4Error::IntegerOverflow)?;
    if description_index >= tracks[track_index].sample_descriptions().len() {
        return Err(Mp4Error::InvalidData(
            "fragment sample-description index is out of range",
        ));
    }

    let next_decode_time = next_decode_times
        .iter_mut()
        .find(|(track_id, _)| *track_id == track_header.track_id)
        .ok_or(Mp4Error::InvalidData(
            "fragment decode-time state has no matching track",
        ))?;
    let mut decode_time = decode_time_box
        .map(|header| parse_track_fragment_decode_time(file, header))
        .transpose()?
        .unwrap_or(next_decode_time.1);
    let base_data_offset = match (
        track_header.base_data_offset,
        track_header.default_base_is_moof,
    ) {
        (Some(offset), _) => offset,
        (None, true) => movie_fragment.offset(),
        (None, false) => {
            return Err(Mp4Error::UnsupportedFeature(
                "fragment tfhd has no explicit or moof-relative data base",
            ));
        }
    };
    let default_duration = track_header
        .default_duration
        .or(track_defaults.map(|entry| entry.duration));
    let default_size = track_header
        .default_size
        .or(track_defaults.map(|entry| entry.size));
    let default_flags = track_header
        .default_flags
        .or(track_defaults.map(|entry| entry.flags));

    let mut data_end = None;
    let mut samples = Vec::new();
    for run in run_boxes {
        parse_track_run(
            file,
            run,
            base_data_offset,
            &mut data_end,
            &mut decode_time,
            description_index,
            default_duration,
            default_size,
            default_flags,
            &mut samples,
        )?;
    }
    if tracks[track_index]
        .samples()
        .len()
        .checked_add(samples.len())
        .ok_or(Mp4Error::IntegerOverflow)?
        > MAX_SAMPLE_COUNT
    {
        return Err(Mp4Error::InvalidData(
            "fragment sample count exceeds its limit",
        ));
    }
    tracks[track_index].extend_fragment_samples(samples);
    next_decode_time.1 = decode_time;
    Ok(())
}

fn parse_track_fragment_header(
    file: Mp4File<'_>,
    header: BoxHeader,
) -> Result<TrackFragmentHeader> {
    let mut reader = payload_reader(file, header)?;
    let (version, flags) = read_full_box(&mut reader)?;
    if version != 0 || flags & !TFHD_KNOWN_FLAGS != 0 {
        return Err(Mp4Error::InvalidData("unsupported tfhd version or flags"));
    }
    if flags & TFHD_BASE_DATA_OFFSET_PRESENT != 0 && flags & TFHD_DEFAULT_BASE_IS_MOOF != 0 {
        return Err(Mp4Error::InvalidData("tfhd declares two data-base modes"));
    }
    let track_id = reader.read_u32()?;
    if track_id == 0 {
        return Err(Mp4Error::InvalidData("tfhd track id is zero"));
    }
    let value = TrackFragmentHeader {
        track_id,
        base_data_offset: (flags & TFHD_BASE_DATA_OFFSET_PRESENT != 0)
            .then(|| reader.read_u64())
            .transpose()?,
        description_index: (flags & TFHD_SAMPLE_DESCRIPTION_INDEX_PRESENT != 0)
            .then(|| reader.read_u32())
            .transpose()?,
        default_duration: (flags & TFHD_DEFAULT_SAMPLE_DURATION_PRESENT != 0)
            .then(|| reader.read_u32())
            .transpose()?,
        default_size: (flags & TFHD_DEFAULT_SAMPLE_SIZE_PRESENT != 0)
            .then(|| reader.read_u32())
            .transpose()?,
        default_flags: (flags & TFHD_DEFAULT_SAMPLE_FLAGS_PRESENT != 0)
            .then(|| reader.read_u32())
            .transpose()?,
        duration_is_empty: flags & TFHD_DURATION_IS_EMPTY != 0,
        default_base_is_moof: flags & TFHD_DEFAULT_BASE_IS_MOOF != 0,
    };
    require_finished(&reader, "tfhd has trailing bytes")?;
    if value.description_index == Some(0) {
        return Err(Mp4Error::InvalidData(
            "tfhd sample-description index is zero",
        ));
    }
    Ok(value)
}

fn parse_track_fragment_decode_time(file: Mp4File<'_>, header: BoxHeader) -> Result<i64> {
    let mut reader = payload_reader(file, header)?;
    let (version, flags) = read_full_box(&mut reader)?;
    if flags != 0 || version > 1 {
        return Err(Mp4Error::InvalidData("unsupported tfdt version or flags"));
    }
    let decode_time = if version == 0 {
        u64::from(reader.read_u32()?)
    } else {
        reader.read_u64()?
    };
    require_finished(&reader, "tfdt has trailing bytes")?;
    i64::try_from(decode_time).map_err(|_| Mp4Error::InvalidData("tfdt exceeds i64"))
}

#[allow(clippy::too_many_arguments)]
fn parse_track_run(
    file: Mp4File<'_>,
    header: BoxHeader,
    base_data_offset: u64,
    previous_data_end: &mut Option<u64>,
    decode_time: &mut i64,
    description_index: usize,
    default_duration: Option<u32>,
    default_size: Option<u32>,
    default_flags: Option<u32>,
    output: &mut Vec<Sample>,
) -> Result<()> {
    let mut reader = payload_reader(file, header)?;
    let (version, flags) = read_full_box(&mut reader)?;
    if version > 1 || flags & !TRUN_KNOWN_FLAGS != 0 {
        return Err(Mp4Error::InvalidData("unsupported trun version or flags"));
    }
    if flags & TRUN_FIRST_SAMPLE_FLAGS_PRESENT != 0 && flags & TRUN_SAMPLE_FLAGS_PRESENT != 0 {
        return Err(Mp4Error::InvalidData(
            "trun has both first-sample and per-sample flags",
        ));
    }
    let sample_count =
        usize::try_from(reader.read_u32()?).map_err(|_| Mp4Error::IntegerOverflow)?;
    if output
        .len()
        .checked_add(sample_count)
        .ok_or(Mp4Error::IntegerOverflow)?
        > MAX_SAMPLE_COUNT
    {
        return Err(Mp4Error::InvalidData(
            "fragment sample count exceeds its limit",
        ));
    }
    let data_offset = (flags & TRUN_DATA_OFFSET_PRESENT != 0)
        .then(|| reader.read_i32())
        .transpose()?;
    let first_sample_flags = (flags & TRUN_FIRST_SAMPLE_FLAGS_PRESENT != 0)
        .then(|| reader.read_u32())
        .transpose()?;
    let mut sample_offset = if let Some(data_offset) = data_offset {
        add_signed(base_data_offset, data_offset)?
    } else {
        previous_data_end.ok_or(Mp4Error::UnsupportedFeature(
            "first trun has no data offset",
        ))?
    };

    for sample_index in 0..sample_count {
        let duration = if flags & TRUN_SAMPLE_DURATION_PRESENT != 0 {
            reader.read_u32()?
        } else {
            default_duration.ok_or(Mp4Error::InvalidData("fragment sample has no duration"))?
        };
        let size = if flags & TRUN_SAMPLE_SIZE_PRESENT != 0 {
            reader.read_u32()?
        } else {
            default_size.ok_or(Mp4Error::InvalidData("fragment sample has no size"))?
        };
        let sample_flags = if flags & TRUN_SAMPLE_FLAGS_PRESENT != 0 {
            reader.read_u32()?
        } else if sample_index == 0 {
            first_sample_flags
                .or(default_flags)
                .ok_or(Mp4Error::InvalidData(
                    "fragment sample has no dependency flags",
                ))?
        } else {
            default_flags.ok_or(Mp4Error::InvalidData(
                "fragment sample has no dependency flags",
            ))?
        };
        let composition_offset = if flags & TRUN_SAMPLE_COMPOSITION_TIME_OFFSET_PRESENT == 0 {
            0
        } else if version == 0 {
            i64::from(reader.read_u32()?)
        } else {
            i64::from(reader.read_i32()?)
        };
        if duration == 0 {
            return Err(Mp4Error::InvalidData("fragment sample duration is zero"));
        }
        if size == 0 {
            return Err(Mp4Error::InvalidData("fragment sample size is zero"));
        }
        let sample_end = sample_offset
            .checked_add(u64::from(size))
            .ok_or(Mp4Error::IntegerOverflow)?;
        if sample_end > file.length() {
            return Err(Mp4Error::InvalidData(
                "fragment sample data exceeds the input",
            ));
        }
        let presentation_time = decode_time
            .checked_add(composition_offset)
            .ok_or(Mp4Error::IntegerOverflow)?;
        output.push(Sample::new_fragment(
            sample_offset,
            size,
            *decode_time,
            presentation_time,
            duration,
            description_index,
            sample_flags & SAMPLE_IS_NON_SYNC == 0,
        ));
        sample_offset = sample_end;
        *decode_time = decode_time
            .checked_add(i64::from(duration))
            .ok_or(Mp4Error::IntegerOverflow)?;
    }
    require_finished(&reader, "trun has trailing bytes")?;
    *previous_data_end = Some(sample_offset);
    Ok(())
}

fn add_signed(base: u64, offset: i32) -> Result<u64> {
    if offset >= 0 {
        base.checked_add(u64::from(offset.unsigned_abs()))
    } else {
        base.checked_sub(u64::from(offset.unsigned_abs()))
    }
    .ok_or(Mp4Error::InvalidData(
        "fragment data offset exceeds the input",
    ))
}

fn payload_reader(file: Mp4File<'_>, header: BoxHeader) -> Result<BoundedReader<'_>> {
    let range = header.payload_range()?;
    BoundedReader::new(file.input(), range.start, range.end)
}

fn require_finished(reader: &BoundedReader<'_>, message: &'static str) -> Result<()> {
    if reader.remaining()? != 0 {
        return Err(Mp4Error::InvalidData(message));
    }
    Ok(())
}

fn set_once<T>(slot: &mut Option<T>, value: T, duplicate: &'static str) -> Result<()> {
    if slot.replace(value).is_some() {
        return Err(Mp4Error::InvalidData(duplicate));
    }
    Ok(())
}
