use crate::{
    BoxHeader, FourCc, Mp4Error, Mp4File, Result,
    reader::{BoundedReader, read_full_box},
};

const STTS: FourCc = FourCc::new(*b"stts");
const CTTS: FourCc = FourCc::new(*b"ctts");
const STSC: FourCc = FourCc::new(*b"stsc");
const STSZ: FourCc = FourCc::new(*b"stsz");
const STZ2: FourCc = FourCc::new(*b"stz2");
const STCO: FourCc = FourCc::new(*b"stco");
const CO64: FourCc = FourCc::new(*b"co64");
const STSS: FourCc = FourCc::new(*b"stss");

const MAX_SAMPLE_COUNT: usize = 2_000_000;
const MAX_TABLE_ENTRY_COUNT: usize = 2_000_000;

/// One compressed sample located and timed by an MP4 sample table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Sample {
    offset: u64,
    size: u32,
    decode_time: i64,
    presentation_time: i64,
    duration: u32,
    description_index: usize,
    sync: bool,
}

impl Sample {
    #[inline]
    pub const fn offset(self) -> u64 {
        self.offset
    }

    #[inline]
    pub const fn size(self) -> u32 {
        self.size
    }

    #[inline]
    pub const fn decode_time(self) -> i64 {
        self.decode_time
    }

    #[inline]
    pub const fn presentation_time(self) -> i64 {
        self.presentation_time
    }

    #[inline]
    pub const fn duration(self) -> u32 {
        self.duration
    }

    /// Zero-based index into the track's sample-description list.
    #[inline]
    pub const fn description_index(self) -> usize {
        self.description_index
    }

    #[inline]
    pub const fn is_sync(self) -> bool {
        self.sync
    }
}

#[derive(Debug, Clone, Copy)]
struct TimeToSampleEntry {
    count: u32,
    delta: u32,
}

#[derive(Debug, Clone, Copy)]
struct CompositionOffsetEntry {
    count: u32,
    offset: i64,
}

#[derive(Debug, Clone, Copy)]
struct SampleToChunkEntry {
    first_chunk: u32,
    samples_per_chunk: u32,
    description_index: u32,
}

pub(crate) fn parse_sample_table(
    file: Mp4File<'_>,
    sample_table: BoxHeader,
    description_count: usize,
) -> Result<Vec<Sample>> {
    if description_count == 0 {
        return Err(Mp4Error::InvalidData(
            "sample table has no sample descriptions",
        ));
    }

    let mut time_to_sample = None;
    let mut composition_offsets = None;
    let mut sample_to_chunk = None;
    let mut sample_sizes = None;
    let mut chunk_offsets = None;
    let mut sync_samples = None;
    for child in file.children(sample_table)? {
        let child = child?;
        match child.kind() {
            STTS => set_once(&mut time_to_sample, child, "duplicate stts box")?,
            CTTS => set_once(&mut composition_offsets, child, "duplicate ctts box")?,
            STSC => set_once(&mut sample_to_chunk, child, "duplicate stsc box")?,
            STSZ | STZ2 => set_once(&mut sample_sizes, child, "multiple sample-size boxes")?,
            STCO | CO64 => set_once(&mut chunk_offsets, child, "multiple chunk-offset boxes")?,
            STSS => set_once(&mut sync_samples, child, "duplicate stss box")?,
            _ => {}
        }
    }

    let sample_sizes = parse_sample_sizes(
        file,
        sample_sizes.ok_or(Mp4Error::InvalidData("sample table has no sample-size box"))?,
    )?;
    let sample_count = sample_sizes.len();
    let time_to_sample = parse_time_to_sample(
        file,
        time_to_sample.ok_or(Mp4Error::InvalidData("sample table has no stts box"))?,
        sample_count,
    )?;
    let composition_offsets = composition_offsets
        .map(|header| parse_composition_offsets(file, header, sample_count))
        .transpose()?
        .unwrap_or_else(|| {
            vec![CompositionOffsetEntry {
                count: u32::try_from(sample_count).expect("sample count capped at two million"),
                offset: 0,
            }]
        });
    let sample_to_chunk = parse_sample_to_chunk(
        file,
        sample_to_chunk.ok_or(Mp4Error::InvalidData("sample table has no stsc box"))?,
        description_count,
    )?;
    let chunk_offsets = parse_chunk_offsets(
        file,
        chunk_offsets.ok_or(Mp4Error::InvalidData(
            "sample table has no chunk-offset box",
        ))?,
    )?;
    let sync = parse_sync_samples(file, sync_samples, sample_count)?;

    let mut composition_run = 0usize;
    let mut composition_remaining = composition_offsets.first().map_or(0, |entry| entry.count);
    let mut decode_time = 0u64;
    let mut sample_index = 0usize;
    let mut samples = Vec::with_capacity(sample_count);
    for entry in time_to_sample {
        for _ in 0..entry.count {
            let composition_offset = composition_offsets
                .get(composition_run)
                .ok_or(Mp4Error::InvalidData("ctts has too few samples"))?
                .offset;
            let decode_time_i64 = i64::try_from(decode_time)
                .map_err(|_| Mp4Error::InvalidData("sample DTS exceeds i64"))?;
            let presentation_time = i128::from(decode_time_i64)
                .checked_add(i128::from(composition_offset))
                .and_then(|value| i64::try_from(value).ok())
                .ok_or(Mp4Error::InvalidData("sample PTS exceeds i64"))?;
            samples.push(Sample {
                offset: 0,
                size: sample_sizes[sample_index],
                decode_time: decode_time_i64,
                presentation_time,
                duration: entry.delta,
                description_index: 0,
                sync: sync[sample_index],
            });
            sample_index += 1;
            decode_time = decode_time
                .checked_add(u64::from(entry.delta))
                .ok_or(Mp4Error::IntegerOverflow)?;

            composition_remaining -= 1;
            if composition_remaining == 0 && sample_index < sample_count {
                composition_run += 1;
                composition_remaining = composition_offsets
                    .get(composition_run)
                    .ok_or(Mp4Error::InvalidData("ctts has too few samples"))?
                    .count;
            }
        }
    }
    if sample_index != sample_count {
        return Err(Mp4Error::InvalidData(
            "stts sample count does not match stsz",
        ));
    }

    assign_sample_offsets(
        &mut samples,
        &sample_to_chunk,
        &chunk_offsets,
        file.length(),
    )?;
    Ok(samples)
}

fn parse_time_to_sample(
    file: Mp4File<'_>,
    header: BoxHeader,
    sample_count: usize,
) -> Result<Vec<TimeToSampleEntry>> {
    let mut reader = payload_reader(file, header)?;
    require_full_box_version_zero(&mut reader, "stts")?;
    let entry_count = read_count(&mut reader, MAX_TABLE_ENTRY_COUNT, "stts")?;
    let mut entries = Vec::with_capacity(entry_count);
    let mut total = 0u64;
    for _ in 0..entry_count {
        let count = reader.read_u32()?;
        if count == 0 {
            return Err(Mp4Error::InvalidData("stts entry has zero samples"));
        }
        let delta = reader.read_u32()?;
        total = total
            .checked_add(u64::from(count))
            .ok_or(Mp4Error::IntegerOverflow)?;
        entries.push(TimeToSampleEntry { count, delta });
    }
    require_finished(&reader, "stts has trailing bytes")?;
    if total != u64::try_from(sample_count).map_err(|_| Mp4Error::IntegerOverflow)? {
        return Err(Mp4Error::InvalidData(
            "stts sample count does not match stsz",
        ));
    }
    Ok(entries)
}

fn parse_composition_offsets(
    file: Mp4File<'_>,
    header: BoxHeader,
    sample_count: usize,
) -> Result<Vec<CompositionOffsetEntry>> {
    let mut reader = payload_reader(file, header)?;
    let (version, flags) = read_full_box(&mut reader)?;
    if flags != 0 || version > 1 {
        return Err(Mp4Error::InvalidData("unsupported ctts version or flags"));
    }
    let entry_count = read_count(&mut reader, MAX_TABLE_ENTRY_COUNT, "ctts")?;
    let mut entries = Vec::with_capacity(entry_count);
    let mut total = 0u64;
    for _ in 0..entry_count {
        let count = reader.read_u32()?;
        if count == 0 {
            return Err(Mp4Error::InvalidData("ctts entry has zero samples"));
        }
        let offset = if version == 0 {
            i64::from(reader.read_u32()?)
        } else {
            i64::from(reader.read_i32()?)
        };
        total = total
            .checked_add(u64::from(count))
            .ok_or(Mp4Error::IntegerOverflow)?;
        entries.push(CompositionOffsetEntry { count, offset });
    }
    require_finished(&reader, "ctts has trailing bytes")?;
    if total != u64::try_from(sample_count).map_err(|_| Mp4Error::IntegerOverflow)? {
        return Err(Mp4Error::InvalidData(
            "ctts sample count does not match stsz",
        ));
    }
    Ok(entries)
}

fn parse_sample_to_chunk(
    file: Mp4File<'_>,
    header: BoxHeader,
    description_count: usize,
) -> Result<Vec<SampleToChunkEntry>> {
    let mut reader = payload_reader(file, header)?;
    require_full_box_version_zero(&mut reader, "stsc")?;
    let entry_count = read_count(&mut reader, MAX_TABLE_ENTRY_COUNT, "stsc")?;
    let mut entries = Vec::with_capacity(entry_count);
    for index in 0..entry_count {
        let first_chunk = reader.read_u32()?;
        let samples_per_chunk = reader.read_u32()?;
        let description_index = reader.read_u32()?;
        if (index == 0 && first_chunk != 1)
            || entries
                .last()
                .is_some_and(|previous: &SampleToChunkEntry| first_chunk <= previous.first_chunk)
        {
            return Err(Mp4Error::InvalidData(
                "stsc first_chunk values are not strictly increasing from one",
            ));
        }
        if samples_per_chunk == 0 {
            return Err(Mp4Error::InvalidData("stsc entry has zero samples"));
        }
        if description_index == 0
            || usize::try_from(description_index).map_err(|_| Mp4Error::IntegerOverflow)?
                > description_count
        {
            return Err(Mp4Error::InvalidData(
                "stsc sample-description index is out of range",
            ));
        }
        entries.push(SampleToChunkEntry {
            first_chunk,
            samples_per_chunk,
            description_index,
        });
    }
    require_finished(&reader, "stsc has trailing bytes")?;
    Ok(entries)
}

fn parse_sample_sizes(file: Mp4File<'_>, header: BoxHeader) -> Result<Vec<u32>> {
    if header.kind() == STZ2 {
        return parse_compact_sample_sizes(file, header);
    }
    let mut reader = payload_reader(file, header)?;
    require_full_box_version_zero(&mut reader, "stsz")?;
    let default_size = reader.read_u32()?;
    let sample_count =
        usize::try_from(reader.read_u32()?).map_err(|_| Mp4Error::IntegerOverflow)?;
    if sample_count > MAX_SAMPLE_COUNT {
        return Err(Mp4Error::InvalidData("sample count exceeds its limit"));
    }
    let sizes = if default_size == 0 {
        let mut sizes = Vec::with_capacity(sample_count);
        for _ in 0..sample_count {
            sizes.push(reader.read_u32()?);
        }
        sizes
    } else {
        vec![default_size; sample_count]
    };
    require_finished(&reader, "stsz has trailing bytes")?;
    Ok(sizes)
}

fn parse_compact_sample_sizes(file: Mp4File<'_>, header: BoxHeader) -> Result<Vec<u32>> {
    let mut reader = payload_reader(file, header)?;
    require_full_box_version_zero(&mut reader, "stz2")?;
    if reader.read_u24()? != 0 {
        return Err(Mp4Error::InvalidData("stz2 reserved bits are not zero"));
    }
    let field_size = reader.read_u8()?;
    if !matches!(field_size, 4 | 8 | 16) {
        return Err(Mp4Error::InvalidData(
            "stz2 field size is not 4, 8, or 16 bits",
        ));
    }
    let sample_count =
        usize::try_from(reader.read_u32()?).map_err(|_| Mp4Error::IntegerOverflow)?;
    if sample_count > MAX_SAMPLE_COUNT {
        return Err(Mp4Error::InvalidData("sample count exceeds its limit"));
    }

    let mut sizes = Vec::with_capacity(sample_count);
    match field_size {
        4 => {
            for index in 0..sample_count {
                if index & 1 == 0 {
                    let packed = reader.read_u8()?;
                    sizes.push(u32::from(packed >> 4));
                    if index + 1 < sample_count {
                        sizes.push(u32::from(packed & 0x0f));
                    }
                }
            }
        }
        8 => {
            for _ in 0..sample_count {
                sizes.push(u32::from(reader.read_u8()?));
            }
        }
        16 => {
            for _ in 0..sample_count {
                sizes.push(u32::from(reader.read_u16()?));
            }
        }
        _ => unreachable!("field size was validated above"),
    }
    require_finished(&reader, "stz2 has trailing bytes")?;
    Ok(sizes)
}

fn parse_chunk_offsets(file: Mp4File<'_>, header: BoxHeader) -> Result<Vec<u64>> {
    let mut reader = payload_reader(file, header)?;
    require_full_box_version_zero(&mut reader, "chunk offsets")?;
    let entry_count = read_count(&mut reader, MAX_TABLE_ENTRY_COUNT, "chunk offsets")?;
    let mut offsets = Vec::with_capacity(entry_count);
    for _ in 0..entry_count {
        offsets.push(if header.kind() == STCO {
            u64::from(reader.read_u32()?)
        } else {
            reader.read_u64()?
        });
    }
    require_finished(&reader, "chunk-offset box has trailing bytes")?;
    Ok(offsets)
}

fn parse_sync_samples(
    file: Mp4File<'_>,
    header: Option<BoxHeader>,
    sample_count: usize,
) -> Result<Vec<bool>> {
    let Some(header) = header else {
        return Ok(vec![true; sample_count]);
    };
    let mut reader = payload_reader(file, header)?;
    require_full_box_version_zero(&mut reader, "stss")?;
    let entry_count = read_count(&mut reader, sample_count, "stss")?;
    let mut sync = vec![false; sample_count];
    let mut previous = 0u32;
    for _ in 0..entry_count {
        let sample_number = reader.read_u32()?;
        if sample_number == 0
            || sample_number <= previous
            || usize::try_from(sample_number).map_err(|_| Mp4Error::IntegerOverflow)? > sample_count
        {
            return Err(Mp4Error::InvalidData(
                "stss sample numbers are not strictly increasing and in range",
            ));
        }
        sync[usize::try_from(sample_number - 1).map_err(|_| Mp4Error::IntegerOverflow)?] = true;
        previous = sample_number;
    }
    require_finished(&reader, "stss has trailing bytes")?;
    Ok(sync)
}

fn assign_sample_offsets(
    samples: &mut [Sample],
    entries: &[SampleToChunkEntry],
    chunk_offsets: &[u64],
    file_length: u64,
) -> Result<()> {
    if samples.is_empty() {
        if entries.is_empty() && chunk_offsets.is_empty() {
            return Ok(());
        }
        return Err(Mp4Error::InvalidData(
            "chunk table describes samples absent from stsz",
        ));
    }
    if chunk_offsets.is_empty() {
        return Err(Mp4Error::InvalidData("non-empty track has no chunks"));
    }
    if entries.last().is_some_and(|entry| {
        u64::from(entry.first_chunk) > u64::try_from(chunk_offsets.len()).unwrap_or(u64::MAX)
    }) {
        return Err(Mp4Error::InvalidData(
            "stsc first_chunk exceeds the chunk count",
        ));
    }

    let mut sample_index = 0usize;
    let mut entry_index = 0usize;
    for (chunk_index, &chunk_offset) in chunk_offsets.iter().enumerate() {
        let chunk_number = u32::try_from(chunk_index + 1).map_err(|_| Mp4Error::IntegerOverflow)?;
        while entries
            .get(entry_index + 1)
            .is_some_and(|next| next.first_chunk <= chunk_number)
        {
            entry_index += 1;
        }
        let entry = entries
            .get(entry_index)
            .ok_or(Mp4Error::InvalidData("stsc does not describe a chunk"))?;
        let mut offset = chunk_offset;
        for _ in 0..entry.samples_per_chunk {
            let sample = samples.get_mut(sample_index).ok_or(Mp4Error::InvalidData(
                "chunk table describes more samples than stsz",
            ))?;
            sample.offset = offset;
            sample.description_index = usize::try_from(entry.description_index - 1)
                .map_err(|_| Mp4Error::IntegerOverflow)?;
            offset = offset
                .checked_add(u64::from(sample.size))
                .ok_or(Mp4Error::IntegerOverflow)?;
            if offset > file_length {
                return Err(Mp4Error::InvalidData("sample data exceeds the input"));
            }
            sample_index += 1;
        }
    }
    if sample_index != samples.len() {
        return Err(Mp4Error::InvalidData(
            "chunk table describes fewer samples than stsz",
        ));
    }
    Ok(())
}

fn payload_reader(file: Mp4File<'_>, header: BoxHeader) -> Result<BoundedReader<'_>> {
    let range = header.payload_range()?;
    BoundedReader::new(file.input(), range.start, range.end)
}

fn require_full_box_version_zero(reader: &mut BoundedReader<'_>, name: &'static str) -> Result<()> {
    let (version, flags) = read_full_box(reader)?;
    if version != 0 || flags != 0 {
        return Err(Mp4Error::InvalidData(match name {
            "stts" => "unsupported stts version or flags",
            "stsc" => "unsupported stsc version or flags",
            "stsz" => "unsupported stsz version or flags",
            "stz2" => "unsupported stz2 version or flags",
            "chunk offsets" => "unsupported chunk-offset version or flags",
            "stss" => "unsupported stss version or flags",
            _ => "unsupported full-box version or flags",
        }));
    }
    Ok(())
}

fn read_count(
    reader: &mut BoundedReader<'_>,
    maximum: usize,
    message_name: &'static str,
) -> Result<usize> {
    let count = usize::try_from(reader.read_u32()?).map_err(|_| Mp4Error::IntegerOverflow)?;
    if count > maximum {
        return Err(Mp4Error::InvalidData(match message_name {
            "stts" => "stts entry count exceeds its limit",
            "ctts" => "ctts entry count exceeds its limit",
            "stsc" => "stsc entry count exceeds its limit",
            "chunk offsets" => "chunk count exceeds its limit",
            "stss" => "sync-sample count exceeds the sample count",
            _ => "MP4 table entry count exceeds its limit",
        }));
    }
    Ok(count)
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
            self.0.as_slice().read_at(offset, buffer)
        }
    }

    fn boxed(kind: [u8; 4], payload: &[u8]) -> Vec<u8> {
        let mut bytes = Vec::from(u32::try_from(8 + payload.len()).unwrap().to_be_bytes());
        bytes.extend_from_slice(&kind);
        bytes.extend_from_slice(payload);
        bytes
    }

    fn full(body: &[u8]) -> Vec<u8> {
        let mut bytes = vec![0; 4];
        bytes.extend_from_slice(body);
        bytes
    }

    fn table_box(kind: [u8; 4], entries: &[[u32; 3]]) -> Vec<u8> {
        let mut body = Vec::from(u32::try_from(entries.len()).unwrap().to_be_bytes());
        for entry in entries {
            for value in entry {
                body.extend_from_slice(&value.to_be_bytes());
            }
        }
        boxed(kind, &full(&body))
    }

    fn synthetic_table() -> (MemoryInput, BoxHeader) {
        let mut stts_body = Vec::from(2u32.to_be_bytes());
        stts_body.extend_from_slice(&3u32.to_be_bytes());
        stts_body.extend_from_slice(&100u32.to_be_bytes());
        stts_body.extend_from_slice(&2u32.to_be_bytes());
        stts_body.extend_from_slice(&50u32.to_be_bytes());
        let stts = boxed(*b"stts", &full(&stts_body));
        let mut ctts_body = Vec::from(2u32.to_be_bytes());
        ctts_body.extend_from_slice(&2u32.to_be_bytes());
        ctts_body.extend_from_slice(&10u32.to_be_bytes());
        ctts_body.extend_from_slice(&3u32.to_be_bytes());
        ctts_body.extend_from_slice(&20u32.to_be_bytes());
        let ctts = boxed(*b"ctts", &full(&ctts_body));
        let stsc = table_box(*b"stsc", &[[1, 2, 1], [2, 3, 1]]);
        let mut stsz_body = Vec::from(0u32.to_be_bytes());
        stsz_body.extend_from_slice(&5u32.to_be_bytes());
        for size in [10u32, 11, 12, 13, 14] {
            stsz_body.extend_from_slice(&size.to_be_bytes());
        }
        let stsz = boxed(*b"stsz", &full(&stsz_body));
        let mut stco_body = Vec::from(2u32.to_be_bytes());
        stco_body.extend_from_slice(&1_000u32.to_be_bytes());
        stco_body.extend_from_slice(&2_000u32.to_be_bytes());
        let stco = boxed(*b"stco", &full(&stco_body));
        let mut stss_body = Vec::from(2u32.to_be_bytes());
        stss_body.extend_from_slice(&1u32.to_be_bytes());
        stss_body.extend_from_slice(&4u32.to_be_bytes());
        let stss = boxed(*b"stss", &full(&stss_body));

        let mut payload = stts;
        payload.extend_from_slice(&ctts);
        payload.extend_from_slice(&stsc);
        payload.extend_from_slice(&stsz);
        payload.extend_from_slice(&stco);
        payload.extend_from_slice(&stss);
        let mut bytes = boxed(*b"stbl", &payload);
        bytes.resize(4_096, 0);
        let input = MemoryInput(bytes);
        let file = Mp4File::open(&input).unwrap();
        let header = file.boxes().next().unwrap().unwrap();
        (input, header)
    }

    fn parse_compact_sizes(field_size: u8, sample_count: u32, packed: &[u8]) -> Result<Vec<u32>> {
        let mut body = vec![0, 0, 0, field_size];
        body.extend_from_slice(&sample_count.to_be_bytes());
        body.extend_from_slice(packed);
        let input = MemoryInput(boxed(*b"stz2", &full(&body)));
        let file = Mp4File::open(&input)?;
        let header = file
            .boxes()
            .next()
            .ok_or(Mp4Error::InvalidData("missing test stz2 box"))??;
        parse_sample_sizes(file, header)
    }

    #[test]
    fn combines_timing_chunks_sizes_and_sync_samples() {
        let (input, header) = synthetic_table();
        let samples = parse_sample_table(Mp4File::open(&input).unwrap(), header, 1).unwrap();
        assert_eq!(samples.len(), 5);
        assert_eq!(
            samples
                .iter()
                .map(|sample| (
                    sample.offset(),
                    sample.size(),
                    sample.decode_time(),
                    sample.presentation_time(),
                    sample.duration(),
                    sample.is_sync(),
                ))
                .collect::<Vec<_>>(),
            [
                (1_000, 10, 0, 10, 100, true),
                (1_010, 11, 100, 110, 100, false),
                (2_000, 12, 200, 220, 100, false),
                (2_012, 13, 300, 320, 50, true),
                (2_025, 14, 350, 370, 50, false),
            ]
        );
    }

    #[test]
    fn supports_signed_composition_offsets() {
        let (mut input, _) = synthetic_table();
        let position = input
            .0
            .windows(4)
            .position(|window| window == b"ctts")
            .unwrap();
        input.0[position + 4] = 1;
        input.0[position + 16..position + 20].copy_from_slice(&(-10i32).to_be_bytes());
        let file = Mp4File::open(&input).unwrap();
        let header = file.boxes().next().unwrap().unwrap();
        let samples = parse_sample_table(file, header, 1).unwrap();
        assert_eq!(samples[0].presentation_time(), -10);
        assert_eq!(samples[1].presentation_time(), 90);
    }

    #[test]
    fn parses_each_compact_sample_size_width() {
        assert_eq!(
            parse_compact_sizes(4, 5, &[0x12, 0xf0, 0x70]).unwrap(),
            [1, 2, 15, 0, 7]
        );
        assert_eq!(
            parse_compact_sizes(8, 3, &[1, 200, 0]).unwrap(),
            [1, 200, 0]
        );
        assert_eq!(
            parse_compact_sizes(16, 3, &[0, 1, 1, 44, 0xff, 0xff]).unwrap(),
            [1, 300, 65_535]
        );
    }

    #[test]
    fn rejects_invalid_or_trailing_compact_sample_sizes() {
        assert!(matches!(
            parse_compact_sizes(12, 1, &[0]),
            Err(Mp4Error::InvalidData(
                "stz2 field size is not 4, 8, or 16 bits"
            ))
        ));
        assert!(matches!(
            parse_compact_sizes(8, 1, &[7, 8]),
            Err(Mp4Error::InvalidData("stz2 has trailing bytes"))
        ));
        assert!(parse_compact_sizes(16, 1, &[0]).is_err());
    }

    #[test]
    fn supports_an_empty_sample_table() {
        let empty_count = full(&0u32.to_be_bytes());
        let mut stsz_body = Vec::from(0u32.to_be_bytes());
        stsz_body.extend_from_slice(&0u32.to_be_bytes());

        let mut payload = boxed(*b"stts", &empty_count);
        payload.extend_from_slice(&boxed(*b"stsc", &empty_count));
        payload.extend_from_slice(&boxed(*b"stsz", &full(&stsz_body)));
        payload.extend_from_slice(&boxed(*b"stco", &empty_count));
        let input = MemoryInput(boxed(*b"stbl", &payload));
        let file = Mp4File::open(&input).unwrap();
        let header = file.boxes().next().unwrap().unwrap();
        assert!(parse_sample_table(file, header, 1).unwrap().is_empty());
    }

    #[test]
    fn rejects_missing_sample_descriptions() {
        let (input, header) = synthetic_table();
        let file = Mp4File::open(&input).unwrap();
        assert!(matches!(
            parse_sample_table(file, header, 0),
            Err(Mp4Error::InvalidData(
                "sample table has no sample descriptions"
            ))
        ));
    }
}
