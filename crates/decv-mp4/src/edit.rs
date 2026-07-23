use std::num::NonZeroU32;

use crate::{
    BoxHeader, FourCc, Mp4Error, Mp4File, Result,
    reader::{BoundedReader, read_full_box},
};

pub(crate) const EDTS: FourCc = FourCc::new(*b"edts");
const ELST: FourCc = FourCc::new(*b"elst");
const MAX_EDIT_COUNT: usize = 1024;

/// One mapping from a movie-timeline segment to media time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Edit {
    segment_duration: u64,
    media_time: Option<i64>,
    media_rate_integer: i16,
    media_rate_fraction: i16,
}

impl Edit {
    /// Duration in the movie's time scale.
    #[inline]
    pub const fn segment_duration(self) -> u64 {
        self.segment_duration
    }

    /// Start time in the media time scale, or `None` for an empty edit.
    #[inline]
    pub const fn media_time(self) -> Option<i64> {
        self.media_time
    }

    /// Integer half of the signed 16.16 playback rate.
    #[inline]
    pub const fn media_rate_integer(self) -> i16 {
        self.media_rate_integer
    }

    /// Fractional half of the signed 16.16 playback rate.
    #[inline]
    pub const fn media_rate_fraction(self) -> i16 {
        self.media_rate_fraction
    }
}

pub(crate) fn parse_edit_container(file: Mp4File<'_>, edit_box: BoxHeader) -> Result<Vec<Edit>> {
    let mut edit_list = None;
    for child in file.children(edit_box)? {
        let child = child?;
        if child.kind() == ELST && edit_list.replace(child).is_some() {
            return Err(Mp4Error::InvalidData("edts has multiple elst boxes"));
        }
    }
    let Some(edit_list) = edit_list else {
        return Ok(Vec::new());
    };
    parse_edit_list(file, edit_list)
}

fn parse_edit_list(file: Mp4File<'_>, header: BoxHeader) -> Result<Vec<Edit>> {
    let range = header.payload_range()?;
    let mut reader = BoundedReader::new(file.input(), range.start, range.end)?;
    let (version, flags) = read_full_box(&mut reader)?;
    if flags != 0 || version > 1 {
        return Err(Mp4Error::InvalidData("unsupported elst version or flags"));
    }
    let count = usize::try_from(reader.read_u32()?).map_err(|_| Mp4Error::IntegerOverflow)?;
    if count > MAX_EDIT_COUNT {
        return Err(Mp4Error::InvalidData("edit-list count exceeds its limit"));
    }

    let mut edits = Vec::with_capacity(count);
    for _ in 0..count {
        let (segment_duration, media_time) = if version == 0 {
            (u64::from(reader.read_u32()?), i64::from(reader.read_i32()?))
        } else {
            (reader.read_u64()?, reader.read_i64()?)
        };
        if media_time < -1 {
            return Err(Mp4Error::InvalidData(
                "edit-list media time is less than negative one",
            ));
        }
        edits.push(Edit {
            segment_duration,
            media_time: (media_time != -1).then_some(media_time),
            media_rate_integer: reader.read_i16()?,
            media_rate_fraction: reader.read_i16()?,
        });
    }
    if reader.remaining()? != 0 {
        return Err(Mp4Error::InvalidData("elst has trailing bytes"));
    }
    Ok(edits)
}

/// Returns the media-timescale offset for a single linear media edit,
/// optionally preceded by one or more empty edits.
pub(crate) fn linear_timeline_offset(
    edits: &[Edit],
    movie_timescale: NonZeroU32,
    media_timescale: NonZeroU32,
) -> Result<i64> {
    if edits.is_empty() {
        return Ok(0);
    }

    let mut empty_duration = 0u64;
    let mut media_start = None;
    for edit in edits {
        match edit.media_time {
            None if media_start.is_none() => {
                empty_duration = empty_duration
                    .checked_add(edit.segment_duration)
                    .ok_or(Mp4Error::IntegerOverflow)?;
            }
            None => {
                return Err(Mp4Error::InvalidData(
                    "non-linear edit lists are not supported",
                ));
            }
            Some(_) if media_start.is_some() => {
                return Err(Mp4Error::InvalidData(
                    "repeated media edits are not supported",
                ));
            }
            Some(start) => {
                if edit.media_rate_integer != 1 || edit.media_rate_fraction != 0 {
                    return Err(Mp4Error::InvalidData(
                        "non-unit edit-list rates are not supported",
                    ));
                }
                media_start = Some(start);
            }
        }
    }
    let media_start = media_start.ok_or(Mp4Error::InvalidData(
        "an all-empty edit list has no media timeline",
    ))?;

    let scaled_numerator = u128::from(empty_duration)
        .checked_mul(u128::from(media_timescale.get()))
        .ok_or(Mp4Error::IntegerOverflow)?;
    let movie_scale = u128::from(movie_timescale.get());
    if scaled_numerator % movie_scale != 0 {
        return Err(Mp4Error::InvalidData(
            "edit-list offset is not exact in the media time scale",
        ));
    }
    let presentation_start =
        i64::try_from(scaled_numerator / movie_scale).map_err(|_| Mp4Error::IntegerOverflow)?;
    presentation_start
        .checked_sub(media_start)
        .ok_or(Mp4Error::IntegerOverflow)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn boxed(kind: [u8; 4], payload: &[u8]) -> Vec<u8> {
        let mut bytes = Vec::from(u32::try_from(8 + payload.len()).unwrap().to_be_bytes());
        bytes.extend_from_slice(&kind);
        bytes.extend_from_slice(payload);
        bytes
    }

    fn edit_container(version: u8, entries: &[(u64, i64, i16, i16)]) -> Vec<u8> {
        let mut payload = vec![version, 0, 0, 0];
        payload.extend_from_slice(&u32::try_from(entries.len()).unwrap().to_be_bytes());
        for &(duration, time, integer, fraction) in entries {
            if version == 0 {
                payload.extend_from_slice(&u32::try_from(duration).unwrap().to_be_bytes());
                payload.extend_from_slice(&i32::try_from(time).unwrap().to_be_bytes());
            } else {
                payload.extend_from_slice(&duration.to_be_bytes());
                payload.extend_from_slice(&time.to_be_bytes());
            }
            payload.extend_from_slice(&integer.to_be_bytes());
            payload.extend_from_slice(&fraction.to_be_bytes());
        }
        boxed(*b"edts", &boxed(*b"elst", &payload))
    }

    fn parse(bytes: Vec<u8>) -> Result<Vec<Edit>> {
        let file = Mp4File::open(&bytes)?;
        let header = file
            .boxes()
            .next()
            .ok_or(Mp4Error::InvalidData("missing test box"))??;
        parse_edit_container(file, header)
    }

    #[test]
    fn parses_version_zero_and_one_edit_lists() {
        for version in [0, 1] {
            let bytes = edit_container(version, &[(600, -1, 1, 0), (1_200, 25, 1, 0)]);
            assert_eq!(
                parse(bytes).unwrap(),
                [
                    Edit {
                        segment_duration: 600,
                        media_time: None,
                        media_rate_integer: 1,
                        media_rate_fraction: 0,
                    },
                    Edit {
                        segment_duration: 1_200,
                        media_time: Some(25),
                        media_rate_integer: 1,
                        media_rate_fraction: 0,
                    },
                ]
            );
        }
    }

    #[test]
    fn maps_a_linear_edit_into_media_time() {
        let edits = parse(edit_container(0, &[(500, -1, 1, 0), (1_000, 1_024, 1, 0)])).unwrap();
        assert_eq!(
            linear_timeline_offset(
                &edits,
                NonZeroU32::new(1_000).unwrap(),
                NonZeroU32::new(12_000).unwrap()
            )
            .unwrap(),
            4_976
        );
    }

    #[test]
    fn rejects_lossy_and_non_linear_mappings() {
        let lossy = parse(edit_container(0, &[(1, -1, 1, 0), (1, 0, 1, 0)])).unwrap();
        assert!(matches!(
            linear_timeline_offset(
                &lossy,
                NonZeroU32::new(3).unwrap(),
                NonZeroU32::new(2).unwrap()
            ),
            Err(Mp4Error::InvalidData(
                "edit-list offset is not exact in the media time scale"
            ))
        ));

        let repeated = parse(edit_container(0, &[(10, 0, 1, 0), (10, 0, 1, 0)])).unwrap();
        assert!(matches!(
            linear_timeline_offset(
                &repeated,
                NonZeroU32::new(1).unwrap(),
                NonZeroU32::new(1).unwrap()
            ),
            Err(Mp4Error::InvalidData(
                "repeated media edits are not supported"
            ))
        ));
    }
}
