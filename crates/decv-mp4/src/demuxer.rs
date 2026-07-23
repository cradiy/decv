use std::sync::Arc;

use decv_core::{EncodedVideoPacket, MediaInput, MediaTime};

use crate::{Movie, Mp4Error, Mp4File, Result, Track};

const MAX_PACKET_SIZE: usize = 256 * 1024 * 1024;

/// An MP4 movie coupled to the random-access input from which its samples are
/// read. Parsing does not retain cursors or require an async runtime.
#[derive(Debug)]
pub struct Mp4Demuxer<I> {
    input: I,
    movie: Movie,
}

impl<I> Mp4Demuxer<I>
where
    I: MediaInput,
{
    pub fn open(input: I) -> Result<Self> {
        let movie = Movie::parse(Mp4File::open(&input)?)?;
        Ok(Self { input, movie })
    }

    #[inline]
    pub const fn input(&self) -> &I {
        &self.input
    }

    #[inline]
    pub const fn movie(&self) -> &Movie {
        &self.movie
    }

    pub fn read_packet(
        &self,
        track_index: usize,
        sample_index: usize,
    ) -> Result<EncodedVideoPacket> {
        let track = self
            .movie
            .tracks()
            .get(track_index)
            .ok_or(Mp4Error::IndexOutOfRange {
                kind: "track",
                index: track_index,
            })?;
        track.read_packet(&self.input, sample_index)
    }

    pub fn into_input(self) -> I {
        self.input
    }
}

impl Track {
    /// Reads one compressed sample and maps its raw media timestamps through
    /// the track's supported edit-list timeline.
    pub fn read_packet(
        &self,
        input: &dyn MediaInput,
        sample_index: usize,
    ) -> Result<EncodedVideoPacket> {
        let sample = self
            .samples()
            .get(sample_index)
            .ok_or(Mp4Error::IndexOutOfRange {
                kind: "sample",
                index: sample_index,
            })?;
        let size = usize::try_from(sample.size()).map_err(|_| Mp4Error::IntegerOverflow)?;
        if size > MAX_PACKET_SIZE {
            return Err(Mp4Error::InvalidData(
                "MP4 sample exceeds the packet allocation limit",
            ));
        }

        let mut data = vec![0; size];
        read_exact_at(input, sample.offset(), &mut data)?;

        let offset = self.presentation_time_offset()?.value;
        let pts = sample
            .presentation_time()
            .checked_add(offset)
            .ok_or(Mp4Error::IntegerOverflow)?;
        let dts = sample
            .decode_time()
            .checked_add(offset)
            .ok_or(Mp4Error::IntegerOverflow)?;
        let timescale = self.media_timescale();

        let mut packet = EncodedVideoPacket::new(Arc::<[u8]>::from(data));
        packet.pts = Some(MediaTime::new(pts, timescale));
        packet.dts = Some(MediaTime::new(dts, timescale));
        packet.duration = Some(MediaTime::new(i64::from(sample.duration()), timescale));
        packet.keyframe = sample.is_sync();
        Ok(packet)
    }
}

fn read_exact_at(input: &dyn MediaInput, offset: u64, mut buffer: &mut [u8]) -> Result<()> {
    let mut position = offset;
    while !buffer.is_empty() {
        let read = input.read_at(position, buffer)?;
        if read == 0 {
            return Err(Mp4Error::InvalidData("unexpected end of MP4 sample data"));
        }
        if read > buffer.len() {
            return Err(Mp4Error::InvalidData(
                "MediaInput returned more bytes than requested",
            ));
        }
        position = position
            .checked_add(u64::try_from(read).map_err(|_| Mp4Error::IntegerOverflow)?)
            .ok_or(Mp4Error::IntegerOverflow)?;
        buffer = &mut buffer[read..];
    }
    Ok(())
}
