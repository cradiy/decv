use std::sync::Arc;

use decv_core::{
    AudioDecoderConfig, EncodedAudioPacket, EncodedVideoPacket, MediaInput, MediaTime,
    VideoDecoderConfig,
};

use crate::{Movie, Mp4Error, Mp4File, Result, Track, TrackKind};

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

    /// Synchronously prefetches a continuous source range containing samples
    /// from one track.
    ///
    /// This index-based form can be scheduled on a separate I/O worker without
    /// sharing a mutable packet cursor. `maximum_span_bytes` includes gaps
    /// occupied by interleaved tracks. The first sample is included even when
    /// it alone exceeds the budget. Returns the number of samples covered.
    pub fn prefetch_track_bytes(
        &self,
        track_index: usize,
        first_sample_index: usize,
        maximum_span_bytes: usize,
    ) -> Result<usize> {
        let track = self
            .movie
            .tracks()
            .get(track_index)
            .ok_or(Mp4Error::IndexOutOfRange {
                kind: "track",
                index: track_index,
            })?;
        prefetch_samples(
            self.input(),
            track.samples(),
            first_sample_index,
            maximum_span_bytes,
        )
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
        if track.kind() != TrackKind::Video {
            return Err(Mp4Error::UnsupportedFeature(
                "video packet read requires a video track",
            ));
        }
        track.read_packet(&self.input, sample_index)
    }

    pub fn packet_cursor(&self, track_index: usize) -> Result<PacketCursor<'_, I>> {
        let track = self
            .movie
            .tracks()
            .get(track_index)
            .ok_or(Mp4Error::IndexOutOfRange {
                kind: "track",
                index: track_index,
            })?;
        if track.kind() != TrackKind::Video {
            return Err(Mp4Error::UnsupportedFeature(
                "video packet cursor requires a video track",
            ));
        }
        Ok(PacketCursor {
            demuxer: self,
            track_index,
            next_sample_index: 0,
        })
    }

    pub fn read_audio_packet(
        &self,
        track_index: usize,
        sample_index: usize,
    ) -> Result<EncodedAudioPacket> {
        let track = self
            .movie
            .tracks()
            .get(track_index)
            .ok_or(Mp4Error::IndexOutOfRange {
                kind: "track",
                index: track_index,
            })?;
        track.read_audio_packet(&self.input, sample_index)
    }

    pub fn audio_packet_cursor(&self, track_index: usize) -> Result<AudioPacketCursor<'_, I>> {
        let track = self
            .movie
            .tracks()
            .get(track_index)
            .ok_or(Mp4Error::IndexOutOfRange {
                kind: "track",
                index: track_index,
            })?;
        if track.kind() != TrackKind::Audio {
            return Err(Mp4Error::UnsupportedFeature(
                "audio packet cursor requires an audio track",
            ));
        }
        Ok(AudioPacketCursor {
            demuxer: self,
            track_index,
            next_sample_index: 0,
        })
    }

    pub fn into_input(self) -> I {
        self.input
    }
}

/// A lightweight sequential view over one track. Multiple cursors can read
/// independently from the same immutable random-access input.
#[derive(Debug)]
pub struct PacketCursor<'demuxer, I> {
    demuxer: &'demuxer Mp4Demuxer<I>,
    track_index: usize,
    next_sample_index: usize,
}

impl<I> PacketCursor<'_, I>
where
    I: MediaInput,
{
    #[inline]
    pub const fn track_index(&self) -> usize {
        self.track_index
    }

    #[inline]
    pub const fn next_sample_index(&self) -> usize {
        self.next_sample_index
    }

    #[inline]
    pub fn track(&self) -> &Track {
        &self.demuxer.movie().tracks()[self.track_index]
    }

    pub fn next_packet(&mut self) -> Result<Option<EncodedVideoPacket>> {
        if self.next_sample_index == self.track().samples().len() {
            return Ok(None);
        }
        let sample_index = self.next_sample_index;
        let packet = self.demuxer.read_packet(self.track_index, sample_index)?;
        self.next_sample_index = self
            .next_sample_index
            .checked_add(1)
            .ok_or(Mp4Error::IntegerOverflow)?;
        Ok(Some(packet))
    }

    /// Synchronously prefetches a continuous source range containing upcoming
    /// compressed packets.
    ///
    /// `maximum_span_bytes` bounds the complete source range, including gaps
    /// occupied by interleaved tracks. The first packet is included even when
    /// it alone exceeds the budget. Returns the number of upcoming packets
    /// covered by the hint. Network-backed inputs should call this from a
    /// playback worker.
    pub fn prefetch_next_bytes(&self, maximum_span_bytes: usize) -> Result<usize> {
        self.demuxer.prefetch_track_bytes(
            self.track_index,
            self.next_sample_index,
            maximum_span_bytes,
        )
    }

    pub fn decoder_config(&self) -> Result<Option<VideoDecoderConfig>> {
        let Some(sample) = self.track().samples().get(self.next_sample_index) else {
            return Ok(None);
        };
        self.track()
            .decoder_config(sample.description_index())
            .map(Some)
    }

    /// Repositions to the closest sync sample at or before `target`.
    ///
    /// After this call, the decoder must be flushed and packets should be
    /// decoded until the first output frame at or after the requested target.
    pub fn seek_to_keyframe(&mut self, target: MediaTime) -> Result<Option<usize>> {
        let sample_index = self.track().keyframe_at_or_before(target)?;
        if let Some(sample_index) = sample_index {
            self.next_sample_index = sample_index;
        }
        Ok(sample_index)
    }

    /// Repositions to the closest sync sample at or after `target`.
    ///
    /// Unlike [`Self::seek_to_keyframe`], this is an approximate forward seek:
    /// it avoids decoder preroll but may start presentation after `target`.
    pub fn seek_to_keyframe_at_or_after(&mut self, target: MediaTime) -> Result<Option<usize>> {
        let sample_index = self.track().keyframe_at_or_after(target)?;
        if let Some(sample_index) = sample_index {
            self.next_sample_index = sample_index;
        }
        Ok(sample_index)
    }

    /// Repositions to the sync sample nearest to `target`.
    ///
    /// This is an approximate preview seek: the selected picture may be
    /// earlier or later than `target`, but decoding can begin without preroll.
    /// Use [`Self::seek_to_keyframe`] for an exact seek.
    pub fn seek_to_nearest_keyframe(&mut self, target: MediaTime) -> Result<Option<usize>> {
        let sample_index = self.track().keyframe_nearest(target)?;
        if let Some(sample_index) = sample_index {
            self.next_sample_index = sample_index;
        }
        Ok(sample_index)
    }

    /// Repositions the cursor to an exact sample boundary.
    ///
    /// Starting at an arbitrary non-sync sample is only decodable when the
    /// codec state that precedes that sample is restored as well. This is
    /// intended for pairing a saved cursor index with a decoder seek
    /// checkpoint. The index one past the final sample is the valid EOF cursor.
    pub fn seek_to_sample(&mut self, sample_index: usize) -> Result<()> {
        if sample_index > self.track().samples().len() {
            return Err(Mp4Error::IndexOutOfRange {
                kind: "sample cursor",
                index: sample_index,
            });
        }
        self.next_sample_index = sample_index;
        Ok(())
    }

    #[inline]
    pub const fn rewind(&mut self) {
        self.next_sample_index = 0;
    }
}

/// A lightweight sequential view over one audio track.
#[derive(Debug)]
pub struct AudioPacketCursor<'demuxer, I> {
    demuxer: &'demuxer Mp4Demuxer<I>,
    track_index: usize,
    next_sample_index: usize,
}

impl<I> AudioPacketCursor<'_, I>
where
    I: MediaInput,
{
    #[inline]
    pub const fn track_index(&self) -> usize {
        self.track_index
    }

    #[inline]
    pub const fn next_sample_index(&self) -> usize {
        self.next_sample_index
    }

    #[inline]
    pub fn track(&self) -> &Track {
        &self.demuxer.movie().tracks()[self.track_index]
    }

    pub fn decoder_config(&self) -> Result<Option<AudioDecoderConfig>> {
        let Some(sample) = self.track().samples().get(self.next_sample_index) else {
            return Ok(None);
        };
        self.track()
            .audio_decoder_config(sample.description_index())
            .map(Some)
    }

    pub fn next_packet(&mut self) -> Result<Option<EncodedAudioPacket>> {
        if self.next_sample_index == self.track().samples().len() {
            return Ok(None);
        }
        let sample_index = self.next_sample_index;
        let packet = self
            .demuxer
            .read_audio_packet(self.track_index, sample_index)?;
        self.next_sample_index = self
            .next_sample_index
            .checked_add(1)
            .ok_or(Mp4Error::IntegerOverflow)?;
        Ok(Some(packet))
    }

    /// Synchronously prefetches a continuous source range containing upcoming
    /// compressed audio packets.
    ///
    /// See [`PacketCursor::prefetch_next_bytes`] for budget semantics.
    pub fn prefetch_next_bytes(&self, maximum_span_bytes: usize) -> Result<usize> {
        self.demuxer.prefetch_track_bytes(
            self.track_index,
            self.next_sample_index,
            maximum_span_bytes,
        )
    }

    /// Repositions to the closest complete audio sample at or before `target`.
    pub fn seek_to_time(&mut self, target: MediaTime) -> Result<Option<usize>> {
        let sample_index = self.track().audio_sample_at_or_before(target)?;
        if let Some(sample_index) = sample_index {
            self.next_sample_index = sample_index;
        }
        Ok(sample_index)
    }

    #[inline]
    pub const fn rewind(&mut self) {
        self.next_sample_index = 0;
    }
}

fn prefetch_samples(
    input: &dyn MediaInput,
    samples: &[crate::Sample],
    first_sample_index: usize,
    maximum_span_bytes: usize,
) -> Result<usize> {
    if maximum_span_bytes == 0 {
        return Ok(0);
    }
    let Some(upcoming) = samples.get(first_sample_index..) else {
        return Err(Mp4Error::IndexOutOfRange {
            kind: "sample cursor",
            index: first_sample_index,
        });
    };
    let Some(first) = upcoming.first() else {
        return Ok(0);
    };

    let maximum_span = u64::try_from(maximum_span_bytes).map_err(|_| Mp4Error::IntegerOverflow)?;
    let mut range_start = first.offset();
    let mut range_end = first
        .offset()
        .checked_add(u64::from(first.size()))
        .ok_or(Mp4Error::IntegerOverflow)?;
    let mut packet_count = 1usize;

    for sample in &upcoming[1..] {
        let sample_end = sample
            .offset()
            .checked_add(u64::from(sample.size()))
            .ok_or(Mp4Error::IntegerOverflow)?;
        let candidate_start = range_start.min(sample.offset());
        let candidate_end = range_end.max(sample_end);
        if candidate_end - candidate_start > maximum_span {
            break;
        }
        range_start = candidate_start;
        range_end = candidate_end;
        packet_count = packet_count
            .checked_add(1)
            .ok_or(Mp4Error::IntegerOverflow)?;
    }

    let length = usize::try_from(range_end - range_start).map_err(|_| Mp4Error::IntegerOverflow)?;
    input.prefetch(range_start, length)?;
    Ok(packet_count)
}

impl Track {
    /// Reads one compressed sample and maps its raw media timestamps through
    /// the track's supported edit-list timeline.
    pub fn read_packet(
        &self,
        input: &dyn MediaInput,
        sample_index: usize,
    ) -> Result<EncodedVideoPacket> {
        if self.kind() != TrackKind::Video {
            return Err(Mp4Error::UnsupportedFeature(
                "video packet read requires a video track",
            ));
        }
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

    /// Reads one raw compressed audio access unit and maps its timestamps
    /// through the track's supported edit-list timeline.
    pub fn read_audio_packet(
        &self,
        input: &dyn MediaInput,
        sample_index: usize,
    ) -> Result<EncodedAudioPacket> {
        if self.kind() != TrackKind::Audio {
            return Err(Mp4Error::UnsupportedFeature(
                "audio packet read requires an audio track",
            ));
        }
        let sample = self
            .samples()
            .get(sample_index)
            .ok_or(Mp4Error::IndexOutOfRange {
                kind: "sample",
                index: sample_index,
            })?;
        let data = read_sample_data(input, sample.offset(), sample.size())?;
        let (pts, dts, duration) = self.packet_times(sample)?;
        let mut packet = EncodedAudioPacket::new(data);
        packet.pts = Some(pts);
        packet.dts = Some(dts);
        packet.duration = Some(duration);
        Ok(packet)
    }

    fn packet_times(&self, sample: &crate::Sample) -> Result<(MediaTime, MediaTime, MediaTime)> {
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
        Ok((
            MediaTime::new(pts, timescale),
            MediaTime::new(dts, timescale),
            MediaTime::new(i64::from(sample.duration()), timescale),
        ))
    }
}

fn read_sample_data(input: &dyn MediaInput, offset: u64, size: u32) -> Result<Arc<[u8]>> {
    let size = usize::try_from(size).map_err(|_| Mp4Error::IntegerOverflow)?;
    if size > MAX_PACKET_SIZE {
        return Err(Mp4Error::InvalidData(
            "MP4 sample exceeds the packet allocation limit",
        ));
    }
    let mut data = vec![0; size];
    read_exact_at(input, offset, &mut data)?;
    Ok(Arc::from(data))
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

#[cfg(test)]
mod tests {
    use std::{
        io,
        sync::{Arc, Mutex},
    };

    use decv_core::MediaInput;

    use crate::{Mp4Demuxer, TrackKind};

    #[derive(Debug)]
    struct TrackingInput {
        bytes: Arc<[u8]>,
        prefetches: Mutex<Vec<(u64, usize)>>,
    }

    impl MediaInput for TrackingInput {
        fn len(&self) -> io::Result<Option<u64>> {
            Ok(Some(u64::try_from(self.bytes.as_ref().len()).unwrap()))
        }

        fn prefetch(&self, offset: u64, length: usize) -> io::Result<()> {
            self.prefetches.lock().unwrap().push((offset, length));
            Ok(())
        }

        fn read_at(&self, offset: u64, buffer: &mut [u8]) -> io::Result<usize> {
            self.bytes.as_ref().read_at(offset, buffer)
        }
    }

    #[test]
    fn packet_cursor_prefetches_upcoming_sample_span() {
        let input = TrackingInput {
            bytes: Arc::from(decode_hex(include_str!(
                "../../decv/tests/fixtures/three-frame-high-b.mp4.hex"
            ))),
            prefetches: Mutex::new(Vec::new()),
        };
        let demuxer = Mp4Demuxer::open(input).unwrap();
        let track_index = demuxer
            .movie()
            .tracks()
            .iter()
            .position(|track| track.kind() == TrackKind::Video)
            .unwrap();
        let first_sample = demuxer.movie().tracks()[track_index].samples()[0];

        let cursor = demuxer.packet_cursor(track_index).unwrap();
        let packet_count = cursor.prefetch_next_bytes(1024).unwrap();
        assert!(packet_count >= 1);

        let prefetches = demuxer.input().prefetches.lock().unwrap();
        assert_eq!(prefetches.len(), 1);
        let (offset, length) = prefetches[0];
        assert!(offset <= first_sample.offset());
        assert!(
            offset + u64::try_from(length).unwrap()
                >= first_sample.offset() + u64::from(first_sample.size())
        );
    }

    fn decode_hex(text: &str) -> Vec<u8> {
        let digits = text
            .bytes()
            .filter(|byte| !byte.is_ascii_whitespace())
            .collect::<Vec<_>>();
        assert!(digits.len().is_multiple_of(2));
        digits
            .chunks_exact(2)
            .map(|pair| hex_digit(pair[0]) << 4 | hex_digit(pair[1]))
            .collect()
    }

    fn hex_digit(byte: u8) -> u8 {
        match byte {
            b'0'..=b'9' => byte - b'0',
            b'a'..=b'f' => byte - b'a' + 10,
            b'A'..=b'F' => byte - b'A' + 10,
            _ => panic!("fixture contains a non-hex byte"),
        }
    }
}
