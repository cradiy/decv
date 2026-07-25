use std::fmt;

use decv_core::{MediaInput, MediaTime, VideoDecoder};
use decv_h264::{H264Decoder, H264Error, H264SeekCheckpointCache};
use decv_mp4::{Mp4Error, PacketCursor};

/// The decoder/container state selected for an exact H.264 MP4 seek.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum H264Mp4SeekOutcome {
    /// Continue from the current packet position and decoded-picture buffer.
    ForwardRetarget,
    /// Resume from a cached decoder state and its matching MP4 sample.
    Checkpoint { sample_index: usize },
    /// Restart from the preceding independently decodable MP4 sample.
    Keyframe { sample_index: usize },
}

impl H264Mp4SeekOutcome {
    /// Whether the first packet after the seek must be marked discontinuous.
    ///
    /// A checkpoint and a forward retarget preserve the decoder timeline.
    /// Marking their next packet discontinuous would discard the retained DPB.
    pub const fn requires_discontinuity(self) -> bool {
        matches!(self, Self::Keyframe { .. })
    }
}

/// Errors produced while coordinating an MP4 cursor and H.264 decoder seek.
#[derive(Debug)]
#[non_exhaustive]
pub enum H264Mp4SeekError {
    Decoder(H264Error),
    Demuxer(Mp4Error),
    TrackMismatch { expected: usize, actual: usize },
    NoKeyframeBeforeTarget,
    NoPreviewKeyframe,
}

impl fmt::Display for H264Mp4SeekError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Decoder(error) => error.fmt(formatter),
            Self::Demuxer(error) => error.fmt(formatter),
            Self::TrackMismatch { expected, actual } => write!(
                formatter,
                "H.264 MP4 seek controller is bound to track {expected}, not track {actual}"
            ),
            Self::NoKeyframeBeforeTarget => {
                formatter.write_str("MP4 has no keyframe at or before the seek target")
            }
            Self::NoPreviewKeyframe => {
                formatter.write_str("MP4 has no keyframe available for seek preview")
            }
        }
    }
}

impl std::error::Error for H264Mp4SeekError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Decoder(error) => Some(error),
            Self::Demuxer(error) => Some(error),
            Self::TrackMismatch { .. } | Self::NoKeyframeBeforeTarget | Self::NoPreviewKeyframe => {
                None
            }
        }
    }
}

impl From<H264Error> for H264Mp4SeekError {
    fn from(error: H264Error) -> Self {
        Self::Decoder(error)
    }
}

impl From<Mp4Error> for H264Mp4SeekError {
    fn from(error: Mp4Error) -> Self {
        Self::Demuxer(error)
    }
}

/// Coordinates low-latency repeated seeks for one H.264 MP4 track.
///
/// The controller preserves the least expensive valid state transition:
///
/// 1. an explicitly allowed forward retarget keeps the live DPB and cursor;
/// 2. otherwise the latest cached checkpoint before the target is restored;
/// 3. a cold seek falls back to the preceding keyframe.
///
/// Checkpoints are not captured automatically because applications have
/// different latency and memory budgets. Call [`Self::capture_checkpoint`]
/// sparsely after complete MP4 samples have been accepted by the decoder.
#[derive(Debug)]
pub struct H264Mp4SeekController {
    track_index: usize,
    checkpoints: H264SeekCheckpointCache<usize>,
    active_exact_target: Option<MediaTime>,
}

impl H264Mp4SeekController {
    pub const fn new(
        track_index: usize,
        maximum_checkpoints: usize,
        maximum_estimated_reference_bytes: usize,
    ) -> Self {
        Self {
            track_index,
            checkpoints: H264SeekCheckpointCache::new(
                maximum_checkpoints,
                maximum_estimated_reference_bytes,
            ),
            active_exact_target: None,
        }
    }

    pub const fn track_index(&self) -> usize {
        self.track_index
    }

    pub const fn active_exact_target(&self) -> Option<MediaTime> {
        self.active_exact_target
    }

    pub const fn checkpoint_count(&self) -> usize {
        self.checkpoints.len()
    }

    pub const fn estimated_retained_reference_bytes(&self) -> usize {
        self.checkpoints.estimated_retained_reference_bytes()
    }

    /// Captures the decoder state after the most recently accepted MP4 sample.
    ///
    /// The cursor's next sample index is stored atomically with the checkpoint.
    /// A `false` result means the configured cache limits immediately evicted
    /// or rejected the checkpoint.
    pub fn capture_checkpoint<I>(
        &mut self,
        decoder: &mut H264Decoder,
        cursor: &PacketCursor<'_, I>,
    ) -> Result<bool, H264Mp4SeekError>
    where
        I: MediaInput,
    {
        self.ensure_track(cursor)?;
        Ok(self
            .checkpoints
            .capture(decoder, cursor.next_sample_index())?)
    }

    /// Starts an exact seek using the cheapest state transition allowed.
    ///
    /// Set `allow_forward_retarget` only while the current compressed-input
    /// loop can safely continue forward. This is normally true for a newer
    /// scrub request that supersedes an in-progress request before its result
    /// is presented. If it is false, a cached checkpoint or keyframe restart is
    /// selected even when `target` is later.
    pub fn begin_exact_seek<I>(
        &mut self,
        decoder: &mut H264Decoder,
        cursor: &mut PacketCursor<'_, I>,
        target: MediaTime,
        allow_forward_retarget: bool,
    ) -> Result<H264Mp4SeekOutcome, H264Mp4SeekError>
    where
        I: MediaInput,
    {
        self.ensure_track(cursor)?;

        if allow_forward_retarget
            && self
                .active_exact_target
                .is_some_and(|current| target > current)
        {
            decoder.retarget_seek_forward(target)?;
            self.active_exact_target = Some(target);
            return Ok(H264Mp4SeekOutcome::ForwardRetarget);
        }

        if let Some(entry) = self.checkpoints.latest_before(target) {
            let previous_sample_index = cursor.next_sample_index();
            cursor.seek_to_sample(*entry.input_position())?;
            if let Err(error) = decoder.restore_seek_checkpoint(entry.checkpoint(), target) {
                // The stored position came from this cursor, so rollback should
                // only fail if the caller changed the track's sample table.
                let _ = cursor.seek_to_sample(previous_sample_index);
                return Err(error.into());
            }
            self.active_exact_target = Some(target);
            return Ok(H264Mp4SeekOutcome::Checkpoint {
                sample_index: cursor.next_sample_index(),
            });
        }

        let sample_index = cursor
            .seek_to_keyframe(target)?
            .ok_or(H264Mp4SeekError::NoKeyframeBeforeTarget)?;
        decoder.flush_for_seek(target);
        self.active_exact_target = Some(target);
        Ok(H264Mp4SeekOutcome::Keyframe { sample_index })
    }

    /// Repositions to the presentation-nearest keyframe for immediate preview.
    ///
    /// This intentionally abandons exact output filtering. The returned sample
    /// is independently decodable and the first packet must be discontinuous.
    /// Call [`Self::begin_exact_seek`] after scrubbing settles.
    pub fn begin_nearest_preview<I>(
        &mut self,
        decoder: &mut H264Decoder,
        cursor: &mut PacketCursor<'_, I>,
        target: MediaTime,
    ) -> Result<usize, H264Mp4SeekError>
    where
        I: MediaInput,
    {
        self.ensure_track(cursor)?;
        let sample_index = cursor
            .seek_to_nearest_keyframe(target)?
            .ok_or(H264Mp4SeekError::NoPreviewKeyframe)?;
        decoder.flush();
        self.active_exact_target = None;
        Ok(sample_index)
    }

    /// Invalidates all retained state after changing media or decoder config.
    pub fn clear(&mut self) {
        self.checkpoints.clear();
        self.active_exact_target = None;
    }

    fn ensure_track<I>(&self, cursor: &PacketCursor<'_, I>) -> Result<(), H264Mp4SeekError>
    where
        I: MediaInput,
    {
        let actual = cursor.track_index();
        if actual == self.track_index {
            Ok(())
        } else {
            Err(H264Mp4SeekError::TrackMismatch {
                expected: self.track_index,
                actual,
            })
        }
    }
}
