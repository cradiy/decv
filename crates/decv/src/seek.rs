use std::{fmt, num::NonZeroUsize};

use decv_core::{MediaInput, MediaTime, VideoDecoder};
use decv_h264::{H264Decoder, H264Error, H264SeekCheckpointCache};
use decv_mp4::{Mp4Error, PacketCursor, Track};

/// Decoder state from which an exact H.264 MP4 seek will resume.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum H264Mp4SeekSource {
    ForwardRetarget,
    Checkpoint,
    Keyframe,
}

impl H264Mp4SeekSource {
    pub const fn requires_discontinuity(self) -> bool {
        matches!(self, Self::Keyframe)
    }
}

/// Read-only cost estimate for an exact H.264 MP4 seek.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct H264Mp4SeekPlan {
    source: H264Mp4SeekSource,
    resume_sample_index: usize,
    selected_sample_index: Option<usize>,
    estimated_input_samples: usize,
}

impl H264Mp4SeekPlan {
    pub const fn source(self) -> H264Mp4SeekSource {
        self.source
    }

    /// First compressed sample used by the selected resume strategy.
    pub const fn resume_sample_index(self) -> usize {
        self.resume_sample_index
    }

    /// Presentation-first sample at or after the requested target.
    ///
    /// `None` means the target has no following presentation sample and an
    /// exact seek would consume the remaining input without producing a frame.
    pub const fn selected_sample_index(self) -> Option<usize> {
        self.selected_sample_index
    }

    /// Estimated number of compressed samples needed before selected output.
    ///
    /// This includes decode-ordered B pictures needed to release the selected
    /// frame from presentation reordering. Zero means retained decoder output
    /// may already satisfy the newer forward target without additional input.
    pub const fn estimated_input_samples(self) -> usize {
        self.estimated_input_samples
    }

    pub const fn requires_discontinuity(self) -> bool {
        self.source.requires_discontinuity()
    }

    /// Whether an interactive seek should use a keyframe preview instead.
    pub const fn exceeds_exact_sample_budget(self, maximum_input_samples: usize) -> bool {
        self.selected_sample_index.is_none() || self.estimated_input_samples > maximum_input_samples
    }
}

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
    pub const fn source(self) -> H264Mp4SeekSource {
        match self {
            Self::ForwardRetarget => H264Mp4SeekSource::ForwardRetarget,
            Self::Checkpoint { .. } => H264Mp4SeekSource::Checkpoint,
            Self::Keyframe { .. } => H264Mp4SeekSource::Keyframe,
        }
    }

    /// Whether the first packet after the seek must be marked discontinuous.
    ///
    /// A checkpoint and a forward retarget preserve the decoder timeline.
    /// Marking their next packet discontinuous would discard the retained DPB.
    pub const fn requires_discontinuity(self) -> bool {
        self.source().requires_discontinuity()
    }
}

/// Result of a budget-aware seek intended for pointer scrubbing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum H264Mp4InteractiveSeekOutcome {
    /// The estimated input work fit the supplied budget.
    Exact {
        plan: H264Mp4SeekPlan,
        outcome: H264Mp4SeekOutcome,
    },
    /// Exact work exceeded the budget, so the cursor selected a nearby sync
    /// sample for immediate approximate presentation.
    Preview {
        exact_plan: H264Mp4SeekPlan,
        sample_index: usize,
    },
}

impl H264Mp4InteractiveSeekOutcome {
    pub const fn exact_plan(self) -> H264Mp4SeekPlan {
        match self {
            Self::Exact { plan, .. } => plan,
            Self::Preview { exact_plan, .. } => exact_plan,
        }
    }

    pub const fn is_exact(self) -> bool {
        matches!(self, Self::Exact { .. })
    }

    pub const fn exact_outcome(self) -> Option<H264Mp4SeekOutcome> {
        match self {
            Self::Exact { outcome, .. } => Some(outcome),
            Self::Preview { .. } => None,
        }
    }

    pub const fn preview_sample_index(self) -> Option<usize> {
        match self {
            Self::Exact { .. } => None,
            Self::Preview { sample_index, .. } => Some(sample_index),
        }
    }

    /// Whether the first packet after this operation must be discontinuous.
    pub const fn requires_discontinuity(self) -> bool {
        match self {
            Self::Exact { outcome, .. } => outcome.requires_discontinuity(),
            Self::Preview { .. } => true,
        }
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
/// The controller estimates the least expensive valid state transition from
/// compressed-sample positions:
///
/// 1. an allowed forward retarget is used only when the live cursor is already
///    at least as far forward as every restart candidate;
/// 2. a cached checkpoint is used only when it resumes after the preceding
///    keyframe;
/// 3. otherwise the decoder restarts from that keyframe.
///
/// Checkpoints are not captured automatically because applications have
/// different latency and memory budgets. Call [`Self::capture_checkpoint`]
/// sparsely after complete MP4 samples have been accepted by the decoder.
#[derive(Debug)]
pub struct H264Mp4SeekController {
    track_index: usize,
    checkpoints: H264SeekCheckpointCache<usize>,
    active_exact_target: Option<MediaTime>,
    minimum_checkpoint_sample_distance: usize,
    last_checkpoint_sample_index: Option<usize>,
}

impl H264Mp4SeekController {
    /// Default spacing between retained checkpoints.
    ///
    /// This is one second at 30 samples per second and half a second at 60.
    /// Applications can override it with
    /// [`Self::with_minimum_checkpoint_sample_distance`].
    pub const DEFAULT_MINIMUM_CHECKPOINT_SAMPLE_DISTANCE: usize = 30;

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
            minimum_checkpoint_sample_distance: Self::DEFAULT_MINIMUM_CHECKPOINT_SAMPLE_DISTANCE,
            last_checkpoint_sample_index: None,
        }
    }

    /// Sets the minimum compressed-sample distance between checkpoints.
    ///
    /// Checkpoint creation finishes the current access unit and waits for
    /// pending finalization, so capturing every video sample can reduce decode
    /// throughput. A value of one explicitly permits per-sample capture.
    pub const fn with_minimum_checkpoint_sample_distance(
        mut self,
        minimum_distance: NonZeroUsize,
    ) -> Self {
        self.minimum_checkpoint_sample_distance = minimum_distance.get();
        self
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

    pub const fn minimum_checkpoint_sample_distance(&self) -> usize {
        self.minimum_checkpoint_sample_distance
    }

    /// Estimates the resume source and compressed-input work for an exact seek
    /// without modifying the decoder, cursor, cache recency, or active target.
    pub fn plan_exact_seek<I>(
        &self,
        cursor: &PacketCursor<'_, I>,
        target: MediaTime,
        allow_forward_retarget: bool,
    ) -> Result<H264Mp4SeekPlan, H264Mp4SeekError>
    where
        I: MediaInput,
    {
        self.ensure_track(cursor)?;

        let track = cursor.track();
        let keyframe_sample_index = track
            .keyframe_at_or_before(target)?
            .ok_or(H264Mp4SeekError::NoKeyframeBeforeTarget)?;
        let forward_sample_index = (allow_forward_retarget
            && self
                .active_exact_target
                .is_some_and(|current| target > current))
        .then(|| cursor.next_sample_index());
        let checkpoint_sample_index = self
            .checkpoints
            .latest_before(target)
            .map(|entry| *entry.input_position());
        let source = select_exact_seek_source(
            keyframe_sample_index,
            checkpoint_sample_index,
            forward_sample_index,
        );
        let resume_sample_index = match source {
            ExactSeekSource::Forward => {
                forward_sample_index.expect("forward source requires an active cursor")
            }
            ExactSeekSource::Checkpoint => {
                checkpoint_sample_index.expect("checkpoint source requires a cache entry")
            }
            ExactSeekSource::Keyframe => keyframe_sample_index,
        };
        let target_sample = estimate_target_sample(track, keyframe_sample_index, target)?;
        let estimated_input_samples = target_sample.map_or_else(
            || track.samples().len().saturating_sub(resume_sample_index),
            |target| {
                if resume_sample_index > target.required_decode_sample_index {
                    0
                } else {
                    target.required_decode_sample_index - resume_sample_index + 1
                }
            },
        );

        Ok(H264Mp4SeekPlan {
            source: source.public(),
            resume_sample_index,
            selected_sample_index: target_sample.map(|target| target.selected_sample_index),
            estimated_input_samples,
        })
    }

    /// Captures the decoder state after the most recently accepted MP4 sample.
    ///
    /// The cursor's next sample index is stored atomically with the checkpoint.
    /// A `false` result means the sample is too close to the previous retained
    /// checkpoint or the configured cache limits rejected the new checkpoint.
    pub fn capture_checkpoint<I>(
        &mut self,
        decoder: &mut H264Decoder,
        cursor: &PacketCursor<'_, I>,
    ) -> Result<bool, H264Mp4SeekError>
    where
        I: MediaInput,
    {
        self.ensure_track(cursor)?;
        let sample_index = cursor.next_sample_index();
        if self.last_checkpoint_sample_index.is_some_and(|previous| {
            previous.abs_diff(sample_index) < self.minimum_checkpoint_sample_distance
        }) {
            return Ok(false);
        }

        let retained = self.checkpoints.capture(decoder, sample_index)?;
        if retained {
            self.last_checkpoint_sample_index = Some(sample_index);
        }
        Ok(retained)
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
        let plan = self.plan_exact_seek(cursor, target, allow_forward_retarget)?;
        self.execute_exact_seek_plan(decoder, cursor, target, plan)
    }

    /// Starts a budget-aware seek for interactive timeline scrubbing.
    ///
    /// Exact seek is used when its estimated compressed-input count is within
    /// `maximum_exact_input_samples`. More expensive requests select the
    /// nearest keyframe for immediate preview. After pointer movement settles,
    /// call [`Self::begin_exact_seek`] to resolve the final timestamp exactly.
    pub fn begin_interactive_seek<I>(
        &mut self,
        decoder: &mut H264Decoder,
        cursor: &mut PacketCursor<'_, I>,
        target: MediaTime,
        allow_forward_retarget: bool,
        maximum_exact_input_samples: usize,
    ) -> Result<H264Mp4InteractiveSeekOutcome, H264Mp4SeekError>
    where
        I: MediaInput,
    {
        let plan = self.plan_exact_seek(cursor, target, allow_forward_retarget)?;
        if plan.exceeds_exact_sample_budget(maximum_exact_input_samples) {
            let sample_index = self.begin_nearest_preview(decoder, cursor, target)?;
            Ok(H264Mp4InteractiveSeekOutcome::Preview {
                exact_plan: plan,
                sample_index,
            })
        } else {
            let outcome = self.execute_exact_seek_plan(decoder, cursor, target, plan)?;
            Ok(H264Mp4InteractiveSeekOutcome::Exact { plan, outcome })
        }
    }

    fn execute_exact_seek_plan<I>(
        &mut self,
        decoder: &mut H264Decoder,
        cursor: &mut PacketCursor<'_, I>,
        target: MediaTime,
        plan: H264Mp4SeekPlan,
    ) -> Result<H264Mp4SeekOutcome, H264Mp4SeekError>
    where
        I: MediaInput,
    {
        let outcome = match plan.source {
            H264Mp4SeekSource::ForwardRetarget => {
                decoder.retarget_seek_forward(target)?;
                H264Mp4SeekOutcome::ForwardRetarget
            }
            H264Mp4SeekSource::Checkpoint => {
                let previous_sample_index = cursor.next_sample_index();
                cursor.seek_to_sample(plan.resume_sample_index)?;
                let restored_sample_index = match self
                    .checkpoints
                    .restore_latest_before(decoder, target)
                {
                    Ok(Some(sample_index)) => *sample_index,
                    Ok(None) => unreachable!("checkpoint source requires a matching cache entry"),
                    Err(error) => {
                        let _ = cursor.seek_to_sample(previous_sample_index);
                        return Err(error.into());
                    }
                };
                if restored_sample_index != plan.resume_sample_index {
                    // The stored position came from this cursor, so rollback
                    // should only fail if the caller changed the sample table.
                    let _ = cursor.seek_to_sample(previous_sample_index);
                    return Err(H264Error::InvalidSyntax(
                        "H.264 seek checkpoint cache changed during restore",
                    )
                    .into());
                }
                H264Mp4SeekOutcome::Checkpoint {
                    sample_index: cursor.next_sample_index(),
                }
            }
            H264Mp4SeekSource::Keyframe => {
                cursor.seek_to_sample(plan.resume_sample_index)?;
                decoder.flush_for_seek(target);
                H264Mp4SeekOutcome::Keyframe {
                    sample_index: plan.resume_sample_index,
                }
            }
        };

        self.active_exact_target = Some(target);
        Ok(outcome)
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
        self.last_checkpoint_sample_index = None;
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExactSeekSource {
    Forward,
    Checkpoint,
    Keyframe,
}

impl ExactSeekSource {
    const fn public(self) -> H264Mp4SeekSource {
        match self {
            Self::Forward => H264Mp4SeekSource::ForwardRetarget,
            Self::Checkpoint => H264Mp4SeekSource::Checkpoint,
            Self::Keyframe => H264Mp4SeekSource::Keyframe,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct EstimatedTargetSample {
    selected_sample_index: usize,
    required_decode_sample_index: usize,
}

fn estimate_target_sample(
    track: &Track,
    keyframe_sample_index: usize,
    target: MediaTime,
) -> Result<Option<EstimatedTargetSample>, Mp4Error> {
    let samples = track.samples();
    let next_keyframe_position = track
        .sync_sample_indices()
        .partition_point(|&sample_index| sample_index <= keyframe_sample_index);
    let scan_end = track
        .sync_sample_indices()
        .get(next_keyframe_position)
        .and_then(|sample_index| sample_index.checked_add(1))
        .unwrap_or(samples.len())
        .min(samples.len());
    let presentation_offset = track.presentation_time_offset()?.value;
    let timescale = track.media_timescale();

    let mut selected: Option<(MediaTime, usize)> = None;
    for (relative_index, sample) in samples[keyframe_sample_index..scan_end].iter().enumerate() {
        let sample_index = keyframe_sample_index + relative_index;
        let presentation_value = sample
            .presentation_time()
            .checked_add(presentation_offset)
            .ok_or(Mp4Error::IntegerOverflow)?;
        let presentation_time = MediaTime::new(presentation_value, timescale);
        if presentation_time >= target
            && selected.is_none_or(|current| (presentation_time, sample_index) < current)
        {
            selected = Some((presentation_time, sample_index));
        }
    }
    let Some((selected_time, selected_sample_index)) = selected else {
        return Ok(None);
    };

    let mut required_decode_sample_index = selected_sample_index;
    for (relative_index, sample) in samples[keyframe_sample_index..scan_end].iter().enumerate() {
        let presentation_value = sample
            .presentation_time()
            .checked_add(presentation_offset)
            .ok_or(Mp4Error::IntegerOverflow)?;
        if MediaTime::new(presentation_value, timescale) <= selected_time {
            required_decode_sample_index =
                required_decode_sample_index.max(keyframe_sample_index + relative_index);
        }
    }

    Ok(Some(EstimatedTargetSample {
        selected_sample_index,
        required_decode_sample_index,
    }))
}

fn select_exact_seek_source(
    keyframe_sample_index: usize,
    checkpoint_sample_index: Option<usize>,
    forward_sample_index: Option<usize>,
) -> ExactSeekSource {
    let checkpoint_is_closer =
        checkpoint_sample_index.is_some_and(|checkpoint| checkpoint > keyframe_sample_index);
    let best_restart_sample_index = checkpoint_sample_index
        .filter(|_| checkpoint_is_closer)
        .unwrap_or(keyframe_sample_index);

    if forward_sample_index.is_some_and(|current| current >= best_restart_sample_index) {
        ExactSeekSource::Forward
    } else if checkpoint_is_closer {
        ExactSeekSource::Checkpoint
    } else {
        ExactSeekSource::Keyframe
    }
}

#[cfg(test)]
mod tests {
    use super::{ExactSeekSource, select_exact_seek_source};

    #[test]
    fn exact_seek_selects_the_latest_decodable_sample_position() {
        assert_eq!(
            select_exact_seek_source(20, Some(30), Some(40)),
            ExactSeekSource::Forward
        );
        assert_eq!(
            select_exact_seek_source(20, Some(30), Some(10)),
            ExactSeekSource::Checkpoint
        );
        assert_eq!(
            select_exact_seek_source(20, Some(15), Some(10)),
            ExactSeekSource::Keyframe
        );
        assert_eq!(
            select_exact_seek_source(20, Some(20), None),
            ExactSeekSource::Keyframe
        );
    }
}
