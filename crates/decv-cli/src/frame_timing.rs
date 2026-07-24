use std::time::{Duration, Instant};

use decv_core::{
    DecodeInputStatus, DecodeOutput, EncodedVideoPacket, VideoDecoder, VideoDecoderConfig,
};
use decv_h264::{H264Decoder, H264Error, H264Parallelism};

#[derive(Debug, Default)]
struct FrameTiming {
    pending_decoder_work: Duration,
    samples: Vec<Duration>,
}

impl FrameTiming {
    fn measure<T>(&mut self, operation: impl FnOnce() -> T) -> T {
        let started = Instant::now();
        let result = operation();
        self.pending_decoder_work += started.elapsed();
        result
    }

    fn finish_frame(&mut self) {
        self.samples.push(self.pending_decoder_work);
        self.pending_decoder_work = Duration::ZERO;
    }

    fn summary(&self) -> Option<FrameTimingSummary> {
        FrameTimingSummary::from_samples(&self.samples)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FrameTimingSummary {
    count: usize,
    mean: Duration,
    p50: Duration,
    p95: Duration,
    p99: Duration,
    max: Duration,
}

impl FrameTimingSummary {
    fn from_samples(samples: &[Duration]) -> Option<Self> {
        if samples.is_empty() {
            return None;
        }
        let mut sorted = samples.to_vec();
        sorted.sort_unstable();
        let total = sorted.iter().map(Duration::as_nanos).sum::<u128>();
        Some(Self {
            count: sorted.len(),
            mean: duration_from_nanos(total / sorted.len() as u128),
            p50: nearest_rank(&sorted, 50),
            p95: nearest_rank(&sorted, 95),
            p99: nearest_rank(&sorted, 99),
            max: *sorted.last().expect("non-empty timing sample set"),
        })
    }
}

impl std::fmt::Display for FrameTimingSummary {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "decoder frame service: count={} mean={:.3}ms p50={:.3}ms \
             p95={:.3}ms p99={:.3}ms max={:.3}ms",
            self.count,
            milliseconds(self.mean),
            milliseconds(self.p50),
            milliseconds(self.p95),
            milliseconds(self.p99),
            milliseconds(self.max)
        )
    }
}

fn nearest_rank(sorted: &[Duration], percentile: usize) -> Duration {
    debug_assert!(!sorted.is_empty());
    debug_assert!((1..=100).contains(&percentile));
    let rank = sorted.len().saturating_mul(percentile).div_ceil(100);
    sorted[rank.saturating_sub(1)]
}

fn duration_from_nanos(nanos: u128) -> Duration {
    Duration::from_nanos(u64::try_from(nanos).unwrap_or(u64::MAX))
}

fn milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

pub(crate) struct CliDecoder {
    decoder: H264Decoder,
    timing: Option<FrameTiming>,
}

impl CliDecoder {
    pub(crate) fn new(frame_timing: bool) -> Self {
        Self {
            decoder: H264Decoder::new(),
            timing: frame_timing.then(FrameTiming::default),
        }
    }

    pub(crate) fn set_parallelism(
        &mut self,
        parallelism: H264Parallelism,
    ) -> Result<(), H264Error> {
        self.decoder.set_parallelism(parallelism)
    }

    pub(crate) fn configure(&mut self, config: VideoDecoderConfig) -> Result<(), H264Error> {
        self.decoder.configure(config)
    }

    pub(crate) fn send_packet(
        &mut self,
        packet: EncodedVideoPacket,
    ) -> Result<DecodeInputStatus, H264Error> {
        if let Some(timing) = self.timing.as_mut() {
            timing.measure(|| self.decoder.send_packet(packet))
        } else {
            self.decoder.send_packet(packet)
        }
    }

    pub(crate) fn receive_frame(&mut self) -> Result<DecodeOutput, H264Error> {
        let output = if let Some(timing) = self.timing.as_mut() {
            timing.measure(|| self.decoder.receive_frame())
        } else {
            self.decoder.receive_frame()
        }?;
        if matches!(output, DecodeOutput::Frame(_))
            && let Some(timing) = self.timing.as_mut()
        {
            timing.finish_frame();
        }
        Ok(output)
    }

    pub(crate) fn flush(&mut self) {
        if let Some(timing) = self.timing.as_mut() {
            timing.measure(|| self.decoder.flush());
        } else {
            self.decoder.flush();
        }
    }

    pub(crate) fn drain(&mut self) -> Result<(), H264Error> {
        if let Some(timing) = self.timing.as_mut() {
            timing.measure(|| self.decoder.drain())
        } else {
            self.decoder.drain()
        }
    }

    pub(crate) fn frame_timing_summary(&self) -> Option<FrameTimingSummary> {
        self.timing.as_ref().and_then(FrameTiming::summary)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summarizes_frame_service_time_with_nearest_rank_percentiles() {
        let samples = [1, 2, 3, 4, 100].map(Duration::from_millis);
        let summary = FrameTimingSummary::from_samples(&samples).unwrap();
        assert_eq!(summary.count, 5);
        assert_eq!(summary.mean, Duration::from_millis(22));
        assert_eq!(summary.p50, Duration::from_millis(3));
        assert_eq!(summary.p95, Duration::from_millis(100));
        assert_eq!(summary.p99, Duration::from_millis(100));
        assert_eq!(summary.max, Duration::from_millis(100));
        assert_eq!(nearest_rank(&samples, 1), Duration::from_millis(1));
        assert_eq!(nearest_rank(&samples, 100), Duration::from_millis(100));
    }
}
