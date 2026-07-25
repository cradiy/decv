use std::{fmt, num::NonZeroUsize, sync::Arc};

use decv_core::Size;
use rayon::ThreadPool;

use crate::{H264Error, Result};

pub(crate) const WIDE_AUTO_PARALLELISM_MIN_PIXELS: u64 = 8_000_000;

/// CPU parallelism used by the H.264 reconstruction backend.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum H264Parallelism {
    /// Keep all reconstruction work on the caller thread.
    Serial,
    /// Select a conservative worker count for the current implementation.
    ///
    /// CABAC parsing remains serial. `Auto` uses at most two workers below
    /// roughly eight megapixels and at most four workers for 4K-class pictures.
    #[default]
    Auto,
    /// Use exactly this many reconstruction threads.
    Threads(NonZeroUsize),
}

#[derive(Clone)]
pub(crate) enum ReconstructionExecutor {
    Serial,
    Parallel(Arc<ThreadPool>),
}

impl ReconstructionExecutor {
    pub(crate) fn serial() -> Self {
        Self::Serial
    }

    pub(crate) fn try_new(parallelism: H264Parallelism) -> Result<Self> {
        Self::try_new_with_auto_cap(parallelism, 2)
    }

    pub(crate) fn try_new_for_coded_size(
        parallelism: H264Parallelism,
        coded_size: Size,
    ) -> Result<Self> {
        let pixels = u64::from(coded_size.width) * u64::from(coded_size.height);
        let auto_cap = if pixels >= WIDE_AUTO_PARALLELISM_MIN_PIXELS {
            4
        } else {
            2
        };
        Self::try_new_with_auto_cap(parallelism, auto_cap)
    }

    fn try_new_with_auto_cap(parallelism: H264Parallelism, auto_cap: usize) -> Result<Self> {
        let threads = match parallelism {
            H264Parallelism::Serial => return Ok(Self::Serial),
            H264Parallelism::Auto => std::thread::available_parallelism()
                .map(NonZeroUsize::get)
                .unwrap_or(1)
                .min(auto_cap),
            H264Parallelism::Threads(threads) => threads.get(),
        };
        if threads == 1 {
            return Ok(Self::Serial);
        }

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|index| format!("decv-h264-{index}"))
            .build()
            .map_err(|_| H264Error::UnsupportedFeature("failed to create H.264 worker pool"))?;
        Ok(Self::Parallel(Arc::new(pool)))
    }

    pub(crate) fn pool(&self) -> Option<&ThreadPool> {
        match self {
            Self::Serial => None,
            Self::Parallel(pool) => Some(pool),
        }
    }

    #[cfg(test)]
    fn worker_count(&self) -> usize {
        self.pool().map_or(1, ThreadPool::current_num_threads)
    }
}

impl fmt::Debug for ReconstructionExecutor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Serial => formatter.write_str("Serial"),
            Self::Parallel(pool) => formatter
                .debug_tuple("Parallel")
                .field(&pool.current_num_threads())
                .finish(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_uses_a_larger_cap_only_for_4k_class_pictures() {
        let available = std::thread::available_parallelism()
            .map(NonZeroUsize::get)
            .unwrap_or(1);
        let hd = ReconstructionExecutor::try_new_for_coded_size(
            H264Parallelism::Auto,
            Size::new(1920, 1088),
        )
        .unwrap();
        let uhd = ReconstructionExecutor::try_new_for_coded_size(
            H264Parallelism::Auto,
            Size::new(3840, 2176),
        )
        .unwrap();
        assert_eq!(hd.worker_count(), available.min(2));
        assert_eq!(uhd.worker_count(), available.min(4));
    }
}
