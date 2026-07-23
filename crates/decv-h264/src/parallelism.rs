use std::{fmt, num::NonZeroUsize, sync::Arc};

use rayon::ThreadPool;

use crate::{H264Error, Result};

/// CPU parallelism used by the H.264 reconstruction backend.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum H264Parallelism {
    /// Keep all reconstruction work on the caller thread.
    Serial,
    /// Select a conservative worker count for the current implementation.
    ///
    /// CABAC parsing remains serial. Four workers can help when they are pinned
    /// to four performance cores, but unpinned measurements still add CPU work
    /// without reducing wall time. `Auto` therefore caps the pool at two
    /// threads for use inside an interactive application.
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
        let threads = match parallelism {
            H264Parallelism::Serial => return Ok(Self::Serial),
            H264Parallelism::Auto => std::thread::available_parallelism()
                .map(NonZeroUsize::get)
                .unwrap_or(1)
                .min(2),
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
