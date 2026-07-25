use std::{
    collections::HashMap,
    io,
    num::NonZeroUsize,
    ops::Range,
    sync::{
        Arc, Condvar, Mutex, MutexGuard,
        atomic::{AtomicU64, Ordering},
    },
};

use decv_core::MediaInput;

/// Synchronous source of exact byte ranges from one immutable media object.
pub trait RangeFetcher: Send + Sync {
    /// Stable total length of the remote object.
    fn len(&self) -> io::Result<u64>;

    /// Whether the remote object has no bytes.
    fn is_empty(&self) -> io::Result<bool> {
        self.len().map(|length| length == 0)
    }

    /// Fetches exactly `range.end - range.start` bytes.
    ///
    /// Implementations must reject partial responses and responses for a
    /// different range. Sources with object validators should also reject
    /// version changes instead of silently combining different versions.
    fn fetch_range(&self, range: Range<u64>) -> io::Result<Vec<u8>>;
}

/// Memory and request granularity for [`CachedRangeInput`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RangeCacheConfig {
    block_size: NonZeroUsize,
    maximum_blocks: NonZeroUsize,
}

impl RangeCacheConfig {
    pub const fn new(block_size: NonZeroUsize, maximum_blocks: NonZeroUsize) -> Self {
        Self {
            block_size,
            maximum_blocks,
        }
    }

    pub const fn block_size(self) -> NonZeroUsize {
        self.block_size
    }

    pub const fn maximum_blocks(self) -> NonZeroUsize {
        self.maximum_blocks
    }

    pub const fn maximum_cached_bytes(self) -> usize {
        self.block_size
            .get()
            .saturating_mul(self.maximum_blocks.get())
    }
}

impl Default for RangeCacheConfig {
    fn default() -> Self {
        Self::new(
            NonZeroUsize::new(256 * 1024).expect("default block size is non-zero"),
            NonZeroUsize::new(32).expect("default block count is non-zero"),
        )
    }
}

/// Snapshot of cache activity since input construction or the last reset.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct RangeInputStats {
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub range_requests: u64,
    pub fetched_bytes: u64,
    pub evicted_blocks: u64,
}

/// Shared atomic counters used by [`CachedRangeInput`].
#[derive(Debug, Default)]
pub struct RangeCacheStats {
    cache_hits: AtomicU64,
    cache_misses: AtomicU64,
    range_requests: AtomicU64,
    fetched_bytes: AtomicU64,
    evicted_blocks: AtomicU64,
}

impl RangeCacheStats {
    pub fn snapshot(&self) -> RangeInputStats {
        RangeInputStats {
            cache_hits: self.cache_hits.load(Ordering::Relaxed),
            cache_misses: self.cache_misses.load(Ordering::Relaxed),
            range_requests: self.range_requests.load(Ordering::Relaxed),
            fetched_bytes: self.fetched_bytes.load(Ordering::Relaxed),
            evicted_blocks: self.evicted_blocks.load(Ordering::Relaxed),
        }
    }

    pub fn reset(&self) {
        self.cache_hits.store(0, Ordering::Relaxed);
        self.cache_misses.store(0, Ordering::Relaxed);
        self.range_requests.store(0, Ordering::Relaxed);
        self.fetched_bytes.store(0, Ordering::Relaxed);
        self.evicted_blocks.store(0, Ordering::Relaxed);
    }
}

/// A bounded block cache that implements [`MediaInput`] over a range source.
///
/// Concurrent reads for the same missing block share one fetch. Reads for
/// different blocks may proceed concurrently. Ready blocks are retained using
/// least-recently-used eviction.
#[derive(Debug)]
pub struct CachedRangeInput<F> {
    fetcher: F,
    length: u64,
    config: RangeCacheConfig,
    cache: Mutex<CacheState>,
    stats: RangeCacheStats,
}

impl<F> CachedRangeInput<F>
where
    F: RangeFetcher,
{
    pub fn new(fetcher: F, config: RangeCacheConfig) -> io::Result<Self> {
        let length = fetcher.len()?;
        Ok(Self {
            fetcher,
            length,
            config,
            cache: Mutex::new(CacheState::default()),
            stats: RangeCacheStats::default(),
        })
    }

    pub const fn config(&self) -> RangeCacheConfig {
        self.config
    }

    pub const fn content_length(&self) -> u64 {
        self.length
    }

    pub const fn stats(&self) -> &RangeCacheStats {
        &self.stats
    }

    pub fn clear(&self) -> io::Result<()> {
        lock(&self.cache)?.blocks.clear();
        Ok(())
    }

    /// Synchronously warms every block intersecting the requested range.
    ///
    /// Applications should normally call this on a worker thread. The range is
    /// clipped to the known media length.
    pub fn prefetch(&self, offset: u64, length: usize) -> io::Result<()> {
        let requested = clipped_length(self.length, offset, length)?;
        if requested == 0 {
            return Ok(());
        }
        let block_size =
            u64::try_from(self.config.block_size.get()).map_err(|_| integer_overflow())?;
        let end = offset
            .checked_add(u64::try_from(requested).map_err(|_| integer_overflow())?)
            .ok_or_else(integer_overflow)?;
        let first = offset / block_size;
        let last = (end - 1) / block_size;
        for block_index in first..=last {
            self.get_block(block_index)?;
        }
        Ok(())
    }

    fn get_block(&self, block_index: u64) -> io::Result<Arc<[u8]>> {
        let (entry, fetch) = {
            let mut cache = lock(&self.cache)?;
            let recency = cache.take_recency();
            if let Some(record) = cache.blocks.get_mut(&block_index) {
                record.last_used = recency;
                self.stats.cache_hits.fetch_add(1, Ordering::Relaxed);
                (record.entry.clone(), false)
            } else {
                self.stats.cache_misses.fetch_add(1, Ordering::Relaxed);
                self.evict_for_insert(&mut cache)?;
                let entry = Arc::new(BlockEntry::loading());
                cache.blocks.insert(
                    block_index,
                    CacheRecord {
                        entry: entry.clone(),
                        last_used: recency,
                    },
                );
                (entry, true)
            }
        };

        if fetch {
            self.fetch_block(block_index, &entry)
        } else {
            entry.wait()
        }
    }

    fn evict_for_insert(&self, cache: &mut CacheState) -> io::Result<()> {
        while cache.blocks.len() >= self.config.maximum_blocks.get() {
            let candidate = oldest_ready_block(cache)?;
            let Some(index) = candidate else {
                // All retained blocks are in flight. Temporary overflow avoids
                // evicting a request that another reader is waiting on.
                break;
            };
            cache.blocks.remove(&index);
            self.stats.evicted_blocks.fetch_add(1, Ordering::Relaxed);
        }
        Ok(())
    }

    fn fetch_block(&self, block_index: u64, entry: &Arc<BlockEntry>) -> io::Result<Arc<[u8]>> {
        let block_size =
            u64::try_from(self.config.block_size.get()).map_err(|_| integer_overflow())?;
        let start = block_index
            .checked_mul(block_size)
            .ok_or_else(integer_overflow)?;
        let end = start
            .checked_add(block_size)
            .ok_or_else(integer_overflow)?
            .min(self.length);
        let expected = usize::try_from(end - start).map_err(|_| integer_overflow())?;

        self.stats.range_requests.fetch_add(1, Ordering::Relaxed);
        match self.fetcher.fetch_range(start..end) {
            Ok(bytes) if bytes.len() == expected => {
                self.stats.fetched_bytes.fetch_add(
                    u64::try_from(bytes.len()).unwrap_or(u64::MAX),
                    Ordering::Relaxed,
                );
                let bytes = Arc::<[u8]>::from(bytes);
                entry.complete(bytes.clone())?;
                self.trim_after_fetch()?;
                Ok(bytes)
            }
            Ok(bytes) => {
                let error = io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    format!(
                        "range source returned {} bytes for expected {expected}-byte block",
                        bytes.len()
                    ),
                );
                self.fail_block(block_index, entry, &error)?;
                Err(error)
            }
            Err(error) => {
                self.fail_block(block_index, entry, &error)?;
                Err(error)
            }
        }
    }

    fn fail_block(
        &self,
        block_index: u64,
        entry: &Arc<BlockEntry>,
        error: &io::Error,
    ) -> io::Result<()> {
        {
            let mut cache = lock(&self.cache)?;
            if cache
                .blocks
                .get(&block_index)
                .is_some_and(|record| Arc::ptr_eq(&record.entry, entry))
            {
                cache.blocks.remove(&block_index);
            }
        }
        entry.fail(error)
    }

    fn trim_after_fetch(&self) -> io::Result<()> {
        let mut cache = lock(&self.cache)?;
        while cache.blocks.len() > self.config.maximum_blocks.get() {
            let candidate = oldest_ready_block(&cache)?;
            let Some(index) = candidate else {
                break;
            };
            cache.blocks.remove(&index);
            self.stats.evicted_blocks.fetch_add(1, Ordering::Relaxed);
        }
        Ok(())
    }
}

impl<F> MediaInput for CachedRangeInput<F>
where
    F: RangeFetcher,
{
    fn len(&self) -> io::Result<Option<u64>> {
        Ok(Some(self.length))
    }

    fn read_at(&self, offset: u64, buffer: &mut [u8]) -> io::Result<usize> {
        let requested = clipped_length(self.length, offset, buffer.len())?;
        if requested == 0 {
            return Ok(0);
        }

        let block_size =
            u64::try_from(self.config.block_size.get()).map_err(|_| integer_overflow())?;
        let mut position = offset;
        let mut written = 0usize;
        while written < requested {
            let block_index = position / block_size;
            let bytes = self.get_block(block_index)?;
            let within_block =
                usize::try_from(position % block_size).map_err(|_| integer_overflow())?;
            let available = bytes
                .as_ref()
                .len()
                .checked_sub(within_block)
                .ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        "cached range block does not contain the requested offset",
                    )
                })?;
            let count = available.min(requested - written);
            buffer[written..written + count]
                .copy_from_slice(&bytes[within_block..within_block + count]);
            position = position
                .checked_add(u64::try_from(count).map_err(|_| integer_overflow())?)
                .ok_or_else(integer_overflow)?;
            written += count;
        }
        Ok(written)
    }
}

#[derive(Debug, Default)]
struct CacheState {
    blocks: HashMap<u64, CacheRecord>,
    next_recency: u64,
}

impl CacheState {
    fn take_recency(&mut self) -> u64 {
        let recency = self.next_recency;
        self.next_recency = self.next_recency.saturating_add(1);
        recency
    }
}

#[derive(Debug)]
struct CacheRecord {
    entry: Arc<BlockEntry>,
    last_used: u64,
}

fn oldest_ready_block(cache: &CacheState) -> io::Result<Option<u64>> {
    let mut candidate = None;
    for (&index, record) in &cache.blocks {
        if !record.entry.is_loading()?
            && candidate.is_none_or(|(_, last_used)| record.last_used < last_used)
        {
            candidate = Some((index, record.last_used));
        }
    }
    Ok(candidate.map(|(index, _)| index))
}

#[derive(Debug)]
struct BlockEntry {
    state: Mutex<BlockState>,
    ready: Condvar,
}

impl BlockEntry {
    fn loading() -> Self {
        Self {
            state: Mutex::new(BlockState::Loading),
            ready: Condvar::new(),
        }
    }

    fn is_loading(&self) -> io::Result<bool> {
        Ok(matches!(*lock(&self.state)?, BlockState::Loading))
    }

    fn wait(&self) -> io::Result<Arc<[u8]>> {
        let mut state = lock(&self.state)?;
        loop {
            match &*state {
                BlockState::Loading => {
                    state = self
                        .ready
                        .wait(state)
                        .map_err(|_| poisoned("range block wait"))?;
                }
                BlockState::Ready(bytes) => return Ok(bytes.clone()),
                BlockState::Failed(error) => return Err(error.to_io_error()),
            }
        }
    }

    fn complete(&self, bytes: Arc<[u8]>) -> io::Result<()> {
        *lock(&self.state)? = BlockState::Ready(bytes);
        self.ready.notify_all();
        Ok(())
    }

    fn fail(&self, error: &io::Error) -> io::Result<()> {
        *lock(&self.state)? = BlockState::Failed(SharedIoError::new(error));
        self.ready.notify_all();
        Ok(())
    }
}

#[derive(Debug)]
enum BlockState {
    Loading,
    Ready(Arc<[u8]>),
    Failed(SharedIoError),
}

#[derive(Debug)]
struct SharedIoError {
    kind: io::ErrorKind,
    message: Arc<str>,
}

impl SharedIoError {
    fn new(error: &io::Error) -> Self {
        Self {
            kind: error.kind(),
            message: Arc::from(error.to_string()),
        }
    }

    fn to_io_error(&self) -> io::Error {
        io::Error::new(self.kind, self.message.to_string())
    }
}

fn clipped_length(total: u64, offset: u64, requested: usize) -> io::Result<usize> {
    if offset >= total || requested == 0 {
        return Ok(0);
    }
    let available = total - offset;
    let requested_u64 = u64::try_from(requested).map_err(|_| integer_overflow())?;
    usize::try_from(available.min(requested_u64)).map_err(|_| integer_overflow())
}

fn lock<T>(mutex: &Mutex<T>) -> io::Result<MutexGuard<'_, T>> {
    mutex.lock().map_err(|_| poisoned("range cache"))
}

fn poisoned(component: &'static str) -> io::Error {
    io::Error::other(format!("{component} mutex is poisoned"))
}

fn integer_overflow() -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, "range input integer overflow")
}

#[cfg(test)]
mod tests {
    use std::{
        io,
        ops::Range,
        sync::{
            Arc, Barrier, Mutex,
            atomic::{AtomicUsize, Ordering},
            mpsc,
        },
        thread,
    };

    use decv_core::MediaInput;

    use super::{CachedRangeInput, RangeCacheConfig, RangeFetcher};

    #[derive(Debug)]
    struct MemoryFetcher {
        bytes: Arc<[u8]>,
        requests: AtomicUsize,
    }

    impl MemoryFetcher {
        fn new(bytes: impl Into<Arc<[u8]>>) -> Self {
            Self {
                bytes: bytes.into(),
                requests: AtomicUsize::new(0),
            }
        }
    }

    impl RangeFetcher for MemoryFetcher {
        fn len(&self) -> io::Result<u64> {
            Ok(u64::try_from(self.bytes.as_ref().len()).unwrap())
        }

        fn fetch_range(&self, range: Range<u64>) -> io::Result<Vec<u8>> {
            self.requests.fetch_add(1, Ordering::SeqCst);
            let start = usize::try_from(range.start).unwrap();
            let end = usize::try_from(range.end).unwrap();
            Ok(self.bytes[start..end].to_vec())
        }
    }

    fn config(block_size: usize, maximum_blocks: usize) -> RangeCacheConfig {
        RangeCacheConfig::new(
            block_size.try_into().unwrap(),
            maximum_blocks.try_into().unwrap(),
        )
    }

    #[test]
    fn reads_across_blocks_and_reuses_cached_bytes() {
        let input =
            CachedRangeInput::new(MemoryFetcher::new(*b"abcdefghijkl"), config(4, 2)).unwrap();
        let mut output = [0; 7];
        assert_eq!(input.read_at(2, &mut output).unwrap(), 7);
        assert_eq!(&output, b"cdefghi");
        assert_eq!(input.stats().snapshot().range_requests, 3);

        let mut cached = [0; 2];
        assert_eq!(input.read_at(5, &mut cached).unwrap(), 2);
        assert_eq!(&cached, b"fg");
        assert_eq!(input.stats().snapshot().range_requests, 3);
        assert!(input.stats().snapshot().cache_hits > 0);
    }

    #[test]
    fn clips_reads_and_prefetch_to_the_known_length() {
        let input = CachedRangeInput::new(MemoryFetcher::new(*b"abcdef"), config(4, 4)).unwrap();
        input.prefetch(4, 20).unwrap();
        let mut output = [0; 8];
        assert_eq!(input.read_at(4, &mut output).unwrap(), 2);
        assert_eq!(&output[..2], b"ef");
        assert_eq!(input.read_at(7, &mut output).unwrap(), 0);
    }

    #[test]
    fn lru_eviction_retains_the_recently_read_block() {
        let input =
            CachedRangeInput::new(MemoryFetcher::new(*b"abcdefghijkl"), config(4, 2)).unwrap();
        let mut byte = [0; 1];
        input.read_at(0, &mut byte).unwrap();
        input.read_at(4, &mut byte).unwrap();
        input.read_at(0, &mut byte).unwrap();
        input.read_at(8, &mut byte).unwrap();
        input.read_at(4, &mut byte).unwrap();
        assert_eq!(input.stats().snapshot().range_requests, 4);
        assert_eq!(input.stats().snapshot().evicted_blocks, 2);
    }

    #[derive(Debug)]
    struct GatedFetcher {
        bytes: Arc<[u8]>,
        requests: AtomicUsize,
        started: mpsc::Sender<()>,
        release: Mutex<mpsc::Receiver<()>>,
        clients_ready: Barrier,
    }

    impl RangeFetcher for GatedFetcher {
        fn len(&self) -> io::Result<u64> {
            Ok(u64::try_from(self.bytes.as_ref().len()).unwrap())
        }

        fn fetch_range(&self, range: Range<u64>) -> io::Result<Vec<u8>> {
            self.requests.fetch_add(1, Ordering::SeqCst);
            self.started.send(()).unwrap();
            self.clients_ready.wait();
            self.release.lock().unwrap().recv().unwrap();
            let start = usize::try_from(range.start).unwrap();
            let end = usize::try_from(range.end).unwrap();
            Ok(self.bytes[start..end].to_vec())
        }
    }

    #[test]
    fn concurrent_reads_share_one_in_flight_block() {
        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let input = Arc::new(
            CachedRangeInput::new(
                GatedFetcher {
                    bytes: Arc::from(*b"abcdefgh"),
                    requests: AtomicUsize::new(0),
                    started: started_tx,
                    release: Mutex::new(release_rx),
                    clients_ready: Barrier::new(2),
                },
                config(8, 2),
            )
            .unwrap(),
        );

        let first = {
            let input = input.clone();
            thread::spawn(move || {
                let mut output = [0; 4];
                input.read_at(0, &mut output).unwrap();
                output
            })
        };
        started_rx.recv().unwrap();
        let second = {
            let input = input.clone();
            thread::spawn(move || {
                let mut output = [0; 4];
                input.read_at(2, &mut output).unwrap();
                output
            })
        };
        input.fetcher.clients_ready.wait();
        release_tx.send(()).unwrap();

        assert_eq!(&first.join().unwrap(), b"abcd");
        assert_eq!(&second.join().unwrap(), b"cdef");
        assert_eq!(input.stats().snapshot().range_requests, 1);
    }

    #[derive(Debug)]
    struct ParallelFetcher {
        bytes: Arc<[u8]>,
        started: Barrier,
    }

    impl RangeFetcher for ParallelFetcher {
        fn len(&self) -> io::Result<u64> {
            Ok(u64::try_from(self.bytes.as_ref().len()).unwrap())
        }

        fn fetch_range(&self, range: Range<u64>) -> io::Result<Vec<u8>> {
            self.started.wait();
            let start = usize::try_from(range.start).unwrap();
            let end = usize::try_from(range.end).unwrap();
            Ok(self.bytes[start..end].to_vec())
        }
    }

    #[test]
    fn concurrent_block_overflow_is_trimmed_after_fetches_finish() {
        let input = Arc::new(
            CachedRangeInput::new(
                ParallelFetcher {
                    bytes: Arc::from(*b"abcdefgh"),
                    started: Barrier::new(3),
                },
                config(4, 1),
            )
            .unwrap(),
        );
        let first = {
            let input = input.clone();
            thread::spawn(move || {
                let mut byte = [0];
                input.read_at(0, &mut byte).unwrap();
            })
        };
        let second = {
            let input = input.clone();
            thread::spawn(move || {
                let mut byte = [0];
                input.read_at(4, &mut byte).unwrap();
            })
        };
        input.fetcher.started.wait();
        first.join().unwrap();
        second.join().unwrap();

        assert_eq!(input.cache.lock().unwrap().blocks.len(), 1);
        assert_eq!(input.stats().snapshot().evicted_blocks, 1);
    }

    #[test]
    fn cached_range_input_drives_the_mp4_demuxer() {
        let bytes = decode_hex(include_str!(
            "../../decv/tests/fixtures/three-frame-high-b.mp4.hex"
        ));
        let input = CachedRangeInput::new(MemoryFetcher::new(bytes), config(64, 4)).unwrap();
        let demuxer = decv_mp4::Mp4Demuxer::open(input).unwrap();
        let video_track = demuxer
            .movie()
            .tracks()
            .iter()
            .position(|track| track.kind() == decv_mp4::TrackKind::Video)
            .unwrap();

        let packet = demuxer.read_packet(video_track, 0).unwrap();
        assert!(!packet.data.as_ref().is_empty());
        assert!(demuxer.input().stats().snapshot().range_requests > 0);
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
