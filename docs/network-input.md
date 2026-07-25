# Network media input

`decv-network` is an optional synchronous transport adapter. It turns an
immutable, known-length remote object into `decv_core::MediaInput`, so the MP4
demuxer and codec crates remain independent of HTTP clients, authentication,
and async runtimes.

This is intentionally not a streaming-protocol implementation. It supports
ordinary MP4 files exposed by a server with correct HTTP byte-range semantics.
HLS, DASH, live streams, unknown-length bodies, and servers that always return
the full object are outside this layer.

## Boundary

The responsibilities are split as follows:

| Layer | Responsibility |
| --- | --- |
| application/player | choose the URL, supply credentials, schedule blocking work, cancel obsolete playback work, select prefetch policy, and refresh expired authorization |
| `decv-network` | issue exact range requests, validate the remote object, coalesce concurrent block reads, bound cached data, and expose `MediaInput` |
| `decv-mp4` | parse the container, index samples, seek, and read packet bytes through `MediaInput` |
| codec | decode packets without knowing whether their bytes came from memory, a file, or a network |

This keeps network policy in the host application while avoiding a separate
HTTP implementation in every caller.

## HTTP contract

Opening `HttpRangeInput` performs a `Range: bytes=0-0` request. The response
must be `206 Partial Content`, use identity content encoding, and contain a
consistent `Content-Range` with a concrete total length. A server that returns
`200 OK` is rejected instead of being mistaken for random-access input.

If the probe returns a strong `ETag`, later requests use it in `If-Range`.
Otherwise, `Last-Modified` is used when available. A later full response,
changed validator, changed total length, wrong range, or wrong body length is
reported as invalid data. When neither validator is available, range and
length consistency are still enforced, but an in-place same-length object
change cannot be detected; applications should use immutable URLs for those
servers.

Custom headers may be used for authorization. `Range`, `If-Range`, and
`Accept-Encoding` are managed internally. Debug output does not include the
URL or header values.

## Use

Enable the optional facade feature:

```toml
[dependencies]
decv = { path = "../decv", features = ["network"] }
```

Construct the network input and pass it directly to the MP4 demuxer:

```rust,ignore
use decv::{HttpRangeInput, Mp4Demuxer, RangeCacheConfig};
use std::num::NonZeroUsize;

let cache = RangeCacheConfig::new(
    NonZeroUsize::new(256 * 1024).unwrap(),
    NonZeroUsize::new(32).unwrap(),
);
let input = HttpRangeInput::builder(download_url)
    .header("Authorization", format!("Bearer {token}"))?
    .cache_config(cache)
    .build()?;
let demuxer = Mp4Demuxer::open(input)?;
```

`HttpRangeInput` is synchronous because `MediaInput` and the current demuxer
are synchronous. Opening, demuxing, seeking, packet reads, and explicit
`prefetch` calls should run on a playback worker rather than a UI or async
executor thread. Applications that already own an HTTP configuration may pass
a configured `ureq::Agent` through the builder; cloned agents share their
connection pool.

## Cache behavior

The default cache uses 256 KiB blocks and retains up to 32 ready blocks. A
read that crosses blocks assembles the requested bytes transparently.
Concurrent reads of the same missing block share one request, while different
blocks can be fetched concurrently. Ready blocks use least-recently-used
eviction.

`prefetch(offset, length)` warms blocks synchronously. It is a mechanism, not
a playback policy: the upper layer decides whether to prefetch around the
current packet cursor, a seek target, or the next expected media segment.
`stats_snapshot()` exposes cache hits, misses, request count, fetched bytes,
and evictions for tuning.

The cache is also reusable without HTTP. Implement `RangeFetcher` for another
transport, then wrap it in `CachedRangeInput`. This is the intended extension
point for WebDAV clients, signed object storage, or an application's existing
request stack.
