# Consumer API Boundary

The `decv` crate is the stable-candidate facade for ordinary library
consumers. It intentionally exposes a smaller surface than the codec
implementation crates:

- the synchronous `VideoDecoder` push/pull contract;
- compressed packets, exact timestamps, video formats, and immutable frames;
- `H264Decoder` and its reconstruction-parallelism policy;
- random-access MP4 demuxing and sequential packet cursors.

Code that only consumes decoded video should depend on `decv`, not directly on
the H.264 syntax, CABAC, motion, transform, prediction, or reconstruction
types exported by `decv-h264`. Those lower-level exports exist for decoder
development and may change while the codec implementation evolves.

The workspace is still version `0.1.0`, so consumers that need reproducible
builds should pin an exact repository revision. The facade defines the API
intended to converge toward semantic-versioning stability; it is not yet a
promise that no `0.1` source changes will occur.

## Decoder Lifecycle

Create the decoder, select H.264 parallelism before decoding starts, then
configure the compressed packet framing:

```rust
use decv::{
    BitstreamFormat, H264Decoder, H264Parallelism, VideoCodec, VideoDecoder,
    VideoDecoderConfig,
};

let mut decoder = H264Decoder::new();
decoder.set_parallelism(H264Parallelism::Auto)?;
decoder.configure(VideoDecoderConfig::new(
    VideoCodec::H264,
    BitstreamFormat::ByteStream,
))?;
# Ok::<(), decv::H264Error>(())
```

Annex-B input uses `BitstreamFormat::ByteStream`. MP4 normally supplies
length-prefixed AVC samples and an out-of-band `avcC` payload; obtain the
matching `VideoDecoderConfig` from `PacketCursor::decoder_config` instead of
constructing it manually.

Input ownership follows an explicit backpressure loop:

```rust
use decv::{
    DecodeInputStatus, DecodeOutput, EncodedVideoPacket, H264Decoder,
    VideoDecoder,
};

fn send_and_drain(
    decoder: &mut H264Decoder,
    mut packet: EncodedVideoPacket,
    mut output: impl FnMut(DecodeOutput),
) -> Result<(), decv::H264Error> {
    loop {
        match decoder.send_packet(packet)? {
            DecodeInputStatus::Accepted => break,
            DecodeInputStatus::NeedOutput(unconsumed) => {
                packet = unconsumed;
                loop {
                    match decoder.receive_frame()? {
                        DecodeOutput::NeedInput => break,
                        DecodeOutput::EndOfStream => {
                            output(DecodeOutput::EndOfStream);
                            return Ok(());
                        }
                        event => output(event),
                    }
                }
            }
            _ => {
                return Err(decv::H264Error::UnsupportedFeature(
                    "unknown decoder input status",
                ));
            }
        }
    }

    loop {
        match decoder.receive_frame()? {
            DecodeOutput::NeedInput => return Ok(()),
            DecodeOutput::EndOfStream => {
                output(DecodeOutput::EndOfStream);
                return Ok(());
            }
            event => output(event),
        }
    }
}
```

`FormatChanged` is emitted before the first frame using a new format. At end
of input, call `drain` and continue receiving until `EndOfStream`. After a
seek, reposition the packet cursor to a preceding keyframe, call `flush`, mark
the first packet as discontinuous, and discard decoded frames whose PTS is
before the requested target.

When using `H264Decoder` directly, prefer `flush_for_seek(target)` over plain
`flush` for exact seek preroll. It still reconstructs reference pictures and
maintains display reordering, but suppresses pre-target output materialization
and skips pixel reconstruction for pre-target non-reference pictures. This
avoids work whose result cannot affect the selected frame. The ordinary
trait-level `flush` clears this filter.

If an exact seek is already decoding forward and the requested target moves
later, call `retarget_seek_forward(later_target)` and keep feeding packets from
the current cursor. The decoder keeps its DPB, parser history, current picture,
and reorder state while dropping output below the newer target. This avoids
restarting the same GOP from its preceding keyframe for every forward scrub
update. Retargeting cannot move backward because frames suppressed for the old
target are no longer recoverable; use a container seek followed by
`flush_for_seek` for a backward target or a different decode timeline.

For repeated seeks in both directions, cache a decoder checkpoint together
with its exact next-sample cursor position:

```rust,ignore
let mut checkpoints = H264SeekCheckpointCache::new(
    4,
    128 * 1024 * 1024,
);
let resume_sample = cursor.next_sample_index();
let checkpoint = decoder.create_seek_checkpoint()?;
checkpoints.insert(checkpoint, resume_sample);

// Later, the cache enforces the strict resume-time bound:
if let Some(cached) = checkpoints.latest_before(target) {
    cursor.seek_to_sample(*cached.input_position())?;
    decoder.restore_seek_checkpoint(cached.checkpoint(), target)?;
}
```

Call `create_seek_checkpoint` only after feeding a complete access unit. It
finishes that unit before snapshotting parser history, the DPB, and display
reordering. The decoder derives `checkpoint.resume_time()` as the maximum PTS
of every completed access unit, rather than trusting decode order to match
presentation order. Every completed picture must therefore carry a PTS, and
the restored target must be later than that bound. Reference pictures and
motion fields are stored behind `Arc`, so checkpoint cloning does not copy full
pixel planes. Checkpoints can still retain old reference pictures, so
consumers should use a bounded, sparsely sampled cache rather than keeping one
for every frame. `retained_reference_count()` and
`estimated_retained_reference_bytes()` expose a conservative per-checkpoint
cache cost. Summing the byte estimate can overcount allocations shared by
multiple checkpoints, making it suitable for a simple upper-budget eviction
policy. `H264SeekCheckpointCache<T>` implements that policy with independent
entry and byte limits. It stores a caller-defined input-position token, keeps
entries ordered by resume time, selects the strict predecessor required by
restore, replaces duplicate resume points, and evicts the oldest checkpoints
first to retain the most recent decoded window.

For one active decoder, choose the least destructive valid transition in this
order:

1. If the new target is later and the current packet cursor can continue,
   call `retarget_seek_forward`.
2. Otherwise, restore the latest cached checkpoint whose `resume_time()` is
   strictly before the target and reposition the cursor with
   `seek_to_sample`.
3. If neither applies, seek the container to the preceding keyframe and call
   `flush_for_seek`.

Decoder mutation remains single-threaded and synchronous. A request scheduler
should stop feeding stale work between compressed packets and tag
consumer-owned results with its own request generation, because a frame
already returned to the consumer cannot be recalled. The first packet after a
keyframe seek must carry `discontinuity = true`. The first packet after
`restore_seek_checkpoint` must not: discontinuity handling deliberately clears
the DPB and would destroy the state that was just restored.

For a low-latency scrub preview, callers may instead use
`seek_to_nearest_keyframe`. That path begins at the independently decodable
picture closest to the requested presentation time and avoids preroll. It may
display an earlier or later timestamp, with equidistant keyframes preferring
the earlier one. `seek_to_keyframe_at_or_after` remains available when a
preview must never move backward. A typical timeline uses an approximate mode
while the pointer is moving, cancels stale requests, and performs the
preceding-keyframe exact seek after the interaction settles.

## Frame Ownership

`DecodedVideoFrame` is immutable and cheaply cloneable. CPU planes carry an
independent backing `Arc<[u8]>`, offset, stride, and row count. Consumers must
not assume that planes are adjacent or tightly packed.

Extensible facade enums, including `DecodeInputStatus`, `DecodeOutput`,
`FrameStorage`, and `PixelFormat`, are non-exhaustive. Matches must retain a
fallback arm. Extensible data structures provide constructors; consumers
should use those constructors instead of struct literals. This lets later
versions add device storage, packet side data, configuration, and metadata
without invalidating existing construction sites.

The current H.264 backend produces NV12:

- plane zero contains one 8-bit luma sample per coded pixel;
- plane one contains interleaved 8-bit Cb/Cr pairs at half width and height;
- `visible_rect` selects coded pixels that belong to the displayed image;
- `display_size` carries the final sample-aspect-ratio-adjusted dimensions;
- `ColorInfo` identifies range, matrix, primaries, and transfer metadata.

The decoder is synchronous and `Send`. A latency-sensitive consumer should
own it on a dedicated worker and pass immutable frames through a bounded
queue. Presentation order is already restored by the decoder; scheduling
should use frame PTS and duration rather than decode completion time.
