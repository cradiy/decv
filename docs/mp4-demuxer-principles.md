# MP4 Demuxer Principles

This document explains the ordinary and fragmented MP4 paths implemented by
`decv-mp4`. It focuses on the relationships that matter when maintaining the
code:

- how box traversal stays bounded;
- how several compact sample tables become one packet index;
- why DTS and PTS are different;
- how edit lists move media time onto the movie timeline;
- how an `avc1`/`avc3` track configures the H.264 decoder;
- what seeking can and cannot do by itself.

The main rule is:

> An MP4 demuxer does not decode video. It converts container metadata plus
> random-access bytes into correctly framed, timed compressed packets.

## 1. Input, container, and codec are separate layers

The current MP4 path moves data through three layers:

```text
MediaInput
    |
    v
decv-mp4
    |  EncodedVideoPacket
    |  data + PTS + DTS + duration + keyframe
    v
decv-h264
    |  DecodedVideoFrame
    |  NV12 + geometry + color + presentation time
```

Each layer answers a different question:

- `MediaInput`: where are the bytes?
- `decv-mp4`: which bytes form the next compressed sample, and when is it used?
- `decv-h264`: how do those compressed bytes reconstruct pictures?

Keeping these boundaries strict prevents storage policy and codec
reconstruction from leaking into the demuxer.

## 2. MP4 is a tree of bounded boxes

An ordinary box starts with:

```text
32-bit size
32-bit four-character type
payload...
```

For example:

```text
moov
├── mvhd
└── trak
    ├── tkhd
    ├── edts
    │   └── elst
    └── mdia
        ├── mdhd
        ├── hdlr
        └── minf
            └── stbl
                ├── stsd
                ├── stts
                ├── ctts
                ├── stsc
                ├── stsz or stz2
                ├── stco or co64
                └── stss
```

Some size encodings need special handling:

- size `1`: an additional 64-bit extended size follows;
- size `0`: the box extends to the end of its parent;
- type `uuid`: a 16-byte user type is part of the header.

Every child must fit completely inside its parent:

```text
parent_start <= child_start
child_end <= parent_end
child_size >= child_header_size
```

This is both a correctness rule and a security boundary. A malformed child
size must never let a parser read fields from a sibling box or from arbitrary
file offsets.

`Mp4File`, `BoxHeader`, `BoxIter`, and `BoundedReader` enforce this boundary.
Higher-level parsers should not perform unbounded reads themselves.

## 3. Why `MediaInput` is random-access

MP4 metadata and sample data are often far apart:

```text
ftyp | moov | free | mdat
```

or:

```text
ftyp | mdat | moov
```

A cursor-only API would force the demuxer to seek mutable state or copy the
whole file. The project instead uses:

```rust
pub trait MediaInput: Send + Sync {
    fn len(&self) -> std::io::Result<Option<u64>>;
    fn read_at(
        &self,
        offset: u64,
        buffer: &mut [u8],
    ) -> std::io::Result<usize>;
}
```

The consequences are useful:

- a local file can use positional reads;
- memory inputs need no cursor;
- HTTP or WebDAV implementations can use range requests;
- multiple packet cursors can read one immutable input independently;
- one extraction cursor does not disturb another cursor.

`read_at` is allowed to return a short read. All exact-field and exact-sample
paths therefore loop until the requested range is complete or a zero-byte read
proves truncation.

The current top-level MP4 parser requires a known length. It needs that length
to validate size-zero boxes and every absolute box range.

## 4. Movie time and media time are different

`mvhd` defines the movie time scale.

`mdhd` defines one track's media time scale.

They do not have to match. A value is meaningful only together with its scale:

```text
seconds = value / timescale
```

For example:

```text
movie:  500 / 1000  = 0.5 seconds
video: 6144 / 12288 = 0.5 seconds
```

`MediaTime` preserves the signed integer and explicit nonzero time scale:

```rust
pub struct MediaTime {
    pub value: i64,
    pub timescale: NonZeroU32,
}
```

Signed values are necessary because an edit list can make early decode
timestamps negative.

Never compare raw timestamp integers from different tracks. `MediaTime`
compares their exact rational values with cross multiplication.

## 5. A sample is the unit passed to the decoder

For an AVC video track, one MP4 sample normally contains one access unit:

```text
[NAL length][NAL bytes][NAL length][NAL bytes]...
```

The MP4 sample is not necessarily one NAL unit, and it is not necessarily one
slice. It can contain the NAL units needed for one compressed picture.

The demuxer's final `Sample` record contains:

```text
offset
size
decode_time
presentation_time
duration
sample_description_index
is_sync
```

No single MP4 box stores that complete record. It must be reconstructed by
combining several tables.

## 6. How the sample tables combine

### 6.1 `stsz` or `stz2`: sample sizes

`stsz` gives:

- one constant size used by every sample; or
- one explicit size for each sample.

After expanding `stsz`, the demuxer knows the total sample count and:

```text
size[sample_index]
```

This count becomes the reference count against which timing and chunk tables
are checked.

`stz2` is the compact alternative. It stores every size in a fixed 4-, 8-, or
16-bit field. Four-bit entries are packed most-significant nibble first. The
parser expands either representation into the same `Vec<u32>`, so all later
table-fusion logic is shared.

### 6.2 `stts`: decode times and durations

`stts` is run-length encoded:

```text
(sample_count, sample_delta)
```

Expanding its runs gives each sample duration and builds DTS:

```text
DTS(0) = 0
DTS(i + 1) = DTS(i) + duration(i)
```

The total number of samples described by `stts` must exactly match `stsz`.

### 6.3 `ctts`: presentation offsets

With frame reordering, decode order and presentation order differ.

`ctts` stores:

```text
composition_offset = PTS - DTS
```

Therefore:

```text
PTS(i) = DTS(i) + composition_offset(i)
```

Version 0 uses unsigned offsets. Version 1 uses signed offsets.

If `ctts` is absent:

```text
PTS = DTS
```

For a stream with B pictures, sample-table order remains decode order even
when PTS values appear out of order:

```text
decode order:       I  P  B  B
DTS:                0  1  2  3
PTS:                0  3  1  2
```

Packets must be sent to the decoder in sample/DTS order, not sorted by PTS.

### 6.4 `stco` or `co64`: chunk offsets

These tables give the absolute file offset of each chunk:

- `stco`: 32-bit offsets;
- `co64`: 64-bit offsets.

A chunk can contain one or more consecutive samples.

### 6.5 `stsc`: samples per chunk

`stsc` is another run table:

```text
(first_chunk, samples_per_chunk, sample_description_index)
```

Chunk and description indices stored in MP4 are one-based. Rust vectors are
zero-based, so the parser validates the stored value first and then subtracts
one.

To assign sample offsets:

```text
sample_index = 0

for each chunk:
    select the active stsc run
    offset = chunk_offset[chunk]

    repeat samples_per_chunk times:
        sample[sample_index].offset = offset
        sample[sample_index].description = description_index
        offset += sample[sample_index].size
        sample_index += 1
```

The algorithm rejects all count mismatches:

- chunks describe more samples than `stsz`;
- chunks describe fewer samples than `stsz`;
- an `stsc` description index does not exist;
- computed sample data exceeds the input length.

### 6.6 `stss`: sync samples

`stss` lists one-based sync-sample numbers.

For AVC, a sync sample is a random-access/keyframe candidate. If `stss` is
absent, every sample is treated as a sync sample, as required by the format.

The demuxer stores a compact list of zero-based sync sample indices for seek
queries.

### 6.7 Fragmented sample indexes

A fragmented MP4 can carry a valid but empty initialization `stbl`. Its real
samples arrive later in top-level movie fragments:

```text
moov/mvex/trex       track-level defaults
moof/traf/tfhd       fragment track and overridden defaults
moof/traf/tfdt       first decode time
moof/traf/trun       sample fields and media-data offset
mdat                 compressed sample bytes
```

`decv-mp4` expands each `trun` into the same `Sample` representation used by
ordinary tables. Duration, size, dependency flags, and sample-description
index are selected from the per-sample field, `tfhd`, or `trex` in that order.
Version 0 composition offsets are unsigned; version 1 offsets are signed.
The first sample's data position is derived from the validated fragment data
base plus `trun.data_offset`, and subsequent positions advance by their
declared sizes.

This normalization keeps packet cursors, edit-list mapping, decoder
configuration, and keyframe search independent of the source index shape. The
fragment parser rejects missing defaults, invalid description indices,
out-of-file data ranges, unsupported implicit data bases, and arithmetic
overflow before exposing samples.

## 7. `stsd`, `avc1`/`avc3`, and `avcC`

`stsd` contains sample descriptions. `stsc` tells each chunk which description
its samples use.

For the current video path:

```text
stsd
└── avc1 or avc3
    └── avcC
```

The visual sample entry provides coded metadata such as width and height.

The `avcC` payload is an `AVCDecoderConfigurationRecord`. It provides:

- H.264 profile and level metadata;
- the length-prefix width, from one to four bytes;
- out-of-band SPS and PPS NAL units.

The decoder configuration derived by `decv-mp4` is:

```rust
VideoDecoderConfig::new(
    VideoCodec::H264,
    BitstreamFormat::LengthPrefixed { length_size },
)
.with_codec_data(avcc)
```

MP4 AVC samples must not be treated as Annex-B. Their NAL boundaries are
length-prefixed, not marked by `00 00 01` start codes.

`decv-mp4` validates enough of the `avcC` header to derive framing. The H.264
decoder performs the codec-specific validation and parses the parameter sets.

## 8. Edit lists map media time onto presentation time

The raw sample tables describe the media timeline. They do not include the
track's `elst` mapping.

Each edit contains:

```text
segment_duration  // movie time scale
media_time        // media time scale, or -1 for an empty edit
media_rate        // signed 16.16 fixed point
```

An empty edit delays the start of a track. A normal media edit chooses the
point in the media timeline that appears at the current movie time.

For the common linear case supported today:

```text
presentation_offset =
    empty_duration converted to media ticks
    - media_start

adjusted_PTS = raw_PTS + presentation_offset
adjusted_DTS = raw_DTS + presentation_offset
```

For a real file inspected during development:

```text
movie timescale = 1000
media timescale = 12288
media_start = 1024
empty_duration = 0

presentation_offset = -1024
raw first PTS = 1024  -> adjusted PTS = 0
raw first DTS = 0     -> adjusted DTS = -1024
```

This exactly matches FFmpeg's packet timeline.

The current mapping accepts:

- no edit list;
- one unit-rate media edit;
- one or more initial empty edits followed by one unit-rate media edit;
- version 0 and version 1 `elst` records.

It explicitly rejects:

- repeated media segments;
- an empty edit after a media segment has begun;
- non-unit media rates;
- an empty duration that cannot be represented exactly in the media scale.

Those are valid container features, but silently flattening them into one
offset would be incorrect.

## 9. Reading a packet

`Track::read_packet` performs the following operation:

```text
1. Validate the sample index.
2. Enforce the packet allocation limit.
3. Read exactly sample.size bytes at sample.offset.
4. Apply the supported edit-list timestamp offset.
5. Build MediaTime values using the track media time scale.
6. Copy the sync-sample flag into packet.keyframe.
```

The result is:

```rust
EncodedVideoPacket {
    data,
    pts,
    dts,
    duration,
    keyframe,
    discontinuity: false,
}
```

Compressed bytes are owned by an `Arc<[u8]>`, so transferring a packet into
the decoder does not require another full payload copy.

The allocation cap is important. A corrupt four-byte sample size must not be
allowed to request an arbitrary multi-gigabyte allocation.

## 10. Packet cursors

`Mp4Demuxer<I>` owns:

```text
input + parsed Movie
```

It has no mutable file cursor. `PacketCursor` contains only:

```text
demuxer reference
track index
next sample index
```

This makes cursors cheap and independent. Multiple extraction operations can
create separate cursors over the same random-access source.

Sequential reading still follows sample-table/decode order:

```rust
let mut cursor = demuxer.packet_cursor(track_index)?;
let config = cursor.decoder_config()?.expect("non-empty track");
decoder.configure(config)?;

while let Some(packet) = cursor.next_packet()? {
    // Push packet and drain decoder output as required.
}
```

## 11. Seeking is a two-stage operation

Container seek and accurate video seek are not the same thing.

### Stage 1: demuxer seek

The demuxer finds the sync sample with the greatest adjusted PTS not after the
requested time:

```rust
let sample_index = cursor.seek_to_keyframe(target)?;
```

The cursor is repositioned to that sample in decode order.

Tracks retain a presentation-sorted sync-sample index, so both preceding and
following keyframe lookup are `O(log keyframe_count)` even for long media.

### Stage 2: decoder preroll

Accurate seeking is completed by:

```text
1. flush the decoder for the requested output start time;
2. start feeding packets from the selected sync sample;
3. decode reference and reordered pictures normally;
4. discard output frames whose PTS is before the exact target;
5. retain the first output frame at or after the target;
6. establish the new presentation-timeline origin.
```

Starting directly at an arbitrary non-keyframe packet is not accurate seek.
That packet may depend on reference pictures that have not been decoded.

`H264Decoder::flush_for_seek(target)` implements the first and fourth steps
inside the decoder. Pre-target reference pictures still participate in
reconstruction, the DPB, and deblocking. Pre-target non-reference pictures
cannot affect later reconstruction, so only their parsed picture timing and a
lightweight reorder marker are retained. The CLI uses this path and retains
its PTS filter as a defensive check.

The demuxer does not flush the decoder itself. That would couple the container
crate to a particular codec instance and higher-level state.

When an in-progress exact seek is retargeted to a later presentation time, the
packet cursor does not need to move backward. A direct H.264 consumer can call
`H264Decoder::retarget_seek_forward(later_target)` and continue feeding from
its current sample. The decoder preserves parsed history, reference pictures,
and display reordering, then filters pending output against the newer target.
This reuses work already completed within the GOP. Moving the target backward
requires either an older saved decoder checkpoint or selecting a preceding
sync sample and calling `flush_for_seek`, because pictures suppressed after
the retained state cannot be recreated from the current decoder state.

An H.264 checkpoint must be paired with the exact cursor position immediately
after its completed access unit:

```rust,ignore
let resume_sample = cursor.next_sample_index();
let checkpoint = decoder.create_seek_checkpoint()?;

cursor.seek_to_sample(resume_sample)?;
decoder.restore_seek_checkpoint(&checkpoint, new_target)?;
```

`PacketCursor::seek_to_sample` does not make an arbitrary non-sync sample
independently decodable; it only restores the container half of the saved
state. The decoder computes `checkpoint.resume_time()` as an exclusive lower
bound from the maximum PTS of all completed pictures, including future
reference pictures encountered before earlier B pictures in decode order. A
bounded sparse cache can retain several points within a long GOP. Each
checkpoint clones compact state and `Arc` handles rather than complete
reference pixels, although those handles keep their referenced pictures alive.
`H264SeekCheckpoint::estimated_retained_reference_bytes` provides a
conservative logical cost for an upper-budget eviction policy; shared
allocations may be counted by more than one checkpoint.

The first packet read after a checkpoint restore is a continuation of the
saved decode timeline and must not be marked discontinuous. By contrast, a
packet read after repositioning to a keyframe for a fresh exact seek must be
marked discontinuous. Mixing these cases silently discards the restored DPB
and turns the following inter picture into an invalid random-access start.

Cancellation is performed between samples: stop the stale packet loop, select
forward retarget, checkpoint restore, or keyframe restart in that order, then
resume feeding on the decoder's owning thread. Container reads are random
access and decoder calls are synchronous, so no container or codec task needs
to mutate the cursor concurrently.

### Low-latency preview seek

Interactive scrubbing often values response time over exact frame selection.
For that case:

```rust
let sample_index = cursor.seek_to_nearest_keyframe(target)?;
```

This starts at the presentation-nearest sync sample and therefore requires no
earlier GOP preroll. It uses the same presentation-sorted index and remains
`O(log keyframe_count)`. The first frame may precede or follow `target`; a tie
selects the earlier keyframe. Choosing the nearest of the adjacent sync
samples limits ordinary preview timestamp error to half of that keyframe
interval.

`seek_to_keyframe_at_or_after` remains useful when a preview must never move
backward, at the cost of potentially jumping by the complete keyframe
interval. A player can use either approximate mode while the pointer is
moving, cancel stale requests, and perform the exact preceding-keyframe seek
after the interaction settles.

## 12. Error handling and atomicity

Malformed input is normal parser input, not a reason to panic.

The MP4 path checks:

- integer addition and multiplication overflow;
- field reads crossing box bounds;
- child boxes crossing parent bounds;
- duplicate mandatory boxes;
- unsupported full-box versions and flags;
- declared table counts versus available bytes;
- sample-count agreement between tables;
- one-based indices before subtraction;
- sample ends against the known input length;
- allocation caps;
- short reads and truncation.

The unit suite also feeds every prefix truncation of a complete synthetic movie
through the owned demuxer, then applies deterministic single-byte corruptions
throughout the file. Any successfully parsed mutation is exercised through
decoder-configuration lookup and packet reads. Errors are ordinary outcomes;
panics are test failures.

Valid but currently unimplemented format behavior returns
`Mp4Error::UnsupportedFeature`. Corrupt structure returns
`Mp4Error::InvalidData`.

That distinction matters to callers:

- invalid data means the file is inconsistent;
- unsupported means the file may be valid, but this implementation cannot
  preserve its semantics yet.

## 13. Performance model

The current non-fragmented parser intentionally spends memory once to build a
flat sample index:

```text
parse compact tables once -> O(sample_count)
read a known sample        -> O(sample_size)
sequential packet access   -> O(1) index step + I/O
```

This is a good trade for random-access local files:

- timing runs are not expanded repeatedly;
- chunk lookup is not repeated for every seek;
- packet reads go directly to absolute offsets;
- no whole-file compressed buffer is retained;
- compressed working memory is bounded by one packet plus decoder state.

Potential future optimizations should be measured before implementation:

- preserve compact timing runs for extremely long tracks;
- pool packet buffers if allocation profiles show packet allocation as hot;
- add a range cache above `MediaInput` for network sources;
- parse `moov` lazily only if startup profiling justifies the complexity.

Do not optimize away validation. Container parsing handles attacker-controlled
sizes, counts, and offsets.

## 14. Current support boundary

Implemented:

- known-length random-access inputs;
- bounded ordinary, extended-size, size-zero, and UUID box traversal;
- `moov`, movie headers, video tracks, and AVC sample descriptions;
- `stts`, `ctts` version 0/1, `stsc`, `stsz`, `stz2`, `stco`, `co64`, and
  `stss`;
- `mvex`/`trex` fragment defaults;
- `moof`/`traf`, `tfhd`, version 0/1 `tfdt`, and version 0/1 `trun`;
- `edts`/`elst` version 0/1 parsing;
- common linear edit-list timestamp mapping;
- `avc1` and `avc3` configuration through `avcC`;
- indexed packet reads with PTS, DTS, duration, and keyframe status;
- sequential packet cursors and previous-keyframe seek;
- end-to-end MP4 AVC decoding through `decv-cli`.

Not implemented yet:

- fragment runs without an explicit base-data-offset or
  `default-base-is-moof`;
- encrypted/protected sample entries and CENC metadata;
- audio sample descriptions and interleaved A/V demuxing;
- complex/repeated/variable-rate edit lists;
- mid-stream sample-description switching in the CLI;
- subtitle and metadata packet APIs;
- progressive parsing when the total input length is unknown.

These limits should remain explicit. Falling back to guessed offsets or guessed
timestamps would create output that looks plausible while being wrong.

## 15. Verification

The reproducible MP4 regression is:

```bash
./scripts/verify_real_mp4.sh
```

It:

1. generates a 48-frame High Profile AVC MP4 with CABAC, B pictures, multiple
   references, weighted prediction, and edit-list timing;
2. decodes the MP4 with FFmpeg to visible NV12;
3. decodes the same MP4 through `decv-mp4` and `decv-h264`;
4. compares the complete raw outputs byte for byte;
5. seeks both implementations to a non-frame-aligned time after a later
   keyframe and compares the complete decoded suffix byte for byte.

The same accurate-seek path is available manually:

```bash
cargo run -p decv-cli -- --seek 1.37 file.mp4 output.nv12
```

For a first-frame-only latency probe, add `--max-frames 1`. The CLI stops
feeding compressed packets as soon as the selected frame is emitted instead
of decoding and draining the remaining suffix:

```bash
cargo run --release -p decv-cli -- \
    --seek 1.37 --max-frames 1 file.mp4
```

Useful local checks are:

```bash
cargo test --workspace --release
cargo clippy --workspace --all-targets -- -D warnings
cargo run -p decv-mp4 --example inspect -- file.mp4 --samples
```

The `inspect` example exposes both raw sample-table times and edit-adjusted
packet times. Comparing them to `ffprobe -show_packets` is a direct way to
diagnose timing or table-fusion mistakes.

## 16. References

- Apple QuickTime File Format documentation:
  <https://developer.apple.com/documentation/quicktime-file-format>
- Apple edit-list overview:
  <https://developer.apple.com/documentation/quicktime-file-format/edit_list_atom>
- Apple time-to-sample documentation:
  <https://developer.apple.com/documentation/quicktime-file-format/time-to-sample_atom>
- FFmpeg MOV/MP4 demuxer source:
  <https://ffmpeg.org/doxygen/trunk/mov_8c_source.html>
