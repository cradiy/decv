# decv

`decv` is an experimental, pure-Rust video decoding and demuxing library. Its
current software pipeline decodes H.264/AVC from Annex-B streams or ordinary
and fragmented MP4 files into immutable CPU NV12 frames.

The workspace provides deterministic media primitives:

- codec-independent packet, timestamp, format, color, and frame types;
- synchronous `Send` decoder objects with no async-runtime dependency;
- random-access input implemented for files and memory and extensible to other
  storage sources;
- explicit errors for unsupported, corrupt, and truncated input;
- byte-exact software reconstruction with bounded decoder state.

The API and supported bitstream surface are still evolving. Do not yet treat
the `0.1.0` crates as a complete H.264 conformance implementation.

## Workspace

| Crate | Responsibility |
| --- | --- |
| `decv` | Narrow consumer facade for decoding, immutable frames, H.264 configuration, and MP4 packet access |
| `decv-core` | Codec-independent time, packet, frame, color, input, and synchronous decoder contracts |
| `decv-h264` | Pure-Rust H.264 parsing, reconstruction, deblocking, DPB management, and NV12 output |
| `decv-mp4` | Synchronous random-access MP4 parsing, track/sample indexing, packet timestamps, and keyframe seek |
| `bit-readers` | Allocation-free MSB-first bit reading and Exp-Golomb primitives |
| `decv-cli` | Annex-B/MP4 decoding, seek, verification, and benchmark command |

The library crates and command-line tool are independently buildable.

Ordinary consumers should prefer the `decv` facade. The codec implementation
crates also expose lower-level parsing and reconstruction types for decoder
development; those internals are not part of the stable-candidate consumer
boundary.

## Current Video Support

The connected H.264 path currently supports:

- Annex-B and one-to-four-byte AVCC length-prefixed NAL units;
- out-of-band `avcC` SPS/PPS configuration;
- progressive frame-coded 8-bit YUV 4:2:0;
- Baseline, Main, and High profiles;
- CAVLC and CABAC;
- I, P, and B slices, including Direct and weighted prediction;
- multiple slices and multiple reference pictures;
- POC display reordering, decoded-picture-buffer marking, drain, and flush;
- SPS/PPS updates, coded size, crop, sample aspect ratio, and VUI color
  metadata;
- immutable strided CPU NV12 output with separate Y and UV allocations;
- serial, automatic, or explicitly sized reconstruction worker pools.

Valid H.264 features that are still rejected include:

- field pictures and MBAFF;
- FMO slice groups;
- SP and SI reconstruction;
- transform-bypass reconstruction;
- data-partition and slice-extension NAL units;
- chroma formats other than 4:2:0;
- bit depths other than 8-bit.

Unsupported syntax must return an explicit error. Corrupt or truncated
untrusted input must not panic or silently produce a known-wrong frame.

`decv-mp4` currently handles ordinary sample tables and movie fragments,
including track enumeration, AVC sample descriptions, `trex`/`tfhd`/`tfdt`/
`trun` sample indexing, DTS/PTS/duration, binary-searchable sync-sample
indexes, simple linear edit lists, packet cursors, exact-seek preroll from a
preceding keyframe, and low-latency preview seeks to a following keyframe. The
H.264 decoder additionally exposes
forward-target retargeting and reusable decode-state checkpoints for
interactive exact seeks, including a bounded checkpoint cache with generic
container cursor tokens and paired capture/restore operations. Audio decoding
and encrypted media are not yet implemented.

## Decoder Contract

`decv-core::VideoDecoder` is a synchronous push/pull interface:

```rust,ignore
pub trait VideoDecoder: Send {
    type Error: std::error::Error + Send + Sync + 'static;

    fn configure(&mut self, config: VideoDecoderConfig)
        -> Result<(), Self::Error>;
    fn send_packet(&mut self, packet: EncodedVideoPacket)
        -> Result<DecodeInputStatus, Self::Error>;
    fn receive_frame(&mut self) -> Result<DecodeOutput, Self::Error>;
    fn flush(&mut self);
    fn drain(&mut self) -> Result<(), Self::Error>;
}
```

One input packet may produce zero, one, or several output events. Callers must
handle `NeedOutput`, receive `FormatChanged` before frames using the new
format, and call `drain` at end of input. `flush` removes DPB and delayed
frames from the previous timeline after seek or discontinuity.

Decoded frames own or share immutable storage. A `CpuPlane` carries its real
backing allocation, offset, stride, and row count; consumers must not assume
that planes are adjacent or tightly packed.

See [Consumer API boundary](docs/consumer-api.md) for the supported facade,
packet backpressure loop, seek lifecycle, and frame ownership rules.

## Basic Use

Build the portable workspace:

```bash
cargo build --release
```

Decode Annex-B H.264 or MP4, optionally writing visible NV12 bytes:

```bash
cargo run --release -p decv-cli -- input.h264 output.nv12
cargo run --release -p decv-cli -- input.mp4 output.nv12
cargo run --release -p decv-cli -- --seek 12.5 input.mp4 output.nv12
cargo run --release -p decv-cli -- --seek 12.5 --max-frames 1 input.mp4
```

Select reconstruction parallelism with `--parallelism serial`, `auto`, or a
positive worker count. `--max-frames` stops as soon as the requested number of
output frames has been observed, without draining the rest of the input. This
is useful for measuring first-frame seek latency independently of subsequent
playback.

For opt-in frame-service latency statistics, build the CLI with its dedicated
feature:

```bash
cargo run --release -p decv-cli --features frame-timing -- \
    --frame-timing --parallelism auto input.h264
```

The summary reports mean, p50, p95, p99, and maximum wall time accumulated
inside decoder API calls between output-frame events. It excludes input file
reads, raw-frame writes, and per-frame logging. The first sample includes
decoder startup and any presentation-reordering pre-roll, so use a sufficiently
long stream when evaluating steady-state tail latency. The feature is absent
from ordinary release and PGO builds unless explicitly enabled.

## Portable and Tuned Builds

The normal Cargo build remains the portable library baseline. CPU-specific
builds are optional deployment choices:

```bash
./scripts/build_native_release.sh -p decv-cli
```

LLVM PGO can use representative videos to guide branch layout and inlining:

```bash
rustup component add llvm-tools-preview
./scripts/build_pgo_release.sh representative-4k.h264 representative.mp4
```

The PGO script builds and trains `decv-cli`, optimizing its complete linked
dependency graph. It records the toolchain, flags, parallelism modes, canonical
input paths, byte lengths, and SHA-256 hashes in
`target/pgo-data/training-manifest.tsv`. Compare the `input` rows before
attributing an A/B result to a source change: changing the relative number or
kind of training frames can change code layout independently. PGO and
`target-cpu=native` are not API requirements and should not be used for a
binary intended for unrelated CPUs or unrepresented workloads.

## Verification

Useful checks are:

```bash
cargo test --workspace --all-features
cargo clippy --workspace --all-targets --all-features -- -D warnings
./scripts/verify_real_h264.sh
./scripts/verify_real_mp4.sh
./scripts/benchmark_h264_compare.sh
```

The real-stream scripts generate inputs with FFmpeg/libx264 and require
byte-exact visible NV12 output. Performance changes are accepted only after
independent native binaries are measured on the same inputs.

## Design Notes

- [H.264 decoder principles](docs/h264-decoder-principles.md)
- [Consumer API boundary](docs/consumer-api.md)
- [H.264 performance record](docs/h264-performance.md)
- [H.264 reconstruction parallelism](docs/h264-parallel-decoding-plan.md)
- [MP4 demuxer principles](docs/mp4-demuxer-principles.md)
