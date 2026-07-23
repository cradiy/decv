# H.264 Software Decode Performance

This document records repeatable performance comparisons for the current
decoder. Results are snapshots, not permanent claims: rerun the benchmark after
changing reconstruction, deblocking, frame storage, or scheduling.

## Benchmark Method

Run:

```text
./scripts/benchmark_h264_compare.sh 180 3
```

The script generates one deterministic three-second input and reuses it for
every decoder:

- 1920x1080 at 60 frames per second;
- 180 frames of `testsrc2`;
- H.264 High Profile with CABAC;
- three B pictures, three references, weighted P/B prediction, and 8x8 DCT;
- a 60-frame closed measurement GOP;
- three measured runs after one warm-up run.

The NV12 cases include decoding, conversion or packing to NV12, and writing the
visible frame bytes to `/dev/null`. The FFmpeg decode-only cases discard decoded
frames without requesting NV12. This distinction matters because `decv`
currently exposes CPU NV12 frames.

Snapshot environment:

- CPU: AMD Ryzen AI 7 H 350, 8 cores / 16 logical CPUs;
- FFmpeg 8.1.2;
- release-mode `decv-cli`;
- date: 2026-07-24.

## Current Results

Median of three runs:

| Decoder mode | Output | Wall time | User CPU | Peak RSS | Throughput |
| --- | --- | ---: | ---: | ---: | ---: |
| decv Serial | NV12 | 2.63 s | 2.55 s | 80,004 KiB | 68.4 FPS |
| decv Auto (2 workers) | NV12 | 2.61 s | 2.82 s | 79,628 KiB | 69.0 FPS |
| FFmpeg 1 thread | NV12 | 0.64 s | 0.73 s | 152,128 KiB | 281.3 FPS |
| FFmpeg Auto | NV12 | 0.27 s | 1.47 s | 290,468 KiB | 666.7 FPS |
| FFmpeg 1 thread | decode-only | 0.58 s | 0.56 s | 95,664 KiB | 310.3 FPS |
| FFmpeg Auto | decode-only | 0.22 s | 0.98 s | 192,040 KiB | 818.2 FPS |

On this workload:

- decv Serial takes about **4.1x** as much wall time as single-threaded FFmpeg
  when both produce NV12;
- decv Auto takes about **9.7x** as much wall time as FFmpeg Auto when both
  produce NV12;
- decv Auto does about **1.9x** as much total user-CPU work as FFmpeg Auto's
  NV12 path;
- decv uses about **53%** of FFmpeg single-threaded NV12 peak RSS and about
  **28%** of FFmpeg Auto NV12 peak RSS;
- prior measurements with 16 decv workers were slower than the two-worker
  `Auto` policy and consumed far more CPU, confirming that the current parallel
  region is too narrow to scale.

The 60 FPS real-time target requires decoding 180 frames in at most 3.00
seconds. The current Serial result has about 14.1% throughput headroom over that
line, and the measured two-worker Auto result has about 14.9%. The ordering
between Serial and Auto remains sensitive to scheduling and thermal state
because the current parallel region is narrow.

This snapshot includes the removal of repeated by-value copies of the
544-byte `MacroblockDeblockInfo` value from the deblocking traversal. Passing
that metadata by reference reduced the 180-frame Serial median from 4.61 to
3.74 seconds without changing the decoding algorithm. Reusing B-prediction
scratch buffers and removing redundant CABAC rollback snapshots from the
picture-terminal error path subsequently reduced the measured median to 3.63
seconds. Replacing tiny integer-motion `memmove` calls with validated unaligned
fixed-width loads and stores reduced it further to 3.55 seconds. Public
low-level CABAC APIs retain their transactional rollback semantics. Decoding
CABAC luma and chroma residuals directly into their final macroblock object,
instead of returning two large intermediate arrays, then reduced the median to
3.47 seconds. A checked public/unchecked internal split for CABAC coded-block
state recording eliminated repeated coordinate and grid validation, reducing
the measured median to 3.36 seconds. Reusing the fixed-width copy primitive for
B-partition assembly reduced the current result to 3.34 seconds.
Applying the same specialization to P-partition assembly reduced it to 3.31
seconds. Loading 64 bits when the single-bit reservoir is empty and batching
CABAC renormalization reads reduced the current Serial snapshot to 3.29
seconds. Vectorizing single-list prediction weights and compacting each
deblocking motion cell from 32 to 24 bytes then reduced the current snapshot
to 3.19 seconds. Replacing pointer-sized deblocking reference identities with
stable picture-local byte tokens reduced each motion cell from 24 to 10 bytes.
Fusing B-picture block residuals directly into their prediction matrices with
SSE2 then removed a 1.5 KiB assembled residual matrix and a second macroblock
output pass. Together these changes reduced the current snapshot to 3.06
seconds. Packing CABAC decision state, vectorizing two-dimensional luma and
chroma fractional interpolation, and building reference motion fields directly
in their final storage subsequently reduced the fixed result to 2.94 seconds.
Representing B-skip residuals as absent, rather than allocating and traversing
an explicit all-zero residual, reduced it further to 2.79 seconds.

A separate alternating A/B run used the same 300-frame stream and pinned CPU
sets to isolate those two bit-reading changes from run-to-run drift. Serial
median wall time moved from 5.515 to 5.470 seconds (about 0.8%), while Auto
moved from 5.375 to 5.290 seconds (about 1.6%). This confirms that bit reading
has a measurable effect, but is not a dominant explanation for the remaining
FFmpeg gap.

The prediction-weight SIMD change moved the pinned 300-frame Serial median
from 5.470 to 5.365 seconds and Auto from 5.310 to 5.180 seconds. Packing the
two reference identities and two motion vectors in each deblocking cell moved
Serial from 5.420 to 5.260 seconds and Auto from 5.240 to 5.120 seconds. The
metadata layout change reduces both the per-macroblock copy and the later
boundary-strength traversal footprint.

The picture-local reference-token change moved the pinned 300-frame Serial
median from 5.285 to 5.180 seconds (about 2.0%) and Auto from 5.210 to 4.990
seconds (about 4.2%). Direct B residual fusion moved a separate pinned Serial
run from 5.430 to 5.305 seconds (about 2.3%) and Auto from 5.120 to 4.970
seconds (about 2.9%). The residual path is checked against an assembled-matrix
oracle for both 4x4 and 8x8 transforms, including saturating extreme values.

Packing each CABAC context into one byte moved a pinned 300-frame Serial median
from 5.295 to 5.180 seconds (about 2.2%) and Auto from 5.070 to 4.945 seconds
(about 2.5%). The combined fractional-motion SIMD change moved Serial from
5.120 to 4.990 seconds (about 2.5%) and Auto from 4.935 to 4.895 seconds (about
0.8%). Building motion fields in final storage removed a whole-picture
`Option<Cell>` conversion; its Serial wall time was noise-level, but sampled
cycles fell about 1.0% and Auto wall time fell about 1.0%. Finally, omitting
all-zero B-skip residual objects moved Serial from 4.89 to 4.67 seconds (about
4.5%) and Auto from 4.80 to 4.52 seconds (about 5.8%).

Sharing the immutable planar luma allocation with the NV12 output frame avoids
one full-resolution Y-plane copy. A pinned 300-frame Serial A/B moved from 4.66
to 4.60 seconds (about 1.3%) while peak RSS fell by about 4.6 MiB. Auto wall
time remained noise-level in that run, while user CPU fell about 1.2% and peak
RSS fell about 2 MiB. NV12 planes now carry independent immutable backing
allocations, which the `CpuFrame` contract already permits.

Batching the copy-on-write uniqueness check for completed B macroblocks moved
the pinned 300-frame Serial median from about 4.67 to 4.56 seconds. Processing
implicit bidirectional weights per plane rather than per row subsequently
reduced whole-decoder instructions by about 3.2%. Reusing the four spatial
Direct neighbour cells for both reference lists reduced instructions by a
further 2.9% and moved an Auto run from about 5.06 to 4.83 seconds. Finally,
using the construction guarantees of the Direct partition grid when recording
motion cells, and explicitly expanding a four-element neighbour conversion
that LLVM otherwise lowered through `array::try_map`, each reduced pinned
Serial cycles by about 1%. Together, those changes moved the fixed benchmark
to 2.66 seconds, down from the preceding 2.81-second snapshot.

Skipping a complete four-edge vertical deblocking group when all four boundary
strengths are zero then reduced CABAC Serial instructions by about 2.1% and
cycles by about 2.5%. The same change reduced CAVLC instructions by about 2.4%
and cycles by about 1.2%. Applying the grouped check to horizontal edges was
near-neutral for CABAC Serial wall time, reduced CAVLC cycles by about 1.0%,
and moved a pinned Auto median from about 4.89 to 4.75 seconds. The current
fixed benchmark is 2.63 seconds in Serial mode and 2.61 seconds in Auto mode.

## BitReader Checkpoint

The generic `bit-readers` crate is no longer a leading whole-decoder hotspot.
On the current machine, representative Criterion medians are approximately:

| Workload | Median | Effective throughput |
| --- | ---: | ---: |
| mixed runtime-width fields | 45.6 us | 1.34 GiB/s |
| mixed compile-time-width fields | 29.1 us | 2.10 GiB/s |
| unsigned Exp-Golomb | 179.7 us | 348 MiB/s |
| peek and skip | 88.7 us | 704 MiB/s |

Three plausible generic-reader changes were tested and rejected:

- refilling the generic runtime-width path with 64 bits regressed its mixed
  microbenchmark by about 20%;
- reordering the cached-bit bound check improved isolated microbenchmarks by
  roughly 0.5% to 1.3%, but did not reduce real decoder hardware work and
  increased branch misses;
- eliminating the refill loop in favor of one refill plus one extraction made
  the mixed-field microbenchmark about 5.8% faster, but made the real CAVLC
  decoder about 1.8% slower in cycles, 1.5% slower in wall time, and increased
  branch misses by about 6%.

Current `perf` samples do not place a generic BitReader symbol above the 0.5%
reporting threshold on the CABAC comparison stream. Specialized bit ingestion
inside CABAC has produced measurable gains before, but optimizing the generic
reader solely from its microbenchmark would currently trade whole-decoder
performance for a better synthetic number. A future padded/trusted H.264 reader
could remove some atomic end-of-input checks, but it should be attempted only
with exact A/B decoder binaries and both CABAC and CAVLC inputs.

## Interpretation

The wall-time gap is not explained by thread count alone. Single-threaded
FFmpeg is already about 4.1x faster in the comparable NV12 case. FFmpeg then
reduces latency further with mature frame/slice threading, while decv currently
parallelizes only owned CABAC B-macroblock pixel reconstruction. CABAC parsing,
residual reconstruction, most P-picture reconstruction, output packaging, and
deblocking remain serial.

The immediate optimization priority should therefore remain single-thread hot
loops and broader dependency-aware parallelism, not a larger Rayon pool.
Deblocking and motion compensation are the most important measured hot regions.
Any new optimization must keep byte-exact output against FFmpeg and must be
benchmarked in both `Serial` and `Auto` modes.
