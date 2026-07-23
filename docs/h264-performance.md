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
| decv Serial | NV12 | 3.19 s | 3.10 s | 84,776 KiB | 56.4 FPS |
| decv Auto (2 workers) | NV12 | 3.37 s | 3.64 s | 83,648 KiB | 53.4 FPS |
| FFmpeg 1 thread | NV12 | 0.62 s | 0.70 s | 152,008 KiB | 290.3 FPS |
| FFmpeg Auto | NV12 | 0.26 s | 1.43 s | 293,400 KiB | 692.3 FPS |
| FFmpeg 1 thread | decode-only | 0.57 s | 0.55 s | 95,668 KiB | 315.8 FPS |
| FFmpeg Auto | decode-only | 0.22 s | 0.95 s | 192,960 KiB | 818.2 FPS |

On this workload:

- decv Serial takes about **5.1x** as much wall time as single-threaded FFmpeg
  when both produce NV12;
- decv Auto takes about **13.0x** as much wall time as FFmpeg Auto when both
  produce NV12;
- decv Auto does about **2.5x** as much total user-CPU work as FFmpeg Auto's
  NV12 path;
- decv uses about **56%** of FFmpeg single-threaded NV12 peak RSS and about
  **29%** of FFmpeg Auto NV12 peak RSS;
- prior measurements with 16 decv workers were slower than the two-worker
  `Auto` policy and consumed far more CPU, confirming that the current parallel
  region is too narrow to scale.

The 60 FPS real-time target requires decoding 180 frames in at most 3.00
seconds. The current 3.19-second Serial result is about 0.94x real time, or
roughly 6% more wall-clock work than the target permits. The measured
two-worker Auto median is 3.37 seconds, about 12% over the target. The ordering
between Serial and Auto is sensitive to scheduling and thermal state because
the current parallel region is narrow.

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
to 3.19 seconds.

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

## Interpretation

The wall-time gap is not explained by thread count alone. Single-threaded
FFmpeg is already about 5.1x faster in the comparable NV12 case. FFmpeg then
reduces latency further with mature frame/slice threading, while decv currently
parallelizes only owned CABAC B-macroblock pixel reconstruction. CABAC parsing,
residual reconstruction, most P-picture reconstruction, output packaging, and
deblocking remain serial.

The immediate optimization priority should therefore remain single-thread hot
loops and broader dependency-aware parallelism, not a larger Rayon pool.
Deblocking and motion compensation are the most important measured hot regions.
Any new optimization must keep byte-exact output against FFmpeg and must be
benchmarked in both `Serial` and `Auto` modes.
