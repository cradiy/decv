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
- date: 2026-07-23.

## Current Results

Median of three runs:

| Decoder mode | Output | Wall time | User CPU | Peak RSS | Throughput |
| --- | --- | ---: | ---: | ---: | ---: |
| decv Serial | NV12 | 3.74 s | 3.62 s | 84,216 KiB | 48.1 FPS |
| decv Auto (2 workers) | NV12 | 3.80 s | 4.16 s | 84,392 KiB | 47.4 FPS |
| decv 16 workers | NV12 | 3.82 s | 6.75 s | 90,268 KiB | 47.1 FPS |
| FFmpeg 1 thread | NV12 | 0.62 s | 0.70 s | 151,780 KiB | 290.3 FPS |
| FFmpeg Auto | NV12 | 0.27 s | 1.47 s | 293,156 KiB | 666.7 FPS |
| FFmpeg 1 thread | decode-only | 0.58 s | 0.56 s | 95,912 KiB | 310.3 FPS |
| FFmpeg Auto | decode-only | 0.22 s | 0.99 s | 192,436 KiB | 818.2 FPS |

On this workload:

- decv Serial takes about **6.0x** as much wall time as single-threaded FFmpeg
  when both produce NV12;
- decv Auto takes about **14.1x** as much wall time as FFmpeg Auto when both
  produce NV12;
- decv Auto does about **2.8x** as much total user-CPU work as FFmpeg Auto's
  NV12 path;
- decv uses about **56%** of FFmpeg single-threaded NV12 peak RSS and about
  **29%** of FFmpeg Auto NV12 peak RSS;
- 16 decv workers are slower than the two-worker `Auto` policy and consume far
  more CPU, confirming that the current parallel region is too narrow to scale.

The 60 FPS real-time target requires decoding 180 frames in at most 3.00
seconds. The current 3.74-second Serial result is about 0.80x real time, or
roughly 25% more wall-clock work than the target permits.

This snapshot includes the removal of repeated by-value copies of the
544-byte `MacroblockDeblockInfo` value from the deblocking traversal. Passing
that metadata by reference reduced the 180-frame Serial median from 4.61 to
3.74 seconds without changing the decoding algorithm.

## Interpretation

The wall-time gap is not explained by thread count alone. Single-threaded
FFmpeg is already about 6.0x faster in the comparable NV12 case. FFmpeg then
reduces latency further with mature frame/slice threading, while decv currently
parallelizes only owned CABAC B-macroblock pixel reconstruction. CABAC parsing,
residual reconstruction, most P-picture reconstruction, output packaging, and
deblocking remain serial.

The immediate optimization priority should therefore remain single-thread hot
loops and broader dependency-aware parallelism, not a larger Rayon pool.
Deblocking and motion compensation are the most important measured hot regions.
Any new optimization must keep byte-exact output against FFmpeg and must be
benchmarked in both `Serial` and `Auto` modes.
