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
| decv Serial | NV12 | 2.00 s | 1.92 s | 80,216 KiB | 90.0 FPS |
| decv Auto (2 workers) | NV12 | 2.08 s | 2.29 s | 78,960 KiB | 86.5 FPS |
| FFmpeg 1 thread | NV12 | 0.63 s | 0.70 s | 152,240 KiB | 285.7 FPS |
| FFmpeg Auto | NV12 | 0.27 s | 1.50 s | 293,132 KiB | 666.7 FPS |
| FFmpeg 1 thread | decode-only | 0.59 s | 0.57 s | 95,404 KiB | 305.1 FPS |
| FFmpeg Auto | decode-only | 0.23 s | 0.98 s | 192,428 KiB | 782.6 FPS |

On this workload:

- decv Serial takes about **3.2x** as much wall time as single-threaded FFmpeg
  when both produce NV12;
- decv Auto takes about **7.7x** as much wall time as FFmpeg Auto when both
  produce NV12;
- decv Auto does about **1.5x** as much total user-CPU work as FFmpeg Auto's
  NV12 path;
- decv uses about **53%** of FFmpeg single-threaded NV12 peak RSS and about
  **27%** of FFmpeg Auto NV12 peak RSS;
- prior measurements with 16 decv workers were slower than the two-worker
  `Auto` policy and consumed far more CPU, confirming that the current parallel
  region is too narrow to scale.

The 60 FPS real-time target requires decoding 180 frames in at most 3.00
seconds. The current Serial result has about 50.0% throughput headroom over that
line, and the measured two-worker Auto result has about 44.2%. The ordering
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
fixed benchmark at that point was 2.63 seconds in Serial mode and 2.61 seconds
in Auto mode.

For a residual-free inter macroblock whose sixteen deblocking motion cells are
identical, every internal boundary strength is provably zero. Detecting that
case once avoids 8 to 24 bidirectional motion comparisons and all three
internal threshold calculations. It moved a pinned 300-frame CABAC Serial
median from about 4.45 to 4.31 seconds, reduced CAVLC cycles by about 3.1%, and
moved Auto from about 5.25 to 4.95 seconds. Finally, constructing each retained
reference motion macroblock directly in its final array while using a `u16`
coverage mask for overlap and completeness checks reduced CABAC cycles by about
1.5% and CAVLC cycles by about 1.4%. The fixed benchmark at that point was 2.50
seconds in Serial mode and 2.45 seconds in Auto mode.

The reference motion field is always completely overwritten before it becomes
observable, so its Builder now keeps the backing allocation as
`MaybeUninit<MotionFieldCell>` and converts it only after the existing
all-macroblocks-complete check. This avoids pre-filling several hundred KiB per
1080p picture. Pinned A/B runs moved CABAC Serial from about 4.16 to 4.11
seconds, reduced CAVLC cycles by about 2.9%, and moved Auto from about 4.51 to
4.41 seconds. A partial-Builder Clone regression test covers the initialization
bitmap invariant.

Direct B macroblocks are normatively represented on an 8x8 or 4x4 motion grid,
but many of those cells carry an identical pair of reference motions. Pixel
reconstruction now proves that such a grid covers the macroblock without
overlap and coalesces it into one 16x16 prediction, while preserving the
original grid for motion-field recording and deblocking. On the pinned
300-frame inputs this reduced CABAC instructions by about 9.0% and cycles by
about 7.5%; CAVLC instructions fell about 10.4% and cycles about 9.6%. The
current fixed benchmark is 2.29 seconds in Serial mode and 2.30 seconds in Auto
mode.

The Direct resolvers now perform the same coalescing immediately after their
normative 4x4 motion-state cells have been recorded. Consequently reference
motion retention and deblocking metadata construction also see one 16x16
partition instead of revisiting 4 or 16 identical entries. Pinned CABAC
instructions fell a further 3.3%, cycles 2.1%, and cache misses 6.4%; CAVLC
instructions fell 3.8% and cycles 2.5%. The fixed Serial benchmark is now 2.23
seconds. Auto measured 2.37 seconds in this run, reinforcing that the current
narrow two-worker region remains scheduling-sensitive.

Spatial Direct's co-located zero flag can only change a non-zero predicted
motion vector that selects reference index zero. When neither list satisfies
that condition, the resolver now validates the co-located macroblock bounds
once and directly emits the already-coalesced 16x16 result without reading and
expanding its 4 or 16 co-located cells. Pinned CABAC cycles fell another 4.5%
and wall time 3.7%; CAVLC cycles fell 8.0% and wall time 7.8%. A regression test
ensures that the shortcut still rejects an undersized co-located motion field
without committing the macroblock. The fixed benchmark is now 2.19 seconds in
Serial mode and 2.26 seconds in Auto mode.

NV12 packaging now interleaves 16 Cb and Cr samples at a time with the
x86-64-baseline SSE2 unpack operations, using unaligned loads and stores into
the final initialized allocation. Exhaustive vector-boundary and tail tests,
the complete workspace suite, Clippy, and the real FFmpeg byte-exact verifier
all pass. On pinned 300-frame runs, CABAC instructions fell about 6.0% and
cycles 1.6%; CAVLC instructions fell about 7.0% and cycles 2.9%. The fixed
benchmark is now 2.11 seconds in Serial mode and 2.22 seconds in Auto mode.

CABAC P inter macroblocks now use the same four-row owned-job pipeline as
CABAC B macroblocks. Pinned 300-frame A/B samples reduced Serial cycles by
about 2.2% and wall time by about 1.7%. The two-worker Auto sample reduced wall
time by about 5.3%, although total cycles, instructions, and cache misses rose
because of worker and staging overhead. The fixed benchmark is now 2.10
seconds in Serial mode and 2.23 seconds in Auto mode, effectively unchanged in
Auto at this measurement resolution. The CAVLC path deliberately retains its
direct reconstruction path and remains neutral against the pre-change binary.

Worker-count scaling depends strongly on CPU placement. With the same
300-frame stream pinned to four performance cores, the medians were 3.52
seconds for Serial, 3.47 for two workers, 3.44 for three, and 3.31 for four.
Without affinity, the corresponding Serial/two/four medians were 3.51, 3.63,
and 3.59 seconds, while four workers used about 31% more user CPU than Serial.
`Auto` therefore remains capped at two; callers that control affinity may
explicitly request four workers.

Outlining the co-located-zero branch of spatial Direct motion reduced its hot
function body from roughly 8 KiB to 4.9 KiB, but did not reduce whole-decoder
instructions and increased sampled cache misses by about 1.2%. Six alternating
pinned runs were about 0.8% slower at the median, so the layout change was
rejected.

CABAC luma coded-block-pattern bits cover four 4x4 blocks, and a zero chroma
pattern infers ten DC/AC block states at once. Recording those normative
inferred-zero groups directly avoids repeating block-enum dispatch, macroblock
coordinate division, and pattern checks for every member. On the pinned
300-frame CABAC stream, instructions fell about 3.3%, branches 3.6%, cycles
1.7%, and the six-run Serial median improved about 1.8%. CAVLC is unaffected.
The fixed Serial benchmark is now 2.05 seconds; Auto remains scheduling-noisy
at 2.24 seconds.

Chroma deblocking now processes one complete eight-sample macroblock edge with
SSE2 while preserving the four independent two-sample boundary strengths.
Mixed-strength scalar-oracle tests cover both horizontal and vertical edges,
weak and strong filtering, zero-strength lanes, threshold rejection, and
saturating output. Horizontal edge vectorization reduced the pinned CABAC
median by about 2.0%, instructions by 2.9%, and cycles by 1.8%; the CAVLC
median improved about 3.8% with cycles down 4.8%. Vertical vectorization then
reduced CABAC by another 1.2% to 1.5% and CAVLC by about 0.7%, with instructions
down a further 2.2% to 2.4%. The complete workspace suite, Clippy, and every
real FFmpeg byte-exact stream pass. The fixed benchmark is now 2.00 seconds in
Serial mode and 2.11 seconds in Auto mode.

Completing a CABAC reconstruction batch now borrows each pending job before
clearing the vector instead of consuming it through `drain(..)`. This prevents
LLVM from moving each 776- or 848-byte owned job to the stack merely to retain
its address and deblocking metadata. Pinned CABAC cycles fell about 0.7% and
wall time about 0.4%. Packing the sixteen luma residual-presence booleans into
one `u16` then reduced the deblocking record footprint and moved the CABAC and
CAVLC medians by about 0.6% and 0.3%, respectively; CAVLC cycles fell about
1.2%. Finally, forcing the high-frequency boundary-strength derivation into
the sole picture traversal enabled constant propagation of internal versus
external edge rules. CABAC instructions and branches fell about 0.9%, with an
eight-pair wall-time median improvement of about 1.2%. CAVLC was neutral to
slightly faster. The current fixed benchmark remains 2.00 seconds in Serial
mode at its 0.01-second reporting resolution and measures 2.08 seconds in Auto
mode.

Writing deblocking records into the whole-picture array as soon as each CABAC
macroblock was parsed removed a later 188-byte copy, but worsened the
eight-pair pinned median by about 1.2%. It likely displaced hotter entropy and
reconstruction state before the batch completed, so that write-timing change
was rejected.

## BitReader Checkpoint

The generic `bit-readers` crate is no longer a leading whole-decoder hotspot.
On the current machine, representative Criterion medians are approximately:

| Workload | Median | Effective throughput |
| --- | ---: | ---: |
| mixed runtime-width fields | 45.6 us | 1.34 GiB/s |
| mixed compile-time-width fields | 29.1 us | 2.10 GiB/s |
| unsigned Exp-Golomb | 179.7 us | 348 MiB/s |
| peek and skip | 88.7 us | 704 MiB/s |

Four plausible reader changes were tested and rejected:

- refilling the generic runtime-width path with 64 bits regressed its mixed
  microbenchmark by about 20%;
- reordering the cached-bit bound check improved isolated microbenchmarks by
  roughly 0.5% to 1.3%, but did not reduce real decoder hardware work and
  increased branch misses;
- eliminating the refill loop in favor of one refill plus one extraction made
  the mixed-field microbenchmark about 5.8% faster, but made the real CAVLC
  decoder about 1.8% slower in cycles, 1.5% slower in wall time, and increased
  branch misses by about 6%;
- outlining the rare 1-to-7-bit CABAC refill path reduced the
  `decode_decision` machine-code body from 388 to 312-329 bytes and removed
  about 1.4% to 1.7% of whole-decoder instructions. Despite also reducing
  branches and sampled cache misses, six alternating pinned runs were
  consistently about 1% slower because the refill call lengthened the
  dependency path.

Two adjacent CABAC-core experiments were also rejected. Returning the internal
decision result through a compact `Option<u8>` reduced instructions by about
0.3% but made alternating wall-time runs roughly 1% slower. Combining the LPS
range and both context transitions into one 2 KiB table left instructions
nearly unchanged while increasing cycles by about 1.6% and wall time by about
2.5%.

Current `perf` samples do not place a generic BitReader symbol above the 0.5%
reporting threshold on the CABAC comparison stream. Specialized bit ingestion
inside CABAC has produced measurable gains before, but optimizing the generic
reader solely from its microbenchmark would currently trade whole-decoder
performance for a better synthetic number. A future padded/trusted H.264 reader
could remove some atomic end-of-input checks, but it should be attempted only
with exact A/B decoder binaries and both CABAC and CAVLC inputs.

## Interpretation

The wall-time gap is not explained by thread count alone. Single-threaded
FFmpeg is already about 3.2x faster in the comparable NV12 case. FFmpeg then
reduces latency further with mature frame/slice threading, while decv currently
parallelizes owned CABAC P- and B-macroblock pixel reconstruction. CABAC
parsing, residual reconstruction, output packaging, and deblocking remain
serial.

The immediate optimization priority should therefore remain single-thread hot
loops and broader dependency-aware parallelism, not a larger Rayon pool.
Deblocking and motion compensation are the most important measured hot regions.
Any new optimization must keep byte-exact output against FFmpeg and must be
benchmarked in both `Serial` and `Auto` modes.
