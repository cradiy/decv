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
- native release-mode `decv-cli` (`-C target-cpu=native`);
- date: 2026-07-24.

## Current Results

Median of three runs:

| Decoder mode | Output | Wall time | User CPU | Peak RSS | Throughput |
| --- | --- | ---: | ---: | ---: | ---: |
| decv Serial | NV12 | 1.41 s | 1.33 s | 80,332 KiB | 127.7 FPS |
| decv Auto (2 workers) | NV12 | 1.57 s | 1.59 s | 79,996 KiB | 114.6 FPS |
| FFmpeg 1 thread | NV12 | 0.61 s | 0.68 s | 152,172 KiB | 295.1 FPS |
| FFmpeg Auto | NV12 | 0.26 s | 1.44 s | 290,248 KiB | 692.3 FPS |
| FFmpeg 1 thread | decode-only | 0.57 s | 0.55 s | 95,844 KiB | 315.8 FPS |
| FFmpeg Auto | decode-only | 0.22 s | 0.95 s | 192,328 KiB | 818.2 FPS |

On this workload:

- decv Serial takes about **2.3x** as much wall time as single-threaded FFmpeg
  when both produce NV12;
- decv Auto takes about **6.0x** as much wall time as FFmpeg Auto when both
  produce NV12;
- decv Auto does about **1.10x** as much total user-CPU work as FFmpeg Auto's
  NV12 path;
- decv uses about **53%** of FFmpeg single-threaded NV12 peak RSS and about
  **28%** of FFmpeg Auto NV12 peak RSS;
- prior measurements with 16 decv workers were slower than the two-worker
  `Auto` policy and consumed far more CPU, confirming that the current parallel
  region is too narrow to scale.

The 60 FPS real-time target requires decoding 180 frames in at most 3.00
seconds. The current Serial result has about 112.8% throughput headroom over
that line, and the measured two-worker Auto result has about 91.1%. The ordering
between Serial and Auto remains sensitive to scheduling and thermal state
because the current parallel region is narrow.

The Serial result represents about 264.7 million luma pixels per second. A
3840x2160 120 FPS stream represents about 995.3 million luma pixels per
second, so the current measured pixel throughput is still roughly 3.76x short
of the 4K120 target. Resolution scaling is not perfectly linear, but this is a
useful lower-bound checkpoint: 1080p120 is now demonstrated, while 4K120 still
requires materially faster kernels and broader frame/slice parallelism.

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

Strong luma deblocking now evaluates four adjacent sample sets with SSE2 in
both orientations. The shared vector kernel retains the normative narrow
`p0/q0` path and independently selects the wide three-sample tap set on each
side of each lane. A scalar-oracle test covers strengths 1 through 4, five QP
regions, horizontal and vertical traversal, smooth edges, and threshold
rejection. Pinned CABAC wall time improved about 1.8%, instructions and
branches about 0.7%, and branch misses about 1.8%. Seven-run CAVLC invariant
cycles were neutral to slightly lower, with instructions down about 0.4%. The
fixed benchmark now measures 1.99 seconds in Serial mode and 2.06 seconds in
Auto mode.

Inlining the small luma edge wrappers looked attractive after SIMD moved the
large kernels out of line. It improved CABAC wall time by about 2.1%, but
increased CAVLC invariant cycles about 1.2% when both orientations were
inlined. Vertical-only and horizontal-only variants were also slower on CAVLC,
by about 2.0% and 0.5%, respectively. The resulting picture-traversal code
layout is workload-sensitive, so all wrapper-inline variants were rejected.

Building the serial CABAC P- and B-reconstruction batches with explicit
preallocated loops instead of fallible iterator collection stopped
`Iterator::try_process` from repeatedly moving the 392-byte staged macroblock
value. On the pinned CABAC stream, the median improved about 1.4% while
instructions and branches fell about 0.6%; sampled `memmove` overhead fell
from roughly 11% to 8.4%. CAVLC does not use this owned pending-job pipeline
and was unaffected.

The 4x4 partition-coverage grid used by inter prediction is now one `u16`
instead of sixteen booleans. CABAC reference cycles were neutral while
branches fell about 1.1%; CAVLC reference cycles improved about 1.6% with a
similar branch reduction. More importantly, the serial CABAC path now
constructs each 384-byte macroblock pixel result directly inside its
preallocated staged slot. The owned return-value API remains in place for
parallel workers, but serial reconstruction no longer returns and wraps that
large value before pushing it into the batch. Alternating pinned samples
reduced CABAC reference cycles about 3.0%, instructions about 0.9%, and
branches about 1.1%; CAVLC remained neutral to slightly faster. Sampled
`memmove` overhead fell further to about 7.8%. The fixed benchmark now measures
1.92 seconds in Serial mode and 2.09 seconds in Auto mode.

Whether all internal deblocking edges are zero is now derived while each
inter-macroblock motion grid is constructed and stored in previously unused
padding in `MacroblockDeblockInfo`; the structure remains 176 bytes. This
avoids comparing fifteen ten-byte motion cells during the later picture pass.
CABAC instructions fell about 0.6%, branches about 2.7%, and reference cycles
about 0.9%; CAVLC showed a similar branch reduction and slightly lower
reference cycles. A single full 16x16 P/B partition also fills the deblocking
motion array as one uniform grid instead of using dynamic nested loops. That
removed a further 0.4% of instructions and reduced
`b_inter_deblock_info` from roughly 2.6% to 1.8% of sampled cycles.

Reference motion retention now has an equivalent final-storage fast path.
Intra macroblocks and single full 16x16 inter partitions write one
`MotionFieldCell` directly to the sixteen destination cells, bypassing the
large local array, coverage walk, and second copy. CABAC instructions fell
about 3.0% and reference cycles about 2.4%; CAVLC instructions fell about 3.4%
and reference cycles about 2.8%. `MotionFieldBuilder::record_b` consequently
disappeared from the greater-than-1% profile. The fixed benchmark now measures
1.89 seconds in Serial mode and 2.10 seconds in Auto mode.

The uniform writer subsequently stopped materializing a sixteen-element global
index array and instead writes four contiguous cells on each of the four
motion-field rows. CABAC instructions fell another 0.8%, branches 0.6%, and
reference cycles about 2.3%; CAVLC instructions fell 0.9% with reference
cycles down about 0.5%. This keeps the same per-row bounds established by the
validated picture and macroblock address.

On AVX2-capable x86-64 CPUs, a sixteen-pixel single-axis luma partition now
evaluates each six-tap row in one 256-bit operation instead of two eight-pixel
SSE2 chunks. Runtime feature detection retains the SSE2 fallback for other
x86-64 CPUs, and a direct scalar-oracle test covers all six horizontal and
vertical quarter-sample positions. Alternating samples reduced CABAC reference
cycles about 1.1% and CAVLC about 0.9%. The fixed benchmark now measures 1.87
seconds in Serial mode and 2.09 seconds in Auto mode.

The 4x4 inverse-quantization path now validates the QP and scaling list,
inverse-scans the scaling weights, and derives the sixteen position-specific
level scales once per luma or chroma block group rather than once per block.
The prepared level scales fit in a compact `u16[4][4]`; checked coefficient
arithmetic and the public transform API retain their existing overflow
semantics. Five alternating pinned runs reduced CABAC instructions about 2.2%
and reference cycles about 0.3%, while CAVLC instructions fell about 2.8% and
reference cycles about 1.0%.

Two more aggressive layouts were rejected. Storing the prepared scales as
`i32` or `i64` still removed roughly 2% of instructions, but slightly increased
CABAC reference cycles because the larger live context outweighed the saved
work. Fusing inverse scan and scaling into one iterator loop increased
whole-decoder instructions by about 0.4% and regressed CAVLC reference cycles,
so the small intermediate coefficient block remains.

The separable 4x4 inverse transform no longer performs checked `i64`
addition and subtraction at every butterfly. Starting from `i32`
coefficients, one pass grows the absolute bound by less than 3.5x and the
second pass therefore remains below `2^36`; the final checked conversion is
unchanged. An `i128` oracle covers uniform extrema, a maximum/minimum
checkerboard, and an extreme impulse at every coefficient position. Seven
alternating pinned runs reduced CABAC reference cycles about 1.4% and CAVLC
about 1.2%, despite reducing whole-decoder instructions by only 0.1% to 0.2%.
The gain comes mainly from shortening the dependency chain through each
butterfly.

Decoder-internal 4x4 reconstruction now writes into its final luma or chroma
block slot and returns `Result<()>`. The public API still returns an owned
block transactionally, but the hot macroblock loops no longer materialize a
`Result<Block4x4>`, inspect its discriminant, and copy the 64-byte success
payload. Seven alternating pinned runs reduced CABAC reference cycles about
1.0% and instructions 0.3%; CAVLC reference cycles fell about 0.5% and
instructions 0.4%. After the three 4x4 residual changes, the fixed benchmark
measures 1.78 seconds in Serial mode and 1.98 seconds in Auto mode. FFmpeg
takes 0.61 seconds with one thread for the comparable NV12 output, leaving a
roughly 2.9x single-thread gap.

The same internal output pattern now applies between inverse scaling and the
integer transform, removing the remaining `Result<Block4x4>` temporary inside
the prepared reconstruction path. CABAC reference cycles were neutral while
instructions fell 0.6%; CAVLC instructions fell 0.8% and reference cycles
about 3.5%.

Full-macroblock Spatial Direct prediction now loads its fixed A, B, C, and D
4x4 neighbours directly from cells 3, 12, 12, and 15 of the adjacent
macroblocks. The old general partition helper converted four pixel
coordinates back into macroblock and local-cell coordinates with repeated
bounds checks, division, and remainder operations. Slice filtering and
top-right-to-top-left fallback are unchanged. Seven alternating pinned runs
reduced CABAC reference cycles about 1.3%, instructions 0.3%, and branches
0.4%; CAVLC reference cycles fell about 1.8% with a similar instruction
reduction.

A diagnostic run over the 300-frame CABAC stream counted 1,135,509 Spatial
Direct macroblocks on the existing zero-change fast path, 353,134 grids that
were constructed and then coalesced to one uniform partition, and only 1,777
genuinely non-uniform grids. An attempted uniform-zero-flag shortcut therefore
removed about 0.6% of whole-decoder instructions and 0.9% of branches, and
made CAVLC faster. It was still rejected: inlining the shortcut grew the
already-large function from about 8 KiB to 10 KiB and regressed CABAC
reference cycles about 0.5%; outlining the rare grid builder shrank the main
body below 8 KiB but regressed CABAC about 0.9%. This path is sensitive to
front-end layout and branch prediction, so fewer derived partitions alone are
not sufficient evidence of a win.

On AVX2-capable x86-64 CPUs, the common sixteen-luma-pixel chroma partition now
interpolates its eight Cb and eight Cr samples together as sixteen `u16` lanes.
This shares the four bilinear weights, rounding, and packing work across both
planes; runtime feature detection retains the SSE2 path for other x86-64 CPUs
and for narrower partitions. The SIMD oracle covers every one of the 64
fractional positions and compares AVX2, SSE2, and per-sample scalar results.
Seven alternating pinned runs reduced CABAC reference cycles about 1.4% and
CAVLC about 0.5%. Whole-decoder instructions were effectively neutral, while
branches fell about 0.3%.

Eight-luma-pixel chroma partitions now pack four Cb and four Cr source samples
into one SSE2 vector, sharing each bilinear operation across both planes.
One-axis fractional positions also avoid loading and multiplying the two
unused diagonal sources. Scalar and original per-plane SIMD oracles cover all
64 fractional positions. Whole-decoder instructions and branches each fell
about 0.06% to 0.08%; CABAC reference cycles remained within measurement
noise, while CAVLC samples were consistently no slower and often faster.

Applying the same one-axis shortcut inside the AVX2 wide-partition loop was
rejected. LLVM expanded the helper from about 405 bytes to 2.3 KiB, while
CABAC did not show a stable cycle improvement. The compact four-source AVX2
kernel remains preferable to trading instruction count for front-end pressure.

Reconstructed inter luma and both chroma planes now live in one allocation,
and the owned residual handle is one machine word instead of roughly 520
bytes. Reconstruction initializes the luma and chroma blocks directly inside
that allocation, so CABAC pending P/B jobs move only a pointer while retaining
the previous one-allocation-per-coded-macroblock behavior. A layout regression
test fixes the pointer-sized handle invariant. Five alternating pinned runs
reduced CABAC instructions about 0.3%, branches about 0.2%, and reference
cycles about 3.4%; every paired run was faster by roughly 3.0% to 4.7%.

This layout deliberately favors the batched CABAC path. CAVLC consumes each
residual immediately and previously benefited from keeping the chroma arrays
on the stack; its instructions increased about 0.1% and reference cycles about
0.5%. That measured tradeoff is accepted for the current high-profile,
high-frame-rate target, where CABAC is the primary workload, rather than
adding a second residual representation and reconstruction pipeline.

CABAC coefficient decoding now writes its 64-entry backing array into a
caller-owned block and returns only whether `coded_block_flag` selected it.
The public owned APIs are unchanged, but the decoder hot path no longer
returns `Result<Option<CabacCoefficientBlock>>` with an approximately
264-byte success payload for every coded transform block. Five alternating
pinned runs reduced CABAC instructions about 0.4%, branches about 0.9%, and
reference cycles about 0.7%. CAVLC instructions were neutral.

CABAC 8x8 luma coefficient splitting also writes its four `ResidualBlock`
values directly into their final macroblock slots. This removes the temporary
four-block return and its subsequent 288-byte copy. Five alternating pinned
runs reduced instructions and branches by about 0.04% to 0.05%; median
reference cycles improved about 0.7%. The full byte-exact FFmpeg corpus still
matches.

Pending P/B reconstruction jobs no longer carry a copy of
`MacroblockDeblockInfo`. The metadata is staged in its final
macroblock-addressed slot while the completion bit remains unset until the
pixel batch commits. Five alternating pinned runs improved median reference
cycles about 1.6% in serial mode and 0.6% with the two-worker `Auto` mode.
Instructions remained effectively flat, while branches fell about 0.06% in
serial mode and 0.08% in `Auto`. The refreshed fixed benchmark now measures
1.76 seconds in Serial mode and 1.89 seconds in Auto mode.

Hoisting integer-motion row-width dispatch out of the luma and chroma copy
loops was rejected. Compile-time-width row kernels removed about 1.9% of
whole-decoder instructions and 5% of branches, but twelve alternating pinned
runs showed reference cycles about 0.8% worse in the reversed-order set. The
existing repeated branches are highly predictable; the shorter loop appears
to lengthen the row-to-row dependency path.

Forcing the four-sample vertical and horizontal luma deblock dispatchers
inline was also rejected. Five alternating pinned runs increased instructions
about 0.2% and median reference cycles about 1.4%. A useful next deblock step
must batch a complete edge rather than expanding the existing small kernels
inside picture traversal.

Picture traversal now submits each complete sixteen-sample luma edge to one
out-of-line batch entry. The x86-64 backend reuses the proven four-lane SSE2
weak and strong kernels inside that compact entry, while the portable backend
retains segmented scalar filtering. Five alternating pinned runs reduced
CABAC instructions about 0.65%, branches about 0.28%, and median reference
cycles about 0.9%. CAVLC improved by about 0.77%, 0.29%, and 1.3%,
respectively. Two-worker `Auto` instructions fell about 0.63%; its reference
cycles were noise-level but slightly lower.

Preparing filter parameters once when all four edge strengths match was
rejected. Applying it to both orientations increased instructions about 0.14%
without improving reference cycles. A vertical-only variant increased
instructions about 0.18% and median reference cycles about 0.5%. Further
vertical work must reduce the number of SIMD arithmetic calls, not only share
their setup.

An eight-row byte transpose then reduced a uniform weak vertical edge from
four four-lane arithmetic passes to two eight-lane passes. CABAC instructions
fell about 0.19% and its reference-cycle median improved about 0.6%, but CAVLC
reference cycles regressed about 1.6%. Restricting the path independently to
strengths one, two, or three did not remove that tradeoff, so the larger
transpose kernel was rejected.

Skipping the second coverage walk for an already coalesced uniform Direct B
partition was rejected in two forms. A branch inside the prediction loop
reduced instructions about 0.06% but made median reference cycles about 1.1%
worse. Outlining all non-uniform coverage validation reduced instructions only
slightly, increased branches about 0.2%, and was slower in every paired run.

The adjacent 65-byte significance-map return was also converted to a
caller-owned buffer and rejected. It increased CABAC instructions about 0.1%
and reference cycles about 0.3%, indicating that LLVM already handles this
smaller return more efficiently than the explicit initialization and mutable
output path.

Parallel P/B reconstruction workers now initialize their address-ordered
macroblock output slots in place. The slots are allocated as
`MaybeUninit<StagedMacroblockPixels>` and become initialized on the worker
that reconstructs them, so the main thread neither receives a roughly
392-byte owned result nor takes initial ownership of its cache lines by
zero-filling them. Ordinary indexed Rayon collection runs every job and
preserves error order before the initialized boxed slice is converted into a
`Vec`. Five alternating runs pinned to the two `Auto` CPUs reduced
instructions about 0.9%, branches about 1.3%, median reference cycles about
1.4%, and median wall time about 0.7%; every paired cycle and wall-time sample
was faster. A preliminary version that zero-filled the slots on the main
thread removed the same result copies but made cycles slightly worse,
confirming that first-touch placement is part of the optimization.

Shrinking `ResolvedBMacroblock` from four inline `SmallVec` partitions to one
was rejected. Although Direct and Skip macroblocks often coalesce to one
partition, real x264 B pictures also use enough multi-partition macroblocks
that spilling them to the allocator increased whole-decoder instructions
about 0.34%, branches about 0.54%, and branch misses about 3%. Four of five
paired CABAC runs used more reference cycles. The four-partition inline
capacity remains the better representation; future work on its visible
`memmove` cost must eliminate moves without introducing per-macroblock
allocations.

A paired SSE2 kernel for the two Cb and two Cr samples belonging to a 4x4
luma partition was also rejected. It was byte-exact for all 64 fractional
positions and reduced whole-decoder instructions about 0.06% and branches
about 0.09%, but five paired CABAC runs put median reference cycles about
0.16% and wall time about 0.5% higher. Packing four useful samples into an
SSE2 vector does not amortize its setup on this workload.

Changing the 16-cell B-motion commit helper from an owned array parameter to
a borrowed array was rejected after disassembly. LLVM had already lowered the
owned ABI to a single 320-byte copy into the address-indexed motion state, and
the borrowed version emitted the same copy. Serial measurements happened to
improve through code-layout changes, but two-worker `Auto` instructions rose
about 0.08% and branches about 0.16% without a stable wall-time win. The source
change therefore did not remove the operation it was intended to optimize.

Writing a uniform spatial-Direct motion cell into its sixteen final slots with
slice `fill` was also rejected. It avoided the stack array and reduced
whole-decoder instructions about 0.25% and branches about 0.14%, but four of
five CABAC runs used more reference cycles and later paired wall-time samples
regressed roughly 1% to 2.6%. The optimized fixed-size copy has a longer
instruction stream but better throughput than repeated structured stores on
this CPU.

Address-ordered macroblock commit now validates the complete batch once and
copies each fixed 16x16 luma and 8x8 chroma block with a private raw-pointer
row kernel. The prior safe slice loop was fully unrolled by LLVM, including a
multiply and two bounds checks for every row; the validated kernel advances
destination pointers by the stride and emits only the fixed-width loads and
stores. Rollback restoration performs its own one-time macroblock range
assertion before entering the same unsafe boundary. Five alternating CABAC
Serial runs reduced instructions about 1.86%, branches about 3.39%, median
reference cycles about 0.9%, and median wall time about 1.7%. CAVLC improved
about 1.51%, 2.6%, 1.6%, and 1.5%, respectively. Two-worker `Auto`
instructions fell about 1.66%, branches about 2.98%, and all five paired cycle
samples improved, with the median about 1.0% lower; its wall time remained
within scheduling noise.

Consecutive staged macroblocks now derive division-based picture coordinates
only for the first entry and after a real address gap. The common CABAC batch
advances x/y counters across rows, while a regression test covers both a row
transition and a non-consecutive address. Five alternating Serial runs reduced
CABAC instructions about 0.06%, branches about 0.24%, median reference cycles
about 1.3%, and median wall time about 1.6%. CAVLC's mostly single-entry commit
path gained about 0.09% instructions but kept slightly lower median cycles and
wall time. Two-worker `Auto` reference cycles improved about 1.1%; its roughly
0.12% higher instruction count and wall time remained scheduling-sensitive.
The tradeoff is accepted for the current CABAC-first target.

CABAC P/B deblocking metadata is now constructed directly in its final
picture-owned slot instead of returning a large `MacroblockDeblockInfo` and
copying it into place. The direct writers rely on the internal motion
resolvers' full 4x4-grid coverage, guarded by release-free debug assertions and
an equivalence test against the value-returning builders. The CAVLC builders
remain separate because routing them through the shared out-parameter path
measurably increased their instruction count. Five alternating final-binary
runs reduced CABAC Serial instructions about 0.61%, branches about 1.20%, and
median reference cycles about 0.76%, with all five cycle samples improving.
Two-worker `Auto` instructions fell about 0.44%, branches about 0.84%, and
median reference cycles about 2.1%; four of five cycle samples improved, with
the wider median reflecting scheduler variance. Five CAVLC runs remained
neutral: instructions and branches changed by less than 0.01%, while median
cycles were about 0.2% lower.

Moving spatial-Direct B neighbour preparation into one non-inlined helper was
rejected. It reduced `resolve_spatial_direct_macroblock` from roughly 8.1 KiB
to 6.9 KiB, but introduced a roughly 1.9 KiB helper and an out-of-line
64-byte neighbour-pair return. Five alternating CABAC Serial runs increased
instructions about 0.44% and median reference cycles about 0.39%; the code was
fully reverted. Reducing one symbol's size is not useful when the split adds
more aggregate code and return-value traffic.

The workspace release profile now uses fat LTO and one code-generation unit.
This gives LLVM visibility across crate boundaries and favors runtime
throughput over build speed. The release CLI shrank from about 1.5 MiB to
1.3 MiB. Five alternating CABAC Serial runs reduced instructions about 7.0%,
branches about 11.7%, and median reference cycles about 2.1%, with all five
cycle samples improving. CAVLC instructions fell about 7.7%, branches about
12.5%, and median cycles about 1.6%, also with five of five improvements.
Two-worker `Auto` instructions fell about 7.7%, branches about 12.2%, and
median cycles about 3.7%, again improving every paired cycle sample.

Release builds also use aborting panics. Normal malformed-stream and decoder
failures continue to use `Result`; only an internal panic no longer unwinds.
On top of fat LTO this reduced the release CLI to about 1.2 MiB. Five CABAC
Serial runs reduced instructions about 0.24%, branches about 0.61%, and median
cycles about 0.50%; CAVLC median cycles likewise fell about 0.51%. `Auto`
instructions fell about 0.17% and branches about 0.34%, while its 0.06% higher
median cycles remained within thread-scheduling noise.

For machine-local deployments, `./scripts/build_native_release.sh` produces an
opt-in `target/native/release/decv-cli` using `-C target-cpu=native`. The normal
release build remains portable; the native binary must not be copied to CPUs
that lack the build machine's instruction-set features. On the Ryzen AI 7 H
350, five alternating native-versus-portable CABAC Serial runs reduced
instructions about 8.5%, branches about 4.8%, and median reference cycles about
4.5%, with every native cycle sample faster. CAVLC instructions fell about
9.0%, branches about 7.7%, and cycles about 1.5%. Two-worker `Auto`
instructions fell about 8.2%, branches about 4.4%, and cycles about 3.3%.
All CAVLC and `Auto` cycle samples improved, and the native binary passed the
complete byte-exact FFmpeg stream corpus via `DECV_VERIFY_BIN`.

The fixed comparison script accepts an already built decoder through
`DECV_BENCH_BIN`. With the native binary, its three-run medians were 1.53
seconds (117.6 FPS) for Serial and 1.67 seconds (107.8 FPS) for `Auto`.
The same run measured FFmpeg NV12 at 0.61 seconds with one thread and 0.26
seconds in automatic mode, leaving native decv gaps of about 2.5x and 6.4x.

After LTO, `predict_inter_420_into` contains the specialized luma and chroma
kernels and is roughly 21.3 KiB. Forcing both kernels out of line reduced the
combined machine code by about 1.6 KiB, but added two calls per predicted
partition. Five alternating CABAC Serial runs increased instructions about
0.78%, branches about 1.69%, and median reference cycles about 1.1%; every
cycle sample was slower. The experiment was fully reverted. A future DSP
dispatch boundary must therefore operate on a coarser unit than one luma or
chroma partition call.

CABAC bypass decoding now computes the shifted offset and selected bin in
locals, then commits the offset once. This lets the optimized x86-64 build
lower the data-dependent bin selection without an unpredictable branch while
retaining failure-atomic end-of-input handling. Seven alternating native
CABAC Serial runs reduced median reference cycles about 2.28%, task-clock
about 2.19%, and branch misses about 9.5%; all seven paired reference-cycle
samples improved. Instructions increased about 0.25%, but branches fell about
0.33%. Five two-worker `Auto` runs reduced median reference cycles about 0.60%,
task-clock about 0.82%, and branch misses about 8.8%, with four of five
reference-cycle pairs improving. CAVLC remained instruction- and
branch-neutral. Packing `codIRange` and `codIOffset` into one `u32` was tested
separately and rejected because its additional shifts and merges increased
whole-decoder instructions about 0.80% without improving the isolated result.
Making the main MPS/LPS decision branchless was also rejected. An explicit
mask-based implementation reduced branches about 1.49% and branch misses about
10%, but it had to load both transition tables and increased register
pressure. Five alternating CABAC Serial runs increased instructions about
1.73% and median reference cycles about 0.77%; only one of five cycle pairs
improved. Unlike bypass bins, the probability-modelled MPS/LPS choice is
biased enough that retaining its branch is cheaper.

Forcing the complete `decode_decision` operation to inline at every optimized
call site was also rejected. It removed the standalone 381-byte symbol and
reduced whole-decoder instructions about 0.36% and branches about 0.49%, but
expanded native `.text` by about 9 KiB. The first seven CABAC Serial pairs
looked promising; extending the same alternating pinned run to fourteen pairs
left only seven improvements, increased average reference cycles about 0.19%,
and increased task-clock about 1.0%. The ordinary `#[inline]` hint was
restored so LTO can keep this moderately large, error-aware operation outlined.

Processing a sixteen-pixel two-dimensional luma interpolation row with one
AVX2 kernel instead of two SSE2 chunks was also rejected. The wider six-tap
path needed extra lane rearrangement and 16-to-32-bit packing for the diagonal
filter. Although an exhaustive SIMD-versus-scalar test and the complete
byte-exact stream corpus passed, five alternating native CABAC Serial runs
increased whole-decoder instructions about 0.84%, branches about 0.60%, and
median reference cycles about 0.39%. The implementation was fully reverted;
AVX2 width alone is not sufficient evidence that an H.264 interpolation
kernel is faster.

Removing redundant checked `i64` arithmetic from prepared 4x4 inverse scaling
was tested but not retained. The mathematical bound is valid: an `i32`
coefficient multiplied by a `u16` level scale and shifted left by at most four
cannot overflow `i64`, and the final `i32` conversion still catches invalid
output. The inlined version reduced CABAC Serial median reference cycles about
0.44% in five runs, but expanded the combined reconstruction symbol from about
2.6 KiB to 3.0 KiB and made four of five two-worker `Auto` reference-cycle
samples slower, with its median about 0.81% higher. Outlining the complete
16-coefficient scaling pass reduced aggregate machine code to about 2.1 KiB
and whole-decoder instructions about 0.37%, but the added block-level call made
all five Serial samples slower and increased median reference cycles about
1.4%. Both variants were fully reverted.

Fusing 4x4 inverse scan and inverse scaling into one coordinate-driven pass
was also rejected. It removed the intermediate coefficient matrix, shrank
`PreparedInverseScale4x4::reconstruct_into` from 2,642 to 1,611 native bytes,
and reduced total `.text` by 672 bytes. However, the irregular scatter loop
prevented the profitable straight-line/vectorized lowering of the two-pass
form. Seven pinned CABAC Serial pairs increased whole-decoder instructions
about 1.76%, branches about 7.50%, and reference cycles about 2.36%, with only
one pair improving. The separate inverse-scan and row-major scale passes were
fully restored.

Inter-prediction fast-path eligibility now tracks horizontal and vertical
filter margins independently. Previously, a horizontal-only filter near the
top or bottom edge unnecessarily selected the clipped scalar path, a
vertical-only filter did the same near the left or right edge, and integer
motion required six-tap margins that it never read. Temporary counters on the
300-frame CABAC stream found 3.77% of luma and 2.14% of chroma predictions
using the clipped path before this change. Seven alternating native CABAC
Serial runs reduced median reference cycles about 2.37%, task-clock about
2.29%, instructions about 3.77%, and branches about 5.20%; all seven paired
cycle samples improved. Five `Auto` runs reduced median reference cycles about
3.35%, with all five pairs improving. Five CAVLC Serial runs reduced median
reference cycles about 3.09%, instructions about 4.18%, and branches about
5.60%, again with all five pairs improving. Edge-focused tests compare the
newly eligible single-axis fast paths against clipped scalar interpolation,
and the complete stream corpus remains byte-exact.

Spatial Direct B-motion derivation now normalizes each of the four neighbouring
motion cells into List0 and List1 forms in one pass. The previous code unpacked
the same `Option<MotionCell>` array twice, duplicating availability and list
selection work in an already large routine. The fused implementation reduced
the native `resolve_spatial_direct_macroblock` symbol from 7,297 to 6,843
bytes. Seven alternating CABAC Serial runs reduced instructions about 0.17%,
branches about 0.43%, and median reference cycles about 0.50%, with five of
seven cycle pairs improving. Five `Auto` runs reduced median reference cycles
about 0.31%, with three pairs improving. Five CAVLC Serial runs reduced median
reference cycles about 1.09%, with four pairs improving. The complete real
stream corpus remains byte-exact.

Replacing the known single-element Spatial Direct `SmallVec::push` with manual
unchecked inline-buffer initialization was tested and rejected. It removed the
generic capacity branch and reduced whole-decoder instructions about 0.09% and
branches about 0.12%, but expanded `resolve_spatial_direct_macroblock` by 88
bytes. Seven alternating CABAC Serial runs increased median branch misses about
0.45% and median reference cycles about 0.11%; only two of seven cycle pairs
improved. The unsafe initialization was fully reverted.

Explicit single-list prediction weighting now initializes its SSE2 weight,
rounding, shift, and offset vectors once per luma or chroma plane instead of
once per row. The same kernel retains scalar tails for arbitrary row widths,
and an exhaustive oracle covers multi-row luma/chroma strides, all normative
partition widths, guarded SIMD parameter extremes, and scalar fallbacks. The
native `apply_prediction_weights_for_list` symbol shrank from about 15.4 KiB
to 10.0 KiB. Seven alternating CABAC Serial runs reduced instructions about
0.67%, branches about 1.38%, task-clock about 0.82%, and median reference
cycles about 0.80%, with five of seven cycle pairs improving. Five `Auto` runs
reduced instructions about 0.67%, branches about 1.35%, and median reference
cycles about 0.61%, with three pairs improving. CAVLC instructions and branches
fell about 0.76% and 1.50%; its median reference cycles increased about 0.13%
despite three of five pairs improving, a noise-level tradeoff accepted for the
primary High Profile CABAC target. The complete stream corpus remains
byte-exact. A follow-up guard that skipped the SSE2 helper for widths below
eight shaved roughly another 0.02% of instructions, but added branches through
the surrounding dispatch and weakened five-run CABAC median reference-cycle
improvement to about 0.07%. That extra width branch was not retained.

A complete YUV420 implicit-biprediction SSE2 entry that shared weight-vector
setup across luma, Cb, and Cr was tested and rejected. It reduced
whole-decoder instructions about 0.10% and branches about 0.19%. Seven CABAC
Serial runs put median reference cycles about 0.10% lower, with four pairs
improving, but five `Auto` runs increased the median about 0.53% and five CAVLC
runs increased it about 0.57%; each of those workloads improved only one of
five pairs. The wrapper around the already inlined plane kernels was fully
reverted. A future DSP entry must fuse more than shared SIMD constant setup to
amortize a full-YUV dispatch boundary.

Removing `TryFrom` and `checked_add` operations from inter-prediction window
classification was tested in two forms and rejected. Computing the bounded
coordinates directly in `i64` reduced CABAC Serial instructions about 0.95%,
branches about 0.41%, and median reference cycles about 0.48%, with five of
seven pairs improving. `Auto` median reference cycles fell about 0.37%, but
CAVLC increased about 0.99% and all five pairs were slower despite roughly
1.1% fewer instructions. Direct `i32` arithmetic is also mathematically safe
under the decoder's 36,864-macroblock picture limit and i16 motion-vector
range; it shrank the combined prediction symbol by 226 bytes and reduced
CABAC instructions about 0.73%. Nevertheless, seven CABAC Serial runs
increased median reference cycles about 0.48%, with only two pairs improving.
Both dependency-chain variants were fully reverted; the existing checked
operations schedule better across the supported workloads.

Heap-owning the 1,208-byte `IntraPictureReconstructor` and consuming it through
a `Box<Self>` frame-finalization entry was tested and rejected. It removed the
1,208-byte `memcpy` from `finish_current_picture`, reduced that function's
native stack frame from 1,960 to 872 bytes, and shrank its inlined machine code
from 22,758 to 18,508 bytes. Four pinned CABAC Serial `perf stat` pairs,
however, left reference cycles and instructions effectively unchanged at
about +0.06% and +0.05%; clean alternating wall-time medians were identical.
Five two-worker `Auto` runs were also median-neutral. Five CAVLC Serial runs
made every candidate pair slower and increased median wall time about 1.3%,
showing that the extra allocation costs more than the once-per-frame move.
The boxed layout was fully reverted. The sampled cost attributed to
`finish_current_picture` is therefore dominated by inlined deblocking and
picture finalization, not by its entry move.

NV12 output now selects a 32-sample AVX2 Cb/Cr interleave kernel once per
picture when the CPU supports it, with the existing SSE2 and scalar paths kept
as fallbacks. The dispatch is deliberately outlined: allowing LTO to inline
the AVX2 path into the already large `finish_current_picture` reduced
instructions but produced inconsistent cycles on CAVLC. Outlining the
full-plane operation shrank `finish_current_picture` from 22,758 to 22,163
bytes and made the workloads agree. Five pinned CABAC Serial pairs reduced
average reference cycles about 0.61%, instructions about 0.33%, and branches
about 0.42%. Five CAVLC Serial pairs reduced reference cycles about 1.66%,
instructions about 0.38%, and branches about 0.46%, with four pairs improving.
The two-worker `Auto` measurement remained scheduling-sensitive: average
reference cycles fell about 1.34% and its wall-time median moved from 2.60 to
2.57 seconds, but only two of five pairs improved. The complete H.264 and MP4
FFmpeg corpora remain byte-exact. At that checkpoint, the fixed 180-frame
benchmark measured 1.58 seconds in Serial mode and 1.70 seconds in Auto mode.

B-picture motion state now uses its atomic macroblock lifecycle when rejecting
duplicate writes. Every successful commit fills all sixteen 4x4 cells and
every rollback clears all sixteen, so release builds inspect the first cell
instead of scanning the complete macroblock; debug builds assert the all-full
or all-empty invariant. Five pinned CABAC Serial pairs reduced reference
cycles about 0.57%, instructions about 0.22%, and branches about 0.82%. Five
CAVLC Serial pairs reduced reference cycles about 0.36%, instructions about
0.25%, and branches about 0.90%. The two-worker `Auto` averages moved in the
same direction: reference cycles fell about 0.38% and branches about 0.81%.
A regression test covers duplicate rejection, clear, and re-recording.

Representing an absent B-motion neighbour reference with `u8::MAX` instead of
signed `-1` was tested and rejected. It removed sign checks and conversions,
shrinking `resolve_spatial_direct_macroblock` by 732 bytes and reducing
whole-decoder instructions about 0.60% on CABAC and 0.68% on CAVLC. CAVLC
reference cycles improved about 0.99%, but the primary CABAC workload increased
average reference cycles about 0.52% across eight alternating pinned pairs.
The unsigned sentinel was fully reverted; fewer instructions alone do not
justify a worse CABAC dependency schedule.

Interior integer-motion prediction now selects its fixed row width once per
luma or chroma rectangle and copies four independent rows per iteration.
Previously every row and both chroma planes re-entered a 2/4/8/16-byte jump
table. A first const-width loop reduced instructions about 2.9% and branches
about 7.8% but left CABAC cycles unchanged because its row pointers formed a
serial dependency chain. Four-row expansion retains the hoisted dispatch while
restoring memory-level parallelism. Five pinned CABAC Serial pairs all
improved, reducing reference cycles about 2.95%, instructions about 2.71%, and
branches about 7.31%. Five CAVLC Serial pairs reduced reference cycles about
1.72%, instructions about 3.08%, and branches about 8.07%, with four pairs
improving. All five two-worker `Auto` pairs improved; reference cycles fell
about 3.31%, instructions about 2.66%, and branches about 7.13%. An exhaustive
rectangle oracle covers every supported fixed width and row count at
unaligned source and destination addresses.

Inter residual reconstruction now uses the decoded block's `total_coeff`
metadata to skip inverse scan, inverse quantization, and inverse transform for
known-zero blocks. The 8x8 path skips only when all four interleaved source
blocks are empty; chroma skips only when both its transformed DC value and AC
block are zero. Block-size syntax validation still runs before every shortcut,
and output storage is zero-initialized. The native binary's `.text` shrank by
about 1.2 KiB. Seven pinned CABAC Serial pairs all improved, reducing average
reference cycles about 3.75%, task-clock about 3.69%, instructions about
6.42%, and branches about 3.98%. Seven CAVLC Serial pairs all improved,
reducing reference cycles about 4.98%, task-clock about 4.81%, instructions
about 6.90%, and branches about 3.91%. All seven two-worker `Auto` pairs also
improved; reference cycles fell about 2.91%, task-clock about 2.58%,
instructions about 6.36%, and branches about 3.91%. The portable and native
H.264 and MP4 corpora remain byte-exact.

Vertical chroma deblocking now gathers each eight-row edge with eight
unaligned 32-bit loads, transposes p1/p0/q0/q1 in SSE2 registers, and writes
the same eight complete words back after filtering. The previous kernel made
32 scalar sample loads and 16 scattered byte stores around the SIMD equation.
The existing randomized scalar oracle covers mixed boundary strengths, and
the native `.text` shrank by 176 bytes. Fourteen pinned CABAC Serial pairs
reduced average reference cycles about 1.13% and task-clock about 1.08%, with
eleven pairs improving. Seven two-worker `Auto` pairs reduced reference cycles
about 1.51% and task-clock about 1.63%, with six pairs improving. CAVLC is a
measured tradeoff: fourteen Serial pairs increased reference cycles about
0.41% and task-clock about 0.37%, despite slightly fewer instructions. The
complete kernel is retained for the High Profile CABAC and high-frame-rate
primary target. Portable and native H.264 and MP4 corpora remain byte-exact.

CABAC P/B reconstruction batches now accumulate up to eight macroblock rows
instead of four before entering the worker pool and committing staged pixels.
At 4K this halves the ordinary pool-entry, staging-allocation, and batch
commit frequency while keeping the serial CABAC state and all intra barriers
unchanged. Across fourteen pinned two-worker 4K `Auto` pairs, task-clock fell
about 1.99%, reference cycles about 0.52%, instructions about 0.22%, branches
about 0.48%, and sampled cache misses about 8.31%; ten pairs improved. Seven
4K Serial pairs reduced task-clock about 0.89% and reference cycles about
0.34%, although only three cycle pairs improved. The 1080p CABAC modes were
cycle-neutral: about -0.01% in Serial and +0.06% in `Auto`. A sixteen-row
follow-up made all seven 4K `Auto` cycle pairs slower and increased reference
cycles about 0.89%, so eight rows remain the measured batching sweet spot.
Serial/Auto CABAC, CAVLC, and 4K outputs remain byte-exact.

`Auto` reconstruction parallelism now selects its pool after the coded picture
size is known. Pictures below roughly eight megapixels retain the conservative
two-worker cap, while 4K-class pictures may use up to four workers. On the
48-frame 4K stream, seven alternating pinned wall-time pairs moved the median
from 1.47 to 1.43 seconds, about 2.7% faster, while average user CPU increased
from 1.52 to 1.63 seconds. The 300-frame 1080p CABAC stream remained exactly
2.43 seconds in both binaries with unchanged user CPU because it still selects
two workers. Eight physical workers were rejected: their 4K median was 1.52
seconds versus 1.44 seconds for four workers, and average user CPU rose from
1.65 to 1.99 seconds. The pool is created once for the active coded size and is
reused across pictures; an IDR resolution change rebuilds it only when the
coded size changes.

Keeping four independent source and destination pointer induction variables
across those rectangle-copy iterations was tested as a follow-up. It shrank
the native `predict_inter_420_into` symbol by 307 bytes and reduced
whole-decoder instructions about 0.15% and branches about 0.31%. Seven pinned
CABAC Serial pairs nevertheless increased average reference cycles about
0.27%, increased branch misses, and improved only three pairs. The extra live
pointers harmed the surrounding register allocation enough to outweigh the
smaller loop, so this variant was fully reverted.

Extending the CABAC residual checked/internal split from state recording to
`coded_block_flag` context lookup was tested and rejected. The internal path
could rely on the already validated macroblock and normative block identifier,
shrinking its coefficient-block symbol by 129 bytes and reducing
whole-decoder instructions about 0.12% and branches about 0.36%. Across
fourteen pinned CABAC Serial pairs, however, exactly seven improved; average
reference cycles increased about 0.24% and branch misses increased. The
checked lookup was restored because removing this validation changed layout
without shortening the arithmetic dependency path.

Replacing release-mode B-partition geometry checks with debug assertions in
both deblocking-metadata builders was also rejected. The partitions are
already produced by validated motion-state code, and the change shrank each
builder by about 0.5 KiB and total native `.text` by about 1 KiB. Fourteen
pinned CABAC Serial pairs nevertheless increased average reference cycles
about 0.29%, task-clock about 1.39%, and branch misses about 0.48%; removing
roughly 0.15% of instructions and branches did not help. The well-predicted
release checks were restored.

Allocating the per-picture deblocking grid with the global allocator's
zero-filled entry point was tested against `vec![Default::default(); count]`.
The explicit implementation kept every Rust value initialized, passed an
all-zero-versus-default layout test, shrank the picture-constructor symbol by
774 bytes, and passed the byte-exact corpus. Across fourteen pinned CABAC
Serial pairs, reference cycles and task-clock were both within 0.01%, page
faults were unchanged, and instructions increased slightly. The platform
allocator/compiler already handles the roughly 1.4 MiB zero fill effectively,
so the manual allocator code was fully reverted.

Moving the prediction-to-macroblock copy dispatch from every row to one
rectangle-level helper was tested as a possible coarser DSP boundary. The
helper selected the fixed 4/8/16-byte luma and 2/4/8-byte chroma widths once,
then copied every row with specialized const-generic kernels. It passed the
reconstruction tests, strict Clippy, and the native byte-exact H.264 corpus.
Whole-decoder instructions fell about 1.88% and branches about 6.20%, but
native `.text` grew by about 11 KiB; seven pinned CABAC Serial pairs increased
average reference cycles about 0.85% and CPU cycles about 1.12%. The wider
inlining boundary traded predictable dispatch for front-end pressure and was
fully reverted. Coarse DSP dispatch should select substantial kernels, not
outline small rectangular copies from already hot reconstruction functions.

A 16-lane AVX2 implementation of two-dimensional fractional luma prediction
was tested against the existing two-chunk SSE2 kernel. It processed a complete
16-sample row at once, including the 32-bit diagonal-filter intermediate, and
matched the scalar oracle for all nine two-axis fractional positions and every
supported partition height. It also passed the native byte-exact H.264 corpus.
The extra lane packing and 32-bit result reordering grew native `.text` by
about 8 KiB; seven pinned CABAC Serial pairs increased instructions about
0.36%, branches about 0.55%, and average reference cycles about 0.54%, with
only two pairs improving. The AVX2 path was fully reverted. Wider SIMD is only
useful when its lane layout matches the codec operation without expensive
cross-lane packing; a DSP backend must keep the narrower SSE2 kernel when that
is faster.

Pre-scanning Spatial Direct co-located-zero flags into a bit mask was tested
to bypass constructing and then coalescing a uniform 4x4/8x8 partition grid.
The uniform case could commit one 16x16 motion partition directly, while the
non-uniform case reused the mask without loading the co-located field twice.
It passed the motion tests and native byte-exact H.264 corpus, and reduced
whole-decoder instructions about 0.69% and branches about 1.12%. The extra
pre-scan and uniformity branch grew native `.text` by about 1.6 KiB, however.
Across fourteen pinned CABAC Serial pairs only six improved; average reference
cycles increased about 0.05% and task-clock about 0.42%. The bit-mask path was
fully reverted. For this stream, producing partitions in one pass and
coalescing afterward schedules better than a speculative classification pass.

Clearing only the category-visible prefix of the reusable 64-entry CABAC
coefficient buffer was tested and rejected. Most blocks expose only 4, 15, or
16 entries, so the candidate avoided zeroing the unused tail while retaining
the complete 64-entry clear for 8x8 transforms. It passed the residual tests
and native byte-exact H.264 corpus, but replaced an efficiently lowered fixed
256-byte clear with a variable-length operation. Native `.text` grew by 48
bytes; seven pinned CABAC Serial pairs increased reference cycles about 0.57%,
task-clock about 1.05%, and branches about 0.12%, with only two pairs
improving. The fixed-size clear was restored.

Representing a CABAC significance map as a partially initialized
`[MaybeUninit<u8>; 64]` was also tested. Significance decoding wrote each
position before increasing `count`, and the safe accessor exposed only that
initialized prefix, eliminating the map's 64-byte zero initialization. It
passed the residual tests and native byte-exact H.264 corpus, reducing
whole-decoder instructions about 0.16% and branches about 0.09%. Across
fourteen pinned CABAC Serial pairs, however, exactly seven improved; average
reference cycles increased about 0.13%, task-clock about 0.33%, and branch
misses about 0.61%. The initialized array and derived value semantics were
restored rather than retaining an unsafe invariant for a slower result.

Splitting inter prediction into checked and prevalidated entries was tested on
the 48-frame 4K120 High Profile CABAC stream. Macroblock reconstruction
validated each partition once and passed an already checked absolute luma
origin to the prediction kernel, avoiding repeated geometry, overflow, and
current-picture boundary checks for the two reference lists of bidirectional
partitions. All H.264 unit tests and native byte-exact CABAC, CAVLC, and 4K
outputs passed. The split nevertheless grew native `.text` by about 3.2 KiB.
Across fourteen alternating pinned 4K Serial pairs, instructions fell about
0.18% and branches about 1.17%, but reference cycles increased about 0.22%;
exactly seven pairs improved. The checked entry and its original code layout
were restored. Validation is not a meaningful part of the 4K motion-
compensation cost, and duplicating a large prediction entry is worse than its
few well-predicted guards.

Outlining only the clipped luma and chroma interpolation paths with
`#[cold]` was tested separately. The normal interior SIMD paths remained
inlined, while the infrequent scalar edge paths moved out of
`predict_inter_420_into`; this reduced that hot symbol from about 23.4 KiB to
18.3 KiB with essentially no total `.text` change. Unit tests and native
byte-exact CABAC, CAVLC, and 4K outputs passed. Fourteen alternating pinned 4K
Serial pairs nevertheless increased average reference cycles about 0.71%,
with only three pairs improving. Instructions fell about 0.30% and sampled
cache misses about 0.65%, but branches increased about 0.24%. The cold wrappers
were removed. Without profile-guided layout, shrinking one symbol does not
justify adding calls and changing branch placement even for a roughly 2% to 4%
edge case.

Implicit B prediction was also tested with its existing SSE2 weighting kernel
writing directly into the destination macroblock instead of updating the
List-0 scratch plane and then copying the finished partition. The 4K workload
uses `weighted_bipred_idc=2`, so this removes an intermediate write and the
following luma/chroma row copies from a common path. Offset-destination SIMD
tests and native Serial/Auto CABAC, CAVLC, and 4K output comparisons were
byte-exact. Fourteen pinned 4K Serial pairs reduced instructions about 2.26%
and branches about 5.76%; the B reconstruction symbol also shrank by about 396
bytes. Reference cycles and task-clock nevertheless increased about 1.1%,
with only six pairs improving. The original contiguous scratch write followed
by fixed-width copies was restored. On this CPU, that store pattern schedules
better than writing the weighted result directly across destination strides;
removing instructions is not sufficient when it lengthens the pixel-data
dependency chain.

Reusing horizontal six-tap rows across adjacent two-dimensional luma outputs
was evaluated in three forms. A rolling six-row SSE2 window reduces a 16-row,
eight-sample chunk from roughly 96 to 112 horizontal filter evaluations to 21.
The fully inlined form passed the SIMD oracle and all native byte-exact stream
checks; fourteen pinned 4K Serial pairs reduced reference cycles and task-clock
about 1.0%, with ten pairs improving. The same binary made all seven 1080p
CABAC pairs slower, however, increasing reference cycles about 1.47%.
Outlining the rolling kernel removed its 4K benefit (about +0.04% reference
cycles, four of seven pairs improving) and still made all 1080p pairs about
1.15% slower. Finally, enabling the inlined kernel only for strides of at least
3840 pixels left the original algorithm active at 1080p, but LTO layout changes
alone made all seven 1080p pairs slower by about 1.68%. All three forms were
reverted. Horizontal-row reuse is algorithmically sound, but adding this large
specialization to the current monolithic prediction unit costs more front-end
performance than it saves; it should be reconsidered only with a backend
layout that isolates inactive code without adding a per-partition call.

Outlining the complete CABAC renormalization path was tested after profiling
showed that the inlined BitReader refill made `decode_decision` save and
restore six callee-saved registers. The split reduced the native decision
symbol from about 381 to 236 bytes. It also reduced whole-decoder instructions
about 0.12%, but renormalization is too frequent to be a cold call: branches
increased about 0.76%, and nine of ten pinned 4K Serial reference-cycle pairs
were slower. This differs from the earlier refill-only split but reaches the
same conclusion: CABAC arithmetic and its common renormalization dependency
chain must remain contiguous.

Outlining Spatial Direct's co-located-grid expansion was also rejected. The
main resolver shrank from about 6.7 KiB to 4.4 KiB, with a separate 2.5 KiB
grid helper. The candidate passed its focused motion tests, but the grid path
was not cold enough: ten pinned native 4K Serial pairs increased instructions
about 0.48% and average reference cycles about 1.34%, with nine cycle pairs
slower. A later four-worker retest was nearly neutral in a native build, but
representative PGO retraining confirmed the dependency cost: instructions
fell about 0.65%, branches about 1.06%, and sampled cache misses about 0.71%,
while seven of nine alternating 4K pairs became slower. Mean wall time
increased about 1.90%, task-clock about 1.38%, and reference cycles about
1.80%. The existing single function was restored.

Three allocation-oriented follow-ups did not justify retention. Constructing
the luma `Arc<[u8]>` directly through a fully initialized
`Arc<[MaybeUninit<u8>]>` was instruction-neutral, showing that the existing
zeroed `Vec` conversion is already effective on this toolchain. Replacing ten
per-slice derived reference vectors with inline-capacity-four `SmallVec`s
reduced instructions about 0.24%, but only one of ten 4K reference-cycle pairs
improved because the eight simultaneous B-list buffers increased stack and
register pressure. Finally, building active DPB metadata directly from
borrowed DPB entries removed intermediate `Arc` clones and pointer searches,
but increased instructions about 0.16% and made most wall-time pairs slightly
slower. All three variants were fully reverted. Future allocation work should
reuse the large per-picture reconstruction state, rather than reshaping these
small temporary objects.

Two attempts to avoid clearing `BMotionState` were rejected because they
weakened its hot neighbour-access layout. A `MaybeUninit<MotionCell>` array
gated by one completion byte per macroblock reduced 4K page faults about 4.1%,
but the required cell-to-macroblock division increased instructions about
1.24% and branches about 0.86%. Splitting slice IDs and both motion lists into
three arrays avoided that division, but destroyed AoS locality: instructions
increased about 2.56%, branches about 3.83%, and page faults increased. A final
AoS variant used one presence byte per cell; it reduced page faults about 10%,
but still increased instructions about 1.66% and branches about 2.83% without
a stable cycle gain. The initialized `Option<MotionCell>` representation was
restored. Its up-front clear is cheaper than adding work to every neighbour
lookup.

DPB reference identities now use a non-zero 32-bit token instead of an
ordinary 64-bit integer. Zero remains the `Option` niche, so
`Option<ReferenceId>` is four bytes, `StoredListMotion` is twelve bytes, and a
retained 4x4 `MotionFieldCell` shrank from 56 to 36 bytes. On the 3840x2176
coded 4K stream, one reference motion field consequently fell from about
27.9 MiB to 17.9 MiB. Tokens wrap while skipping identities still present in
the at-most-16-picture DPB, preserving indefinite decoding without collision;
IDR and DPB clear still restart allocation at one. Layout and wraparound tests
lock down both invariants.

The smaller motion field converted directly into whole-decoder gains. Ten
alternating pinned 4K Serial pairs reduced task-clock about 4.10%, reference
cycles about 3.36%, instructions about 0.21%, branches about 0.18%, and page
faults about 11.3%; nine cycle pairs improved. Ten 4K four-worker `Auto` pairs
reduced task-clock about 4.20% and reference cycles about 3.50%; all ten cycle
pairs and nine task-clock pairs improved, while page faults fell about 9.2%.
Seven 1080p CABAC pairs all improved in both modes: Serial task-clock and
cycles fell about 2.23% and 2.30%, while two-worker `Auto` fell about 3.30%
and 3.26%. Across ten noisier CAVLC Serial pairs, task-clock fell about 1.77%
and reference cycles about 2.24%. The full workspace suite, strict H.264
Clippy, native H.264 corpus, MP4/seek corpus, and 4K Serial/Auto/FFmpeg
three-way comparison remain byte-exact.

The per-picture B-motion workspace now retains its allocation across pictures
of the same decoder. A 4K frame uses 522,240 addressable 4x4 motion cells.
Previously, finishing every picture dropped that large `Vec<Option<MotionCell>>`
and the following picture allocated it again. The retained workspace still
fills every cell with `None` before reuse, so no motion or slice state crosses
the picture boundary; a coded-size change either resizes or replaces the
allocation through the same validated cell-count calculation. A focused test
locks down same-size allocation reuse and the complete clear.

Seven alternating pinned 4K Serial pairs all improved: average task-clock fell
about 2.54% and reference cycles about 2.63%, while instructions were within
0.03%. Four-worker `Auto` was a smaller, scheduling-sensitive gain:
task-clock fell about 0.40% and reference cycles about 0.81%, while ordinary
CPU cycles increased about 0.24%. At 1080p, six of seven Serial pairs improved;
average task-clock fell about 1.69%, reference cycles about 0.76%, and minor
faults about 29.9%. Two-worker `Auto` was effectively neutral at about -0.03%
task-clock and -0.15% reference cycles. Minor faults increased in both 4K
modes and 1080p `Auto`, and one 4K Serial `/usr/bin/time` sample increased
peak RSS by about 1.9 MiB, a measured memory-side tradeoff for the Serial
throughput gain. The full workspace suite, strict H.264 Clippy, native H.264
and MP4/seek corpora, and 4K Serial/Auto/FFmpeg outputs remain byte-exact.

Extending that workspace to retain the roughly 5.5 MiB 4K deblocking-metadata
allocation was tested and rejected. Reusing the allocation still performed
the required full `MacroblockDeblockInfo::default()` fill and reduced minor
faults about 13.6%. The first seven pinned 4K Serial pairs were obscured by two
large baseline frequency outliers. In the following stable seven pairs,
candidate task-clock increased about 0.12%, reference cycles about 0.98%, CPU
cycles about 0.46%, and instructions about 0.034%; only two reference-cycle
pairs improved. The deblock allocation was removed from the reusable workspace.
Lower page-fault counts do not justify retaining a larger live allocation when
the required initialization stores make the primary decode workload slower.

An opt-in `internal-profiling` feature now counts inter-prediction kernel
shapes and represented pixel work across all reconstruction workers. Build it
with:

```text
DECV_NATIVE_TARGET_DIR=/tmp/decv-profile \
    ./scripts/build_native_release.sh -p decv-cli --features internal-profiling
```

The feature deliberately uses relaxed atomic counters and prints one summary
when `decv-cli` exits. It is for path selection, not timing; ordinary builds do
not compile the counters or their hot-path call. The current native 48-frame
4K stream produced 2,647,731 prediction calls. Width 16 accounted for
2,486,419 calls, width 8 for 161,312, and width 4 for none. By represented
luma pixels, integer motion accounted for 90.4%, horizontal-only fractional
motion 3.3%, vertical-only 3.5%, two-dimensional fractional motion 2.8%, and
clipped edge access only 0.4%. Chroma was 82.9% integer and 17.1% bilinear,
with 0.4% clipped.

The 300-frame 1080p CABAC and CAVLC streams agreed on the priority: integer
luma motion represented 94.5% and 94.1% of pixels, two-dimensional motion only
1.9% and 2.0%, and clipping 0.4% in both. Their chroma work was about 93%
integer and 7% bilinear. These counts reject generic edge padding or another
two-dimensional AVX2 attempt as the immediate target for this corpus. The
dominant opportunity is the complete width-16 integer YUV path and its
surrounding per-partition control overhead; the existing two-dimensional SSE2
kernel remains appropriate until a materially different workload proves
otherwise.

A combined width-16 interior integer YUV path was tested against that profile.
It recognized motion vectors with integer luma and chroma positions, validated
one luma rectangle, and copied Y, Cb, and Cr through three fixed-width kernels
before entering the general interpolation path. A focused all-plane oracle and
strict Clippy passed. Seven alternating pinned 4K Serial pairs reduced
instructions about 2.0% and branches about 1.4%, but increased average
task-clock about 0.75%, reference cycles about 1.94%, CPU cycles about 2.45%,
and sampled cache misses about 6.4%; only the first reference-cycle pair
improved. The specialization was fully reverted. Even a dominant path should
not be cloned when its extra entry layout harms the portable release build.

LLVM profile-guided optimization is now available as a separate native build:

```text
rustup component add llvm-tools-preview
./scripts/build_pgo_release.sh representative-4k.h264 representative.mp4
```

The script builds an instrumented decoder, decodes every supplied input once
in `Serial` and once in `Auto`, merges the raw profiles with the
toolchain-matched `llvm-profdata`, and writes the optimized binary to
`target/pgo/release/decv-cli`. Training inputs are intentionally supplied by
the caller: PGO specializes branch probabilities, layout, and inlining for the
workload it observes. Use representative profiles, entropy modes, resolutions,
and GOP structures rather than a single tiny conformance stream.

A mixed 4K High CABAC, 1080p High CABAC, 1080p Main CAVLC, and byte-exact
regression-corpus profile produced a large native win. Seven pinned 4K Serial
pairs reduced average task-clock about 7.0%, reference cycles about 6.7%, and
instructions about 14.4%; every pair improved. Seven pinned 4K four-worker
pairs reduced task-clock about 5.0%, reference cycles about 6.1%, and
instructions about 14.4%. Five pinned 1080p CABAC Serial pairs reduced
task-clock about 6.0%, reference cycles about 6.3%, and instructions about
14.5%, again with every pair improving. Five 1080p CAVLC Serial pairs reduced
average task-clock about 1.0%, reference cycles about 0.9%, and instructions
about 8.7%, though three individual timing pairs were slightly slower and the
profile grew native `.text` about 11.4%. PGO is therefore an effective
throughput build mode, not a replacement for portable release binaries or
representative cross-workload validation.

Non-reference pictures now use a validation-only reference-motion builder.
Motion state is still needed while reconstructing the current picture, but its
finalized reference-motion field is never stored in the decoded picture buffer
and cannot be used as colocated motion by a later picture. The builder retains
macroblock-completion, duplicate-write, partition-alignment, overlap, and
coverage checks while omitting the final 4x4-cell allocation and writes.
Reference pictures retain the original field unchanged. At the 3840x2176
coded size, this removes a roughly 17.9 MiB picture-local allocation from each
non-reference picture.

Seven alternating pinned 4K Serial pairs all improved: average task-clock fell
about 2.56%, reference cycles about 4.36%, and sampled cache misses about
7.93%. Instructions increased about 0.21% and branches about 0.76% because the
discarding path still validates partition coverage; avoiding the memory
traffic was nevertheless a stable net win. Five 4K four-worker pairs reduced
task-clock about 1.87%, reference cycles about 3.14%, and sampled cache misses
about 5.91%. Five 1080p CABAC Serial pairs reduced task-clock about 3.47% and
reference cycles about 4.92%; five CAVLC pairs reduced them about 5.18% and
4.47%. Minor-fault direction varied by workload, so it is not treated as a
reliable benefit. The full H.264 unit suite, strict H.264 Clippy, generated
byte-exact corpus, and 48-frame 4K FFmpeg hash all passed.

Two follow-up attempts to reduce the remaining validation cost were rejected.
Trusting the already resolved motion state and recording only macroblock
completion reduced native instructions about 0.99% and branches about 1.37%,
but was wall-time neutral in Serial and about 1% slower with four workers.
After retraining PGO it remained about 0.4% slower across nine alternating
pairs. Replacing the small partition-mask loop with multiplication and dynamic
shifts reduced instructions about 0.71% and branches about 1.15%, but increased
reference cycles about 1.46% and wall time about 1.27%. Both implementations
were fully reverted; fewer retired instructions did not compensate for their
longer dependency paths and changed layout.

Resolved P-macroblock partitions now use four-entry inline storage, matching
the existing B-macroblock representation. Common P_Skip, 16x16, 16x8, 8x16,
and 8x8 layouts therefore avoid a heap allocation; unusually subdivided 8x8
layouts can still spill without changing semantics. Seven native 4K
four-worker pairs reduced average wall time about 1.21%, reference cycles about
0.82%, instructions about 0.86%, branches about 1.48%, and sampled cache misses
about 1.55%. After PGO, the corresponding reductions were about 0.75%, 0.27%,
0.93%, 1.76%, and 2.04%. The full workspace suite, strict Clippy, and exact
4K hash passed.

CABAC coefficient decoding previously initialized its fixed 64-coefficient
array and then cleared it again inside a generic helper. Removing the second
clear outright made nine native 4K pairs about 3.35% faster, but a retrained
PGO build was about 0.55% slower because PGO had already eliminated almost all
of the redundant work and the source change perturbed layout. That version was
reverted. Keeping the initialization contract and forcing the helper inline
gave the portable build a smaller 0.43% mean wall-time gain. Nine PGO pairs
reduced mean wall time about 0.20%, task-clock about 0.39%, reference cycles
about 0.78%, and sampled cache misses about 0.84%. The one-line inline hint was
retained; the observable double-clear removal was not.

Full-macroblock spatial and temporal Direct motion now iterate static 8x8 and
4x4 partition grids. The selected grid carries both luma offsets and colocated
4x4-cell offsets, replacing two dynamic `step_by` loops plus repeated
coordinate division. All alignment, bounds, temporal scaling, coalescing, and
motion-state commits remain unchanged. Nine native 4K four-worker pairs reduced
mean wall time about 1.31% and task-clock about 0.61%. After PGO, six of nine
pairs improved; mean wall time fell about 0.76%, task-clock about 0.93%, and
reference cycles about 1.29%, with sampled cache misses effectively unchanged.

Three scopes of CABAC partition-allocation removal were tested and rejected.
Inlining the private P/B partition plans and their sub-partition lists with
four-entry `SmallVec`s made nine native 4K four-worker pairs about 0.93% slower
in wall time and 0.41% higher in task-clock, despite reducing reference cycles
about 0.40%. Extending inline storage to the decoded motion headers, first at
the four-entry maximum and then at one- or two-entry common capacities, did not
produce a repeatable 4K gain because the larger values changed stack, boxed
macroblock, and cache behavior. The narrowest version only inlined the
usually-single motion-vector-difference lists. On a stable 600-frame 1080p
CABAC run it reduced instructions about 0.19%, branches about 0.30%, and
sampled cache misses about 1.14%, but changed wall time by only -0.07% and
task-clock by -0.04%; 4K measurements did not retain that direction. That
negligible result does not justify changing the public motion-header field
types, so all three versions were reverted.

Large CABAC non-reference pictures now complete through a bounded cross-frame
pipeline when a reconstruction pool is active. Picture completeness and the
discarded reference-motion field are validated synchronously. Deblocking and
NV12 packaging then run on the existing pool while the decoder begins the next
picture. Completed tasks are consumed strictly in decode order; a reference
picture, drain, ordinary IDR boundary, or end marker waits for prior tasks.
Flush and `no_output_of_prior_pictures` wait and discard them. The pending
queue is bounded to one fewer than the pool's worker count, leaving a worker
available for current-picture reconstruction. Serial mode, CAVLC, and pictures
below two million coded pixels retain the synchronous path.

Against the preceding native binary, seven alternating pinned 4K four-worker
pairs all improved. Average wall time for the 48-frame 3840x2160-visible
stream fell from 1.399 to 1.316 seconds, or from 34.31 to 36.47 FPS:
wall time fell about 5.90%, task-clock about 1.75%, and reference cycles about
0.53%. Five 1080p two-worker CABAC pairs reduced wall time about 4.90%, from
120.16 to 126.35 FPS. The equivalent 4K Serial path was effectively unchanged
at 34.57 FPS, confirming that the gain comes from the bounded overlap rather
than less reconstruction work. The generated H.264 corpus and the 48-frame
4K output remain byte-exact against FFmpeg.

Retraining PGO after adding the pipeline produced a further whole-program
gain. The profile used the 4K CABAC, 1080p CABAC, and 1080p CAVLC streams in
both Serial and Auto modes. Across seven alternating pinned 4K four-worker
pairs, PGO reduced average wall time about 10.77%, task-clock about 7.90%,
reference cycles about 8.52%, and instructions about 14.71%. Same-run
throughput moved from 35.66 to 39.96 FPS. Seven Serial pairs reduced wall time
about 5.17%, reference cycles about 5.58%, and instructions about 14.85%,
moving from 34.27 to 36.14 FPS. The PGO four-worker output retained the exact
597,196,800-byte FFmpeg hash. On this CPU the best measured software result is
therefore approximately 4K40, still roughly three times short of 4K120.

Sampling that PGO four-worker build showed the next scheduling limit clearly:
the caller accounted for 64.14% of sampled cycles, while the four reconstruction
workers accounted for 9.71%, 9.59%, 8.79%, and 7.77%. CABAC B-slice syntax and
residual decoding alone occupied about 23% of the combined sample. Merely adding
workers cannot consume that serial share.

Large CABAC B pictures now overlap those two phases within a picture. One
eight-macroblock-row pixel batch may reconstruct on the existing worker pool
while the caller parses, transforms residuals, and resolves motion for the next
batch. Only owned motion and residual jobs cross the scope. Completed pixels
return through a bounded one-batch channel and still commit in macroblock
address order. An intra macroblock and the end of the slice are strict barriers,
so prediction availability and error ordering retain the synchronous rules.
Serial mode and pictures below eight million coded pixels keep the former
path.

The resolution gate is important. Enabling the overlap on the 1080p two-worker
stream reduced throughput by about 4.0% because scheduling cost exceeded the
saved idle time; disabling it below the 4K class removes that regression. After
retraining PGO, nine alternating pinned 4K four-worker pairs had a median wall
time improvement of about 1.95%. Mean throughput moved from 39.06 to 39.77 FPS;
seven of nine pairs improved. Mean task-clock, reference cycles, and
instructions increased about 2.89%, 1.66%, and 1.85%, respectively, reflecting
intentional overlap and contention, while wall latency still fell about 1.80%.
The full 378-test workspace suite, strict all-target Clippy, and the
597,196,800-byte FFmpeg hash passed.

CABAC residual-neighbour grids now retain their allocations across pictures.
The first reuse attempt cleared every `Option<BlockState>` with `fill(None)`;
although this removed allocator traffic, it compiled into element-wise stores
and increased native whole-decoder instructions by about 8.4%, so that version
was discarded. The retained design carries the reconstructor's monotonically
increasing slice identifier in an internal reusable workspace. Entries from a
previous picture already fail the existing same-slice comparison and therefore
need no clear. A dimension change reallocates the grids, while the practically
unreachable `u32` identifier exhaustion path clears them before restarting the
generation.

Nine alternating native 4K four-worker pairs reduced mean wall time about
1.41%, task-clock about 1.60%, cycles about 1.00%, instructions about 0.36%,
and sampled cache misses about 0.91%. Five 600-frame 1080p two-worker pairs
reduced wall time about 1.51% and task-clock about 1.31%. After retraining the
same mixed CABAC/CAVLC PGO profile, nine 4K pairs reduced mean wall time about
1.58%, task-clock about 1.72%, cycles about 3.50%, instructions about 1.47%,
branches about 1.42%, and sampled cache misses about 3.03%. The 597,196,800-byte
4K output retained its exact SHA-256 hash, and the full 380-test workspace suite
plus strict all-target Clippy passed.

The same cross-picture generation now invalidates reusable B-motion cells
without clearing the complete 16-cell-per-macroblock grid. The grid stores the
first slice identifier belonging to the current picture. Its existing
same-slice neighbour filtering already ignores older entries, while the
duplicate-macroblock guard treats every identifier at or above that boundary
as current-picture state. All sixteen cells are still overwritten atomically.
Identifier exhaustion clears both reusable grids before restarting, preserving
correctness across the `u32` wraparound.

The native build measured this as a small throughput tradeoff: eight stable
fixed-CPU 4K pairs reduced mean wall time about 0.68%, instructions about
0.26%, and branches about 0.12%, while task-clock was neutral, cycles increased
about 0.17%, and sampled cache misses increased about 1.28%. A retrained PGO
build resolved the added generation check more profitably. Nine 4K pairs
reduced mean wall time about 1.47%, task-clock and cycles about 0.90%,
instructions about 0.21%, and branches about 0.09%; sampled cache misses rose
about 0.25%. On the 600-frame 1080p stream, four of five cycle pairs improved
and the median wall time moved from 3.93 to 3.91 seconds, with one large
scheduler outlier. A post-change PGO sample reduced the remaining whole-stream
`memset` share from about 2.38% to 2.03%. The 4K output remains byte-exact.

Full-macroblock Direct resolution now reuses the macroblock coordinates that
the CABAC/CAVLC reconstruction loop has already validated. The low-level
public motion-state methods retain their address-only entry points, while the
decoder's internal path passes `(address, x, y)` to avoid reconstructing
coordinates with another runtime-width integer division for every B_Skip or
fully Direct macroblock. Both spatial and temporal Direct use the same
coordinate-preserving path.

The native build reduced the specialized Spatial Direct symbol by about 100
bytes and removed its hot `div`, but four-worker timing remained too noisy to
claim a portable throughput gain. After retraining the same 4K CABAC, 1080p
CABAC, and 1080p CAVLC PGO profile, nine fixed-CPU 4K four-worker pairs reduced
mean wall time about 4.48%, task-clock about 3.99%, cycles about 2.63%, and
instructions about 1.58%; eight wall-time pairs improved. Mean throughput moved
from 42.93 to 44.95 FPS. Five 300-frame 1080p two-worker pairs reduced
task-clock about 1.8% and instructions about 1.6%. The exact
597,196,800-byte 4K hash, the generated H.264 corpus, and the MP4/seek corpus
remain unchanged.

CABAC significance maps are now constructed directly in their final stack
slot. The public split significance/level API is unchanged, while the complete
block decoder avoids returning and then copying the fixed 64-entry map before
level decoding. This reduced the PGO
`decode_cabac_coefficient_block_into` body from 4,423 to 4,327 bytes. The
native build was throughput-neutral, so the optimization does not depend on a
portable-build claim. After retraining the same mixed CABAC/CAVLC profile,
eight of nine alternating fixed-CPU 4K `Auto` wall-time pairs improved. Mean
wall time fell about 2.45%, task-clock about 2.34%, and reference cycles about
1.70%. The 597,196,800-byte 4K output retained SHA-256
`d261aeed6ed16abe634b89afe40017bed59ff9eb8aa1353279300d7ff9689534`;
the generated H.264 and MP4/seek corpora remained byte-exact.

CABAC inter residuals now initialize their integer-only backing storage once
and then write the per-category coefficient maxima in place. This replaces
array-repeat construction of 26 zero `ResidualBlock` values, which previously
made LLVM materialize and copy several large stack regions. The all-zero bit
pattern is valid for every `i32` and `u8` field, and all non-zero maxima are
restored before the value becomes observable.

In the native build, the `decode_inter_residual_inner` stack frame fell from
3,032 to 1,800 bytes and the symbol from 1,692 to 1,124 bytes. Nine alternating
4K `Auto` pairs reduced mean wall time about 1.37% and task-clock about 1.33%;
six wall-time pairs and seven reference-cycle pairs improved. After retraining
PGO, seven of nine wall-time pairs improved. Mean wall time fell about 1.85%,
task-clock about 1.10%, and reference cycles about 1.05%. The exact 4K hash and
both generated verification corpora remained unchanged.

Spatial Direct neighbour lookup now consumes the macroblock coordinates that
the caller already validated instead of deriving them from the address a
second time. This removes the remaining hot integer division from full
macroblock Spatial Direct resolution. The native resolver shrank by 40 bytes,
although eleven native 4K `Auto` pairs were timing-neutral and do not support
a portable-build throughput claim.

With the mixed PGO training counts held byte-for-byte equal to the preceding
baseline, ten of fifteen fixed-CPU 4K `Auto` wall-time pairs improved. Mean
wall time fell about 0.95%, task-clock about 0.92%, reference cycles about
0.35%, instructions about 0.12%, and branches about 0.16%. The inlined PGO
Direct resolver shrank by 45 bytes. The exact 597,196,800-byte 4K output
retained SHA-256
`d261aeed6ed16abe634b89afe40017bed59ff9eb8aa1353279300d7ff9689534`,
and both generated verification corpora remained byte-exact.

P-skip motion derivation now reuses the macroblock coordinates that the
reconstruction loop has already validated. The public address-only
`PMotionState` entry point remains unchanged, while the decoder's internal
path passes `(address, x, y)` through neighbour and median-predictor lookup.
This removes one runtime-width integer division from every skipped P
macroblock and a second division from the non-zero prediction branch.

The native build reduced instructions and branches by about 0.09% and 0.05%,
respectively, but its task-clock and reference-cycle measurements were neutral,
so there is no portable-build throughput claim. With byte-identical 4K CABAC,
1080p CABAC, and 1080p CAVLC PGO training inputs, ten of fifteen fixed-CPU 4K
`Auto` wall-time pairs improved. Mean wall time fell about 1.64%, task-clock
about 1.23%, instructions about 0.09%, branches about 0.07%, and branch misses
about 0.50%. The PGO resolver body shrank by 86 bytes. The exact
597,196,800-byte 4K output retained SHA-256
`d261aeed6ed16abe634b89afe40017bed59ff9eb8aa1353279300d7ff9689534`,
and both generated verification corpora remained byte-exact.

Single-list weighted prediction now skips planes whose decoded operation is
the identity. An omitted weight and an explicit `(1 << denominator, 0)`
weight both reproduce every input sample exactly, so luma, Cb, and Cr can be
checked independently before entering the SIMD loop. The per-component check
also preserves a non-identity chroma operation when only its sibling uses the
default weight.

Fifteen native 4K `Auto` pairs reduced mean instructions about 1.73%,
reference cycles about 1.51%, task-clock about 1.30%, and wall time about
0.98%; all fifteen instruction and branch pairs improved. After retraining
the byte-identical mixed PGO profile, thirty 4K `Auto` pairs reduced
instructions about 2.49%, branches about 3.12%, task-clock about 0.20%, and
reference cycles about 0.21%, while wall time remained neutral. The optimized
work is on the serial critical path: twelve PGO 4K `Serial` pairs reduced mean
wall time about 2.35%, task-clock about 2.80%, reference cycles about 1.99%,
instructions about 2.46%, and branches about 3.27%. Eleven of twelve
reference-cycle pairs improved. The main PGO weighting entry point also
shrunk by about 7.0 KiB, with the less common chroma implementation outlined.
The exact 597,196,800-byte 4K hash and both generated verification corpora
remained byte-exact.

P-skip state recording now uses one immutable empty neighbour grid and fills
the validated destination cells directly. The skipped macroblock is uniform
by definition, so constructing a second 16-cell stack grid, populating it
through the generic partition validator, and copying it into the picture
state was redundant. All fallible neighbour and predictor work still
completes before the destination changes, preserving transactional failures.

The native resolver shrank from 1,312 to 756 bytes. Fifteen native 4K `Auto`
pairs reduced mean instructions about 0.23%, reference cycles about 0.28%,
and wall time about 0.52%; all fifteen instruction pairs improved. With
byte-identical PGO training inputs, eleven of fifteen 4K `Auto` wall-time
pairs improved. Mean wall time fell about 1.84%, task-clock about 2.06%,
reference cycles about 1.47%, and instructions about 0.49%. The PGO resolver
shrunk by 542 bytes. The exact 597,196,800-byte 4K output retained SHA-256
`d261aeed6ed16abe634b89afe40017bed59ff9eb8aa1353279300d7ff9689534`,
and both generated verification corpora remained byte-exact.

Spatial Direct reconstruction now records the co-located zero flags in one
16-bit mask before constructing output partitions. An all-clear or all-set
mask produces one uniform 16x16 partition and one uniform motion-state fill;
only a genuinely mixed mask constructs the 4x4 or 8x8 partition grid. This
also preserves complete co-located-grid validation before any picture state
changes.

The opt-in internal profile explains why the branch matters. On the 48-frame
4K stream, 369,785 of 691,900 Spatial Direct macroblocks already had
predictions that could not be changed by the co-located zero flag. Of the
remaining macroblocks, 318,891 masks were all clear, 1,747 were all set, and
only 1,477 were mixed. The 300-frame 1080p stream reported 1,135,509
prediction-uniform macroblocks, 340,716 all-clear masks, 12,418 all-set masks,
and 1,777 mixed masks out of 1,490,420 total.

Fifteen alternating native 4K `Auto` pairs reduced mean wall time about
1.83%, task-clock about 1.51%, reference cycles about 1.44%, instructions
about 1.35%, and branches about 1.44%. All fifteen instruction and branch
pairs improved. With byte-identical PGO training manifests, the stable final
fourteen pairs reduced mean wall time about 3.00%, task-clock about 2.25%,
reference cycles about 2.31%, instructions about 1.37%, and branches about
1.51%; twelve wall-time pairs improved, while all fourteen instruction and
branch pairs improved. The streamed 597,196,800-byte 4K output retained
SHA-256
`d261aeed6ed16abe634b89afe40017bed59ff9eb8aa1353279300d7ff9689534`,
and the generated H.264 corpus remained byte-exact.

Zero-residual CABAC inter macroblocks now record their inferred neighbour
state as one macroblock transition. P- and B-skips avoid creating a CABAC
syntax facade and an unused `InterResidual`; ordinary inter macroblocks with a
zero coded-block pattern avoid walking the luma and chroma residual loops.
The specialized fill derives the macroblock coordinates once and writes the
three DC states plus the luma, Cb, and Cr grids in contiguous rows. No bitstream
syntax is skipped because a zero coded-block pattern contains no residual
bins.

After excluding one scheduler outlier, fourteen alternating native 4K `Auto`
pairs reduced median wall time about 4.66%, task-clock about 2.97%, reference
cycles about 3.90%, instructions about 4.64%, and branches about 3.49%. Every
wall-time, task-clock, instruction, and branch pair improved. With
byte-identical mixed PGO training manifests, fifteen 4K `Auto` pairs reduced
median task-clock about 0.68% and reference cycles about 0.79%, while wall time
was neutral; PGO instructions increased about 0.62% and branches about 1.54%.
On the 300-frame 1080p stream, seven PGO pairs reduced median wall time about
1.69%, task-clock about 1.42%, and reference cycles about 1.22%, while
instructions increased about 0.42% and branches about 1.23%. The native
critical-path gain is therefore clear, while PGO already optimizes much of the
old zero-pattern loop and should be judged primarily by elapsed CPU time. The
streamed 597,196,800-byte 4K output retained SHA-256
`d261aeed6ed16abe634b89afe40017bed59ff9eb8aa1353279300d7ff9689534`.

Reference-motion recording for B pictures now preserves the macroblock
coordinates already validated by the reconstruction loop. The address-only
test entry point derives the same coordinates, while the decoder's internal
path passes `(address, x, y)` directly. The fixed-row uniform writer is
outlined to keep the B recorder compact and removes one runtime-width integer
division without returning a large helper value. Mixed partitions retain the
existing complete coverage and overlap validation.

Ten alternating native 4K `Auto` pairs reduced mean wall time about 2.18%,
task-clock about 1.30%, and reference cycles about 1.41%; instructions and
branches were effectively neutral. With byte-identical PGO training
manifests, fifteen 4K `Auto` pairs reduced mean wall time about 0.74%,
task-clock about 0.56%, reference cycles about 0.78%, instructions about
0.69%, and branches about 0.23%. All fifteen PGO instruction pairs, fourteen
branch pairs, and nine wall-time pairs improved. The streamed
597,196,800-byte 4K output retained SHA-256
`d261aeed6ed16abe634b89afe40017bed59ff9eb8aa1353279300d7ff9689534`,
and the generated H.264 and MP4/seek corpora remained byte-exact.

## Frame Service Timing

`decv-cli` has an opt-in `frame-timing` feature for measuring decoder
tail latency without changing the library API or ordinary release binaries.
With `--frame-timing`, it accumulates wall time spent in `send_packet`,
`receive_frame`, `flush`, and `drain` between output-frame events, then reports
mean, p50, p95, p99, and maximum service time. File reads, visible-frame
writes, and CLI logging are outside the measured interval. The first sample
includes decoder initialization and presentation-reordering pre-roll; long
streams are required for representative steady-state percentiles.

On the native development build, the current 48-frame 4K sample reported a
25.019 ms mean and 22.337 ms p50. Two runs of the 300-frame 1080p CABAC sample
reported 7.515-8.223 ms means, 7.078-7.921 ms p50s, 11.555-12.210 ms p95s,
21.979-25.608 ms p99s, and 46.628-50.576 ms maxima. The run-to-run spread is
itself a reminder that host scheduling affects wall-time tails. These are
diagnostic service intervals rather than a playback scheduler result. They
expose long decoder stalls that a whole-stream FPS average hides and provide a
second acceptance signal for future optimizations.

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
  dependency path;
- adding an unchecked 1-to-7-bit `BitReader` operation for CABAC
  renormalization produced the same 381-byte native `decode_decision` body as
  the generic runtime-width call. Its only hot machine-code difference was an
  equivalent x86 shift-count immediate (`7` versus `0x47` after hardware
  masking). Seven pinned CABAC Serial pairs changed reference cycles by about
  +0.07% and instructions by about +0.08%. The compiler already proves the
  narrow width from the CABAC range invariant, so the extra unsafe API was
  fully reverted.

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

## SIMD Edge-Read Safety

The x86 chroma bilinear fast paths must not read an interpolation neighbour
whose coefficient is zero. In particular, a zero horizontal fraction permits
the source rectangle to touch the right plane edge, and a zero vertical
fraction permits it to touch the bottom edge. The former AVX2 and SSE2
implementations loaded the right, bottom, and bottom-right vectors
unconditionally, then multiplied the unused vectors by zero. That preserved
pixel values but could read past the backing allocation at the last row.

The optimized paths now select horizontal-only, vertical-only, and
two-dimensional load sets before touching source memory. An exact-size
right/bottom-edge regression is also run under AddressSanitizer: it reported
the old eight-byte heap over-read and passes with the conditional loads.
Seeking made the fault easier to observe by reconstructing many boundary
macroblocks quickly, but decoder parallelism and timeline state were not the
root cause.

## Exact Seek Preroll Output Suppression

An exact seek must reconstruct every dependency from the preceding sync
sample, but frames before the requested timestamp do not need consumer-facing
NV12 storage. `H264Decoder::flush_for_seek` retains a lightweight marker for
each suppressed picture in the presentation reorder buffer. Reference
pictures still enter the DPB. A second optimization skips slice pixel
reconstruction for a pre-target picture when `nal_ref_idc == 0`, because its
pixels cannot be referenced by the selected or any later picture. Slice
headers and POC state are still parsed so the decoder timeline remains exact.

On the 30-second 1920x1080, 120 fps test stream, seeking to 29.90 seconds
requires roughly 228 frames of preroll from the 28-second keyframe. Ten
alternating native release pairs reduced median end-to-end CLI time from
1.315 seconds to 1.230 seconds (about 6.5%) and median user CPU time from
1.510 seconds to 1.445 seconds (about 4.3%). First-frame-only measurements
were noisier but usually saved 35-50 ms. This does not remove the dominant
reconstruction cost of a long GOP; approximate following-keyframe preview,
request cancellation, or a shorter source keyframe interval is still required
for consistently immediate interactive scrubbing.

Skipping pre-target non-reference reconstruction was then measured against
that output-suppression baseline with the same native build settings. Ten
alternating pairs reduced median end-to-end time from 1.255 seconds to
0.810 seconds (about 35.5%) and median user CPU time from 1.46 seconds to
0.83 seconds (about 43%). Five-run `perf stat` means reduced instructions from
16.03 billion to 8.69 billion, cycles from 6.33 billion to 3.62 billion, and
task-clock from 1.70 seconds to 1.04 seconds. The generated MP4 seek corpus and
a 1440x2560, 60 fps, 132-second long-GOP source remained byte-exact against
FFmpeg at the selected suffix. Repeated-seek tests also cover filter
replacement, retained reference reconstruction, skipped B-picture reorder
markers, and restoring ordinary unfiltered output with `flush`.

## Exact Seek Cost Attribution

The MP4 index lookup is not the limiting part of an exact seek. A probe against
a 1440x2560, 60 fps, 132-second MP4 with one sync sample every 4.167 seconds
measured ten complete demuxer opens at 3.9-8.1 ms, with a 6.0 ms median. Creating
the packet cursor and binary-searching the presentation-sorted keyframe index
both completed below the probe's one-microsecond reporting resolution. A real
player normally retains the open demuxer, eliminating even the 6 ms parse cost.

First-frame latency was measured by stopping the native decoder immediately
after its first output. Five alternating runs at targets inside the final GOP
produced these medians:

| Target relative to preceding keyframe | First-frame latency |
| ---: | ---: |
| 0.000 s | 91.7 ms |
| 1.000 s | 389.8 ms |
| 2.000 s | 625.0 ms |
| 2.733 s | 822.2 ms |

The approximately linear growth, about 267 ms for each additional second of
GOP distance over the complete interval, identifies dependency reconstruction
as the dominant cost. Exact H.264 seek must rebuild every reference picture
needed between the sync sample and the selected picture. The decoder already
suppresses pre-target output storage and entirely skips pre-target
non-reference pictures, so the remaining work is chiefly required reference
P-picture reconstruction, entropy/residual decoding, motion compensation, and
deblocking. Further acceleration of normal decode helps this path, but an
interactive sub-100 ms response across a 4.167-second GOP requires a
presentation strategy as well: following-keyframe preview, cached decoder
checkpoints/reference pictures, request cancellation, or shorter source GOPs.
`PacketCursor::seek_to_nearest_keyframe` provides the generic preview lookup:
it chooses the closer adjacent sync sample in logarithmic time, avoids preroll,
and limits the usual preview timestamp error to half of the surrounding
keyframe interval. Exact output still uses the preceding-keyframe path above.

## Integer Bidirectional Prediction Fusion

Default bidirectional prediction with integer luma and chroma motion formerly
copied both reference rectangles into separate scratch predictions, averaged
one scratch into the other, then copied the result into macroblock staging.
The interior fast path now averages both reference rows directly into the
staged luma, Cb, and Cr blocks. SSE2 handles the fixed 2/4/8/16-byte row widths
on x86_64; other targets retain the scalar normative equation. Fractional,
clipped, single-list, explicit-weight, and implicit-weight cases continue
through the general interpolation path.

Eleven-run PGO `perf stat` means on the 48-frame 4K comparison stream reduced
cycles from 6.211 billion to 6.136 billion (1.2%) and task-clock from 1.690 to
1.680 seconds (0.6%). Mean wall time fell from 1.078 to 1.054 seconds (2.2%),
although wall-time noise remains larger than the hardware-counter change.
Instructions increased by 0.2% because of fast-path qualification. The
complete 570 MiB NV12 output retained SHA-256
`d261aeed6ed16abe634b89afe40017bed59ff9eb8aa1353279300d7ff9689534`.
On the long-GOP seek source, seven native first-frame pairs changed the median
from 836.7 to 813.0 ms (2.8%); this is useful but confirms that ordinary
motion-compensation tuning alone cannot make long-GOP exact seek immediate.

## Lazy Deblock Threshold Preparation

Picture deblocking formerly prepared luma, Cb, and Cr threshold tables for
every permitted macroblock boundary before checking whether any derived
boundary strength was non-zero. Direct and skip-heavy inter pictures contain
many legal boundaries that require no sample filtering. The traversal now
derives all boundary strengths first, then prepares left, top, internal-luma,
and internal-chroma thresholds only for edge groups that will actually be
visited. Internal chroma thresholds are also omitted when only luma-only
internal edges are active.

On the 48-frame 4K comparison stream, twelve alternating PGO pairs reduced
median wall time from 1.015 to 0.980 seconds, increasing measured throughput
from about 47.3 to 49.0 fps. Eleven-run `perf stat` means reduced task-clock
from 1.619 to 1.581 seconds (2.3%), instructions from 13.381 to 12.974 billion
(3.0%), branches from 1.726 to 1.687 billion (2.2%), and branch misses by
0.8%. Cycles were effectively unchanged, while mean wall time fell from 1.032
to 0.998 seconds (3.3%). The full 570 MiB NV12 output retained SHA-256
`d261aeed6ed16abe634b89afe40017bed59ff9eb8aa1353279300d7ff9689534`.

The same change was nearly neutral for the 1440x2560 long-GOP exact-seek
first-frame benchmark, where ten PGO pairs changed the median from 751.3 to
748.0 ms. That path remains dominated by required reference-picture
reconstruction rather than threshold preparation.

## Generation-Reused P-Motion and Intra-Mode Grids

The reusable reconstruction workspace now retains the P-slice motion grid and
the intra-prediction mode grid in addition to the existing B-motion and CABAC
residual state. A new picture assigns a monotonically increasing first-slice
generation. Cells from an older generation are treated as empty without
clearing or reallocating the complete grids; an actual fill is required only
when the generation counter wraps or coded dimensions change. Rollback still
clears the affected macroblock immediately, and focused tests cover allocation
retention, stale-cell invalidation, duplicate-write rejection, and the
generation-exhaustion clear.

This is especially valuable during exact seek, where several seconds of
reference pictures may be reconstructed without materializing their output.
On the 1440x2560 long-GOP seek source, fourteen alternating PGO pairs reduced
median end-to-end time from about 0.76 to 0.705 seconds, roughly 7.2%. Eleven
PGO `perf stat` runs reduced mean task-clock from 1.098 to 0.985 seconds
(10.3%) and minor faults from 129,625 to 86,040 (33.6%). The complete six-frame
post-seek NV12 suffix retained SHA-256
`b27258d86f27c0f8d0c0cb8f1fa16b561205b68708e1e83f704dd81292103a51`.

The change also improves ordinary 4K decoding. Fourteen alternating PGO pairs
reduced the 48-frame median from about 0.995 to 0.950 seconds, increasing
throughput from about 48.2 to 50.5 fps. Eleven PGO counter runs reduced
task-clock by 1.4%, cycles by 0.7%, wall time by 2.2%, and minor faults by
17.5%. Instructions were effectively neutral and branches increased about
1.0%, a measured control-flow cost outweighed by lower allocation and
page-fault overhead. The full 570 MiB NV12 output retained SHA-256
`d261aeed6ed16abe634b89afe40017bed59ff9eb8aa1353279300d7ff9689534`.

## Write-Once Deblocking Metadata

Each picture previously initialized the complete deblocking-metadata vector
with `MacroblockDeblockInfo::default()` before reconstruction, even though
every successfully completed macroblock replaces its entire entry. At 4K the
redundant initialization touches about 5.5 MiB per picture. The vector now
stores `MaybeUninit<MacroblockDeblockInfo>` internally and initializes one
entry when that macroblock completes. The existing completion bitmap remains
the safety boundary: neighbour reads require the corresponding completion
flag, and finalization rejects an incomplete picture before converting the
metadata into an initialized slice. Focused tests cover both complete direct
writers and rejection of an incomplete picture.

The native build retained byte-exact output and reduced mean 4K task-clock by
about 1.3%, CPU cycles by 2.0%, and wall time by 1.2% across seven counter
runs. Nine native exact-seek counter runs reduced task-clock and wall time by
about 1.9%. After retraining PGO, nine 4K counter runs reduced task-clock by
2.6% and wall time by 4.1%; instructions were neutral, branch misses fell
0.6%, and sampled cache misses fell 2.1%. Eleven exact-seek counter runs
reduced task-clock by 5.2% and wall time by 6.2%, while instructions remained
within 0.1%. CPU cycles increased about 1.0% in that seek sample because the
candidate ran at a lower average effective frequency, so the retained result
is based on task-clock and alternating end-to-end timings rather than cycles
alone.

The complete 48-frame 4K NV12 stream retained SHA-256
`d261aeed6ed16abe634b89afe40017bed59ff9eb8aa1353279300d7ff9689534`.
The six-frame post-seek suffix retained SHA-256
`b27258d86f27c0f8d0c0cb8f1fa16b561205b68708e1e83f704dd81292103a51`,
and the complete native H.264 and MP4 verification corpora remained
byte-exact.

## Entropy-Selective CAVLC State

The picture reconstructor previously allocated CAVLC neighbour grids for every
picture, including CABAC pictures that cannot use them. A 4K picture's luma
and two chroma coefficient grids contain 783,360 `u32` entries, about 3.0 MiB
of zeroed state. Reconstruction now selects the state from the active PPS:
CAVLC pictures retain the original fully allocated concrete state, while
CABAC pictures construct the same concrete type with inactive, allocation-free
grids. The CABAC-only constructor is kept out of line so the CAVLC decode hot
path gains neither an `Option` branch nor a changed field representation.

After retraining the mixed CABAC/CAVLC PGO profile, nine 4K counter runs
reduced task-clock by 1.9%, cycles by 2.0%, wall time by 2.2%, and page faults
by 10.1%. Eleven long-GOP exact-seek runs reduced task-clock by 2.7%, cycles by
1.7%, wall time by 3.4%, and page faults by 18.1%. Instructions were neutral
in both workloads. A separate fifteen-run, fixed-CPU serial CAVLC comparison
also remained instruction-neutral while reducing task-clock by 2.8%, cycles
by 1.8%, and wall time by 2.4%; this guards against improving CABAC by taxing
the other entropy mode.

The complete 48-frame 4K output and six-frame exact-seek suffix retained their
respective SHA-256 hashes
`d261aeed6ed16abe634b89afe40017bed59ff9eb8aa1353279300d7ff9689534` and
`b27258d86f27c0f8d0c0cb8f1fa16b561205b68708e1e83f704dd81292103a51`.

## Generation-Reused CABAC Inter Syntax Grids

CABAC P and B slice decoding formerly constructed fresh macroblock-summary and
4x4 motion-syntax grids for every picture. The allocator can recycle those
blocks, but zero-initializing `Option` cells repeatedly still discards useful
physical pages during long-GOP preroll. The reconstruction workspace now
retains lazily allocated P and B states. Exact slice IDs make previous-picture
cells invisible without clearing; a first-slice generation additionally keeps
duplicate macroblock writes detectable within the current picture. Dimension
changes reallocate, and generation wrap explicitly clears all retained cells.
State returns to the workspace on both successful and failed slice decoding.

On the 1440x2560 long-GOP exact-seek workload, seven alternating PGO pairs
reduced median end-to-end time from 0.71 to 0.68 seconds and median CPU task
time from about 0.94 to 0.89 seconds. Five-run counters reduced mean task-clock
by 3.0%, elapsed time by 3.7%, and minor faults from 70,481 to 45,497 (35.4%).
A deeper 3.3-second preroll improved first-frame median from 1.36 to 1.34
seconds. Retained state increased peak RSS by about 5 MiB on the shorter seek.

The 48-frame 4K PGO counter sample reduced task-clock by 2.2%, cycles by 1.1%,
elapsed time by 2.6%, and minor faults by 3.2%; seven alternating wall-time
pairs were neutral at a 0.97-second median. Keeping both P and B grids raised
median peak RSS by about 12 MiB on that stream. This is an explicit
throughput/latency tradeoff: the retained memory is bounded by coded
dimensions and avoids repeated page churn across arbitrarily long streams.
The CAVLC control workload remained neutral because it never allocates these
states.

The native H.264 and MP4 verification corpora remained byte-exact. The
complete 4K output and exact-seek suffix retained SHA-256
`d261aeed6ed16abe634b89afe40017bed59ff9eb8aa1353279300d7ff9689534` and
`b27258d86f27c0f8d0c0cb8f1fa16b561205b68708e1e83f704dd81292103a51`.

## Serial Sub-4K Exact-Seek Preroll

Rayon reconstruction batches are valuable for sustained high-resolution
decode, but they were a net loss during sub-4K exact-seek preroll. Output
suppression removes non-reference pictures before the target, leaving short
reference-picture batches separated by serial CABAC, motion, commit, and
deblock work. On the 1440x2560 long-GOP source, increasing reconstruction
workers from one to two, four, six, or eight raised CPU time without reducing
first-frame latency. Serial execution was faster and avoided worker wakeups,
cross-thread staging, and synchronization.

`Auto` now selects serial reconstruction for pictures below eight megapixels
whose PTS precedes the active exact-seek target. The selected picture and
ordinary playback immediately use the normal size-derived executor again.
4K-class preroll keeps its wider executor because its macroblock batches do
amortize parallel scheduling, and an explicit `Threads(n)` policy is never
overridden. A boundary probe favored serial at 3840x2048 (0.66 versus 0.69
seconds for two threads), while 3840x2160 favored four threads (0.62 versus
0.66 seconds for serial), supporting the existing eight-megapixel split.

Nine alternating PGO pairs on the 1440x2560 source reduced median end-to-end
seek time from 0.67 to 0.64 seconds and median CPU task time from about 0.91 to
0.65 seconds. Seven-run counters reduced mean task-clock from 965 to 665 ms
(31.1%), cycles from 3.269 to 2.988 billion (8.6%), instructions from 6.910 to
6.765 billion (2.1%), branch misses from 6.78 to 5.20 million (23.3%), and
elapsed time from 0.730 to 0.650 seconds (11.0%). For a deeper 3.3-second
preroll, first-frame median fell from 1.31 to 1.12 seconds (14.5%) while CPU
time fell from about 1.87 to 1.14 seconds (39%).

A separate 1920x1080, 60 fps MP4 seek reduced median wall time from 0.24 to
0.22 seconds and CPU task time from about 0.29 to 0.22 seconds. Ordinary 4K
decode remained neutral because it does not enter the seek-specific serial
path. The complete native and PGO H.264/MP4 verification corpora remained
byte-exact.

## Interpretation

The wall-time gap is not explained by thread count alone. Single-threaded
FFmpeg is already about 2.9x faster in the comparable NV12 case. FFmpeg then
reduces latency further with mature frame/slice threading, while decv currently
parallelizes owned CABAC P- and B-macroblock pixel reconstruction. CABAC
parsing and residual reconstruction remain ordered on the caller. Non-reference
deblocking and output packaging can overlap a later picture, but reference
pictures still impose their synchronous completion barrier.

The immediate optimization priority should therefore remain single-thread hot
loops and broader dependency-aware parallelism, not a larger Rayon pool. The
new B-batch overlap consumes part of the caller's former idle boundary, but
CABAC arithmetic and residual parsing remain a serial dependency chain.
Motion compensation is the largest combined worker cost, and reference-picture
deblocking remains a visible barrier. Any new optimization must keep byte-exact
output against FFmpeg and must be benchmarked in both `Serial` and `Auto`
modes.
