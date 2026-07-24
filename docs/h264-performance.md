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
| decv Serial | NV12 | 1.56 s | 1.48 s | 80,492 KiB | 115.4 FPS |
| decv Auto (2 workers) | NV12 | 1.73 s | 1.82 s | 79,772 KiB | 104.0 FPS |
| FFmpeg 1 thread | NV12 | 0.61 s | 0.69 s | 152,128 KiB | 295.1 FPS |
| FFmpeg Auto | NV12 | 0.26 s | 1.43 s | 286,788 KiB | 692.3 FPS |
| FFmpeg 1 thread | decode-only | 0.57 s | 0.55 s | 95,772 KiB | 315.8 FPS |
| FFmpeg Auto | decode-only | 0.23 s | 0.97 s | 192,168 KiB | 782.6 FPS |

On this workload:

- decv Serial takes about **2.6x** as much wall time as single-threaded FFmpeg
  when both produce NV12;
- decv Auto takes about **6.7x** as much wall time as FFmpeg Auto when both
  produce NV12;
- decv Auto does about **1.27x** as much total user-CPU work as FFmpeg Auto's
  NV12 path;
- decv uses about **53%** of FFmpeg single-threaded NV12 peak RSS and about
  **28%** of FFmpeg Auto NV12 peak RSS;
- prior measurements with 16 decv workers were slower than the two-worker
  `Auto` policy and consumed far more CPU, confirming that the current parallel
  region is too narrow to scale.

The 60 FPS real-time target requires decoding 180 frames in at most 3.00
seconds. The current Serial result has about 92.3% throughput headroom over that
line, and the measured two-worker Auto result has about 73.4%. The ordering
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
FFmpeg is already about 2.9x faster in the comparable NV12 case. FFmpeg then
reduces latency further with mature frame/slice threading, while decv currently
parallelizes owned CABAC P- and B-macroblock pixel reconstruction. CABAC
parsing, residual reconstruction, output packaging, and deblocking remain
serial.

The immediate optimization priority should therefore remain single-thread hot
loops and broader dependency-aware parallelism, not a larger Rayon pool.
Deblocking and motion compensation are the most important measured hot regions.
Any new optimization must keep byte-exact output against FFmpeg and must be
benchmarked in both `Serial` and `Auto` modes.
