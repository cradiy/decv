# H.264 Parallel Decoding Plan

This document defines the concurrency boundary for the software H.264 decoder.
It is an implementation plan, not a promise that every stream can use every
kind of parallelism.

## Why SIMD Is Not Enough

The current 1920x1080 High Profile benchmark decodes 180 frames in about 2.10
seconds in Serial mode. That is roughly 85.7 frames per second. The decoder has
therefore crossed the 1080p60 line on this machine, but it remains far from the
4K120 stretch goal and from FFmpeg's single-thread throughput. No single
remaining kernel can provide that speedup alone.

The decoder therefore needs multiple independent work items in flight. It must
not make CABAC, neighbour derivation, or reference-picture state concurrent
merely to create work: those states have normative ordering requirements.

## State That Remains Serial

The following operations remain on the decoder's owning thread:

- NAL and slice-header parsing;
- CABAC arithmetic state and context updates;
- CAVLC neighbour coefficient state;
- macroblock address progression and slice termination;
- macroblock QP derivation;
- spatial and temporal direct-motion derivation;
- reference-list construction and DPB marking;
- POC calculation and output reordering;
- intra prediction and any inter/intra mixed slice that needs reconstructed
  neighbouring pixels while syntax is still being consumed;
- committing completion flags, motion summaries, and deblocking metadata.

Making any of these operations race would change the decoded bitstream
semantics.

## First Parallel Work Unit

The first useful work unit is one already-resolved inter macroblock:

```rust
struct InterMacroblockJob {
    address: usize,
    macroblock_x: usize,
    macroblock_y: usize,
    motion: ResolvedBMacroblock,
    residual: ReconstructedInterResidual,
    weight_mode: OwnedBPredictionWeightMode,
    deblock: MacroblockDeblockInfo,
}
```

The exact types may change, but the ownership rule must not:

- a job owns all syntax-derived data needed for reconstruction;
- reference pictures are immutable and remain alive for the whole batch;
- jobs never read the current output picture;
- jobs write disjoint 16x16 Y and 8x8 Cb/Cr macroblock regions;
- a worker returns pixels or writes through pre-split disjoint row regions;
- decoder-visible state is committed only after every job succeeds.

The safest first implementation returns a fixed-capacity `MacroblockPixels`
value from each worker. The decoder thread then copies successful results into
the picture in address order. This adds one small copy per macroblock but
avoids unsafe aliased writes during the initial concurrency change. A later
version may split the three output planes into disjoint macroblock-row slices
and let workers write directly.

## Two-Phase CABAC B-Slice Pipeline

The common single-slice B picture should use two explicit phases.

### Phase 1: serial syntax and derivation

For every macroblock:

1. Decode CABAC syntax and residual coefficients.
2. Advance the transactional QP state.
3. Reconstruct transform coefficients into an owned residual.
4. Resolve List 0 and List 1 motion, including Direct modes.
5. Validate reference indices and derive deblocking metadata.
6. Append an inter job, or immediately process an Intra/PCM barrier.

An Intra or PCM macroblock is a barrier because it can require current-picture
neighbours. Before processing the barrier, finish and commit the pending inter
batch. After the barrier, a new inter batch may begin.

### Phase 2: parallel pixel reconstruction

For each batch:

1. Run motion compensation, prediction weighting, residual addition, and
   sample clipping in parallel.
2. Collect results without mutating decoder-visible completion state.
3. If any job fails, discard every result in the batch and return the first
   error in macroblock-address order.
4. Otherwise copy pixels and commit mode, motion, completion, and deblocking
   state in macroblock-address order.

Deterministic error ordering matters for tests, fuzzing, and callers that log
malformed streams.

## Reference and Weight Ownership

Worker jobs must not borrow temporary slice-header objects. Before dispatch:

- convert prediction weights into a small owned value;
- store reference identity/index in resolved motion;
- hold reference pictures through an immutable batch-level reference table;
- do not clone reference pixel allocations per job.

The worker pool may borrow the batch reference table through scoped tasks.

## Thread-Pool Policy

`decv` remains synchronous and runtime-independent. Internal parallelism must
not require Tokio or another async runtime.

The H.264-specific decoder configuration exposes:

```rust
pub enum H264Parallelism {
    Serial,
    Auto,
    Threads(NonZeroUsize),
}

decoder.set_parallelism(H264Parallelism::Auto)?;
decoder.configure(video_config)?;
```

Rules:

- `Serial` is the deterministic fallback and test oracle.
- `Auto` currently caps the worker count at two instead of consuming every
  logical CPU in a UI application. Four workers improve the 1080p benchmark
  when pinned to four performance cores, but unpinned runs still add CPU work
  without reducing wall time. The cap can be retuned as more stages become
  parallel.
- `set_parallelism` must run before decoding begins. It is separate from the
  codec-independent `VideoDecoderConfig`.
- a decoder owns a persistent pool; it must not create OS threads
  per frame;
- single-job batches stay serial;
- `flush` waits for or cancels all internal work before clearing the DPB;
- no task may outlive the decoder or a borrowed reference picture.

## Deblocking

Deblocking is not an embarrassingly parallel per-macroblock loop. Filters write
samples on both sides of an edge, and the normative macroblock/edge order makes
nearby work overlap.

The first parallel decoder should keep deblocking serial. Later options are:

1. a dependency-counted macroblock wavefront that preserves vertical-then-
   horizontal edge order;
2. multiple independent slices when `disable_deblocking_filter_idc` prevents
   cross-slice filtering;
3. frame-level parallelism when multiple decoded pictures have no unresolved
   reference dependency.

Y, Cb, and Cr are independent planes, but measured chroma deblocking is only
about 50 milliseconds per 60-frame benchmark. Plane-level scheduling alone
does not justify a thread-pool dependency.

## Rollout Stages

1. Extract an allocation-free function that reconstructs one inter macroblock
   into `MacroblockPixels`; keep the existing serial caller.
2. Add address-ordered batch commit and failure rollback tests.
3. Convert CABAC B inter macroblocks into owned jobs; flush at Intra/PCM
   barriers.
4. Add a persistent scoped worker pool and serial/parallel byte-exact tests.
5. Repeat the job representation for CABAC P slices.
6. Evaluate direct disjoint-plane writes after measuring the copy-based
   implementation.
7. Design deblocking wavefront scheduling only if reconstruction parallelism
   still leaves 1080p60 below target.

Current status: stages 1 through 5 are implemented for CABAC B and P inter
macroblocks. Syntax, QP, residual transform, Direct motion, and metadata
derivation remain serial. Four macroblock rows of owned jobs are reconstructed
on a decoder-owned persistent pool, results are collected in address order,
and batch commit validates all addresses before touching the picture.
Intra/PCM macroblocks flush the pending batch before using current-picture
neighbours. The CAVLC P path retains direct in-place reconstruction because
routing it through owned staging measured slower. `Serial`, conservative
`Auto`, and explicit `Threads(n)` policies are available, and embedded real
CABAC P and B fixtures must match byte for byte between serial and two-thread
decoding.

## Acceptance Checks

Every parallel stage must pass:

- byte-exact output against the serial decoder and FFmpeg;
- identical `FormatChanged`, PTS order, drain, seek, and flush behaviour;
- deterministic malformed-input errors with no panic;
- ThreadSanitizer or Miri coverage for newly introduced unsafe boundaries when
  practical;
- bounded memory during long playback;
- benchmarks with 1, 2, 4, and auto worker counts;
- proof that `Serial` remains available on all supported architectures.

The 1080p60 target is reached only when the full 60-frame benchmark completes
in at most one second with byte-exact output. Kernel-only microbenchmarks do
not satisfy that target.
