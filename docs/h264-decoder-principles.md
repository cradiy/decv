# H.264 Decoder Principles

This document records the mental model behind the decoder so that the code can
be understood and reviewed even when much of it is generated with AI.

The most important questions to keep asking are:

1. Which representation layer is the current data in?
2. What representation is this function converting it into?
3. Which state and invariants must remain correct for later decoding stages?

## 1. The Complete Data Pipeline

A raw byte stream does not go directly into pixel reconstruction. An H.264
decoder processes several distinct layers:

```text
File or network bytes
    |
    v
Container demuxing or Annex-B splitting
    |
    v
NAL units
    |
    v
EBSP to RBSP conversion
    |
    v
Bit-level syntax parsing
    |
    v
SPS, PPS, and slice headers
    |
    v
Entropy decoding
    |
    v
Prediction and residual decoding
    |
    v
Inverse quantization and inverse transform
    |
    v
YUV pixel reconstruction
    |
    v
Deblocking
    |
    v
Reference-picture management and output
```

Each layer should have a clear input and output type. Keeping these layers
separate prevents start codes, escape bytes, syntax bits, and decoded pixels
from being mixed together.

## 2. Annex-B and NAL Units

An elementary H.264 Annex-B stream commonly looks like this:

```text
00 00 00 01 67 ...
00 00 00 01 68 ...
00 00 01    65 ...
```

The sequences below are start codes:

```text
00 00 01
00 00 00 01
```

A start code separates NAL units and is not part of the NAL unit itself.

The first byte after the start code is the NAL header. For example:

```text
0x67 = 0110_0111

0 | 11 | 00111
|   |      |
|   |      +-- nal_unit_type = 7
|   +--------- nal_ref_idc = 3
+------------- forbidden_zero_bit = 0
```

It can be parsed as:

```rust
let forbidden_zero_bit = reader.read_bits_const::<1>()?;
let nal_ref_idc = reader.read_bits_const::<2>()?;
let nal_unit_type = reader.read_bits_const::<5>()?;
```

Common NAL unit types include:

| Type | Meaning |
| ---: | --- |
| 1 | Non-IDR coded slice |
| 5 | IDR coded slice |
| 6 | Supplemental Enhancement Information |
| 7 | Sequence Parameter Set |
| 8 | Picture Parameter Set |
| 9 | Access Unit Delimiter |

Common header bytes therefore include:

```text
0x67 -> SPS
0x68 -> PPS
0x65 -> IDR slice
0x61 -> non-IDR slice
```

The first useful decoder component after `BitReader` should be an Annex-B
splitter that returns individual NAL units without their start codes.

## 3. EBSP and RBSP

The byte sequence `00 00 01` has structural meaning in Annex-B. If it naturally
occurred inside compressed payload data, a splitter could mistake it for the
next NAL unit.

The encoder prevents this by inserting an emulation-prevention byte:

```text
Original RBSP:    00 00 01
Transported EBSP: 00 00 03 01
```

After a complete NAL unit has been isolated, the decoder removes valid
emulation-prevention bytes:

```text
00 00 03 00 -> 00 00 00
00 00 03 01 -> 00 00 01
00 00 03 02 -> 00 00 02
00 00 03 03 -> 00 00 03
```

The representation hierarchy is:

```text
Annex-B stream
    `-- start code + NAL unit

NAL unit
    `-- NAL header + EBSP

EBSP
    `-- remove emulation_prevention_three_byte

RBSP
    `-- actual syntax data consumed by BitReader
```

Do not pass start codes or unprocessed EBSP data to the syntax `BitReader`.

## 4. RBSP Trailing Bits

RBSP syntax does not use arbitrary zero padding. Its end is marked by:

```text
rbsp_stop_one_bit = 1
rbsp_alignment_zero_bit = 0...
```

The first `1` marks the end of the syntax, and zero bits then align the stream
to the next byte boundary.

For example:

```text
Syntax bits:  10110
RBSP byte:    10110 100
                    ^^^
                    stop bit followed by alignment zeroes
```

An SPS or PPS parser must recognize and validate these trailing bits instead of
interpreting them as another field.

## 5. SPS, PPS, and Slices

These structures form a dependency chain:

```text
Slice -> PPS -> SPS
```

### 5.1 Sequence Parameter Set

The Sequence Parameter Set describes long-lived properties of a coded video
sequence, including:

- profile and level;
- chroma format and bit depth;
- coded width and height;
- cropping;
- frame-number rules;
- picture-order-count rules;
- progressive or interlaced coding;
- reference-picture limits;
- optional VUI information.

The SPS tells the decoder what kinds of frames to expect and how large its
frame and reference buffers may need to be.

### 5.2 Picture Parameter Set

The Picture Parameter Set configures picture and slice decoding details,
including:

- CAVLC or CABAC entropy coding;
- slice groups;
- default reference counts;
- quantization parameters;
- prediction flags;
- deblocking-filter control.

Each PPS references an SPS by identifier.

### 5.3 Slice

A slice is a portion of a coded picture. A picture may contain multiple slices,
so a slice must not be assumed to equal a frame.

A slice header describes:

- which picture it belongs to;
- whether it is an I, P, or B slice;
- which PPS it uses;
- frame-number and picture-order information;
- reference-list changes;
- quantization parameters;
- deblocking settings.

The slice header is followed by entropy-coded macroblock data.

The decoder should retain parameter sets by identifier:

```rust
struct Decoder {
    sps_by_id: HashMap<u32, Sps>,
    pps_by_id: HashMap<u32, Pps>,
}
```

If a slice references a missing PPS, or a PPS references a missing SPS, the
decoder must report an error instead of guessing.

## 6. Syntax Parsing Is Conditional

H.264 is not a flat sequence of fields. The presence and interpretation of
later fields often depend on earlier values:

```text
read field A
    |
    +-- if A == 0, read field B
    |
    `-- otherwise, read fields C and D
```

The standard is the source of truth for:

- field width;
- signedness;
- Exp-Golomb versus fixed-width coding;
- conditional presence;
- valid value ranges;
- resulting decoder state.

Never infer a field width from nearby data or from what happens to work on one
sample stream.

## 7. Compressed Video Does Not Store Complete Pixels

The central reconstruction model is:

```text
reconstructed pixel = predicted pixel + residual
```

The result is clipped to the valid sample range:

```text
sample = clip(prediction + residual, 0, 255)
```

for an 8-bit component.

### 7.1 Intra Prediction

Intra-coded blocks predict pixels from already reconstructed neighbors in the
same picture, commonly using:

- pixels above the block;
- pixels to the left;
- the top-left pixel;
- directional extension;
- DC-like averaging.

An IDR picture can be reconstructed without older reference pictures, but its
blocks still depend on previously reconstructed neighbors within the current
picture.

### 7.2 Inter Prediction

Inter-coded blocks predict from one or more reference pictures. A motion vector
selects an area in a reference picture that approximates the current block.

A motion vector is not necessarily the physical motion of an object. It is a
coding instruction identifying where a useful predictor can be sampled.

Fractional motion-vector positions require interpolation rather than a simple
integer-coordinate copy.

### 7.3 Residual Reconstruction

The encoder transforms and compresses the difference between the original and
predicted block:

```text
Residual samples
    -> transform
    -> quantization
    -> entropy coding
```

The decoder performs the inverse operations:

```text
Entropy decoding
    -> inverse scan
    -> inverse quantization
    -> inverse integer transform
    -> residual samples
    -> add prediction
```

The exact integer arithmetic, rounding points, clipping rules, and operation
order are normative. Mathematically similar arithmetic can still produce
different pixels and break reference-picture decoding.

## 8. CAVLC and CABAC

H.264 defines two important entropy-coding systems.

### CAVLC

- Uses context-adaptive variable-length codes.
- Is relatively approachable.
- Is used by Baseline-profile streams.
- Is the better first implementation target.

A residual block is not decoded as a flat list of signed integers. The CAVLC
pipeline is:

```text
coeff_token
    -> TotalCoeff + TrailingOnes
    -> trailing-one sign bits
    -> remaining level prefixes and suffixes
    -> total_zeros
    -> run_before values
    -> coefficients in scan order
```

`TotalCoeff` says how many non-zero coefficients exist. `TrailingOnes` says how
many of the last non-zero coefficients have magnitude one. `total_zeros` and
`run_before` then place those non-zero levels back among the zero coefficients.

The `coeff_token` table is chosen from `nC`, which is derived from the decoded
non-zero coefficient counts of the left and top transform blocks:

```text
left and top available -> nC = (nA + nB + 1) >> 1
only one available     -> nC = that neighbour's count
neither available      -> nC = 0
```

An unavailable block and an available block with `TotalCoeff == 0` are not the
same state. If the top count is six:

```text
left unavailable -> nC = 6
left available 0  -> nC = (0 + 6 + 1) >> 1 = 3
```

Slice boundaries also affect availability. A geometrically adjacent block in
another slice is unavailable for this derivation. I_PCM neighbours contribute
a count of 16, while skipped and coefficient-free blocks contribute an
available count of zero.

This is why CAVLC cannot be implemented as a stateless function over bytes.
The code should make the state transition explicit:

```text
derive nC
    -> decode complete residual block
    -> record TotalCoeff
```

If decoding fails, neither the bit position nor the stored neighbour count may
advance.

### CABAC

- Uses context-adaptive binary arithmetic coding.
- Usually compresses more efficiently.
- Maintains substantially more decoding state.
- Is significantly harder to implement and optimize correctly.

A practical first target is:

```text
H.264 Baseline
8-bit
YUV 4:2:0
progressive pictures
CAVLC
IDR/I slices first
```

This postpones CABAC, B slices, complex reference reordering, and most
inter-picture prediction.

### Macroblock transactions

A CAVLC I macroblock contains conditional syntax before its coefficients:

```text
mb_type
    -> intra prediction modes
    -> coded_block_pattern
    -> mb_qp_delta when required
    -> luma and chroma residual blocks
```

For an I_4x4 or I_8x8 macroblock, the coded block pattern selects which luma
8x8 regions and which chroma coefficient classes are present. For an
I_16x16 macroblock, prediction mode and coded block pattern are partly encoded
inside `mb_type`, and a separate luma DC block is always decoded.

Treat the complete macroblock as a transaction. A malformed coefficient near
the end must roll back:

- syntax bits read for the macroblock;
- earlier residual blocks from the same macroblock;
- `nC` entries written while decoding those blocks.

Copying the entire picture's neighbour grid per macroblock would be correct
but expensive. Snapshotting only the 16 luma and eight chroma entries touched
by the current 4:2:0 macroblock gives bounded rollback without a per-macroblock
heap allocation.

I_PCM is a separate path. For the current 8-bit 4:2:0 scope it byte-aligns,
reads 256 luma samples and 128 chroma samples, and records neighbour counts of
16.

## 9. Reconstructing the First Picture

For an initial IDR I picture, macroblock reconstruction is approximately:

```text
Read macroblock syntax
    |
    v
Decode intra-prediction modes
    |
    v
Build predictions from reconstructed neighbors
    |
    v
Decode transform coefficients with CAVLC
    |
    v
Inverse scan
    |
    v
Inverse quantization
    |
    v
Inverse integer transform
    |
    v
Add residuals to predictions
    |
    v
Write reconstructed Y, Cb, and Cr samples
```

After all relevant macroblocks have been reconstructed:

```text
Apply the deblocking filter
    |
    v
Store the picture as a reference when required
    |
    v
Output or inspect the YUV frame
```

Deblocking is part of normative reconstruction. A reference picture must
contain the filtered samples expected by subsequent pictures.

### 9.1 Scan, scale, and transform are different operations

Entropy decoding produces a one-dimensional coefficient list. Reconstruction
first maps that list into matrix positions. H.264 defines different 4x4 scans
for frame and field macroblocks; scaling lists always use the zig-zag mapping.

Inverse quantization then applies a position-dependent scale:

```text
LevelScale4x4 =
    scaling_list_weight * norm_adjust[QP % 6][position_class]
```

The shift and rounding depend on `QP / 6`. Their exact order matters,
especially for negative coefficients.

The final 4x4 inverse transform is a separable integer transform:

```text
four horizontal 1-D transforms
    -> four vertical 1-D transforms
    -> add 32 and arithmetic-shift right by 6
```

This is not a floating-point IDCT. Using floating point, changing a rounding
point, or replacing an arithmetic shift with truncating division can produce
different reference pixels.

Two DC paths are special:

- I_16x16 luma uses a 4x4 inverse Hadamard transform before DC scaling.
- 4:2:0 chroma uses a 2x2 inverse Hadamard transform before DC scaling.

Those transformed DC values become coefficient zero of the individual 4x4
blocks. Their AC lists occupy coefficient positions 1 through 15, so the
ordinary 4x4 scaling stage preserves the already-scaled DC value.

Scaling-list selection is also stateful. SPS lists use fallback rule set A.
PPS lists either inherit the effective SPS anchor list or use rule set A,
then later component lists can inherit the preceding list. Resolve these rules
once into six concrete 4x4 lists rather than branching for every coefficient.

## 10. Picture Order and Reference State

Decode order and display order can differ. A decoder must distinguish:

- when a picture is parsed;
- when it is fully reconstructed;
- whether it becomes a reference;
- when it should be displayed;
- when its storage can be released.

This becomes especially important when B pictures and picture-order-count
logic are introduced.

The decoder should treat reference-picture management as explicit state rather
than relying on input order.

## 11. Responsibilities of the BitReader

The current `BitReader` is responsible for:

- MSB-first sequential bit access;
- fixed and dynamic field widths;
- lookahead and skipping;
- byte alignment;
- unsigned Exp-Golomb `ue(v)`;
- signed Exp-Golomb `se(v)`;
- safe handling of truncation and integer overflow.

It should not be responsible for:

- Annex-B start-code detection;
- NAL-unit classification;
- EBSP-to-RBSP conversion;
- SPS/PPS semantic validation;
- macroblock state;
- pixel reconstruction.

Keeping those responsibilities separate makes both correctness testing and
performance profiling easier.

## 12. Recommended Implementation Order

Build the decoder in narrow, testable layers:

1. Split Annex-B data into NAL units.
2. Parse and validate NAL headers.
3. Convert EBSP payloads into RBSP data.
4. Validate RBSP trailing bits.
5. Parse and store SPS structures.
6. Parse and store PPS structures.
7. Parse slice headers.
8. Decode CAVLC syntax and coefficients.
9. Reconstruct intra-coded macroblocks.
10. Produce the first IDR YUV frame.
11. Implement deblocking.
12. Add P slices and motion compensation.
13. Add reference-picture management and output ordering.
14. Add more profiles and optional syntax only when required.

At each stage, compare against known-good streams and a reference decoder.

## 13. What Must Be Understood Even When AI Writes the Code

AI can generate parsing code, lookup tables, tests, and repetitive
implementations, but the human reviewer should still be able to answer:

- Is this input Annex-B, a NAL unit, EBSP, or RBSP?
- Which exact standard syntax element is being parsed?
- Why is this field present in this branch?
- Which SPS and PPS does this slice reference?
- Is a value fixed-width, unsigned Exp-Golomb, or signed Exp-Golomb?
- Can the value cause an overflow or an unreasonable allocation?
- Is the current picture allowed to use this reference picture?
- Is the prediction based on current-picture neighbors or an older picture?
- Have inverse operations followed the standard's exact integer rules?
- Does a failure leave the decoder in a valid state?

The core mental model to retain is:

```text
Separate the representation layers.
Parse syntax according to the standard.
Maintain explicit decoder state.
Reconstruct prediction plus residual.
Validate every boundary before trusting the stream.
Measure performance in the real decoding pipeline.
```

## 14. Current Implementation Checkpoint

At the checkpoint recorded by this document, the Rust implementation includes:

- an optimized MSB-first `BitReader` with atomic failure semantics;
- Annex-B splitting, NAL parsing, EBSP-to-RBSP conversion, and trailing bits;
- SPS, PPS, parameter-set resolution, and ordinary slice-header parsing;
- picture-boundary detection and picture-order-count types 0, 1, and 2;
- complete 4:2:0 CAVLC residual-block decoding;
- lookup-table acceleration plus Criterion benchmarks for CAVLC;
- slice-aware Y/Cb/Cr `nC` neighbour state;
- transactional CAVLC I_4x4, I_8x8, I_16x16, and I_PCM macroblocks;
- transactional frame-coded CAVLC P macroblock headers, including skip runs,
  reference indices, all P partition shapes, and motion-vector differences;
- slice-aware frame-coded P motion-vector prediction at 4x4 granularity,
  including A/B/C/D lookup, C-to-D fallback, directional 16x8/8x16 rules,
  median prediction, P_Skip inference, and transactional state updates;
- allocation-free progressive 8-bit 4:2:0 inter prediction blocks with
  normative quarter-sample luma six-tap filtering, eighth-sample chroma
  bilinear filtering, negative-vector decomposition, and edge extension;
- 4x4 inverse scan, inverse scaling, and inverse integer transform;
- 8x8 frame/field inverse scan, inverse scaling, and inverse integer transform;
- I_16x16 luma DC and 4:2:0 chroma DC inverse Hadamard paths;
- SPS/PPS 4x4 and 8x8 scaling-list fallback resolution;
- transactional macroblock QP derivation with independent Cb and Cr offsets;
- all 8-bit Intra4x4, Intra8x8, Intra16x16, and 4:2:0 chroma prediction modes;
- the normative Intra8x8 reference-sample filter;
- stateful Intra4x4/Intra8x8 mode derivation across block and slice boundaries;
- a mutable planar 4:2:0 reconstruction picture surface;
- 384-byte macroblock snapshots for rollback without copying a whole picture;
- prediction-plus-residual reconstruction for I_4x4, I_8x8, I_16x16, and I_PCM;
- normative 8-bit 4:2:0 intra deblocking, including slice-controlled offsets
  and cross-slice edge suppression;
- one-time I420-to-NV12 interleaving into shared immutable CPU plane storage;
- a CAVLC I-slice driver that joins header position, `nC`, QP, transforms,
  prediction, picture writes, and RBSP trailing-bit validation;
- a complete Annex-B SPS/PPS/IDR-to-NV12 integration test;
- a synchronous `VideoDecoder` implementation with packet ownership,
  backpressure, `FormatChanged`, timestamps, flush, and drain behavior.

The first pixel-producing path is therefore real, but deliberately narrow:

```text
Annex-B CAVLC progressive 8-bit 4:2:0 I/IDR
    -> I_4x4, I_8x8, I_16x16, or I_PCM macroblocks
    -> planar reconstruction and in-loop deblocking
    -> immutable CPU NV12 frame
    -> synchronous pull output
```

This is not yet a generally conforming H.264 decoder. The current implementation
still rejects or has not yet connected:

- CABAC slice data;
- P, B, SP, and SI macroblock reconstruction;
- transform-bypass reconstruction;
- field pictures and MBAFF;
- FMO slice groups;
- motion compensation and weighted prediction;
- decoded-picture-buffer marking, reference lifetime, and display reordering;
- length-prefixed input and out-of-band `avcC` configuration.

The next dependency chain is:

```text
inter residual decoding and reference-picture storage
    -> connect fractional-sample prediction to P-picture reconstruction
    -> DPB reference marking and POC-based output
    -> CABAC
```

The synchronous decoder currently keeps an unfinished picture across packets.
It emits the picture when a new picture boundary, AUD/end marker, or explicit
`drain()` makes completion observable. If output is already queued,
`send_packet` returns the original unconsumed packet through `NeedOutput`.
`flush()` clears timeline-dependent state while preserving active SPS/PPS, so
the next random-access picture does not require parameter sets to be repeated.

Useful verification commands are:

```bash
cargo test --workspace --release
cargo clippy -p decv-h264 --all-targets -- -D warnings
cargo bench -p decv-h264 --bench cavlc
```
