# AAC Decoder Principles

This document describes the audio path formed by `decv-core`, `decv-mp4`, and
`decv-aac`. It covers the boundaries that callers and maintainers must preserve
without exposing codec internals.

## 1. The container and decoder have different jobs

`decv-mp4` owns container work:

- identify a `soun` track and its `mp4a` sample entry;
- extract AAC AudioSpecificConfig from `esds`;
- index each raw AAC access unit;
- apply a supported edit list to PTS and DTS;
- seek to a complete audio sample at or before a requested media time.

`decv-aac` owns codec work:

- validate AAC-LC mono/stereo configuration;
- decode one raw MP4 access unit at a time;
- convert planar codec output to owned interleaved `f32`;
- attach the input packet PTS and an exact PCM-derived duration;
- implement backpressure, format announcements, flush, and drain.

MP4 AAC packets do not contain ADTS headers. Adding one inside the demuxer
would mix transport framing into the codec-independent packet contract.

## 2. AudioSpecificConfig is the codec boundary

The `DecoderSpecificInfo` descriptor inside `esds` becomes
`AudioDecoderConfig.codec_data`. The demuxer parses enough bits to reject
unsupported object types and to cross-check:

```text
audioObjectType       AAC-LC
samplingFrequency     matches mp4a sample rate
channelConfiguration  matches mp4a channel count
```

The complete byte sequence is retained because the decoder may need fields
beyond those container-level checks.

## 3. PCM ownership and layout

The AAC backend produces planar floating-point channels. `decv-aac` copies one
decoded block into presentation-friendly interleaved order:

```text
L0, R0, L1, R1, ...  // stereo
M0, M1, M2, ...      // mono
```

The final `Arc<[f32]>` belongs to `DecodedAudioFrame`. The decoder can accept
and decode later packets while an earlier frame remains alive.

AAC-LC normally produces 1024 sample frames per raw access unit. Duration is
derived from the actual decoded sample-frame count:

```text
duration = sample_frames / sample_rate
```

This remains an exact `MediaTime` rational value. No floating-point clock is
accumulated across packets.

## 4. Push/pull state and backpressure

After a packet is accepted, the decoder may have two queued events:

```text
FormatChanged -> Frame
```

Until callers receive those events, `send_packet` returns
`NeedOutput(unconsumed_packet)`. Returning ownership is important: compressed
data does not need to be cloned merely because the consumer pulled output too
slowly.

`flush` clears queued PCM, resets the AAC overlap/filter state, starts frame
IDs from a new timeline, and causes the next frame format to be announced
again. `drain` stops further input and makes `receive_frame` return
`EndOfStream` after queued output is exhausted.

## 5. Seek timing

Audio seek does not use the video sync-sample table. Every indexed AAC access
unit is a complete packet boundary. The cursor chooses the packet with the
greatest adjusted PTS not after the target.

That packet can start before the requested time. A higher layer may discard or
trim the leading PCM while keeping the packet boundary and decoder state
valid. Video and audio cursors carry independent sample indices and share only
immutable random-access input.

## 6. Private pure-Rust backend

The current entropy, inverse-transform, and filter-bank implementation comes
from the pure-Rust `symphonia-codec-aac` 0.6.0 crate under MPL-2.0. It remains
private to `decv-aac`; public decv types never expose Symphonia buffers,
packets, errors, or channel types.

This boundary permits replacing or specializing the backend later without
changing container cursors, timestamps, PCM ownership, or the `AudioDecoder`
contract.

## 7. Verification

`scripts/verify_real_aac.sh` generates AAC-LC stereo, decodes the complete
untrimmed track through decv and FFmpeg, and compares every interleaved `f32`
sample. It requires equal sample counts, finite output, and a maximum absolute
error no greater than `1e-4`.
