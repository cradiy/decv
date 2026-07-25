# Audio Decoder Principles

This document describes the codec-independent audio path formed by
`decv-core`, `decv-mp4`, and `decv-audio`.

## 1. Container and codec responsibilities

A container layer owns track discovery, codec configuration, packet boundaries,
timestamps, edit-list mapping, and seeking. `decv-audio` owns compressed-audio
decoding, sample conversion, frame ownership, backpressure, flush, and drain.

The built-in MP4 demuxer currently constructs decoder configuration and packets
for AAC-LC. The decoder itself is not MP4- or AAC-specific; another container
layer can use the same contract for every codec listed below.

## 2. Supported codecs

`SoftwareAudioDecoder` enables every stable audio decoder in Symphonia 0.6:

| decv codec | Required packet/config information |
| --- | --- |
| AAC-LC | Raw access units and AudioSpecificConfig |
| Microsoft ADPCM | Complete blocks, channels, sample rate, frames per block, maximum frames per packet |
| IMA ADPCM (WAV/QuickTime) | Complete blocks, channels, sample rate, frames per block, maximum frames per packet |
| ALAC | Complete packets and ALAC magic cookie |
| FLAC | Complete frames and STREAMINFO |
| MP1/MP2/MP3 | One complete MPEG audio frame per packet |
| Integer/float PCM, A-law, mu-law | Interleaved samples, channel count, sample rate, and bit widths when applicable |
| Vorbis | Audio packets plus identification and setup header data |

Codec registration does not imply container support. A compressed byte stream
must already be split at valid codec packet boundaries, and initialization data
must use the representation expected by that codec.

## 3. Public configuration boundary

`AudioDecoderConfig` carries only codec-independent values:

- `AudioCodec`;
- sample rate and `ChannelLayout`;
- optional codec-private initialization bytes;
- decoded/coded sample bit widths;
- packet and block frame counts for block codecs.

No Symphonia codec ID, channel map, packet, buffer, or error type appears in
`decv-core` or the `decv` facade. `decv-audio` translates the public
configuration into private Symphonia parameters and creates the decoder through
the enabled codec registry.

AAC, ALAC, FLAC, and Vorbis require codec initialization bytes. ADPCM requires
both frame-count fields. PCM and MPEG audio do not require codec-private data.

## 4. PCM normalization and ownership

Symphonia decoders can produce unsigned integer, signed integer, 24-bit integer,
`f32`, or `f64` planar buffers. `decv-audio` converts every backend format to
owned interleaved `f32`:

```text
L0, R0, L1, R1, ...  // stereo
M0, M1, M2, ...      // mono
```

The resulting `Arc<[f32]>` belongs to `DecodedAudioFrame`; it never borrows a
backend buffer. Integer samples use Symphonia's normalized sample conversion.
The decoder does not resample or clip PCM.

The output format is derived from each decoded buffer. A changed sample rate or
channel count produces `FormatChanged` before the first frame with the new
format. Layouts beyond mono and stereo are currently represented as
`ChannelLayout::Discrete(channel_count)`.

## 5. Time and packet duration

Input PTS remains in decv's exact `MediaTime` representation and is copied to
the decoded frame. Output duration is derived from the actual number of decoded
sample frames:

```text
duration = sample_frames / sample_rate
```

No floating-point clock is accumulated across packets. Input packet duration is
also converted to sample-frame units for block decoders such as ADPCM.

Gapless trim metadata is not yet part of `EncodedAudioPacket`, so the private
backend is configured without automatic gapless trimming.

## 6. Push/pull state

One accepted packet may enqueue:

```text
FormatChanged -> Frame
```

Until queued events are received, `send_packet` returns the original packet in
`NeedOutput`. A codec may accept a packet without producing PCM, for example
while establishing overlap state.

`flush` clears queued output, resets codec state, restarts frame IDs, and causes
the next actual format to be announced. `drain` stops new input and reports
`EndOfStream` after queued frames have been consumed.

## 7. Current MP4 connection

For AAC-LC, `decv-mp4`:

- identifies `soun`/`mp4a`;
- extracts AudioSpecificConfig from `esds`;
- validates sample rate and channel configuration;
- indexes raw AAC access units without adding ADTS;
- applies supported edit lists to timestamps;
- seeks to complete audio samples independently from video.

ALAC, FLAC, MPEG audio, PCM, ADPCM, and Vorbis decoder availability must not be
reported as MP4 sample-entry support until `decv-mp4` parses their corresponding
sample entries and initialization boxes.

## 8. Verification

Unit tests verify registry coverage for every stable Symphonia audio codec,
integer PCM-to-`f32` conversion, configuration validation, backpressure, flush,
drain, and AAC access-unit decoding.

`scripts/verify_real_aac.sh` additionally generates an AAC-LC stereo MP4,
decodes it through decv and FFmpeg, and compares every interleaved `f32` sample
with a maximum absolute error of `1e-4`.
