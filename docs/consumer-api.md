# Consumer API Boundary

The `decv` crate is the stable-candidate facade for ordinary library
consumers. It intentionally exposes a smaller surface than the codec
implementation crates:

- the synchronous `VideoDecoder` push/pull contract;
- compressed packets, exact timestamps, video formats, and immutable frames;
- `H264Decoder` and its reconstruction-parallelism policy;
- random-access MP4 demuxing and sequential packet cursors.

Code that only consumes decoded video should depend on `decv`, not directly on
the H.264 syntax, CABAC, motion, transform, prediction, or reconstruction
types exported by `decv-h264`. Those lower-level exports exist for decoder
development and may change while the codec implementation evolves.

The workspace is still version `0.1.0`, so consumers that need reproducible
builds should pin an exact repository revision. The facade defines the API
intended to converge toward semantic-versioning stability; it is not yet a
promise that no `0.1` source changes will occur.

## Decoder Lifecycle

Create the decoder, select H.264 parallelism before decoding starts, then
configure the compressed packet framing:

```rust
use decv::{
    BitstreamFormat, H264Decoder, H264Parallelism, VideoCodec, VideoDecoder,
    VideoDecoderConfig,
};

let mut decoder = H264Decoder::new();
decoder.set_parallelism(H264Parallelism::Auto)?;
decoder.configure(VideoDecoderConfig {
    codec: VideoCodec::H264,
    bitstream_format: BitstreamFormat::ByteStream,
    codec_data: None,
})?;
# Ok::<(), decv::H264Error>(())
```

Annex-B input uses `BitstreamFormat::ByteStream`. MP4 normally supplies
length-prefixed AVC samples and an out-of-band `avcC` payload; obtain the
matching `VideoDecoderConfig` from `PacketCursor::decoder_config` instead of
constructing it manually.

Input ownership follows an explicit backpressure loop:

```rust
use decv::{
    DecodeInputStatus, DecodeOutput, EncodedVideoPacket, H264Decoder,
    VideoDecoder,
};

fn send_and_drain(
    decoder: &mut H264Decoder,
    mut packet: EncodedVideoPacket,
    mut output: impl FnMut(DecodeOutput),
) -> Result<(), decv::H264Error> {
    loop {
        match decoder.send_packet(packet)? {
            DecodeInputStatus::Accepted => break,
            DecodeInputStatus::NeedOutput(unconsumed) => {
                packet = unconsumed;
                loop {
                    match decoder.receive_frame()? {
                        DecodeOutput::NeedInput => break,
                        DecodeOutput::EndOfStream => {
                            output(DecodeOutput::EndOfStream);
                            return Ok(());
                        }
                        event => output(event),
                    }
                }
            }
        }
    }

    loop {
        match decoder.receive_frame()? {
            DecodeOutput::NeedInput => return Ok(()),
            DecodeOutput::EndOfStream => {
                output(DecodeOutput::EndOfStream);
                return Ok(());
            }
            event => output(event),
        }
    }
}
```

`FormatChanged` is emitted before the first frame using a new format. At end
of input, call `drain` and continue receiving until `EndOfStream`. After a
seek, reposition the packet cursor to a preceding keyframe, call `flush`, mark
the first packet as discontinuous, and discard decoded frames whose PTS is
before the requested target.

## Frame Ownership

`DecodedVideoFrame` is immutable and cheaply cloneable. CPU planes carry an
independent backing `Arc<[u8]>`, offset, stride, and row count. Consumers must
not assume that planes are adjacent or tightly packed.

`FrameStorage` and `PixelFormat` are non-exhaustive. Matches must retain a
fallback arm so that native or device-backed storage can be added without
changing the frame metadata contract.

The current H.264 backend produces NV12:

- plane zero contains one 8-bit luma sample per coded pixel;
- plane one contains interleaved 8-bit Cb/Cr pairs at half width and height;
- `visible_rect` selects coded pixels that belong to the displayed image;
- `display_size` carries the final sample-aspect-ratio-adjusted dimensions;
- `ColorInfo` identifies range, matrix, primaries, and transfer metadata.

The decoder is synchronous and `Send`. A latency-sensitive consumer should
own it on a dedicated worker and pass immutable frames through a bounded
queue. Presentation order is already restored by the decoder; scheduling
should use frame PTS and duration rather than decode completion time.
