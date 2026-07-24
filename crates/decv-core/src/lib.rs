//! Codec-independent media types and synchronous decoder contracts.
//!
//! This crate deliberately has no asynchronous-runtime dependency. A decoder
//! remains synchronous and can run on any caller-owned worker thread.

mod color;
mod decoder;
mod error;
mod format;
mod frame;
mod input;
mod packet;
mod time;

pub use color::{ColorInfo, ColorMatrix, ColorPrimaries, ColorRange, TransferFunction};
pub use decoder::{
    BitstreamFormat, DecodeInputStatus, DecodeOutput, VideoCodec, VideoDecoder, VideoDecoderConfig,
};
pub use error::{MediaError, Result};
pub use format::{PixelFormat, Rect, Size, VideoFormat};
pub use frame::{CpuFrame, CpuPlane, DecodedVideoFrame, FrameStorage};
pub use input::MediaInput;
pub use packet::EncodedVideoPacket;
pub use time::MediaTime;
