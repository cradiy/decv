//! Native VP9 decoder implementation for `decv`.
//!
//! This crate owns its parser, entropy decoder, reconstruction pipeline, and
//! reference-frame state. It does not call another codec implementation.

mod block;
mod bool_decoder;
mod compressed_header;
mod context;
mod decoder;
mod error;
mod header;
mod inter;
mod inverse_transform;
mod loop_filter;
mod reconstruct;
mod superframe;
mod tables;
mod tile;

pub use compressed_header::{
    CompressedHeader, ProbabilityUpdate, ProbabilityUpdateKind, ReferenceMode, TransformMode,
};
pub use decoder::Vp9Decoder;
pub use error::{Result, Vp9Error};
pub use header::{
    BitDepth, ChromaSubsampling, ColorSpace, FrameHeader, FrameType, HeaderParser,
    InterpolationFilter,
};
pub use inter::{InterSyntaxSummary, decode_inter_picture};
pub use reconstruct::IntraPicture;
pub use superframe::{Superframe, SuperframeFrames};
pub use tile::{IntraSyntaxSummary, TileLayout, decode_intra_picture};
