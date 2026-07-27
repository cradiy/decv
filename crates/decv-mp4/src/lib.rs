//! Synchronous, random-access ISO Base Media File Format demuxing.
//!
//! The parser starts with a bounded box layer. Higher-level movie, track, and
//! sample-table readers build on these offsets without loading the whole file
//! or depending on a particular storage/network implementation.

mod audio;
mod boxes;
mod demuxer;
mod edit;
mod error;
mod fourcc;
mod fragment;
mod movie;
mod reader;
mod sample_table;

pub use audio::AacSampleEntry;
pub use boxes::{BoxHeader, BoxIter, Mp4File};
pub use demuxer::{AudioPacketCursor, Mp4Demuxer, PacketCursor};
pub use edit::Edit;
pub use error::{Mp4Error, Result};
pub use fourcc::FourCc;
pub use movie::{
    AvcSampleEntry, Movie, SampleDescription, Track, TrackKind, Vp9SampleEntry,
    VpCodecConfiguration,
};
pub use sample_table::Sample;
