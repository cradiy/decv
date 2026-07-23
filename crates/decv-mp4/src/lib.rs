//! Synchronous, random-access ISO Base Media File Format demuxing.
//!
//! The parser starts with a bounded box layer. Higher-level movie, track, and
//! sample-table readers build on these offsets without loading the whole file
//! or depending on a particular storage/network implementation.

mod boxes;
mod demuxer;
mod edit;
mod error;
mod fourcc;
mod movie;
mod reader;
mod sample_table;

pub use boxes::{BoxHeader, BoxIter, Mp4File};
pub use demuxer::{Mp4Demuxer, PacketCursor};
pub use edit::Edit;
pub use error::{Mp4Error, Result};
pub use fourcc::FourCc;
pub use movie::{AvcSampleEntry, Movie, SampleDescription, Track};
pub use sample_table::Sample;
