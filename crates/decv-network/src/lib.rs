//! Cached random-access inputs for remote media.
//!
//! Container and codec crates remain transport-independent. This crate adapts
//! exact byte-range fetchers, including HTTP Range servers, to
//! [`decv_core::MediaInput`].

#![forbid(unsafe_code)]

mod cache;
#[cfg(feature = "http")]
mod http;

pub use cache::{
    CachedRangeInput, RangeCacheConfig, RangeCacheStats, RangeFetcher, RangeInputStats,
};
#[cfg(feature = "http")]
pub use http::{HttpRangeInput, HttpRangeInputBuilder};
