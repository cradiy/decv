#![cfg(feature = "network")]

use std::num::NonZeroUsize;

use decv::{HttpRangeInput, RangeCacheConfig};

#[test]
fn facade_exposes_network_input_configuration_without_opening_a_connection() {
    let config = RangeCacheConfig::new(
        NonZeroUsize::new(64 * 1024).unwrap(),
        NonZeroUsize::new(8).unwrap(),
    );
    let builder =
        HttpRangeInput::builder("https://media.example.invalid/movie.mp4").cache_config(config);

    let debug = format!("{builder:?}");
    assert!(debug.contains("HttpRangeInputBuilder"));
    assert!(!debug.contains("media.example.invalid"));
}
