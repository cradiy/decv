use std::fmt;

/// Errors shared by codec-independent media contracts.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum MediaError {
    InvalidDecoderConfig(&'static str),
    InvalidVideoFormat(&'static str),
    InvalidFrameStorage(&'static str),
    IntegerOverflow,
}

impl fmt::Display for MediaError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDecoderConfig(message) => {
                write!(formatter, "invalid decoder configuration: {message}")
            }
            Self::InvalidVideoFormat(message) => {
                write!(formatter, "invalid video format: {message}")
            }
            Self::InvalidFrameStorage(message) => {
                write!(formatter, "invalid frame storage: {message}")
            }
            Self::IntegerOverflow => {
                formatter.write_str("integer overflow while processing media data")
            }
        }
    }
}

impl std::error::Error for MediaError {}

pub type Result<T> = std::result::Result<T, MediaError>;
