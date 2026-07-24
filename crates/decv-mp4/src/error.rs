use std::{fmt, io};

#[derive(Debug)]
#[non_exhaustive]
pub enum Mp4Error {
    Io(io::Error),
    InvalidData(&'static str),
    UnsupportedFeature(&'static str),
    IndexOutOfRange { kind: &'static str, index: usize },
    IntegerOverflow,
    UnknownInputLength,
}

impl fmt::Display for Mp4Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "MP4 input error: {error}"),
            Self::InvalidData(message) => write!(formatter, "invalid MP4 data: {message}"),
            Self::UnsupportedFeature(message) => {
                write!(formatter, "unsupported MP4 feature: {message}")
            }
            Self::IndexOutOfRange { kind, index } => {
                write!(formatter, "MP4 {kind} index {index} is out of range")
            }
            Self::IntegerOverflow => formatter.write_str("MP4 integer overflow"),
            Self::UnknownInputLength => {
                formatter.write_str("MP4 top-level parsing requires a known input length")
            }
        }
    }
}

impl std::error::Error for Mp4Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::InvalidData(_)
            | Self::UnsupportedFeature(_)
            | Self::IndexOutOfRange { .. }
            | Self::IntegerOverflow
            | Self::UnknownInputLength => None,
        }
    }
}

impl From<io::Error> for Mp4Error {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

pub type Result<T> = std::result::Result<T, Mp4Error>;
