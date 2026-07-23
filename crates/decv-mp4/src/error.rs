use std::{fmt, io};

#[derive(Debug)]
pub enum Mp4Error {
    Io(io::Error),
    InvalidData(&'static str),
    IntegerOverflow,
    UnknownInputLength,
}

impl fmt::Display for Mp4Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "MP4 input error: {error}"),
            Self::InvalidData(message) => write!(formatter, "invalid MP4 data: {message}"),
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
            Self::InvalidData(_) | Self::IntegerOverflow | Self::UnknownInputLength => None,
        }
    }
}

impl From<io::Error> for Mp4Error {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

pub type Result<T> = std::result::Result<T, Mp4Error>;
