use std::{error::Error, fmt};

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Vp9Error {
    Core(decv_core::MediaError),
    Truncated(&'static str),
    InvalidData(&'static str),
    UnsupportedFeature(&'static str),
    MissingReference(usize),
    TileDecode {
        tile: usize,
        blocks: usize,
        transform_blocks: usize,
        source: Box<Vp9Error>,
    },
    IntegerOverflow,
}

impl fmt::Display for Vp9Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Core(error) => error.fmt(formatter),
            Self::Truncated(field) => write!(formatter, "truncated VP9 {field}"),
            Self::InvalidData(message) => write!(formatter, "invalid VP9 data: {message}"),
            Self::UnsupportedFeature(message) => {
                write!(formatter, "unsupported VP9 feature: {message}")
            }
            Self::MissingReference(index) => {
                write!(
                    formatter,
                    "VP9 frame references an unavailable slot {index}"
                )
            }
            Self::TileDecode {
                tile,
                blocks,
                transform_blocks,
                source,
            } => {
                write!(
                    formatter,
                    "VP9 tile {tile} after {blocks} blocks and {transform_blocks} transform blocks: {source}"
                )
            }
            Self::IntegerOverflow => formatter.write_str("VP9 integer overflow"),
        }
    }
}

impl Error for Vp9Error {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Core(error) => Some(error),
            Self::TileDecode { source, .. } => Some(source),
            _ => None,
        }
    }
}

impl From<decv_core::MediaError> for Vp9Error {
    fn from(error: decv_core::MediaError) -> Self {
        Self::Core(error)
    }
}

pub type Result<T> = std::result::Result<T, Vp9Error>;
