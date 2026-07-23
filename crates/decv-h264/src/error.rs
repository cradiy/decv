use std::fmt;

/// Errors produced while splitting, parsing, or decoding H.264 data.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum H264Error {
    Core(decv_core::MediaError),
    UnexpectedEof,
    InvalidStartCode,
    InvalidNalHeader,
    InvalidRbspEscape,
    InvalidTrailingBits,
    MissingSps(u32),
    MissingPps(u32),
    UnsupportedProfile(u8),
    UnsupportedFeature(&'static str),
    InvalidSyntax(&'static str),
    IntegerOverflow,
}

impl fmt::Display for H264Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Core(error) => error.fmt(formatter),
            Self::UnexpectedEof => formatter.write_str("unexpected end of H.264 data"),
            Self::InvalidStartCode => formatter.write_str("invalid Annex-B start code"),
            Self::InvalidNalHeader => formatter.write_str("invalid H.264 NAL header"),
            Self::InvalidRbspEscape => {
                formatter.write_str("invalid H.264 emulation-prevention sequence")
            }
            Self::InvalidTrailingBits => formatter.write_str("invalid RBSP trailing bits"),
            Self::MissingSps(id) => write!(formatter, "missing SPS {id}"),
            Self::MissingPps(id) => write!(formatter, "missing PPS {id}"),
            Self::UnsupportedProfile(profile) => {
                write!(formatter, "unsupported H.264 profile {profile}")
            }
            Self::UnsupportedFeature(feature) => {
                write!(formatter, "unsupported H.264 feature: {feature}")
            }
            Self::InvalidSyntax(syntax) => write!(formatter, "invalid H.264 syntax: {syntax}"),
            Self::IntegerOverflow => {
                formatter.write_str("integer overflow while decoding H.264 data")
            }
        }
    }
}

impl std::error::Error for H264Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Core(error) => Some(error),
            _ => None,
        }
    }
}

impl From<decv_core::MediaError> for H264Error {
    fn from(error: decv_core::MediaError) -> Self {
        Self::Core(error)
    }
}

pub type Result<T> = std::result::Result<T, H264Error>;
