use std::fmt;

/// One ISO BMFF four-character box or codec identifier.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FourCc([u8; 4]);

impl FourCc {
    #[inline]
    pub const fn new(bytes: [u8; 4]) -> Self {
        Self(bytes)
    }

    #[inline]
    pub const fn bytes(self) -> [u8; 4] {
        self.0
    }
}

impl From<[u8; 4]> for FourCc {
    fn from(bytes: [u8; 4]) -> Self {
        Self::new(bytes)
    }
}

impl fmt::Display for FourCc {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in self.0 {
            for escaped in std::ascii::escape_default(byte) {
                formatter.write_str(char::from(escaped).encode_utf8(&mut [0; 4]))?;
            }
        }
        Ok(())
    }
}

impl fmt::Debug for FourCc {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "FourCc(\"{self}\")")
    }
}

#[cfg(test)]
mod tests {
    use super::FourCc;

    #[test]
    fn formats_printable_and_binary_codes_without_loss() {
        assert_eq!(FourCc::new(*b"moov").to_string(), "moov");
        assert_eq!(
            FourCc::new([0, b'\\', b'"', 0xff]).to_string(),
            "\\x00\\\\\\\"\\xff"
        );
        assert_eq!(format!("{:?}", FourCc::new(*b"avc1")), "FourCc(\"avc1\")");
    }
}
