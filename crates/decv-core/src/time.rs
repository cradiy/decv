use std::cmp::Ordering;
use std::num::NonZeroU32;

/// A signed media timestamp represented in an explicit time base.
///
/// The signed value is required because decode timestamps and edit-list
/// results may be negative. Two timestamps with different time scales compare
/// by their exact rational values rather than by their raw integer fields.
#[derive(Debug, Clone, Copy)]
pub struct MediaTime {
    pub value: i64,
    pub timescale: NonZeroU32,
}

impl MediaTime {
    #[inline]
    pub const fn new(value: i64, timescale: NonZeroU32) -> Self {
        Self { value, timescale }
    }

    #[inline]
    pub const fn from_parts(value: i64, timescale: u32) -> Option<Self> {
        match NonZeroU32::new(timescale) {
            Some(timescale) => Some(Self::new(value, timescale)),
            None => None,
        }
    }

    #[inline]
    pub fn as_seconds_f64(self) -> f64 {
        self.value as f64 / self.timescale.get() as f64
    }

    #[inline]
    fn compare(self, other: Self) -> Ordering {
        let left = self.value as i128 * other.timescale.get() as i128;
        let right = other.value as i128 * self.timescale.get() as i128;
        left.cmp(&right)
    }
}

impl PartialEq for MediaTime {
    fn eq(&self, other: &Self) -> bool {
        self.compare(*other) == Ordering::Equal
    }
}

impl Eq for MediaTime {}

impl PartialOrd for MediaTime {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MediaTime {
    fn cmp(&self, other: &Self) -> Ordering {
        self.compare(*other)
    }
}

#[cfg(test)]
mod tests {
    use super::MediaTime;

    #[test]
    fn compares_exact_values_across_time_scales() {
        let half = MediaTime::from_parts(1, 2).unwrap();
        let equivalent = MediaTime::from_parts(500, 1_000).unwrap();
        let later = MediaTime::from_parts(501, 1_000).unwrap();
        let negative = MediaTime::from_parts(-1, 1).unwrap();

        assert_eq!(half, equivalent);
        assert!(half < later);
        assert!(negative < half);
    }

    #[test]
    fn rejects_a_zero_time_scale() {
        assert_eq!(MediaTime::from_parts(1, 0), None);
    }
}
