//! Opt-in counters for selecting whole-decoder optimization targets.

use std::{
    fmt,
    sync::atomic::{AtomicU64, Ordering},
};

const CALLS: usize = 0;
const LUMA_PIXELS: usize = 1;
const CHROMA_PIXELS: usize = 2;
const WIDTH_4: usize = 3;
const WIDTH_8: usize = 4;
const WIDTH_16: usize = 5;
const LUMA_INTEGER_CALLS: usize = 6;
const LUMA_HORIZONTAL_CALLS: usize = 7;
const LUMA_VERTICAL_CALLS: usize = 8;
const LUMA_TWO_DIMENSIONAL_CALLS: usize = 9;
const LUMA_CLIPPED_CALLS: usize = 10;
const LUMA_INTEGER_PIXELS: usize = 11;
const LUMA_HORIZONTAL_PIXELS: usize = 12;
const LUMA_VERTICAL_PIXELS: usize = 13;
const LUMA_TWO_DIMENSIONAL_PIXELS: usize = 14;
const LUMA_CLIPPED_PIXELS: usize = 15;
const CHROMA_INTEGER_CALLS: usize = 16;
const CHROMA_BILINEAR_CALLS: usize = 17;
const CHROMA_CLIPPED_CALLS: usize = 18;
const CHROMA_INTEGER_PIXELS: usize = 19;
const CHROMA_BILINEAR_PIXELS: usize = 20;
const CHROMA_CLIPPED_PIXELS: usize = 21;
const SPATIAL_DIRECT_MACROBLOCKS: usize = 22;
const SPATIAL_DIRECT_UNIFORM_PREDICTION: usize = 23;
const SPATIAL_DIRECT_COL_ZERO_CLEAR: usize = 24;
const SPATIAL_DIRECT_COL_ZERO_SET: usize = 25;
const SPATIAL_DIRECT_COL_ZERO_MIXED: usize = 26;
const SPATIAL_DIRECT_BOTH_ZERO: usize = 27;
const SPATIAL_DIRECT_LIST0_ZERO: usize = 28;
const SPATIAL_DIRECT_LIST1_ZERO: usize = 29;
const SPATIAL_DIRECT_BOTH_REFERENCE_ZERO: usize = 30;
const SPATIAL_DIRECT_ZERO_NEIGHBOUR_TRIPLET: usize = 31;
const CABAC_DECISIONS: usize = 32;
const CABAC_MPS_NO_RENORMALIZATION: usize = 33;
const CABAC_MPS_RENORMALIZATION: usize = 34;
const CABAC_LPS: usize = 35;
const COUNTER_COUNT: usize = 36;

static COUNTERS: [AtomicU64; COUNTER_COUNT] = [const { AtomicU64::new(0) }; COUNTER_COUNT];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[doc(hidden)]
pub struct InterPredictionProfile {
    counters: [u64; COUNTER_COUNT],
}

impl Default for InterPredictionProfile {
    fn default() -> Self {
        Self {
            counters: [0; COUNTER_COUNT],
        }
    }
}

impl InterPredictionProfile {
    fn counter(self, index: usize) -> u64 {
        self.counters[index]
    }
}

impl fmt::Display for InterPredictionProfile {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let calls = self.counter(CALLS);
        let luma_pixels = self.counter(LUMA_PIXELS);
        let chroma_pixels = self.counter(CHROMA_PIXELS);
        writeln!(
            formatter,
            "inter prediction: {calls} calls, {luma_pixels} luma pixels, \
             {chroma_pixels} chroma pixels"
        )?;
        writeln!(
            formatter,
            "  widths: 4={} 8={} 16={}",
            self.counter(WIDTH_4),
            self.counter(WIDTH_8),
            self.counter(WIDTH_16)
        )?;
        writeln!(
            formatter,
            "  luma calls: integer={} horizontal={} vertical={} 2d={} clipped={}",
            self.counter(LUMA_INTEGER_CALLS),
            self.counter(LUMA_HORIZONTAL_CALLS),
            self.counter(LUMA_VERTICAL_CALLS),
            self.counter(LUMA_TWO_DIMENSIONAL_CALLS),
            self.counter(LUMA_CLIPPED_CALLS)
        )?;
        writeln!(
            formatter,
            "  luma pixels: integer={} ({:.1}%) horizontal={} ({:.1}%) \
             vertical={} ({:.1}%) 2d={} ({:.1}%) clipped={} ({:.1}%)",
            self.counter(LUMA_INTEGER_PIXELS),
            percent(self.counter(LUMA_INTEGER_PIXELS), luma_pixels),
            self.counter(LUMA_HORIZONTAL_PIXELS),
            percent(self.counter(LUMA_HORIZONTAL_PIXELS), luma_pixels),
            self.counter(LUMA_VERTICAL_PIXELS),
            percent(self.counter(LUMA_VERTICAL_PIXELS), luma_pixels),
            self.counter(LUMA_TWO_DIMENSIONAL_PIXELS),
            percent(self.counter(LUMA_TWO_DIMENSIONAL_PIXELS), luma_pixels),
            self.counter(LUMA_CLIPPED_PIXELS),
            percent(self.counter(LUMA_CLIPPED_PIXELS), luma_pixels)
        )?;
        writeln!(
            formatter,
            "  chroma calls: integer={} bilinear={} clipped={}",
            self.counter(CHROMA_INTEGER_CALLS),
            self.counter(CHROMA_BILINEAR_CALLS),
            self.counter(CHROMA_CLIPPED_CALLS)
        )?;
        writeln!(
            formatter,
            "  chroma pixels: integer={} ({:.1}%) bilinear={} ({:.1}%) \
             clipped={} ({:.1}%)",
            self.counter(CHROMA_INTEGER_PIXELS),
            percent(self.counter(CHROMA_INTEGER_PIXELS), chroma_pixels),
            self.counter(CHROMA_BILINEAR_PIXELS),
            percent(self.counter(CHROMA_BILINEAR_PIXELS), chroma_pixels),
            self.counter(CHROMA_CLIPPED_PIXELS),
            percent(self.counter(CHROMA_CLIPPED_PIXELS), chroma_pixels)
        )?;
        writeln!(
            formatter,
            "spatial Direct: macroblocks={} prediction-uniform={} \
             col-zero-clear={} col-zero-set={} col-zero-mixed={}",
            self.counter(SPATIAL_DIRECT_MACROBLOCKS),
            self.counter(SPATIAL_DIRECT_UNIFORM_PREDICTION),
            self.counter(SPATIAL_DIRECT_COL_ZERO_CLEAR),
            self.counter(SPATIAL_DIRECT_COL_ZERO_SET),
            self.counter(SPATIAL_DIRECT_COL_ZERO_MIXED)
        )?;
        writeln!(
            formatter,
            "  uniform predictions: both-zero={} list0-zero={} list1-zero={} \
             both-reference-zero={} zero-neighbour-triplet={}",
            self.counter(SPATIAL_DIRECT_BOTH_ZERO),
            self.counter(SPATIAL_DIRECT_LIST0_ZERO),
            self.counter(SPATIAL_DIRECT_LIST1_ZERO),
            self.counter(SPATIAL_DIRECT_BOTH_REFERENCE_ZERO),
            self.counter(SPATIAL_DIRECT_ZERO_NEIGHBOUR_TRIPLET)
        )?;
        let cabac_decisions = self.counter(CABAC_DECISIONS);
        write!(
            formatter,
            "CABAC decisions: total={} MPS-no-renorm={} ({:.1}%) \
             MPS-renorm={} ({:.1}%) LPS={} ({:.1}%)",
            cabac_decisions,
            self.counter(CABAC_MPS_NO_RENORMALIZATION),
            percent(self.counter(CABAC_MPS_NO_RENORMALIZATION), cabac_decisions),
            self.counter(CABAC_MPS_RENORMALIZATION),
            percent(self.counter(CABAC_MPS_RENORMALIZATION), cabac_decisions),
            self.counter(CABAC_LPS),
            percent(self.counter(CABAC_LPS), cabac_decisions),
        )
    }
}

fn percent(value: u64, total: u64) -> f64 {
    if total == 0 {
        0.0
    } else {
        value as f64 * 100.0 / total as f64
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn record_inter_prediction(
    width: u8,
    height: u8,
    luma_fraction_x: u8,
    luma_fraction_y: u8,
    luma_clipped: bool,
    chroma_fraction_x: u8,
    chroma_fraction_y: u8,
    chroma_clipped: bool,
) {
    let luma_pixels = u64::from(width) * u64::from(height);
    let chroma_pixels = luma_pixels / 2;
    increment(CALLS, 1);
    increment(LUMA_PIXELS, luma_pixels);
    increment(CHROMA_PIXELS, chroma_pixels);
    if let Some(width_counter) = match width {
        4 => Some(WIDTH_4),
        8 => Some(WIDTH_8),
        16 => Some(WIDTH_16),
        _ => None,
    } {
        increment(width_counter, 1);
    }

    let (luma_calls, luma_work) = match (luma_fraction_x, luma_fraction_y) {
        (0, 0) => (LUMA_INTEGER_CALLS, LUMA_INTEGER_PIXELS),
        (_, 0) => (LUMA_HORIZONTAL_CALLS, LUMA_HORIZONTAL_PIXELS),
        (0, _) => (LUMA_VERTICAL_CALLS, LUMA_VERTICAL_PIXELS),
        (_, _) => (LUMA_TWO_DIMENSIONAL_CALLS, LUMA_TWO_DIMENSIONAL_PIXELS),
    };
    increment(luma_calls, 1);
    increment(luma_work, luma_pixels);
    if luma_clipped {
        increment(LUMA_CLIPPED_CALLS, 1);
        increment(LUMA_CLIPPED_PIXELS, luma_pixels);
    }

    let (chroma_calls, chroma_work) = if chroma_fraction_x == 0 && chroma_fraction_y == 0 {
        (CHROMA_INTEGER_CALLS, CHROMA_INTEGER_PIXELS)
    } else {
        (CHROMA_BILINEAR_CALLS, CHROMA_BILINEAR_PIXELS)
    };
    increment(chroma_calls, 1);
    increment(chroma_work, chroma_pixels);
    if chroma_clipped {
        increment(CHROMA_CLIPPED_CALLS, 1);
        increment(CHROMA_CLIPPED_PIXELS, chroma_pixels);
    }
}

pub(crate) fn record_spatial_direct_uniform_prediction(
    list0_zero: bool,
    list1_zero: bool,
    both_reference_zero: bool,
) {
    increment(SPATIAL_DIRECT_MACROBLOCKS, 1);
    increment(SPATIAL_DIRECT_UNIFORM_PREDICTION, 1);
    increment(SPATIAL_DIRECT_LIST0_ZERO, u64::from(list0_zero));
    increment(SPATIAL_DIRECT_LIST1_ZERO, u64::from(list1_zero));
    increment(
        SPATIAL_DIRECT_BOTH_ZERO,
        u64::from(list0_zero && list1_zero),
    );
    increment(
        SPATIAL_DIRECT_BOTH_REFERENCE_ZERO,
        u64::from(both_reference_zero),
    );
}

pub(crate) fn record_spatial_direct_col_zero_grid(cell_count: usize, mask: u16) {
    increment(SPATIAL_DIRECT_MACROBLOCKS, 1);
    let all_set = u16::MAX >> (u16::BITS as usize - cell_count);
    increment(
        match mask {
            0 => SPATIAL_DIRECT_COL_ZERO_CLEAR,
            mask if mask == all_set => SPATIAL_DIRECT_COL_ZERO_SET,
            _ => SPATIAL_DIRECT_COL_ZERO_MIXED,
        },
        1,
    );
}

pub(crate) fn record_spatial_direct_zero_neighbour_triplet() {
    increment(SPATIAL_DIRECT_ZERO_NEIGHBOUR_TRIPLET, 1);
}

pub(crate) fn record_cabac_decision(lps: bool, renormalized: bool) {
    increment(CABAC_DECISIONS, 1);
    increment(
        match (lps, renormalized) {
            (false, false) => CABAC_MPS_NO_RENORMALIZATION,
            (false, true) => CABAC_MPS_RENORMALIZATION,
            (true, _) => CABAC_LPS,
        },
        1,
    );
}

fn increment(index: usize, value: u64) {
    COUNTERS[index].fetch_add(value, Ordering::Relaxed);
}

#[doc(hidden)]
pub fn reset_inter_prediction_profile() {
    for counter in &COUNTERS {
        counter.store(0, Ordering::Relaxed);
    }
}

#[doc(hidden)]
pub fn inter_prediction_profile() -> InterPredictionProfile {
    InterPredictionProfile {
        counters: std::array::from_fn(|index| COUNTERS[index].load(Ordering::Relaxed)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_profile_formats_without_dividing_by_zero() {
        let formatted = InterPredictionProfile::default().to_string();
        assert!(formatted.contains("0 calls"));
        assert!(formatted.contains("(0.0%)"));
        assert!(formatted.contains("spatial Direct: macroblocks=0"));
        assert!(formatted.contains("zero-neighbour-triplet=0"));
        assert!(formatted.contains("CABAC decisions: total=0"));
    }
}
