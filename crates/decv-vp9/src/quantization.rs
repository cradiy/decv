use crate::{BitDepth, FrameHeader, quant_tables_high, tables};

pub(crate) fn dequant(header: &FrameHeader, plane: usize, segment_id: usize) -> [i32; 2] {
    let quantization = header.quantization.expect("frame quantization");
    let mut qindex = i16::from(quantization.base_q_idx);
    if let Some(segmentation) = &header.segmentation {
        let alternate = segmentation.features[segment_id][0];
        if segmentation.enabled && alternate.enabled {
            qindex = if segmentation.absolute_values {
                alternate.value
            } else {
                qindex + alternate.value
            };
        }
    }
    let dc_delta = if plane == 0 {
        quantization.y_dc_delta
    } else {
        quantization.uv_dc_delta
    };
    let ac_delta = if plane == 0 {
        0
    } else {
        quantization.uv_ac_delta
    };
    let dc_index = quant_index(qindex, dc_delta);
    let ac_index = quant_index(qindex, ac_delta);
    let (dc, ac) = match header.bit_depth() {
        BitDepth::Eight => (tables::DC_QUANT_8[dc_index], tables::AC_QUANT_8[ac_index]),
        BitDepth::Ten => (
            quant_tables_high::DC_QUANT_10[dc_index],
            quant_tables_high::AC_QUANT_10[ac_index],
        ),
        BitDepth::Twelve => (
            quant_tables_high::DC_QUANT_12[dc_index],
            quant_tables_high::AC_QUANT_12[ac_index],
        ),
    };
    [i32::from(dc), i32::from(ac)]
}

#[inline]
fn quant_index(qindex: i16, delta: i8) -> usize {
    usize::try_from((qindex + i16::from(delta)).clamp(0, 255))
        .expect("clamped quantizer index is non-negative")
}

#[cfg(test)]
mod tests {
    use super::quant_index;
    use crate::quant_tables_high;

    #[test]
    fn high_bit_depth_quantizer_tables_match_normative_boundaries() {
        assert_eq!(quant_tables_high::DC_QUANT_10[0], 4);
        assert_eq!(quant_tables_high::DC_QUANT_10[255], 5347);
        assert_eq!(quant_tables_high::AC_QUANT_10[255], 7312);
        assert_eq!(quant_tables_high::DC_QUANT_12[255], 21_387);
        assert_eq!(quant_tables_high::AC_QUANT_12[255], 29_247);
        assert_eq!(quant_index(0, -15), 0);
        assert_eq!(quant_index(255, 15), 255);
    }
}
