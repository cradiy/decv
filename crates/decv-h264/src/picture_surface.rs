//! Mutable pre-deblocking picture storage and immutable CPU-frame packaging.

use std::sync::Arc;

use decv_core::{
    CpuFrame, CpuPlane, DecodedVideoFrame, FrameStorage, MediaTime, PixelFormat, Size, VideoFormat,
};

use crate::{
    Block4x4, H264Error, Intra4x4References, Intra8x8References, Intra16x16References,
    IntraChroma420References, MAX_DECODED_PICTURE_MACROBLOCKS, PcmMacroblock, Prediction4x4,
    Prediction8x8, Prediction16x16, Result,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChromaPlane {
    Cb,
    Cr,
}

/// Availability decisions made by the macroblock/slice layer.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct IntraReferenceAvailability {
    pub top: bool,
    pub left: bool,
    pub top_left: bool,
    pub top_right: bool,
}

/// One mutable 8-bit 4:2:0 picture before deblocking.
///
/// Chroma remains planar during reconstruction because Cb and Cr have
/// independent prediction and residual paths. [`Self::into_nv12_frame`]
/// interleaves them once when the picture becomes immutable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Yuv420Picture {
    coded_size: Size,
    width: usize,
    height: usize,
    luma: Arc<[u8]>,
    cb: Vec<u8>,
    cr: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MacroblockPixels {
    luma: Prediction16x16,
    cb: Prediction8x8,
    cr: Prediction8x8,
}

impl MacroblockPixels {
    pub(crate) const fn new(luma: Prediction16x16, cb: Prediction8x8, cr: Prediction8x8) -> Self {
        Self { luma, cb, cr }
    }

    pub(crate) const fn empty() -> Self {
        Self::new([[0; 16]; 16], [[0; 8]; 8], [[0; 8]; 8])
    }

    pub(crate) fn planes_mut(
        &mut self,
    ) -> (&mut Prediction16x16, &mut Prediction8x8, &mut Prediction8x8) {
        (&mut self.luma, &mut self.cb, &mut self.cr)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StagedMacroblockPixels {
    address: usize,
    pixels: MacroblockPixels,
}

impl StagedMacroblockPixels {
    pub(crate) const fn new(address: usize, pixels: MacroblockPixels) -> Self {
        Self { address, pixels }
    }

    pub(crate) const fn empty(address: usize) -> Self {
        Self::new(address, MacroblockPixels::empty())
    }

    pub(crate) fn pixels_mut(&mut self) -> &mut MacroblockPixels {
        &mut self.pixels
    }
}

impl Yuv420Picture {
    pub fn new(coded_size: Size) -> Result<Self> {
        if coded_size.width == 0 || coded_size.height == 0 {
            return Err(H264Error::InvalidSyntax(
                "picture dimensions must be non-zero",
            ));
        }
        if !coded_size.width.is_multiple_of(16) || !coded_size.height.is_multiple_of(16) {
            return Err(H264Error::InvalidSyntax(
                "4:2:0 H.264 coded dimensions must be macroblock aligned",
            ));
        }
        let macroblock_count = u64::from(coded_size.width / 16)
            .checked_mul(u64::from(coded_size.height / 16))
            .ok_or(H264Error::IntegerOverflow)?;
        if macroblock_count > MAX_DECODED_PICTURE_MACROBLOCKS {
            return Err(H264Error::InvalidSyntax(
                "coded picture exceeds the decoder macroblock limit",
            ));
        }

        let width = usize::try_from(coded_size.width).map_err(|_| H264Error::IntegerOverflow)?;
        let height = usize::try_from(coded_size.height).map_err(|_| H264Error::IntegerOverflow)?;
        let luma_len = width
            .checked_mul(height)
            .ok_or(H264Error::IntegerOverflow)?;
        let chroma_len = (width / 2)
            .checked_mul(height / 2)
            .ok_or(H264Error::IntegerOverflow)?;
        Ok(Self {
            coded_size,
            width,
            height,
            luma: vec![0; luma_len].into(),
            cb: vec![0; chroma_len],
            cr: vec![0; chroma_len],
        })
    }

    #[inline]
    pub const fn coded_size(&self) -> Size {
        self.coded_size
    }

    #[inline]
    pub(crate) const fn dimensions(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    #[inline]
    pub(crate) fn planes_mut(&mut self) -> (&mut [u8], &mut [u8], &mut [u8]) {
        (Arc::make_mut(&mut self.luma), &mut self.cb, &mut self.cr)
    }

    #[inline]
    pub(crate) fn planes(&self) -> (&[u8], &[u8], &[u8]) {
        (&self.luma, &self.cb, &self.cr)
    }

    pub fn intra4x4_references(
        &self,
        x: usize,
        y: usize,
        availability: IntraReferenceAvailability,
    ) -> Result<Intra4x4References> {
        self.validate_luma_block(x, y, 4)?;
        let top = if availability.top {
            let mut samples = [0; 8];
            self.copy_top(&self.luma, self.width, x, y, &mut samples[..4])?;
            if availability.top_right {
                self.copy_top(&self.luma, self.width, x + 4, y, &mut samples[4..])?;
            } else {
                let substitute = samples[3];
                samples[4..].fill(substitute);
            }
            Some(samples)
        } else {
            None
        };
        let left = availability
            .left
            .then(|| self.copy_left::<4>(&self.luma, self.width, x, y))
            .transpose()?;
        let top_left = availability
            .top_left
            .then(|| self.sample_top_left(&self.luma, self.width, x, y))
            .transpose()?;
        Ok(Intra4x4References {
            top,
            left,
            top_left,
        })
    }

    pub fn intra8x8_references(
        &self,
        x: usize,
        y: usize,
        availability: IntraReferenceAvailability,
    ) -> Result<Intra8x8References> {
        self.validate_luma_block(x, y, 8)?;
        let top = if availability.top {
            let mut samples = [0; 16];
            self.copy_top(&self.luma, self.width, x, y, &mut samples[..8])?;
            if availability.top_right {
                self.copy_top(&self.luma, self.width, x + 8, y, &mut samples[8..])?;
            } else {
                let substitute = samples[7];
                samples[8..].fill(substitute);
            }
            Some(samples)
        } else {
            None
        };
        let left = availability
            .left
            .then(|| self.copy_left::<8>(&self.luma, self.width, x, y))
            .transpose()?;
        let top_left = availability
            .top_left
            .then(|| self.sample_top_left(&self.luma, self.width, x, y))
            .transpose()?;
        Ok(Intra8x8References {
            top,
            left,
            top_left,
        })
    }

    pub fn intra16x16_references(
        &self,
        macroblock_x: usize,
        macroblock_y: usize,
        availability: IntraReferenceAvailability,
    ) -> Result<Intra16x16References> {
        let x = macroblock_x
            .checked_mul(16)
            .ok_or(H264Error::IntegerOverflow)?;
        let y = macroblock_y
            .checked_mul(16)
            .ok_or(H264Error::IntegerOverflow)?;
        self.validate_luma_block(x, y, 16)?;
        let top = availability
            .top
            .then(|| {
                let mut samples = [0; 16];
                self.copy_top(&self.luma, self.width, x, y, &mut samples)?;
                Ok::<_, H264Error>(samples)
            })
            .transpose()?;
        let left = availability
            .left
            .then(|| self.copy_left::<16>(&self.luma, self.width, x, y))
            .transpose()?;
        let top_left = availability
            .top_left
            .then(|| self.sample_top_left(&self.luma, self.width, x, y))
            .transpose()?;
        Ok(Intra16x16References {
            top,
            left,
            top_left,
        })
    }

    pub fn intra_chroma_references(
        &self,
        plane: ChromaPlane,
        macroblock_x: usize,
        macroblock_y: usize,
        availability: IntraReferenceAvailability,
    ) -> Result<IntraChroma420References> {
        let x = macroblock_x
            .checked_mul(8)
            .ok_or(H264Error::IntegerOverflow)?;
        let y = macroblock_y
            .checked_mul(8)
            .ok_or(H264Error::IntegerOverflow)?;
        let stride = self.width / 2;
        let height = self.height / 2;
        validate_block_bounds(stride, height, x, y, 8)?;
        let samples = self.chroma(plane);
        let top = availability
            .top
            .then(|| {
                let mut top = [0; 8];
                self.copy_top(samples, stride, x, y, &mut top)?;
                Ok::<_, H264Error>(top)
            })
            .transpose()?;
        let left = availability
            .left
            .then(|| self.copy_left::<8>(samples, stride, x, y))
            .transpose()?;
        let top_left = availability
            .top_left
            .then(|| self.sample_top_left(samples, stride, x, y))
            .transpose()?;
        Ok(IntraChroma420References {
            top,
            left,
            top_left,
        })
    }

    pub fn write_luma_4x4(
        &mut self,
        x: usize,
        y: usize,
        prediction: &Prediction4x4,
        residual: &Block4x4,
    ) -> Result<()> {
        self.validate_luma_block(x, y, 4)?;
        add_block(
            Arc::make_mut(&mut self.luma),
            self.width,
            x,
            y,
            prediction,
            residual,
        );
        Ok(())
    }

    pub fn write_luma_8x8(
        &mut self,
        x: usize,
        y: usize,
        prediction: &Prediction8x8,
        residual: &[[i32; 8]; 8],
    ) -> Result<()> {
        self.validate_luma_block(x, y, 8)?;
        add_block(
            Arc::make_mut(&mut self.luma),
            self.width,
            x,
            y,
            prediction,
            residual,
        );
        Ok(())
    }

    pub fn write_luma_16x16(
        &mut self,
        macroblock_x: usize,
        macroblock_y: usize,
        prediction: &Prediction16x16,
        residual: &[[i32; 16]; 16],
    ) -> Result<()> {
        let x = macroblock_x
            .checked_mul(16)
            .ok_or(H264Error::IntegerOverflow)?;
        let y = macroblock_y
            .checked_mul(16)
            .ok_or(H264Error::IntegerOverflow)?;
        self.validate_luma_block(x, y, 16)?;
        add_block(
            Arc::make_mut(&mut self.luma),
            self.width,
            x,
            y,
            prediction,
            residual,
        );
        Ok(())
    }

    pub fn write_chroma_8x8(
        &mut self,
        plane: ChromaPlane,
        macroblock_x: usize,
        macroblock_y: usize,
        prediction: &Prediction8x8,
        residual: &[[i32; 8]; 8],
    ) -> Result<()> {
        let x = macroblock_x
            .checked_mul(8)
            .ok_or(H264Error::IntegerOverflow)?;
        let y = macroblock_y
            .checked_mul(8)
            .ok_or(H264Error::IntegerOverflow)?;
        let stride = self.width / 2;
        validate_block_bounds(stride, self.height / 2, x, y, 8)?;
        add_block(self.chroma_mut(plane), stride, x, y, prediction, residual);
        Ok(())
    }

    pub fn write_pcm_macroblock(
        &mut self,
        macroblock_x: usize,
        macroblock_y: usize,
        pcm: &PcmMacroblock,
    ) -> Result<()> {
        let luma_x = macroblock_x
            .checked_mul(16)
            .ok_or(H264Error::IntegerOverflow)?;
        let luma_y = macroblock_y
            .checked_mul(16)
            .ok_or(H264Error::IntegerOverflow)?;
        self.validate_luma_block(luma_x, luma_y, 16)?;
        copy_block(
            Arc::make_mut(&mut self.luma),
            self.width,
            luma_x,
            luma_y,
            pcm.luma.as_slice(),
            16,
        );

        let chroma_x = macroblock_x
            .checked_mul(8)
            .ok_or(H264Error::IntegerOverflow)?;
        let chroma_y = macroblock_y
            .checked_mul(8)
            .ok_or(H264Error::IntegerOverflow)?;
        let chroma_stride = self.width / 2;
        copy_block(
            &mut self.cb,
            chroma_stride,
            chroma_x,
            chroma_y,
            &pcm.chroma[..64],
            8,
        );
        copy_block(
            &mut self.cr,
            chroma_stride,
            chroma_x,
            chroma_y,
            &pcm.chroma[64..],
            8,
        );
        Ok(())
    }

    pub(crate) fn snapshot_macroblock(
        &self,
        macroblock_x: usize,
        macroblock_y: usize,
    ) -> Result<MacroblockPixels> {
        let luma_x = macroblock_x
            .checked_mul(16)
            .ok_or(H264Error::IntegerOverflow)?;
        let luma_y = macroblock_y
            .checked_mul(16)
            .ok_or(H264Error::IntegerOverflow)?;
        self.validate_luma_block(luma_x, luma_y, 16)?;
        let chroma_x = macroblock_x
            .checked_mul(8)
            .ok_or(H264Error::IntegerOverflow)?;
        let chroma_y = macroblock_y
            .checked_mul(8)
            .ok_or(H264Error::IntegerOverflow)?;
        Ok(MacroblockPixels {
            luma: read_block(&self.luma, self.width, luma_x, luma_y),
            cb: read_block(&self.cb, self.width / 2, chroma_x, chroma_y),
            cr: read_block(&self.cr, self.width / 2, chroma_x, chroma_y),
        })
    }

    pub(crate) fn restore_macroblock(
        &mut self,
        macroblock_x: usize,
        macroblock_y: usize,
        snapshot: &MacroblockPixels,
    ) {
        assert!(
            macroblock_x < self.width / 16 && macroblock_y < self.height / 16,
            "restored macroblock must lie inside the picture"
        );
        let chroma_stride = self.width / 2;
        // SAFETY: The assertion above and macroblock-aligned picture
        // dimensions prove that all three rectangles fit their planes.
        unsafe {
            write_block(
                Arc::make_mut(&mut self.luma),
                self.width,
                macroblock_x * 16,
                macroblock_y * 16,
                &snapshot.luma,
            );
            write_block(
                &mut self.cb,
                chroma_stride,
                macroblock_x * 8,
                macroblock_y * 8,
                &snapshot.cb,
            );
            write_block(
                &mut self.cr,
                chroma_stride,
                macroblock_x * 8,
                macroblock_y * 8,
                &snapshot.cr,
            );
        }
    }

    pub(crate) fn commit_macroblock_batch(
        &mut self,
        staged: &[StagedMacroblockPixels],
    ) -> Result<()> {
        let width_in_macroblocks = self.width / 16;
        let macroblock_count = width_in_macroblocks
            .checked_mul(self.height / 16)
            .ok_or(H264Error::IntegerOverflow)?;
        let mut previous_address = None;
        for entry in staged {
            if entry.address >= macroblock_count {
                return Err(H264Error::InvalidSyntax(
                    "staged macroblock address exceeds picture",
                ));
            }
            if previous_address.is_some_and(|previous| previous >= entry.address) {
                return Err(H264Error::InvalidSyntax(
                    "staged macroblock addresses are not strictly increasing",
                ));
            }
            previous_address = Some(entry.address);
        }

        if staged.is_empty() {
            return Ok(());
        }

        let luma = Arc::make_mut(&mut self.luma);
        let chroma_stride = self.width / 2;
        let mut expected_address = staged[0].address;
        let mut macroblock_x = expected_address % width_in_macroblocks;
        let mut macroblock_y = expected_address / width_in_macroblocks;
        for entry in staged {
            if entry.address != expected_address {
                macroblock_x = entry.address % width_in_macroblocks;
                macroblock_y = entry.address / width_in_macroblocks;
            }
            // SAFETY: The address pass above proves every staged macroblock is
            // inside the macroblock-aligned picture, so all three rectangles
            // fit their destination planes.
            unsafe {
                write_block(
                    luma,
                    self.width,
                    macroblock_x * 16,
                    macroblock_y * 16,
                    &entry.pixels.luma,
                );
                write_block(
                    &mut self.cb,
                    chroma_stride,
                    macroblock_x * 8,
                    macroblock_y * 8,
                    &entry.pixels.cb,
                );
                write_block(
                    &mut self.cr,
                    chroma_stride,
                    macroblock_x * 8,
                    macroblock_y * 8,
                    &entry.pixels.cr,
                );
            }
            expected_address = entry.address + 1;
            macroblock_x += 1;
            if macroblock_x == width_in_macroblocks {
                macroblock_x = 0;
                macroblock_y += 1;
            }
        }
        Ok(())
    }

    pub fn into_nv12_frame(
        self,
        id: u64,
        pts: Option<MediaTime>,
        duration: Option<MediaTime>,
        format: VideoFormat,
    ) -> Result<DecodedVideoFrame> {
        self.to_nv12_frame(id, pts, duration, format)
    }

    /// Packages NV12 output while retaining the planar picture for future
    /// inter prediction. The immutable luma plane is shared with the output;
    /// only Cb and Cr are interleaved into a new chroma plane.
    pub fn to_nv12_frame(
        &self,
        id: u64,
        pts: Option<MediaTime>,
        duration: Option<MediaTime>,
        format: VideoFormat,
    ) -> Result<DecodedVideoFrame> {
        if format.pixel_format != PixelFormat::Nv12 {
            return Err(H264Error::InvalidSyntax(
                "YUV picture output format must be NV12",
            ));
        }
        if format.coded_size != self.coded_size {
            return Err(H264Error::InvalidSyntax(
                "output format coded size does not match the picture",
            ));
        }
        format.validate()?;

        let chroma_len = self
            .cb
            .len()
            .checked_mul(2)
            .ok_or(H264Error::IntegerOverflow)?;
        let chroma = interleave_chroma(&self.cb, &self.cr, chroma_len);
        let chroma: Arc<[u8]> = chroma.into();
        let frame = DecodedVideoFrame {
            id,
            pts,
            duration,
            format,
            storage: FrameStorage::Cpu(CpuFrame {
                planes: vec![
                    CpuPlane {
                        bytes: self.luma.clone(),
                        offset: 0,
                        stride: self.width,
                        rows: self.height,
                    },
                    CpuPlane {
                        bytes: chroma,
                        offset: 0,
                        stride: self.width,
                        rows: self.height / 2,
                    },
                ],
            }),
        };
        frame.validate()?;
        Ok(frame)
    }

    fn validate_luma_block(&self, x: usize, y: usize, size: usize) -> Result<()> {
        validate_block_bounds(self.width, self.height, x, y, size)
    }

    fn chroma(&self, plane: ChromaPlane) -> &[u8] {
        match plane {
            ChromaPlane::Cb => &self.cb,
            ChromaPlane::Cr => &self.cr,
        }
    }

    fn chroma_mut(&mut self, plane: ChromaPlane) -> &mut [u8] {
        match plane {
            ChromaPlane::Cb => &mut self.cb,
            ChromaPlane::Cr => &mut self.cr,
        }
    }

    fn copy_top(
        &self,
        plane: &[u8],
        stride: usize,
        x: usize,
        y: usize,
        output: &mut [u8],
    ) -> Result<()> {
        let row_end = x
            .checked_add(output.len())
            .ok_or(H264Error::IntegerOverflow)?;
        if row_end > stride {
            return Err(H264Error::InvalidSyntax(
                "top reference crosses the picture row",
            ));
        }
        let row = y.checked_sub(1).ok_or(H264Error::InvalidSyntax(
            "top reference lies outside picture",
        ))?;
        let start = row
            .checked_mul(stride)
            .and_then(|offset| offset.checked_add(x))
            .ok_or(H264Error::IntegerOverflow)?;
        let end = start
            .checked_add(output.len())
            .ok_or(H264Error::IntegerOverflow)?;
        let samples = plane.get(start..end).ok_or(H264Error::InvalidSyntax(
            "top reference lies outside picture",
        ))?;
        output.copy_from_slice(samples);
        Ok(())
    }

    fn copy_left<const LENGTH: usize>(
        &self,
        plane: &[u8],
        stride: usize,
        x: usize,
        y: usize,
    ) -> Result<[u8; LENGTH]> {
        let column = x.checked_sub(1).ok_or(H264Error::InvalidSyntax(
            "left reference lies outside picture",
        ))?;
        let mut output = [0; LENGTH];
        for (row, sample) in output.iter_mut().enumerate() {
            let index = (y + row)
                .checked_mul(stride)
                .and_then(|offset| offset.checked_add(column))
                .ok_or(H264Error::IntegerOverflow)?;
            *sample = *plane.get(index).ok_or(H264Error::InvalidSyntax(
                "left reference lies outside picture",
            ))?;
        }
        Ok(output)
    }

    fn sample_top_left(&self, plane: &[u8], stride: usize, x: usize, y: usize) -> Result<u8> {
        let column = x.checked_sub(1).ok_or(H264Error::InvalidSyntax(
            "top-left reference lies outside picture",
        ))?;
        let row = y.checked_sub(1).ok_or(H264Error::InvalidSyntax(
            "top-left reference lies outside picture",
        ))?;
        let index = row
            .checked_mul(stride)
            .and_then(|offset| offset.checked_add(column))
            .ok_or(H264Error::IntegerOverflow)?;
        plane.get(index).copied().ok_or(H264Error::InvalidSyntax(
            "top-left reference lies outside picture",
        ))
    }
}

fn interleave_chroma(cb: &[u8], cr: &[u8], output_len: usize) -> Vec<u8> {
    assert_eq!(cb.len(), cr.len());
    assert_eq!(cb.len().checked_mul(2), Some(output_len));

    let mut output = Vec::with_capacity(output_len);
    // SAFETY: `output` has capacity for exactly two bytes per input sample.
    // The selected implementation writes every byte before `set_len` exposes
    // the initialized allocation.
    unsafe {
        interleave_chroma_into(cb.as_ptr(), cr.as_ptr(), output.as_mut_ptr(), cb.len());
        output.set_len(output_len);
    }
    output
}

#[cfg(target_arch = "x86_64")]
#[inline(never)]
unsafe fn interleave_chroma_into(cb: *const u8, cr: *const u8, output: *mut u8, len: usize) {
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: Runtime feature detection proves AVX2 support. The caller
        // provides equal readable inputs and two output bytes per sample.
        unsafe {
            interleave_chroma_avx2(cb, cr, output, len);
        }
    } else {
        // SAFETY: SSE2 is part of the x86_64 architecture baseline and the
        // caller provides the bounds described above.
        unsafe {
            interleave_chroma_sse2(cb, cr, output, len);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn interleave_chroma_sse2(cb: *const u8, cr: *const u8, output: *mut u8, len: usize) {
    use std::arch::x86_64::{
        __m128i, _mm_loadu_si128, _mm_storeu_si128, _mm_unpackhi_epi8, _mm_unpacklo_epi8,
    };

    let mut index = 0;
    while index + 16 <= len {
        // SAFETY: The loop bound leaves 16 readable input bytes and 32
        // writable output bytes. Unaligned SIMD loads and stores are used.
        unsafe {
            let cb_samples = _mm_loadu_si128(cb.add(index).cast::<__m128i>());
            let cr_samples = _mm_loadu_si128(cr.add(index).cast::<__m128i>());
            _mm_storeu_si128(
                output.add(index * 2).cast::<__m128i>(),
                _mm_unpacklo_epi8(cb_samples, cr_samples),
            );
            _mm_storeu_si128(
                output.add(index * 2 + 16).cast::<__m128i>(),
                _mm_unpackhi_epi8(cb_samples, cr_samples),
            );
        }
        index += 16;
    }

    while index < len {
        // SAFETY: `index` is within both equal-length inputs and the output
        // reserves two bytes for every input sample.
        unsafe {
            output.add(index * 2).write(*cb.add(index));
            output.add(index * 2 + 1).write(*cr.add(index));
        }
        index += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn interleave_chroma_avx2(cb: *const u8, cr: *const u8, output: *mut u8, len: usize) {
    use std::arch::x86_64::{
        __m256i, _mm256_loadu_si256, _mm256_permute2x128_si256, _mm256_storeu_si256,
        _mm256_unpackhi_epi8, _mm256_unpacklo_epi8,
    };

    let mut index = 0;
    while index + 32 <= len {
        // SAFETY: The loop bound leaves 32 readable input bytes and 64
        // writable output bytes. Unaligned SIMD loads and stores are used.
        unsafe {
            let cb_samples = _mm256_loadu_si256(cb.add(index).cast::<__m256i>());
            let cr_samples = _mm256_loadu_si256(cr.add(index).cast::<__m256i>());
            let low = _mm256_unpacklo_epi8(cb_samples, cr_samples);
            let high = _mm256_unpackhi_epi8(cb_samples, cr_samples);
            _mm256_storeu_si256(
                output.add(index * 2).cast::<__m256i>(),
                _mm256_permute2x128_si256::<0x20>(low, high),
            );
            _mm256_storeu_si256(
                output.add(index * 2 + 32).cast::<__m256i>(),
                _mm256_permute2x128_si256::<0x31>(low, high),
            );
        }
        index += 32;
    }

    // SAFETY: The completed AVX2 prefix leaves equal readable tails and two
    // output bytes per remaining sample. SSE2 is guaranteed on x86_64.
    unsafe {
        interleave_chroma_sse2(
            cb.add(index),
            cr.add(index),
            output.add(index * 2),
            len - index,
        );
    }
}

#[cfg(not(target_arch = "x86_64"))]
unsafe fn interleave_chroma_into(cb: *const u8, cr: *const u8, output: *mut u8, len: usize) {
    for index in 0..len {
        // SAFETY: `index` is within both equal-length inputs and the output
        // reserves two bytes for every input sample.
        unsafe {
            output.add(index * 2).write(*cb.add(index));
            output.add(index * 2 + 1).write(*cr.add(index));
        }
    }
}

fn validate_block_bounds(
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    size: usize,
) -> Result<()> {
    let right = x.checked_add(size).ok_or(H264Error::IntegerOverflow)?;
    let bottom = y.checked_add(size).ok_or(H264Error::IntegerOverflow)?;
    if right > width || bottom > height {
        return Err(H264Error::InvalidSyntax(
            "reconstructed block lies outside picture",
        ));
    }
    Ok(())
}

fn add_block<const SIZE: usize>(
    plane: &mut [u8],
    stride: usize,
    x: usize,
    y: usize,
    prediction: &[[u8; SIZE]; SIZE],
    residual: &[[i32; SIZE]; SIZE],
) {
    for row in 0..SIZE {
        let start = (y + row) * stride + x;
        for column in 0..SIZE {
            plane[start + column] =
                (i32::from(prediction[row][column]) + residual[row][column]).clamp(0, 255) as u8;
        }
    }
}

fn copy_block(plane: &mut [u8], stride: usize, x: usize, y: usize, samples: &[u8], size: usize) {
    for row in 0..size {
        let source = &samples[row * size..(row + 1) * size];
        let start = (y + row) * stride + x;
        plane[start..start + size].copy_from_slice(source);
    }
}

fn read_block<const SIZE: usize>(
    plane: &[u8],
    stride: usize,
    x: usize,
    y: usize,
) -> [[u8; SIZE]; SIZE] {
    std::array::from_fn(|row| {
        plane[(y + row) * stride + x..(y + row) * stride + x + SIZE]
            .try_into()
            .expect("validated block bounds guarantee a complete row")
    })
}

unsafe fn write_block<const SIZE: usize>(
    plane: &mut [u8],
    stride: usize,
    x: usize,
    y: usize,
    block: &[[u8; SIZE]; SIZE],
) {
    // SAFETY: The caller guarantees that the first destination row begins
    // within `plane`.
    let mut destination = unsafe { plane.as_mut_ptr().add(y * stride + x) };
    let mut source = block.as_ptr().cast::<u8>();
    for _ in 0..SIZE {
        // SAFETY: The caller guarantees that every SIZE-byte destination row
        // is in `plane`. Exclusive `plane` and shared `block` references prove
        // that source and destination do not overlap.
        unsafe {
            std::ptr::copy_nonoverlapping(source, destination, SIZE);
            source = source.add(SIZE);
            destination = destination.add(stride);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use decv_core::{ColorInfo, FrameStorage, MediaTime, PixelFormat, Rect, Size, VideoFormat};

    use super::{
        ChromaPlane, IntraReferenceAvailability, MacroblockPixels, StagedMacroblockPixels,
        Yuv420Picture, interleave_chroma,
    };
    #[cfg(target_arch = "x86_64")]
    use super::{interleave_chroma_avx2, interleave_chroma_sse2};
    use crate::{H264Error, PcmMacroblock};

    const ALL_REFERENCES: IntraReferenceAvailability = IntraReferenceAvailability {
        top: true,
        left: true,
        top_left: true,
        top_right: true,
    };

    #[test]
    fn chroma_interleave_matches_scalar_for_vector_and_tail_lengths() {
        for len in 0..=65 {
            let cb: Vec<_> = (0..len).map(|index| index as u8).collect();
            let cr: Vec<_> = (0..len)
                .map(|index| (index as u8).wrapping_mul(17).wrapping_add(3))
                .collect();
            let expected: Vec<_> = cb.iter().zip(&cr).flat_map(|(&cb, &cr)| [cb, cr]).collect();
            assert_eq!(interleave_chroma(&cb, &cr, len * 2), expected);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn x86_chroma_interleave_backends_match_scalar_for_vector_and_tail_lengths() {
        for len in 0..=65 {
            let cb: Vec<_> = (0..len).map(|index| index as u8).collect();
            let cr: Vec<_> = (0..len)
                .map(|index| (index as u8).wrapping_mul(17).wrapping_add(3))
                .collect();
            let expected: Vec<_> = cb.iter().zip(&cr).flat_map(|(&cb, &cr)| [cb, cr]).collect();

            let mut sse2 = vec![0; len * 2];
            // SAFETY: SSE2 is part of the x86_64 baseline and all three
            // allocations have the exact lengths required by the backend.
            unsafe {
                interleave_chroma_sse2(cb.as_ptr(), cr.as_ptr(), sse2.as_mut_ptr(), cb.len());
            }
            assert_eq!(sse2, expected);

            if std::arch::is_x86_feature_detected!("avx2") {
                let mut avx2 = vec![0; len * 2];
                // SAFETY: Runtime detection proves AVX2 support and the
                // allocations have the exact lengths required by the backend.
                unsafe {
                    interleave_chroma_avx2(cb.as_ptr(), cr.as_ptr(), avx2.as_mut_ptr(), cb.len());
                }
                assert_eq!(avx2, expected);
            }
        }
    }

    #[test]
    fn validates_macroblock_aligned_picture_dimensions() {
        assert!(Yuv420Picture::new(Size::new(16, 16)).is_ok());
        assert_eq!(
            Yuv420Picture::new(Size::new(17, 16)),
            Err(H264Error::InvalidSyntax(
                "4:2:0 H.264 coded dimensions must be macroblock aligned"
            ))
        );
        assert!(Yuv420Picture::new(Size::new(0, 16)).is_err());
        assert_eq!(
            Yuv420Picture::new(Size::new(4096, 2320)),
            Err(H264Error::InvalidSyntax(
                "coded picture exceeds the decoder macroblock limit"
            ))
        );
    }

    #[test]
    fn gathers_macroblock_luma_and_chroma_references() {
        let mut picture = Yuv420Picture::new(Size::new(32, 32)).unwrap();
        {
            let luma = Arc::make_mut(&mut picture.luma);
            for column in 0..16 {
                luma[15 * 32 + 16 + column] = 10 + column as u8;
            }
            for row in 0..16 {
                luma[(16 + row) * 32 + 15] = 40 + row as u8;
            }
            luma[15 * 32 + 15] = 99;
        }

        let references = picture.intra16x16_references(1, 1, ALL_REFERENCES).unwrap();
        assert_eq!(
            references.top,
            Some(std::array::from_fn(|index| 10 + index as u8))
        );
        assert_eq!(
            references.left,
            Some(std::array::from_fn(|index| 40 + index as u8))
        );
        assert_eq!(references.top_left, Some(99));

        for column in 0..8 {
            picture.cb[7 * 16 + 8 + column] = 60 + column as u8;
        }
        for row in 0..8 {
            picture.cb[(8 + row) * 16 + 7] = 80 + row as u8;
        }
        picture.cb[7 * 16 + 7] = 70;
        let chroma = picture
            .intra_chroma_references(ChromaPlane::Cb, 1, 1, ALL_REFERENCES)
            .unwrap();
        assert_eq!(chroma.top, Some([60, 61, 62, 63, 64, 65, 66, 67]));
        assert_eq!(chroma.left, Some([80, 81, 82, 83, 84, 85, 86, 87]));
        assert_eq!(chroma.top_left, Some(70));
    }

    #[test]
    fn substitutes_unavailable_top_right_samples() {
        let mut picture = Yuv420Picture::new(Size::new(32, 32)).unwrap();
        let luma = Arc::make_mut(&mut picture.luma);
        for column in 0..16 {
            luma[7 * 32 + 8 + column] = 20 + column as u8;
        }
        let availability = IntraReferenceAvailability {
            top: true,
            top_right: false,
            ..IntraReferenceAvailability::default()
        };
        let four = picture.intra4x4_references(8, 8, availability).unwrap();
        assert_eq!(four.top, Some([20, 21, 22, 23, 23, 23, 23, 23]));
        let eight = picture.intra8x8_references(8, 8, availability).unwrap();
        assert_eq!(
            eight.top,
            Some([
                20, 21, 22, 23, 24, 25, 26, 27, 27, 27, 27, 27, 27, 27, 27, 27
            ])
        );
        assert!(
            picture
                .intra8x8_references(
                    24,
                    8,
                    IntraReferenceAvailability {
                        top: true,
                        top_right: true,
                        ..IntraReferenceAvailability::default()
                    }
                )
                .is_err()
        );

        assert!(
            picture
                .intra4x4_references(
                    0,
                    0,
                    IntraReferenceAvailability {
                        top: true,
                        ..IntraReferenceAvailability::default()
                    }
                )
                .is_err()
        );
    }

    #[test]
    fn adds_residuals_with_eight_bit_saturation() {
        let mut picture = Yuv420Picture::new(Size::new(16, 16)).unwrap();
        let mut residual = [[0; 4]; 4];
        residual[0][0] = -10;
        residual[0][1] = 300;
        picture
            .write_luma_4x4(0, 0, &[[5; 4]; 4], &residual)
            .unwrap();
        assert_eq!(&picture.luma[..4], &[0, 255, 5, 5]);

        let mut chroma_residual = [[0; 8]; 8];
        chroma_residual[7][7] = -200;
        picture
            .write_chroma_8x8(ChromaPlane::Cr, 0, 0, &[[100; 8]; 8], &chroma_residual)
            .unwrap();
        assert_eq!(picture.cr[0], 100);
        assert_eq!(picture.cr[63], 0);
    }

    #[test]
    fn validates_a_staged_macroblock_batch_before_committing_pixels() {
        let pixels =
            |luma, cb, cr| MacroblockPixels::new([[luma; 16]; 16], [[cb; 8]; 8], [[cr; 8]; 8]);
        let mut picture = Yuv420Picture::new(Size::new(32, 16)).unwrap();
        picture
            .commit_macroblock_batch(&[
                StagedMacroblockPixels::new(0, pixels(10, 20, 30)),
                StagedMacroblockPixels::new(1, pixels(40, 50, 60)),
            ])
            .unwrap();
        assert_eq!(&picture.luma[..16], &[10; 16]);
        assert_eq!(&picture.luma[16..32], &[40; 16]);

        let before = picture.clone();
        assert_eq!(
            picture.commit_macroblock_batch(&[
                StagedMacroblockPixels::new(1, pixels(1, 2, 3)),
                StagedMacroblockPixels::new(1, pixels(4, 5, 6)),
            ]),
            Err(H264Error::InvalidSyntax(
                "staged macroblock addresses are not strictly increasing"
            ))
        );
        assert_eq!(picture, before);

        assert_eq!(
            picture.commit_macroblock_batch(&[StagedMacroblockPixels::new(2, pixels(7, 8, 9))]),
            Err(H264Error::InvalidSyntax(
                "staged macroblock address exceeds picture"
            ))
        );
        assert_eq!(picture, before);

        picture
            .commit_macroblock_batch(&[StagedMacroblockPixels::new(0, pixels(70, 80, 90))])
            .unwrap();
        assert_eq!(&picture.luma[..16], &[70; 16]);
        assert_eq!(&before.luma[..16], &[10; 16]);
    }

    #[test]
    fn commits_contiguous_rows_and_recomputes_after_address_gaps() {
        let pixels = |value| MacroblockPixels::new([[value; 16]; 16], [[0; 8]; 8], [[0; 8]; 8]);
        let mut picture = Yuv420Picture::new(Size::new(48, 32)).unwrap();
        picture
            .commit_macroblock_batch(&[
                StagedMacroblockPixels::new(1, pixels(10)),
                StagedMacroblockPixels::new(2, pixels(20)),
                StagedMacroblockPixels::new(3, pixels(30)),
                StagedMacroblockPixels::new(5, pixels(50)),
            ])
            .unwrap();

        assert_eq!(picture.luma[16], 10);
        assert_eq!(picture.luma[32], 20);
        assert_eq!(picture.luma[16 * 48], 30);
        assert_eq!(picture.luma[16 * 48 + 16], 0);
        assert_eq!(picture.luma[16 * 48 + 32], 50);
    }

    #[test]
    fn writes_pcm_and_packages_independent_nv12_planes() {
        let mut picture = Yuv420Picture::new(Size::new(16, 16)).unwrap();
        let pcm = PcmMacroblock {
            luma: Box::new(std::array::from_fn(|index| index as u8)),
            chroma: Box::new(std::array::from_fn(
                |index| {
                    if index < 64 { 10 } else { 20 }
                },
            )),
        };
        picture.write_pcm_macroblock(0, 0, &pcm).unwrap();
        let format = VideoFormat {
            coded_size: Size::new(16, 16),
            visible_rect: Rect::new(0, 0, 16, 16),
            display_size: Size::new(16, 16),
            pixel_format: PixelFormat::Nv12,
            color: ColorInfo::default(),
        };
        let frame = picture
            .into_nv12_frame(
                7,
                MediaTime::from_parts(1, 30),
                MediaTime::from_parts(1, 30),
                format,
            )
            .unwrap();
        assert_eq!(frame.id, 7);
        assert_eq!(frame.validate(), Ok(()));

        let cpu = match &frame.storage {
            FrameStorage::Cpu(cpu) => cpu,
            _ => panic!("expected CPU frame"),
        };
        assert_eq!(cpu.planes.len(), 2);
        assert_eq!(cpu.planes[0].bytes.len(), 256);
        assert_eq!(cpu.planes[1].bytes.len(), 128);
        assert!(!Arc::ptr_eq(&cpu.planes[0].bytes, &cpu.planes[1].bytes));
        assert_eq!(&cpu.planes[0].bytes[..4], &[0, 1, 2, 3]);
        assert_eq!(cpu.planes[1].offset, 0);
        assert_eq!(&cpu.planes[1].bytes[..8], &[10, 20, 10, 20, 10, 20, 10, 20]);
    }
}
