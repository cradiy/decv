use std::ops::Range;

use crate::{
    BitDepth, CompressedHeader, FrameHeader, Result, TransformMode, Vp9Error,
    block::{BlockSize, IntraMode, Partition, TransformSize, TransformType},
    bool_decoder::BoolDecoder,
    context::{CoefficientCounts, FrameCounts, ProbabilityContext},
    loop_filter::{FilterMode, FilterModeMap, apply_loop_filter},
    quantization::dequant,
    reconstruct::IntraPicture,
    tables,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TileLayout {
    ranges: Vec<Range<usize>>,
    columns: usize,
    rows: usize,
}

impl TileLayout {
    pub fn parse(frame: &[u8], header: &FrameHeader) -> Result<Self> {
        let payload_start = header
            .uncompressed_header_size
            .checked_add(header.compressed_header_size)
            .ok_or(Vp9Error::IntegerOverflow)?;
        if payload_start > frame.len() {
            return Err(Vp9Error::Truncated("tile payload"));
        }
        let columns = 1usize << header.tile_columns_log2;
        let rows = 1usize << header.tile_rows_log2;
        let tile_count = columns.checked_mul(rows).ok_or(Vp9Error::IntegerOverflow)?;
        let mut cursor = payload_start;
        let mut ranges = Vec::with_capacity(tile_count);
        for tile_index in 0..tile_count {
            let size = if tile_index + 1 == tile_count {
                frame.len() - cursor
            } else {
                let bytes = frame
                    .get(cursor..cursor + 4)
                    .ok_or(Vp9Error::Truncated("tile size"))?;
                cursor += 4;
                usize::try_from(u32::from_be_bytes(bytes.try_into().unwrap()))
                    .map_err(|_| Vp9Error::IntegerOverflow)?
            };
            if size == 0 {
                return Err(Vp9Error::InvalidData("tile is empty"));
            }
            let end = cursor.checked_add(size).ok_or(Vp9Error::IntegerOverflow)?;
            if end > frame.len() {
                return Err(Vp9Error::Truncated("tile payload"));
            }
            ranges.push(cursor..end);
            cursor = end;
        }
        if cursor != frame.len() {
            return Err(Vp9Error::InvalidData(
                "tile sizes do not cover the coded frame",
            ));
        }
        Ok(Self {
            ranges,
            columns,
            rows,
        })
    }

    #[inline]
    pub fn columns(&self) -> usize {
        self.columns
    }

    #[inline]
    pub fn rows(&self) -> usize {
        self.rows
    }

    #[inline]
    pub fn tile_count(&self) -> usize {
        self.ranges.len()
    }

    pub fn tiles<'a>(&'a self, frame: &'a [u8]) -> impl ExactSizeIterator<Item = &'a [u8]> {
        self.ranges.iter().map(|range| &frame[range.clone()])
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct IntraSyntaxSummary {
    pub blocks: usize,
    pub transform_blocks: usize,
    pub nonzero_transform_blocks: usize,
    pub coefficients: usize,
    pub coefficient_sum_abs: u64,
}

impl IntraSyntaxSummary {
    pub fn parse(
        frame: &[u8],
        header: &FrameHeader,
        compressed: &CompressedHeader,
    ) -> Result<Self> {
        let mut context = ProbabilityContext::default();
        context.apply(compressed)?;
        parse_intra_syntax(frame, header, compressed, &context, None, None)
            .map(|(summary, _)| summary)
    }
}

/// Decodes and reconstructs one intra-only VP9 frame at its native bit depth.
pub fn decode_intra_picture(
    frame: &[u8],
    header: &FrameHeader,
    compressed: &CompressedHeader,
) -> Result<IntraPicture> {
    let size = header
        .size
        .ok_or(Vp9Error::InvalidData("frame has no dimensions"))?;
    let width = usize::try_from(size.width).map_err(|_| Vp9Error::IntegerOverflow)?;
    let height = usize::try_from(size.height).map_err(|_| Vp9Error::IntegerOverflow)?;
    let mut context = ProbabilityContext::default();
    context.apply(compressed)?;
    let mut picture = IntraPicture::new(
        width,
        height,
        header.chroma_subsampling(),
        header.bit_depth(),
    );
    let (_, modes) = parse_intra_syntax(
        frame,
        header,
        compressed,
        &context,
        Some(&mut picture),
        None,
    )?;
    apply_loop_filter(&mut picture, header, &modes)?;
    Ok(picture)
}

pub(crate) fn decode_intra_picture_with_context(
    frame: &[u8],
    header: &FrameHeader,
    compressed: &CompressedHeader,
    context: &ProbabilityContext,
    counts: &mut FrameCounts,
) -> Result<(IntraPicture, Vec<u8>)> {
    let size = header
        .size
        .ok_or(Vp9Error::InvalidData("frame has no dimensions"))?;
    let width = usize::try_from(size.width).map_err(|_| Vp9Error::IntegerOverflow)?;
    let height = usize::try_from(size.height).map_err(|_| Vp9Error::IntegerOverflow)?;
    let mut picture = IntraPicture::new(
        width,
        height,
        header.chroma_subsampling(),
        header.bit_depth(),
    );
    let (_, modes) = parse_intra_syntax(
        frame,
        header,
        compressed,
        context,
        Some(&mut picture),
        Some(counts),
    )?;
    apply_loop_filter(&mut picture, header, &modes)?;
    Ok((picture, modes.segment_ids()))
}

/// Parses every mode and coefficient token of an intra-only VP9 frame.
///
/// This is the syntax half of reconstruction. Keeping it separately
/// verifiable prevents predictor or inverse-transform bugs from being
/// misdiagnosed as entropy-decoder failures.
pub(crate) fn parse_intra_syntax(
    frame: &[u8],
    header: &FrameHeader,
    compressed: &CompressedHeader,
    context: &ProbabilityContext,
    mut picture: Option<&mut IntraPicture>,
    mut counts: Option<&mut FrameCounts>,
) -> Result<(IntraSyntaxSummary, FilterModeMap)> {
    if !header.intra_only {
        return Err(Vp9Error::UnsupportedFeature(
            "intra syntax parser received an inter frame",
        ));
    }
    let size = header
        .size
        .ok_or(Vp9Error::InvalidData("frame has no dimensions"))?;
    let mi_columns =
        usize::try_from(size.width.div_ceil(8)).map_err(|_| Vp9Error::IntegerOverflow)?;
    let mi_rows =
        usize::try_from(size.height.div_ceil(8)).map_err(|_| Vp9Error::IntegerOverflow)?;
    let layout = TileLayout::parse(frame, header)?;
    let mut modes = vec![None; mi_columns * mi_rows];
    let mut summary = IntraSyntaxSummary::default();

    if layout.rows() == 1
        && layout.columns() > 1
        && let Some(picture) = picture.take()
    {
        return parse_intra_tiles_parallel(
            frame, header, compressed, context, counts, &layout, picture, mi_columns, mi_rows,
        );
    }

    for (tile_index, tile) in layout.tiles(frame).enumerate() {
        let tile_row = tile_index / layout.columns;
        let tile_column = tile_index % layout.columns;
        let (row_start, row_end) = tile_offsets(tile_row, mi_rows, header.tile_rows_log2);
        let (column_start, column_end) =
            tile_offsets(tile_column, mi_columns, header.tile_columns_log2);
        let mut decoder = IntraTileDecoder::new(
            tile,
            header,
            compressed,
            context,
            mi_columns,
            mi_rows,
            row_start,
            row_end,
            column_start,
            column_end,
            &mut modes,
            picture.as_deref_mut(),
            counts.as_deref_mut(),
        )?;
        match decoder.parse() {
            Ok(tile_summary) => summary += tile_summary,
            Err(source) => {
                return Err(Vp9Error::TileDecode {
                    tile: tile_index,
                    blocks: decoder.summary.blocks,
                    transform_blocks: decoder.summary.transform_blocks,
                    source: Box::new(source),
                });
            }
        }
    }
    let filter_modes = modes
        .into_iter()
        .map(|mode| {
            let mode = mode.ok_or(Vp9Error::InvalidData(
                "intra mode map has an undecoded block",
            ))?;
            Ok(FilterMode {
                block_size: mode.block_size,
                transform_size: mode.transform_size,
                skip: mode.skip,
                segment_id: mode.segment_id,
                reference: 0,
                mode_class: 0,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((
        summary,
        FilterModeMap::new(mi_columns, mi_rows, filter_modes)?,
    ))
}

#[allow(clippy::too_many_arguments)]
fn parse_intra_tiles_parallel(
    frame: &[u8],
    header: &FrameHeader,
    compressed: &CompressedHeader,
    context: &ProbabilityContext,
    mut counts: Option<&mut FrameCounts>,
    layout: &TileLayout,
    picture: &mut IntraPicture,
    mi_columns: usize,
    mi_rows: usize,
) -> Result<(IntraSyntaxSummary, FilterModeMap)> {
    struct TileResult {
        summary: IntraSyntaxSummary,
        modes: Vec<Option<ModeInfo>>,
        counts: FrameCounts,
        picture: IntraPicture,
        column_start: usize,
        column_end: usize,
    }

    let width = picture.width();
    let height = picture.height();
    let collect_counts = counts.is_some();
    let tiles = layout.tiles(frame).collect::<Vec<_>>();
    let tile_results = std::thread::scope(|scope| {
        let mut workers = Vec::with_capacity(tiles.len());
        for (tile_index, tile) in tiles.into_iter().enumerate() {
            let worker = std::thread::Builder::new()
                .name(format!("decv-vp9-intra-tile-{tile_index}"))
                .stack_size(512 * 1024)
                .spawn_scoped(scope, move || -> Result<TileResult> {
                    let (column_start, column_end) =
                        tile_offsets(tile_index, mi_columns, header.tile_columns_log2);
                    let origin_x = column_start * 8;
                    let end_x = (column_end * 8).min(width);
                    let mut tile_picture = IntraPicture::new_strip(
                        width,
                        height,
                        origin_x,
                        end_x - origin_x,
                        header.chroma_subsampling(),
                        header.bit_depth(),
                    );
                    let mut tile_modes = vec![None; mi_columns * mi_rows];
                    let mut tile_counts = FrameCounts::default();
                    let mut decoder = IntraTileDecoder::new(
                        tile,
                        header,
                        compressed,
                        context,
                        mi_columns,
                        mi_rows,
                        0,
                        mi_rows,
                        column_start,
                        column_end,
                        &mut tile_modes,
                        Some(&mut tile_picture),
                        collect_counts.then_some(&mut tile_counts),
                    )?;
                    let summary = match decoder.parse() {
                        Ok(summary) => summary,
                        Err(source) => {
                            return Err(Vp9Error::TileDecode {
                                tile: tile_index,
                                blocks: decoder.summary.blocks,
                                transform_blocks: decoder.summary.transform_blocks,
                                source: Box::new(source),
                            });
                        }
                    };
                    Ok(TileResult {
                        summary,
                        modes: tile_modes,
                        counts: tile_counts,
                        picture: tile_picture,
                        column_start,
                        column_end,
                    })
                })
                .map_err(|_| Vp9Error::InvalidData("failed to spawn VP9 intra tile worker"))?;
            workers.push(worker);
        }
        workers
            .into_iter()
            .map(|worker| {
                worker
                    .join()
                    .map_err(|_| Vp9Error::InvalidData("VP9 intra tile worker panicked"))?
            })
            .collect::<Result<Vec<_>>>()
    })?;

    let mut summary = IntraSyntaxSummary::default();
    let mut modes = vec![None; mi_columns * mi_rows];
    for tile in tile_results {
        summary += tile.summary;
        if let Some(counts) = &mut counts {
            counts.merge_from(&tile.counts);
        }
        for row in 0..mi_rows {
            let start = row * mi_columns + tile.column_start;
            let end = row * mi_columns + tile.column_end;
            modes[start..end].copy_from_slice(&tile.modes[start..end]);
        }
        picture.copy_strip_from(&tile.picture);
    }
    let filter_modes = modes
        .into_iter()
        .map(|mode| {
            let mode = mode.ok_or(Vp9Error::InvalidData(
                "intra mode map has an undecoded block",
            ))?;
            Ok(FilterMode {
                block_size: mode.block_size,
                transform_size: mode.transform_size,
                skip: mode.skip,
                segment_id: mode.segment_id,
                reference: 0,
                mode_class: 0,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((
        summary,
        FilterModeMap::new(mi_columns, mi_rows, filter_modes)?,
    ))
}

impl std::ops::AddAssign for IntraSyntaxSummary {
    fn add_assign(&mut self, rhs: Self) {
        self.blocks += rhs.blocks;
        self.transform_blocks += rhs.transform_blocks;
        self.nonzero_transform_blocks += rhs.nonzero_transform_blocks;
        self.coefficients += rhs.coefficients;
        self.coefficient_sum_abs += rhs.coefficient_sum_abs;
    }
}

pub(crate) fn tile_offsets(index: usize, mi_count: usize, log2_tiles: u8) -> (usize, usize) {
    let superblocks = mi_count.div_ceil(8);
    let start = ((index * superblocks) >> log2_tiles) << 3;
    let end = ((((index + 1) * superblocks) >> log2_tiles) << 3).min(mi_count);
    (start.min(mi_count), end)
}

#[derive(Debug, Clone, Copy)]
struct ModeInfo {
    block_size: BlockSize,
    skip: bool,
    transform_size: TransformSize,
    mode: IntraMode,
    sub_modes: [IntraMode; 4],
    uv_mode: IntraMode,
    segment_id: u8,
}

struct IntraTileDecoder<'a, 'state> {
    bits: BoolDecoder<'a>,
    header: &'a FrameHeader,
    compressed: &'a CompressedHeader,
    context: &'a ProbabilityContext,
    mi_columns: usize,
    mi_rows: usize,
    row_start: usize,
    row_end: usize,
    column_start: usize,
    column_end: usize,
    modes: &'state mut [Option<ModeInfo>],
    above_partition: Vec<u8>,
    left_partition: [u8; 8],
    above_coefficients: [Vec<u8>; 3],
    left_coefficients: [[u8; 16]; 3],
    summary: IntraSyntaxSummary,
    picture: Option<&'state mut IntraPicture>,
    counts: Option<&'state mut FrameCounts>,
}

impl<'a, 'state> IntraTileDecoder<'a, 'state> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        tile: &'a [u8],
        header: &'a FrameHeader,
        compressed: &'a CompressedHeader,
        context: &'a ProbabilityContext,
        mi_columns: usize,
        mi_rows: usize,
        row_start: usize,
        row_end: usize,
        column_start: usize,
        column_end: usize,
        modes: &'state mut [Option<ModeInfo>],
        picture: Option<&'state mut IntraPicture>,
        counts: Option<&'state mut FrameCounts>,
    ) -> Result<Self> {
        let luma_4x4_columns = mi_columns * 2;
        let chroma_4x4_columns = luma_4x4_columns >> header.chroma_subsampling().x_shift();
        Ok(Self {
            bits: BoolDecoder::new(tile)?,
            header,
            compressed,
            context,
            mi_columns,
            mi_rows,
            row_start,
            row_end,
            column_start,
            column_end,
            modes,
            above_partition: vec![0; mi_columns],
            left_partition: [0; 8],
            above_coefficients: [
                vec![0; luma_4x4_columns],
                vec![0; chroma_4x4_columns],
                vec![0; chroma_4x4_columns],
            ],
            left_coefficients: [[0; 16]; 3],
            summary: IntraSyntaxSummary::default(),
            picture,
            counts,
        })
    }

    fn parse(&mut self) -> Result<IntraSyntaxSummary> {
        for mi_row in (self.row_start..self.row_end).step_by(8) {
            self.left_partition.fill(0);
            for contexts in &mut self.left_coefficients {
                contexts.fill(0);
            }
            for mi_column in (self.column_start..self.column_end).step_by(8) {
                self.decode_partition(mi_row, mi_column, BlockSize::B64x64, 3)?;
            }
        }
        Ok(self.summary)
    }

    fn decode_partition(
        &mut self,
        mi_row: usize,
        mi_column: usize,
        block_size: BlockSize,
        level: u8,
    ) -> Result<()> {
        if mi_row >= self.row_end || mi_column >= self.column_end {
            return Ok(());
        }
        let half_mi = if level == 0 { 0 } else { 1usize << (level - 1) };
        let has_rows = mi_row + half_mi < self.row_end;
        let has_columns = mi_column + half_mi < self.column_end;
        let partition = self.read_partition(mi_row, mi_column, level, has_rows, has_columns)?;
        let subsize = block_size
            .partition_subsize(partition)
            .ok_or(Vp9Error::InvalidData("invalid partition for block size"))?;

        if level == 0 {
            self.decode_block(mi_row, mi_column, subsize)?;
        } else {
            match partition {
                Partition::None => self.decode_block(mi_row, mi_column, subsize)?,
                Partition::Horizontal => {
                    self.decode_block(mi_row, mi_column, subsize)?;
                    if has_rows {
                        self.decode_block(mi_row + half_mi, mi_column, subsize)?;
                    }
                }
                Partition::Vertical => {
                    self.decode_block(mi_row, mi_column, subsize)?;
                    if has_columns {
                        self.decode_block(mi_row, mi_column + half_mi, subsize)?;
                    }
                }
                Partition::Split => {
                    self.decode_partition(mi_row, mi_column, subsize, level - 1)?;
                    self.decode_partition(mi_row, mi_column + half_mi, subsize, level - 1)?;
                    self.decode_partition(mi_row + half_mi, mi_column, subsize, level - 1)?;
                    self.decode_partition(
                        mi_row + half_mi,
                        mi_column + half_mi,
                        subsize,
                        level - 1,
                    )?;
                }
            }
        }

        if level == 0 || partition != Partition::Split {
            let width_mi = 1usize << level;
            let (above_value, left_value) = subsize.partition_context();
            for column in mi_column..(mi_column + width_mi).min(self.column_end) {
                self.above_partition[column] = above_value;
            }
            for row in mi_row..(mi_row + width_mi).min(self.row_end) {
                self.left_partition[row & 7] = left_value;
            }
        }
        Ok(())
    }

    fn read_partition(
        &mut self,
        mi_row: usize,
        mi_column: usize,
        level: u8,
        has_rows: bool,
        has_columns: bool,
    ) -> Result<Partition> {
        let above = self.above_partition[mi_column] >> level & 1;
        let left = self.left_partition[mi_row & 7] >> level & 1;
        let probability_context =
            usize::from(level) * 4 + usize::from(left) * 2 + usize::from(above);
        let probabilities = &tables::KF_PARTITION[probability_context * 3..][..3];
        if has_rows && has_columns {
            if !self.bits.read_bool(probabilities[0])? {
                Ok(Partition::None)
            } else if !self.bits.read_bool(probabilities[1])? {
                Ok(Partition::Horizontal)
            } else if !self.bits.read_bool(probabilities[2])? {
                Ok(Partition::Vertical)
            } else {
                Ok(Partition::Split)
            }
        } else if !has_rows && has_columns {
            Ok(if self.bits.read_bool(probabilities[1])? {
                Partition::Split
            } else {
                Partition::Horizontal
            })
        } else if has_rows && !has_columns {
            Ok(if self.bits.read_bool(probabilities[2])? {
                Partition::Split
            } else {
                Partition::Vertical
            })
        } else {
            Ok(Partition::Split)
        }
    }

    fn decode_block(
        &mut self,
        mi_row: usize,
        mi_column: usize,
        block_size: BlockSize,
    ) -> Result<()> {
        let mode = self.read_mode_info(mi_row, mi_column, block_size)?;
        self.store_mode(mi_row, mi_column, mode);
        self.summary.blocks += 1;
        self.read_transform_blocks(mi_row, mi_column, mode)
    }

    fn read_mode_info(
        &mut self,
        mi_row: usize,
        mi_column: usize,
        block_size: BlockSize,
    ) -> Result<ModeInfo> {
        let segment_id = self.read_segment_id()?;
        let segment_skip = self
            .header
            .segmentation
            .as_ref()
            .is_some_and(|segmentation| segmentation.features[usize::from(segment_id)][3].enabled);
        let above = self.mode_above(mi_row, mi_column);
        let left = self.mode_left(mi_row, mi_column);
        let skip_context = usize::from(above.is_some_and(|mode| mode.skip))
            + usize::from(left.is_some_and(|mode| mode.skip));
        let skip = segment_skip || self.bits.read_bool(self.context.skip[skip_context])?;
        let transform_size = self.read_transform_size(block_size, above, left)?;
        let mut sub_modes = [IntraMode::Dc; 4];
        let mode = match block_size {
            BlockSize::B4x4 => {
                for block in 0..4 {
                    sub_modes[block] = self.read_keyframe_y_mode(above, left, &sub_modes, block)?;
                }
                sub_modes[3]
            }
            BlockSize::B4x8 => {
                sub_modes[0] = self.read_keyframe_y_mode(above, left, &sub_modes, 0)?;
                sub_modes[2] = sub_modes[0];
                sub_modes[1] = self.read_keyframe_y_mode(above, left, &sub_modes, 1)?;
                sub_modes[3] = sub_modes[1];
                sub_modes[3]
            }
            BlockSize::B8x4 => {
                sub_modes[0] = self.read_keyframe_y_mode(above, left, &sub_modes, 0)?;
                sub_modes[1] = sub_modes[0];
                sub_modes[2] = self.read_keyframe_y_mode(above, left, &sub_modes, 2)?;
                sub_modes[3] = sub_modes[2];
                sub_modes[3]
            }
            _ => {
                let mode = self.read_keyframe_y_mode(above, left, &sub_modes, 0)?;
                sub_modes.fill(mode);
                mode
            }
        };
        let uv_mode = read_intra_mode(
            &mut self.bits,
            &tables::KF_UV_MODE[mode as usize * 9..][..9],
        )?;
        Ok(ModeInfo {
            block_size,
            skip,
            transform_size,
            mode,
            sub_modes,
            uv_mode,
            segment_id,
        })
    }

    fn read_segment_id(&mut self) -> Result<u8> {
        let Some(segmentation) = &self.header.segmentation else {
            return Ok(0);
        };
        if !segmentation.enabled || !segmentation.update_map {
            return Ok(0);
        }
        read_segment_tree(&mut self.bits, &segmentation.tree_probabilities)
    }

    fn read_transform_size(
        &mut self,
        block_size: BlockSize,
        above: Option<ModeInfo>,
        left: Option<ModeInfo>,
    ) -> Result<TransformSize> {
        let maximum = block_size.maximum_transform();
        if self.compressed.transform_mode != TransformMode::Select || block_size < BlockSize::B8x8 {
            let selected_maximum = match self.compressed.transform_mode {
                TransformMode::Only4x4 => TransformSize::Tx4x4,
                TransformMode::Allow8x8 => TransformSize::Tx8x8,
                TransformMode::Allow16x16 => TransformSize::Tx16x16,
                TransformMode::Allow32x32 | TransformMode::Select => TransformSize::Tx32x32,
            };
            return Ok(maximum.min(selected_maximum));
        }

        let mut above_context = above
            .filter(|mode| !mode.skip)
            .map_or(maximum, |mode| mode.transform_size);
        let mut left_context = left
            .filter(|mode| !mode.skip)
            .map_or(maximum, |mode| mode.transform_size);
        if left.is_none() {
            left_context = above_context;
        }
        if above.is_none() {
            above_context = left_context;
        }
        let context = usize::from(above_context as u8 + left_context as u8 > maximum as u8);
        let probabilities: &[u8] = match maximum {
            TransformSize::Tx8x8 => &self.context.transform[10 + context..][..1],
            TransformSize::Tx16x16 => &self.context.transform[6 + context * 2..][..2],
            TransformSize::Tx32x32 => &self.context.transform[context * 3..][..3],
            TransformSize::Tx4x4 => return Ok(TransformSize::Tx4x4),
        };
        let mut transform = usize::from(self.bits.read_bool(probabilities[0])?);
        if transform != 0 && maximum >= TransformSize::Tx16x16 {
            transform += usize::from(self.bits.read_bool(probabilities[1])?);
            if transform != 1 && maximum >= TransformSize::Tx32x32 {
                transform += usize::from(self.bits.read_bool(probabilities[2])?);
            }
        }
        Ok([
            TransformSize::Tx4x4,
            TransformSize::Tx8x8,
            TransformSize::Tx16x16,
            TransformSize::Tx32x32,
        ][transform])
    }

    fn read_keyframe_y_mode(
        &mut self,
        above: Option<ModeInfo>,
        left: Option<ModeInfo>,
        current: &[IntraMode; 4],
        block: usize,
    ) -> Result<IntraMode> {
        let above_mode = if block < 2 {
            above.map_or(IntraMode::Dc, |mode| mode.y_mode(block + 2))
        } else {
            current[block - 2]
        };
        let left_mode = if block == 0 || block == 2 {
            left.map_or(IntraMode::Dc, |mode| mode.y_mode(block + 1))
        } else {
            current[block - 1]
        };
        let start = (above_mode as usize * 10 + left_mode as usize) * 9;
        read_intra_mode(&mut self.bits, &tables::KF_Y_MODE[start..start + 9])
    }

    fn mode_above(&self, mi_row: usize, mi_column: usize) -> Option<ModeInfo> {
        if mi_row == self.row_start {
            None
        } else {
            self.modes[(mi_row - 1) * self.mi_columns + mi_column]
        }
    }

    fn mode_left(&self, mi_row: usize, mi_column: usize) -> Option<ModeInfo> {
        if mi_column == self.column_start {
            None
        } else {
            self.modes[mi_row * self.mi_columns + mi_column - 1]
        }
    }

    fn store_mode(&mut self, mi_row: usize, mi_column: usize, mode: ModeInfo) {
        let row_end = (mi_row + mode.block_size.height_mi()).min(self.mi_rows);
        let column_end = (mi_column + mode.block_size.width_mi()).min(self.mi_columns);
        for row in mi_row..row_end {
            for column in mi_column..column_end {
                self.modes[row * self.mi_columns + column] = Some(mode);
            }
        }
    }

    fn read_transform_blocks(
        &mut self,
        mi_row: usize,
        mi_column: usize,
        mode: ModeInfo,
    ) -> Result<()> {
        for plane in 0..3 {
            let subsampling_x =
                usize::from(plane != 0) * self.header.chroma_subsampling().x_shift();
            let subsampling_y =
                usize::from(plane != 0) * self.header.chroma_subsampling().y_shift();
            let (block_width, block_height) = if mode.block_size < BlockSize::B8x8 {
                (2 >> subsampling_x, 2 >> subsampling_y)
            } else {
                (
                    mode.block_size.width_4x4().div_ceil(1 << subsampling_x),
                    mode.block_size.height_4x4().div_ceil(1 << subsampling_y),
                )
            };
            let maximum_transform = floor_transform(block_width.min(block_height));
            let transform_size = mode.transform_size.min(maximum_transform);
            let step = transform_size.width_4x4();
            let origin_x = (mi_column * 2) >> subsampling_x;
            let origin_y = (mi_row * 2) >> subsampling_y;
            let plane_width_4x4 = (self.mi_columns * 2) >> subsampling_x;
            let plane_height_4x4 = (self.mi_rows * 2) >> subsampling_y;
            let usable_width = block_width.min(plane_width_4x4.saturating_sub(origin_x));
            let usable_height = block_height.min(plane_height_4x4.saturating_sub(origin_y));

            if mode.skip {
                self.clear_coefficient_contexts(
                    plane,
                    origin_x,
                    origin_y,
                    usable_width,
                    usable_height,
                );
                if let Some(picture) = &mut self.picture {
                    for row in (0..usable_height).step_by(step) {
                        for column in (0..usable_width).step_by(step) {
                            let prediction_mode = if plane == 0 && mode.block_size < BlockSize::B8x8
                            {
                                mode.sub_modes[(row << 1) + column]
                            } else if plane == 0 {
                                mode.mode
                            } else {
                                mode.uv_mode
                            };
                            picture.predict(
                                plane,
                                (origin_x + column) * 4,
                                (origin_y + row) * 4,
                                step * 4,
                                prediction_mode,
                                ((self.column_start * 2) >> subsampling_x) * 4,
                                ((self.row_start * 2) >> subsampling_y) * 4,
                                column + step < block_width,
                            );
                        }
                    }
                }
                continue;
            }
            for row in (0..usable_height).step_by(step) {
                for column in (0..usable_width).step_by(step) {
                    let luma_mode = if plane == 0 && mode.block_size < BlockSize::B8x8 {
                        mode.sub_modes[(row << 1) + column]
                    } else if plane == 0 {
                        mode.mode
                    } else {
                        mode.uv_mode
                    };
                    let transform_type = if transform_size != TransformSize::Tx32x32
                        && plane == 0
                        && !self
                            .header
                            .quantization
                            .expect("decoded frame has quantization")
                            .lossless()
                    {
                        luma_mode.transform_type()
                    } else {
                        TransformType::DctDct
                    };
                    let eob = self.read_coefficients(
                        plane,
                        origin_x + column,
                        origin_y + row,
                        transform_size,
                        transform_type,
                        mode.segment_id,
                    )?;
                    if let Some(picture) = &mut self.picture {
                        let x = (origin_x + column) * 4;
                        let y = (origin_y + row) * 4;
                        let tile_left = ((self.column_start * 2) >> subsampling_x) * 4;
                        let tile_top = ((self.row_start * 2) >> subsampling_y) * 4;
                        picture.predict(
                            plane,
                            x,
                            y,
                            step * 4,
                            luma_mode,
                            tile_left,
                            tile_top,
                            column + step < block_width,
                        );
                        if eob.eob != 0 {
                            picture.add_residual(
                                plane,
                                x,
                                y,
                                transform_size,
                                transform_type,
                                self.header
                                    .quantization
                                    .expect("decoded frame has quantization")
                                    .lossless(),
                                eob.values(),
                            );
                        }
                    }
                    self.summary.transform_blocks += 1;
                    self.summary.nonzero_transform_blocks += usize::from(eob.eob != 0);
                    self.summary.coefficients += eob.eob;
                    self.summary.coefficient_sum_abs =
                        self.summary.coefficient_sum_abs.saturating_add(
                            eob.values()
                                .iter()
                                .map(|value| u64::from(value.unsigned_abs()))
                                .sum(),
                        );
                }
            }
        }
        Ok(())
    }

    fn clear_coefficient_contexts(
        &mut self,
        plane: usize,
        origin_x: usize,
        origin_y: usize,
        width: usize,
        height: usize,
    ) {
        self.above_coefficients[plane][origin_x..origin_x + width].fill(0);
        for row in origin_y..origin_y + height {
            self.left_coefficients[plane][row & 15] = 0;
        }
    }

    fn read_coefficients(
        &mut self,
        plane: usize,
        x: usize,
        y: usize,
        transform_size: TransformSize,
        transform_type: TransformType,
        segment_id: u8,
    ) -> Result<CoefficientBlock> {
        let step = transform_size.width_4x4();
        let above_nonzero = self.above_coefficients[plane][x..x + step]
            .iter()
            .any(|&value| value != 0);
        let left_nonzero = (y..y + step).any(|row| self.left_coefficients[plane][row & 15] != 0);
        let initial_context = usize::from(above_nonzero) + usize::from(left_nonzero);
        let scan = scan_order(transform_size, transform_type);
        let dequant = dequant(self.header, plane, usize::from(segment_id));
        let coefficients = decode_coefficient_tokens(
            &mut self.bits,
            &self.context.coefficient[transform_size as usize],
            usize::from(plane != 0),
            0,
            initial_context,
            scan,
            transform_size,
            dequant,
            self.header.bit_depth(),
            self.counts
                .as_deref_mut()
                .map(|counts| &mut counts.coefficient),
        )?;
        let nonzero = u8::from(coefficients.eob != 0);
        self.above_coefficients[plane][x..x + step].fill(nonzero);
        for row in y..y + step {
            self.left_coefficients[plane][row & 15] = nonzero;
        }
        Ok(coefficients)
    }
}

pub(crate) fn read_segment_tree(bits: &mut BoolDecoder<'_>, probabilities: &[u8; 7]) -> Result<u8> {
    let first = usize::from(bits.read_bool(probabilities[0])?);
    let second_node = 1 + first;
    let second = usize::from(bits.read_bool(probabilities[second_node])?);
    let pair = first * 2 + second;
    let leaf_node = 3 + pair;
    let last = usize::from(bits.read_bool(probabilities[leaf_node])?);
    Ok((pair * 2 + last) as u8)
}

impl ModeInfo {
    fn y_mode(self, block: usize) -> IntraMode {
        if self.block_size < BlockSize::B8x8 {
            self.sub_modes[block]
        } else {
            self.mode
        }
    }
}

pub(crate) fn floor_transform(width_4x4: usize) -> TransformSize {
    if width_4x4 >= 8 {
        TransformSize::Tx32x32
    } else if width_4x4 >= 4 {
        TransformSize::Tx16x16
    } else if width_4x4 >= 2 {
        TransformSize::Tx8x8
    } else {
        TransformSize::Tx4x4
    }
}

pub(crate) fn read_intra_mode(
    bits: &mut BoolDecoder<'_>,
    probabilities: &[u8],
) -> Result<IntraMode> {
    if !bits.read_bool(probabilities[0])? {
        return Ok(IntraMode::Dc);
    }
    if !bits.read_bool(probabilities[1])? {
        return Ok(IntraMode::TrueMotion);
    }
    if !bits.read_bool(probabilities[2])? {
        return Ok(IntraMode::Vertical);
    }
    if !bits.read_bool(probabilities[3])? {
        if !bits.read_bool(probabilities[4])? {
            Ok(IntraMode::Horizontal)
        } else if !bits.read_bool(probabilities[5])? {
            Ok(IntraMode::D135)
        } else {
            Ok(IntraMode::D117)
        }
    } else if !bits.read_bool(probabilities[6])? {
        Ok(IntraMode::D45)
    } else if !bits.read_bool(probabilities[7])? {
        Ok(IntraMode::D63)
    } else if !bits.read_bool(probabilities[8])? {
        Ok(IntraMode::D153)
    } else {
        Ok(IntraMode::D207)
    }
}

#[derive(Clone, Copy)]
pub(crate) struct CoefficientScan {
    scan: &'static [u16],
    neighbors: &'static [u16],
}

pub(crate) fn scan_order(size: TransformSize, transform: TransformType) -> CoefficientScan {
    match (size, transform) {
        (TransformSize::Tx4x4, TransformType::AdstDct) => CoefficientScan {
            scan: &tables::ROW_SCAN_4X4,
            neighbors: &tables::ROW_NEIGHBORS_4X4,
        },
        (TransformSize::Tx4x4, TransformType::DctAdst) => CoefficientScan {
            scan: &tables::COL_SCAN_4X4,
            neighbors: &tables::COL_NEIGHBORS_4X4,
        },
        (TransformSize::Tx4x4, _) => CoefficientScan {
            scan: &tables::SCAN_4X4,
            neighbors: &tables::NEIGHBORS_4X4,
        },
        (TransformSize::Tx8x8, TransformType::AdstDct) => CoefficientScan {
            scan: &tables::ROW_SCAN_8X8,
            neighbors: &tables::ROW_NEIGHBORS_8X8,
        },
        (TransformSize::Tx8x8, TransformType::DctAdst) => CoefficientScan {
            scan: &tables::COL_SCAN_8X8,
            neighbors: &tables::COL_NEIGHBORS_8X8,
        },
        (TransformSize::Tx8x8, _) => CoefficientScan {
            scan: &tables::SCAN_8X8,
            neighbors: &tables::NEIGHBORS_8X8,
        },
        (TransformSize::Tx16x16, TransformType::AdstDct) => CoefficientScan {
            scan: &tables::ROW_SCAN_16X16,
            neighbors: &tables::ROW_NEIGHBORS_16X16,
        },
        (TransformSize::Tx16x16, TransformType::DctAdst) => CoefficientScan {
            scan: &tables::COL_SCAN_16X16,
            neighbors: &tables::COL_NEIGHBORS_16X16,
        },
        (TransformSize::Tx16x16, _) => CoefficientScan {
            scan: &tables::SCAN_16X16,
            neighbors: &tables::NEIGHBORS_16X16,
        },
        (TransformSize::Tx32x32, _) => CoefficientScan {
            scan: &tables::SCAN_32X32,
            neighbors: &tables::NEIGHBORS_32X32,
        },
    }
}

#[derive(Debug)]
pub(crate) struct CoefficientBlock {
    values: CoefficientValues,
    pub(crate) eob: usize,
}

impl CoefficientBlock {
    pub(crate) fn values(&self) -> &[i32] {
        self.values.as_slice()
    }
}

#[derive(Debug)]
// Tx16 deliberately stays inline: it avoids a heap allocation for a common
// transform while still reducing the previous fixed 4 KiB coefficient block.
#[allow(clippy::large_enum_variant)]
enum CoefficientValues {
    Tx4([i32; 4 * 4]),
    Tx8([i32; 8 * 8]),
    Tx16([i32; 16 * 16]),
    Tx32(Box<[i32; 32 * 32]>),
}

impl CoefficientValues {
    fn new(transform_size: TransformSize) -> Self {
        match transform_size {
            TransformSize::Tx4x4 => Self::Tx4([0; 4 * 4]),
            TransformSize::Tx8x8 => Self::Tx8([0; 8 * 8]),
            TransformSize::Tx16x16 => Self::Tx16([0; 16 * 16]),
            // Keep the uncommon largest transform off the tile decoder's
            // stack while preserving exact-size initialization.
            TransformSize::Tx32x32 => Self::Tx32(Box::new([0; 32 * 32])),
        }
    }

    fn as_slice(&self) -> &[i32] {
        match self {
            Self::Tx4(values) => values,
            Self::Tx8(values) => values,
            Self::Tx16(values) => values,
            Self::Tx32(values) => values.as_slice(),
        }
    }

    fn as_mut_slice(&mut self) -> &mut [i32] {
        match self {
            Self::Tx4(values) => values,
            Self::Tx8(values) => values,
            Self::Tx16(values) => values,
            Self::Tx32(values) => values.as_mut_slice(),
        }
    }
}

enum TokenCache {
    Tx4([u8; 4 * 4]),
    Tx8([u8; 8 * 8]),
    Tx16([u8; 16 * 16]),
    Tx32(Box<[u8; 32 * 32]>),
}

impl TokenCache {
    fn new(transform_size: TransformSize) -> Self {
        match transform_size {
            TransformSize::Tx4x4 => Self::Tx4([0; 4 * 4]),
            TransformSize::Tx8x8 => Self::Tx8([0; 8 * 8]),
            TransformSize::Tx16x16 => Self::Tx16([0; 16 * 16]),
            TransformSize::Tx32x32 => Self::Tx32(Box::new([0; 32 * 32])),
        }
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        match self {
            Self::Tx4(values) => values,
            Self::Tx8(values) => values,
            Self::Tx16(values) => values,
            Self::Tx32(values) => values.as_mut_slice(),
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_coefficient_tokens(
    bits: &mut BoolDecoder<'_>,
    probabilities: &[u8; 396],
    plane_type: usize,
    reference_type: usize,
    mut context: usize,
    scan: CoefficientScan,
    transform_size: TransformSize,
    dequant: [i32; 2],
    bit_depth: BitDepth,
    mut counts: Option<&mut CoefficientCounts>,
) -> Result<CoefficientBlock> {
    let maximum = transform_size.coefficient_count();
    let transform_index = transform_size as usize;
    let family = plane_type * 2 + reference_type;
    let quant_shift = u32::from(transform_size == TransformSize::Tx32x32);
    let mut token_cache_storage = TokenCache::new(transform_size);
    let token_cache = token_cache_storage.as_mut_slice();
    let mut value_storage = CoefficientValues::new(transform_size);
    let values = value_storage.as_mut_slice();
    let mut coefficient = 0usize;
    while coefficient < maximum {
        let mut band = coefficient_band(transform_size, coefficient);
        let mut model_start = coefficient_model_start(family, band, context);
        let mut model = &probabilities[model_start..model_start + 3];
        if let Some(counts) = &mut counts {
            counts
                .model_index_mut(transform_index, model_start / 3)
                .record_eob_branch();
        }
        if !bits.read_bool(model[0])? {
            if let Some(counts) = &mut counts {
                counts
                    .model_index_mut(transform_index, model_start / 3)
                    .record_token(3);
            }
            return Ok(CoefficientBlock {
                values: value_storage,
                eob: coefficient,
            });
        }
        while !bits.read_bool(model[1])? {
            if let Some(counts) = &mut counts {
                counts
                    .model_index_mut(transform_index, model_start / 3)
                    .record_token(0);
            }
            token_cache[usize::from(scan.scan[coefficient])] = 0;
            coefficient += 1;
            if coefficient == maximum {
                return Ok(CoefficientBlock {
                    values: value_storage,
                    eob: coefficient,
                });
            }
            context = coefficient_context(token_cache, scan.neighbors, coefficient);
            band = coefficient_band(transform_size, coefficient);
            model_start = coefficient_model_start(family, band, context);
            model = &probabilities[model_start..model_start + 3];
        }
        let (energy, magnitude) = if !bits.read_bool(model[2])? {
            if let Some(counts) = &mut counts {
                counts
                    .model_index_mut(transform_index, model_start / 3)
                    .record_token(1);
            }
            (1, 1)
        } else {
            if let Some(counts) = &mut counts {
                counts
                    .model_index_mut(transform_index, model_start / 3)
                    .record_token(2);
            }
            decode_large_coefficient(bits, model[2], bit_depth)?
        };
        let raster = usize::from(scan.scan[coefficient]);
        token_cache[raster] = energy;
        let quant = dequant[usize::from(coefficient != 0)];
        let value = (magnitude * quant) >> quant_shift;
        values[raster] = if bits.read_bit()? { -value } else { value };
        coefficient += 1;
        if coefficient < maximum {
            context = coefficient_context(token_cache, scan.neighbors, coefficient);
        }
    }
    Ok(CoefficientBlock {
        values: value_storage,
        eob: coefficient,
    })
}

#[inline(always)]
fn coefficient_model_start(family: usize, band: usize, context: usize) -> usize {
    let band_offset = if band == 0 { 0 } else { 9 + (band - 1) * 18 };
    family * 99 + band_offset + context * 3
}

static COEFFICIENT_BANDS_4X4: [u8; 16] = [0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5];
static COEFFICIENT_BANDS_LARGE: [u8; 22] = [
    0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5,
];

#[inline(always)]
fn coefficient_band(size: TransformSize, coefficient: usize) -> usize {
    if size == TransformSize::Tx4x4 {
        usize::from(COEFFICIENT_BANDS_4X4[coefficient])
    } else {
        COEFFICIENT_BANDS_LARGE
            .get(coefficient)
            .copied()
            .map_or(5, usize::from)
    }
}

#[inline(always)]
fn coefficient_context(token_cache: &[u8], neighbors: &[u16], coefficient: usize) -> usize {
    let first = usize::from(neighbors[coefficient * 2]);
    let second = usize::from(neighbors[coefficient * 2 + 1]);
    (usize::from(token_cache[first]) + usize::from(token_cache[second]) + 1) >> 1
}

fn decode_large_coefficient(
    bits: &mut BoolDecoder<'_>,
    pivot_probability: u8,
    bit_depth: BitDepth,
) -> Result<(u8, i32)> {
    let pivot = usize::from(pivot_probability.saturating_sub(1));
    let probabilities = &tables::PARETO8[pivot * 8..pivot * 8 + 8];
    if !bits.read_bool(probabilities[0])? {
        if bits.read_bool(probabilities[1])? {
            Ok((3, 3 + i32::from(bits.read_bool(probabilities[2])?)))
        } else {
            Ok((2, 2))
        }
    } else if !bits.read_bool(probabilities[3])? {
        if bits.read_bool(probabilities[4])? {
            Ok((4, 7 + read_category(bits, &tables::CAT2)?))
        } else {
            Ok((4, 5 + read_category(bits, &tables::CAT1)?))
        }
    } else {
        if bits.read_bool(probabilities[5])? {
            if bits.read_bool(probabilities[7])? {
                Ok((5, 67 + read_category(bits, cat6_probabilities(bit_depth))?))
            } else {
                Ok((5, 35 + read_category(bits, &tables::CAT5)?))
            }
        } else if bits.read_bool(probabilities[6])? {
            Ok((5, 19 + read_category(bits, &tables::CAT4)?))
        } else {
            Ok((5, 11 + read_category(bits, &tables::CAT3)?))
        }
    }
}

fn cat6_probabilities(bit_depth: BitDepth) -> &'static [u8] {
    match bit_depth {
        BitDepth::Eight => &tables::CAT6,
        BitDepth::Ten => &tables::CAT6_HIGH_12[2..],
        BitDepth::Twelve => &tables::CAT6_HIGH_12,
    }
}

fn read_category(bits: &mut BoolDecoder<'_>, probabilities: &[u8]) -> Result<i32> {
    let mut value = 0;
    for &probability in probabilities {
        value = value << 1 | i32::from(bits.read_bool(probability)?);
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::{cat6_probabilities, tile_offsets};
    use crate::BitDepth;

    #[test]
    fn aligns_tile_boundaries_to_superblocks() {
        assert_eq!(tile_offsets(0, 480, 3), (0, 56));
        assert_eq!(tile_offsets(1, 480, 3), (56, 120));
        assert_eq!(tile_offsets(7, 480, 3), (416, 480));
    }

    #[test]
    fn category_six_probability_width_tracks_sample_depth() {
        assert_eq!(cat6_probabilities(BitDepth::Eight).len(), 14);
        assert_eq!(cat6_probabilities(BitDepth::Ten).len(), 16);
        assert_eq!(cat6_probabilities(BitDepth::Twelve).len(), 18);
        assert_eq!(cat6_probabilities(BitDepth::Ten)[..2], [255, 255]);
        assert_eq!(
            &cat6_probabilities(BitDepth::Twelve)[4..],
            cat6_probabilities(BitDepth::Eight)
        );
    }
}
