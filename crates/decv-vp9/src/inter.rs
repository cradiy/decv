use crate::{
    CompressedHeader, FrameHeader, InterpolationFilter, ReferenceMode, Result, TransformMode,
    Vp9Error,
    block::{BlockSize, IntraMode, Partition, TransformSize, TransformType},
    bool_decoder::BoolDecoder,
    context::{FrameCounts, MotionVectorComponentCounts, MotionVectorCounts, ProbabilityContext},
    loop_filter::{FilterMode, FilterModeMap, apply_loop_filter},
    quantization::dequant,
    reconstruct::IntraPicture,
    tables,
    tile::{
        TileLayout, decode_coefficient_tokens, floor_transform, read_intra_mode, read_segment_tree,
        scan_order, tile_offsets,
    },
};
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct InterSyntaxSummary {
    pub blocks: usize,
    pub intra_blocks: usize,
    pub inter_blocks: usize,
    pub compound_blocks: usize,
    pub new_motion_vectors: usize,
    pub transform_blocks: usize,
    pub nonzero_transform_blocks: usize,
    pub coefficients: usize,
    pub padded_bytes: usize,
}

impl InterSyntaxSummary {
    pub fn parse(
        frame: &[u8],
        header: &FrameHeader,
        compressed: &CompressedHeader,
    ) -> Result<Self> {
        if header.intra_only {
            return Err(Vp9Error::UnsupportedFeature(
                "inter syntax parser received an intra-only frame",
            ));
        }
        let mut context = ProbabilityContext::default();
        context.apply(compressed)?;
        parse_inter_syntax(
            frame, header, compressed, &context, None, None, None, None, None,
        )
        .map(|(summary, _)| summary)
    }
}

/// Decodes and reconstructs one inter frame against LAST, GOLDEN, and ALTREF
/// pictures in that order, preserving the coded sample depth.
pub fn decode_inter_picture(
    frame: &[u8],
    header: &FrameHeader,
    compressed: &CompressedHeader,
    references: [&IntraPicture; 3],
) -> Result<IntraPicture> {
    if header.intra_only {
        return Err(Vp9Error::UnsupportedFeature(
            "inter picture decoder received an intra-only frame",
        ));
    }
    let size = header
        .size
        .ok_or(Vp9Error::InvalidData("frame has no dimensions"))?;
    let width = usize::try_from(size.width).map_err(|_| Vp9Error::IntegerOverflow)?;
    let height = usize::try_from(size.height).map_err(|_| Vp9Error::IntegerOverflow)?;
    if references.iter().any(|reference| {
        reference.subsampling() != header.chroma_subsampling()
            || reference.bit_depth() != header.bit_depth()
            || !valid_reference_size(reference, width, height)
    }) {
        return Err(Vp9Error::UnsupportedFeature(
            "VP9 reference layout is incompatible with the current frame",
        ));
    }
    let mut context = ProbabilityContext::default();
    context.apply(compressed)?;
    let mut picture = IntraPicture::new(
        width,
        height,
        header.chroma_subsampling(),
        header.bit_depth(),
    );
    let (_, modes) = parse_inter_syntax(
        frame,
        header,
        compressed,
        &context,
        Some(&mut picture),
        Some(references),
        None,
        None,
        None,
    )?;
    apply_loop_filter(&mut picture, header, &modes.loop_filter_map()?)?;
    Ok(picture)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_inter_picture_with_context(
    frame: &[u8],
    header: &FrameHeader,
    compressed: &CompressedHeader,
    context: &ProbabilityContext,
    references: [&IntraPicture; 3],
    counts: &mut FrameCounts,
    previous_modes: Option<&InterModeMap>,
    previous_segment_ids: Option<&[u8]>,
) -> Result<(IntraPicture, InterModeMap)> {
    let size = header
        .size
        .ok_or(Vp9Error::InvalidData("frame has no dimensions"))?;
    let width = usize::try_from(size.width).map_err(|_| Vp9Error::IntegerOverflow)?;
    let height = usize::try_from(size.height).map_err(|_| Vp9Error::IntegerOverflow)?;
    if references.iter().any(|reference| {
        reference.subsampling() != header.chroma_subsampling()
            || reference.bit_depth() != header.bit_depth()
            || !valid_reference_size(reference, width, height)
    }) {
        return Err(Vp9Error::UnsupportedFeature(
            "VP9 reference layout is incompatible with the current frame",
        ));
    }
    let mut picture = IntraPicture::new(
        width,
        height,
        header.chroma_subsampling(),
        header.bit_depth(),
    );
    let (_, modes) = parse_inter_syntax(
        frame,
        header,
        compressed,
        context,
        Some(&mut picture),
        Some(references),
        Some(counts),
        previous_modes,
        previous_segment_ids,
    )?;
    apply_loop_filter(&mut picture, header, &modes.loop_filter_map()?)?;
    Ok((picture, modes))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum ReferenceFrame {
    Intra,
    Last,
    Golden,
    Alt,
    None,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InterMode {
    Nearest,
    Near,
    Zero,
    New,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct MotionVector {
    row: i16,
    column: i16,
}

impl MotionVector {
    const ZERO: Self = Self { row: 0, column: 0 };

    fn scaled_for_reference(
        self,
        source: ReferenceFrame,
        target: ReferenceFrame,
        header: &FrameHeader,
    ) -> Self {
        if reference_bias(header, source) == reference_bias(header, target) {
            self
        } else {
            Self {
                row: self.row.saturating_neg(),
                column: self.column.saturating_neg(),
            }
        }
    }

    fn lower_precision(self, allow_high_precision: bool) -> Self {
        if allow_high_precision && self.uses_high_precision() {
            return self;
        }
        Self {
            row: round_to_even(self.row),
            column: round_to_even(self.column),
        }
    }

    fn uses_high_precision(self) -> bool {
        self.row.unsigned_abs() < 64 && self.column.unsigned_abs() < 64
    }
}

fn round_to_even(value: i16) -> i16 {
    if value & 1 == 0 {
        value
    } else if value > 0 {
        value - 1
    } else {
        value + 1
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum Interpolation {
    EightTap,
    Smooth,
    Sharp,
    /// The fourth switchable-filter probability context used by intra blocks
    /// or disagreeing neighbors. This is not a reconstruction filter.
    Sentinel,
    Bilinear,
}

#[derive(Debug, Clone, Copy)]
struct ModeInfo {
    block_size: BlockSize,
    segment_id: u8,
    segment_id_predicted: bool,
    skip: bool,
    transform_size: TransformSize,
    is_inter: bool,
    intra_mode: IntraMode,
    sub_intra_modes: [IntraMode; 4],
    uv_mode: IntraMode,
    inter_mode: InterMode,
    references: [ReferenceFrame; 2],
    interpolation: Interpolation,
    motion_vectors: [MotionVector; 2],
    sub_motion_vectors: [[MotionVector; 2]; 4],
}

impl ModeInfo {
    fn has_second_reference(self) -> bool {
        self.references[1] != ReferenceFrame::None
    }
}

#[derive(Debug, Clone)]
pub(crate) struct InterModeMap {
    mi_columns: usize,
    mi_rows: usize,
    modes: Vec<Option<ModeInfo>>,
    segment_ids: Vec<u8>,
}

impl InterModeMap {
    pub(crate) fn intra(width: u32, height: u32, segment_ids: Vec<u8>) -> Result<Self> {
        let mi_columns =
            usize::try_from(width.div_ceil(8)).map_err(|_| Vp9Error::IntegerOverflow)?;
        let mi_rows = usize::try_from(height.div_ceil(8)).map_err(|_| Vp9Error::IntegerOverflow)?;
        if segment_ids.len() != mi_columns.saturating_mul(mi_rows) {
            return Err(Vp9Error::InvalidData(
                "intra segment map dimensions do not match the frame",
            ));
        }
        Ok(Self {
            mi_columns,
            mi_rows,
            modes: vec![None; mi_columns * mi_rows],
            segment_ids,
        })
    }

    pub(crate) fn segment_ids(&self) -> &[u8] {
        &self.segment_ids
    }

    fn loop_filter_map(&self) -> Result<FilterModeMap> {
        let modes = self
            .modes
            .iter()
            .map(|mode| {
                let mode = mode.ok_or(Vp9Error::InvalidData(
                    "inter mode map has an undecoded block",
                ))?;
                let reference = match mode.references[0] {
                    ReferenceFrame::Intra => 0,
                    ReferenceFrame::Last => 1,
                    ReferenceFrame::Golden => 2,
                    ReferenceFrame::Alt => 3,
                    ReferenceFrame::None => {
                        return Err(Vp9Error::InvalidData("primary reference frame is missing"));
                    }
                };
                let mode_class = u8::from(mode.is_inter && mode.inter_mode != InterMode::Zero);
                Ok(FilterMode {
                    block_size: mode.block_size,
                    transform_size: mode.transform_size,
                    skip: mode.skip,
                    segment_id: mode.segment_id,
                    reference,
                    mode_class,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        FilterModeMap::new(self.mi_columns, self.mi_rows, modes)
    }
}

#[allow(clippy::too_many_arguments)]
fn parse_inter_syntax(
    frame: &[u8],
    header: &FrameHeader,
    compressed: &CompressedHeader,
    context: &ProbabilityContext,
    mut picture: Option<&mut IntraPicture>,
    references: Option<[&IntraPicture; 3]>,
    mut counts: Option<&mut FrameCounts>,
    previous_modes: Option<&InterModeMap>,
    previous_segment_ids: Option<&[u8]>,
) -> Result<(InterSyntaxSummary, InterModeMap)> {
    let size = header
        .size
        .ok_or(Vp9Error::InvalidData("frame has no dimensions"))?;
    let mi_columns =
        usize::try_from(size.width.div_ceil(8)).map_err(|_| Vp9Error::IntegerOverflow)?;
    let mi_rows =
        usize::try_from(size.height.div_ceil(8)).map_err(|_| Vp9Error::IntegerOverflow)?;
    if previous_modes
        .is_some_and(|previous| previous.mi_columns != mi_columns || previous.mi_rows != mi_rows)
    {
        return Err(Vp9Error::InvalidData(
            "previous mode map dimensions do not match the frame",
        ));
    }
    if previous_segment_ids.is_some_and(|segments| segments.len() != mi_columns * mi_rows) {
        return Err(Vp9Error::InvalidData(
            "previous segment map dimensions do not match the frame",
        ));
    }
    let layout = TileLayout::parse(frame, header)?;
    let mut modes = vec![None; mi_columns * mi_rows];
    let mut segment_ids = vec![0; mi_columns * mi_rows];
    let mut summary = InterSyntaxSummary::default();

    if layout.rows() == 1
        && layout.columns() > 1
        && let Some(picture) = picture.take()
    {
        return parse_inter_tiles_parallel(
            frame,
            header,
            compressed,
            context,
            references,
            counts,
            previous_modes,
            previous_segment_ids,
            &layout,
            picture,
            mi_columns,
            mi_rows,
        );
    }

    for (tile_index, tile) in layout.tiles(frame).enumerate() {
        let tile_row = tile_index / layout.columns();
        let tile_column = tile_index % layout.columns();
        let (row_start, row_end) = tile_offsets(tile_row, mi_rows, header.tile_rows_log2);
        let (column_start, column_end) =
            tile_offsets(tile_column, mi_columns, header.tile_columns_log2);
        let mut decoder = InterTileDecoder::new(
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
            &mut segment_ids,
            picture.as_deref_mut(),
            references,
            counts.as_deref_mut(),
            previous_modes.map(|modes| modes.modes.as_slice()),
            previous_segment_ids,
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
    Ok((
        summary,
        InterModeMap {
            mi_columns,
            mi_rows,
            modes,
            segment_ids,
        },
    ))
}

#[allow(clippy::too_many_arguments)]
fn parse_inter_tiles_parallel(
    frame: &[u8],
    header: &FrameHeader,
    compressed: &CompressedHeader,
    context: &ProbabilityContext,
    references: Option<[&IntraPicture; 3]>,
    mut counts: Option<&mut FrameCounts>,
    previous_modes: Option<&InterModeMap>,
    previous_segment_ids: Option<&[u8]>,
    layout: &TileLayout,
    picture: &mut IntraPicture,
    mi_columns: usize,
    mi_rows: usize,
) -> Result<(InterSyntaxSummary, InterModeMap)> {
    struct TileResult {
        summary: InterSyntaxSummary,
        modes: Vec<Option<ModeInfo>>,
        segment_ids: Vec<u8>,
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
                .name(format!("decv-vp9-tile-{tile_index}"))
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
                    let mut tile_segment_ids = vec![0; mi_columns * mi_rows];
                    let mut tile_counts = FrameCounts::default();
                    let mut decoder = InterTileDecoder::new(
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
                        &mut tile_segment_ids,
                        Some(&mut tile_picture),
                        references,
                        collect_counts.then_some(&mut tile_counts),
                        previous_modes.map(|modes| modes.modes.as_slice()),
                        previous_segment_ids,
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
                        segment_ids: tile_segment_ids,
                        counts: tile_counts,
                        picture: tile_picture,
                        column_start,
                        column_end,
                    })
                })
                .map_err(|_| Vp9Error::InvalidData("failed to spawn VP9 tile worker"))?;
            workers.push(worker);
        }
        workers
            .into_iter()
            .map(|worker| {
                worker
                    .join()
                    .map_err(|_| Vp9Error::InvalidData("VP9 tile worker panicked"))?
            })
            .collect::<Result<Vec<_>>>()
    })?;

    let mut summary = InterSyntaxSummary::default();
    let mut modes = vec![None; mi_columns * mi_rows];
    let mut segment_ids = vec![0; mi_columns * mi_rows];
    for tile in tile_results {
        summary += tile.summary;
        if let Some(counts) = &mut counts {
            counts.merge_from(&tile.counts);
        }
        for row in 0..mi_rows {
            let start = row * mi_columns + tile.column_start;
            let end = row * mi_columns + tile.column_end;
            modes[start..end].copy_from_slice(&tile.modes[start..end]);
            segment_ids[start..end].copy_from_slice(&tile.segment_ids[start..end]);
        }
        picture.copy_strip_from(&tile.picture);
    }
    Ok((
        summary,
        InterModeMap {
            mi_columns,
            mi_rows,
            modes,
            segment_ids,
        },
    ))
}

impl std::ops::AddAssign for InterSyntaxSummary {
    fn add_assign(&mut self, rhs: Self) {
        self.blocks += rhs.blocks;
        self.intra_blocks += rhs.intra_blocks;
        self.inter_blocks += rhs.inter_blocks;
        self.compound_blocks += rhs.compound_blocks;
        self.new_motion_vectors += rhs.new_motion_vectors;
        self.transform_blocks += rhs.transform_blocks;
        self.nonzero_transform_blocks += rhs.nonzero_transform_blocks;
        self.coefficients += rhs.coefficients;
        self.padded_bytes += rhs.padded_bytes;
    }
}

struct InterTileDecoder<'a, 'state> {
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
    segment_ids: &'state mut [u8],
    above_partition: Vec<u8>,
    left_partition: [u8; 8],
    above_coefficients: [Vec<u8>; 3],
    left_coefficients: [[u8; 16]; 3],
    summary: InterSyntaxSummary,
    picture: Option<&'state mut IntraPicture>,
    references: Option<[&'state IntraPicture; 3]>,
    counts: Option<&'state mut FrameCounts>,
    previous_modes: Option<&'state [Option<ModeInfo>]>,
    previous_segment_ids: Option<&'state [u8]>,
}

impl<'a, 'state> InterTileDecoder<'a, 'state> {
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
        segment_ids: &'state mut [u8],
        picture: Option<&'state mut IntraPicture>,
        references: Option<[&'state IntraPicture; 3]>,
        counts: Option<&'state mut FrameCounts>,
        previous_modes: Option<&'state [Option<ModeInfo>]>,
        previous_segment_ids: Option<&'state [u8]>,
    ) -> Result<Self> {
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
            segment_ids,
            above_partition: vec![0; mi_columns],
            left_partition: [0; 8],
            above_coefficients: {
                let luma_columns = mi_columns * 2;
                let chroma_columns = luma_columns >> header.chroma_subsampling().x_shift();
                [
                    vec![0; luma_columns],
                    vec![0; chroma_columns],
                    vec![0; chroma_columns],
                ]
            },
            left_coefficients: [[0; 16]; 3],
            summary: InterSyntaxSummary::default(),
            picture,
            references,
            counts,
            previous_modes,
            previous_segment_ids,
        })
    }

    fn parse(&mut self) -> Result<InterSyntaxSummary> {
        for mi_row in (self.row_start..self.row_end).step_by(8) {
            self.left_partition.fill(0);
            for contexts in &mut self.left_coefficients {
                contexts.fill(0);
            }
            for mi_column in (self.column_start..self.column_end).step_by(8) {
                self.decode_partition(mi_row, mi_column, BlockSize::B64x64, 3)?;
            }
        }
        self.summary.padded_bytes = self.bits.padded_bytes();
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
            let (above, left) = subsize.partition_context();
            for column in mi_column..(mi_column + width_mi).min(self.column_end) {
                self.above_partition[column] = above;
            }
            for row in mi_row..(mi_row + width_mi).min(self.row_end) {
                self.left_partition[row & 7] = left;
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
        let probabilities = &self.context.partition[probability_context * 3..][..3];
        let partition = if has_rows && has_columns {
            if !self.bits.read_bool(probabilities[0])? {
                Partition::None
            } else if !self.bits.read_bool(probabilities[1])? {
                Partition::Horizontal
            } else if !self.bits.read_bool(probabilities[2])? {
                Partition::Vertical
            } else {
                Partition::Split
            }
        } else if !has_rows && has_columns {
            if self.bits.read_bool(probabilities[1])? {
                Partition::Split
            } else {
                Partition::Horizontal
            }
        } else if has_rows && !has_columns {
            if self.bits.read_bool(probabilities[2])? {
                Partition::Split
            } else {
                Partition::Vertical
            }
        } else {
            Partition::Split
        };
        if let Some(counts) = &mut self.counts {
            counts.partition[probability_context][partition as usize] += 1;
        }
        Ok(partition)
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
        self.summary.intra_blocks += usize::from(!mode.is_inter);
        self.summary.inter_blocks += usize::from(mode.is_inter);
        self.summary.compound_blocks += usize::from(mode.has_second_reference());
        if mode.is_inter && self.picture.is_some() {
            self.predict_inter_block(mi_row, mi_column, mode)?;
        }
        self.read_transform_blocks(mi_row, mi_column, mode)?;
        if self.bits.padded_bytes() > 8 {
            return Err(Vp9Error::InvalidData(
                "inter tile syntax reads beyond its coded partition",
            ));
        }
        Ok(())
    }

    fn predict_inter_block(
        &mut self,
        mi_row: usize,
        mi_column: usize,
        mode: ModeInfo,
    ) -> Result<()> {
        let references = self
            .references
            .ok_or(Vp9Error::MissingReference(usize::MAX))?;
        let picture = self
            .picture
            .as_deref_mut()
            .expect("caller checked inter reconstruction");
        let kernel = interpolation_kernel(mode.interpolation);
        let reference_count = 1 + usize::from(mode.has_second_reference());

        for reference_index in 0..reference_count {
            let reference_type = mode.references[reference_index];
            let reference = references[match reference_type {
                ReferenceFrame::Last => 0,
                ReferenceFrame::Golden => 1,
                ReferenceFrame::Alt => 2,
                _ => {
                    return Err(Vp9Error::InvalidData(
                        "inter block selected a non-inter reference",
                    ));
                }
            }];
            for plane in 0..3 {
                let subsampling_x =
                    usize::from(plane != 0) * self.header.chroma_subsampling().x_shift();
                let subsampling_y =
                    usize::from(plane != 0) * self.header.chroma_subsampling().y_shift();
                let origin_x = (mi_column * 8) >> subsampling_x;
                let origin_y = (mi_row * 8) >> subsampling_y;
                if mode.block_size < BlockSize::B8x8 {
                    let blocks_wide = 2 >> subsampling_x;
                    let blocks_high = 2 >> subsampling_y;
                    for block_y in 0..blocks_high {
                        for block_x in 0..blocks_wide {
                            let motion = split_motion_vector(
                                &mode.sub_motion_vectors,
                                reference_index,
                                subsampling_x,
                                subsampling_y,
                                block_y * blocks_wide + block_x,
                            );
                            picture.predict_inter(
                                reference,
                                plane,
                                origin_x + block_x * 4,
                                origin_y + block_y * 4,
                                4,
                                4,
                                i32::from(motion.row) << (1 - subsampling_y),
                                i32::from(motion.column) << (1 - subsampling_x),
                                kernel,
                                reference_index != 0,
                            );
                        }
                    }
                } else {
                    let width = (mode.block_size.width_4x4() * 4) >> subsampling_x;
                    let height = (mode.block_size.height_4x4() * 4) >> subsampling_y;
                    let motion = mode.motion_vectors[reference_index];
                    picture.predict_inter(
                        reference,
                        plane,
                        origin_x,
                        origin_y,
                        width,
                        height,
                        i32::from(motion.row) << (1 - subsampling_y),
                        i32::from(motion.column) << (1 - subsampling_x),
                        kernel,
                        reference_index != 0,
                    );
                }
            }
        }
        Ok(())
    }

    fn read_mode_info(
        &mut self,
        mi_row: usize,
        mi_column: usize,
        block_size: BlockSize,
    ) -> Result<ModeInfo> {
        let above = self.mode_above(mi_row, mi_column);
        let left = self.mode_left(mi_row, mi_column);
        let (segment_id, segment_id_predicted) =
            self.read_segment_id(mi_row, mi_column, block_size, above, left)?;
        let segment_skip = segment_feature(self.header, segment_id, 3).is_some();
        let skip_context = usize::from(above.is_some_and(|mode| mode.skip))
            + usize::from(left.is_some_and(|mode| mode.skip));
        let skip = if segment_skip {
            true
        } else {
            let skip = self.bits.read_bool(self.context.skip[skip_context])?;
            if let Some(counts) = &mut self.counts {
                counts.skip[skip_context][usize::from(skip)] += 1;
            }
            skip
        };
        let is_inter = if let Some(reference) = segment_feature(self.header, segment_id, 2) {
            reference != 0
        } else {
            let intra_inter_context = intra_inter_context(above, left);
            let is_inter = self
                .bits
                .read_bool(self.context.intra_inter[intra_inter_context])?;
            if let Some(counts) = &mut self.counts {
                counts.intra_inter[intra_inter_context][usize::from(is_inter)] += 1;
            }
            is_inter
        };
        let transform_size =
            self.read_transform_size(block_size, !skip || !is_inter, above, left)?;

        let mut mode = ModeInfo {
            block_size,
            segment_id,
            segment_id_predicted,
            skip,
            transform_size,
            is_inter,
            intra_mode: IntraMode::Dc,
            sub_intra_modes: [IntraMode::Dc; 4],
            uv_mode: IntraMode::Dc,
            inter_mode: InterMode::Zero,
            references: [ReferenceFrame::Intra, ReferenceFrame::None],
            interpolation: Interpolation::Sentinel,
            motion_vectors: [MotionVector::ZERO; 2],
            sub_motion_vectors: [[MotionVector::ZERO; 2]; 4],
        };
        if is_inter {
            mode.references = self.read_references(segment_id, above, left)?;
            let mode_context = self.inter_mode_context(mi_row, mi_column, block_size);
            if segment_skip {
                if block_size < BlockSize::B8x8 {
                    return Err(Vp9Error::InvalidData(
                        "segment skip feature is invalid for sub-8x8 blocks",
                    ));
                }
            } else if block_size >= BlockSize::B8x8 {
                mode.inter_mode =
                    read_inter_mode(&mut self.bits, &self.context.inter_mode[mode_context * 3..])?;
                if let Some(counts) = &mut self.counts {
                    counts.inter_mode[mode_context][inter_mode_symbol(mode.inter_mode)] += 1;
                }
            }
            mode.interpolation = self.read_interpolation(above, left)?;
            if block_size < BlockSize::B8x8 {
                self.read_sub8_motion_vectors(mi_row, mi_column, &mut mode, mode_context)?;
            } else {
                self.read_block_motion_vectors(mi_row, mi_column, &mut mode)?;
            }
        } else {
            let group = size_group(block_size);
            match block_size {
                BlockSize::B4x4 => {
                    for index in 0..4 {
                        mode.sub_intra_modes[index] = self.read_inter_intra_mode(group)?;
                    }
                }
                BlockSize::B4x8 => {
                    let first = self.read_inter_intra_mode(group)?;
                    let second = self.read_inter_intra_mode(group)?;
                    mode.sub_intra_modes = [first, second, first, second];
                }
                BlockSize::B8x4 => {
                    let first = self.read_inter_intra_mode(group)?;
                    let second = self.read_inter_intra_mode(group)?;
                    mode.sub_intra_modes = [first, first, second, second];
                }
                _ => {
                    let selected = self.read_inter_intra_mode(group)?;
                    mode.sub_intra_modes.fill(selected);
                }
            }
            mode.intra_mode = mode.sub_intra_modes[3];
            mode.uv_mode = read_intra_mode(
                &mut self.bits,
                &self.context.uv_mode[mode.intra_mode as usize * 9..][..9],
            )?;
            if let Some(counts) = &mut self.counts {
                counts.uv_mode[mode.intra_mode as usize][mode.uv_mode as usize] += 1;
            }
        }
        Ok(mode)
    }

    fn read_segment_id(
        &mut self,
        mi_row: usize,
        mi_column: usize,
        block_size: BlockSize,
        above: Option<ModeInfo>,
        left: Option<ModeInfo>,
    ) -> Result<(u8, bool)> {
        let row_end = (mi_row + block_size.height_mi()).min(self.mi_rows);
        let column_end = (mi_column + block_size.width_mi()).min(self.mi_columns);
        let Some(segmentation) = self
            .header
            .segmentation
            .as_ref()
            .filter(|segmentation| segmentation.enabled)
        else {
            return Ok((0, false));
        };

        let mi_columns = self.mi_columns;
        let predicted = self.previous_segment_ids.map_or(0, |segments| {
            (mi_row..row_end)
                .flat_map(|row| {
                    (mi_column..column_end).map(move |column| segments[row * mi_columns + column])
                })
                .min()
                .unwrap_or(0)
        });

        if !segmentation.update_map {
            for row in mi_row..row_end {
                for column in mi_column..column_end {
                    let index = row * self.mi_columns + column;
                    self.segment_ids[index] = self
                        .previous_segment_ids
                        .map_or(0, |segments| segments[index]);
                }
            }
            return Ok((predicted, false));
        }

        let segment_id_predicted = if segmentation.temporal_update {
            let context = usize::from(above.is_some_and(|mode| mode.segment_id_predicted))
                + usize::from(left.is_some_and(|mode| mode.segment_id_predicted));
            self.bits
                .read_bool(segmentation.prediction_probabilities[context])?
        } else {
            false
        };
        let segment_id = if segment_id_predicted {
            predicted
        } else {
            read_segment_tree(&mut self.bits, &segmentation.tree_probabilities)?
        };
        for row in mi_row..row_end {
            for column in mi_column..column_end {
                self.segment_ids[row * self.mi_columns + column] = segment_id;
            }
        }
        Ok((segment_id, segment_id_predicted))
    }

    fn read_inter_intra_mode(&mut self, group: usize) -> Result<IntraMode> {
        let mode = read_intra_mode(&mut self.bits, &self.context.y_mode[group * 9..][..9])?;
        if let Some(counts) = &mut self.counts {
            counts.y_mode[group][mode as usize] += 1;
        }
        Ok(mode)
    }

    fn read_transform_size(
        &mut self,
        block_size: BlockSize,
        allow_select: bool,
        above: Option<ModeInfo>,
        left: Option<ModeInfo>,
    ) -> Result<TransformSize> {
        let maximum = block_size.maximum_transform();
        if !allow_select
            || self.compressed.transform_mode != TransformMode::Select
            || block_size < BlockSize::B8x8
        {
            let selected = match self.compressed.transform_mode {
                TransformMode::Only4x4 => TransformSize::Tx4x4,
                TransformMode::Allow8x8 => TransformSize::Tx8x8,
                TransformMode::Allow16x16 => TransformSize::Tx16x16,
                TransformMode::Allow32x32 | TransformMode::Select => TransformSize::Tx32x32,
            };
            return Ok(maximum.min(selected));
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
        let selected = [
            TransformSize::Tx4x4,
            TransformSize::Tx8x8,
            TransformSize::Tx16x16,
            TransformSize::Tx32x32,
        ][transform];
        if let Some(counts) = &mut self.counts {
            match maximum {
                TransformSize::Tx8x8 => counts.transform_8x8[context][transform] += 1,
                TransformSize::Tx16x16 => counts.transform_16x16[context][transform] += 1,
                TransformSize::Tx32x32 => counts.transform_32x32[context][transform] += 1,
                TransformSize::Tx4x4 => {}
            }
        }
        Ok(selected)
    }

    fn mode_above(&self, row: usize, column: usize) -> Option<ModeInfo> {
        (row != self.row_start).then(|| self.modes[(row - 1) * self.mi_columns + column])?
    }

    fn mode_left(&self, row: usize, column: usize) -> Option<ModeInfo> {
        (column != self.column_start).then(|| self.modes[row * self.mi_columns + column - 1])?
    }

    fn store_mode(&mut self, row: usize, column: usize, mode: ModeInfo) {
        let row_end = (row + mode.block_size.height_mi()).min(self.mi_rows);
        let column_end = (column + mode.block_size.width_mi()).min(self.mi_columns);
        for target_row in row..row_end {
            for target_column in column..column_end {
                self.modes[target_row * self.mi_columns + target_column] = Some(mode);
            }
        }
    }

    fn read_references(
        &mut self,
        segment_id: u8,
        above: Option<ModeInfo>,
        left: Option<ModeInfo>,
    ) -> Result<[ReferenceFrame; 2]> {
        if let Some(reference) = segment_feature(self.header, segment_id, 2) {
            return Ok([
                match reference {
                    1 => ReferenceFrame::Last,
                    2 => ReferenceFrame::Golden,
                    3 => ReferenceFrame::Alt,
                    _ => {
                        return Err(Vp9Error::InvalidData(
                            "inter segment forces an invalid reference frame",
                        ));
                    }
                },
                ReferenceFrame::None,
            ]);
        }
        let (fixed, variables) = compound_references(self.header);
        let compound = match self.compressed.reference_mode {
            ReferenceMode::Single => false,
            ReferenceMode::Compound => true,
            ReferenceMode::Select => {
                let context = reference_mode_context(above, left, fixed);
                let compound = self.bits.read_bool(self.context.compound_inter[context])?;
                if let Some(counts) = &mut self.counts {
                    counts.compound_inter[context][usize::from(compound)] += 1;
                }
                compound
            }
        };
        if compound {
            let context = compound_reference_context(
                above,
                left,
                fixed,
                variables,
                reference_bias(self.header, fixed),
            );
            let bit = self
                .bits
                .read_bool(self.context.compound_reference[context])?;
            if let Some(counts) = &mut self.counts {
                counts.compound_reference[context][usize::from(bit)] += 1;
            }
            let variable = variables[usize::from(bit)];
            let mut references = [variable, variable];
            let fixed_index = usize::from(reference_bias(self.header, fixed));
            references[fixed_index] = fixed;
            references[1 - fixed_index] = variable;
            Ok(references)
        } else {
            let first_context = single_reference_context_one(above, left);
            let non_last = self
                .bits
                .read_bool(self.context.single_reference[first_context * 2])?;
            if let Some(counts) = &mut self.counts {
                counts.single_reference[first_context][0][usize::from(non_last)] += 1;
            }
            let reference = if non_last {
                let second_context = single_reference_context_two(above, left);
                let alternate = self
                    .bits
                    .read_bool(self.context.single_reference[second_context * 2 + 1])?;
                if let Some(counts) = &mut self.counts {
                    counts.single_reference[second_context][1][usize::from(alternate)] += 1;
                }
                if alternate {
                    ReferenceFrame::Alt
                } else {
                    ReferenceFrame::Golden
                }
            } else {
                ReferenceFrame::Last
            };
            Ok([reference, ReferenceFrame::None])
        }
    }

    fn read_interpolation(
        &mut self,
        above: Option<ModeInfo>,
        left: Option<ModeInfo>,
    ) -> Result<Interpolation> {
        if self.header.interpolation_filter != InterpolationFilter::Switchable {
            return Ok(interpolation_from_header(self.header.interpolation_filter));
        }
        let left_filter = left.map_or(Interpolation::Sentinel, |mode| mode.interpolation);
        let above_filter = above.map_or(Interpolation::Sentinel, |mode| mode.interpolation);
        let context = if left_filter == above_filter {
            left_filter as usize
        } else if left_filter == Interpolation::Sentinel {
            above_filter as usize
        } else if above_filter == Interpolation::Sentinel {
            left_filter as usize
        } else {
            Interpolation::Sentinel as usize
        };
        let symbol = read_tree(
            &mut self.bits,
            &[0, 2, -1, -2],
            &self.context.interpolation[context * 2..][..2],
        )?;
        if let Some(counts) = &mut self.counts {
            counts.interpolation[context][symbol] += 1;
        }
        Ok(match symbol {
            0 => Interpolation::EightTap,
            1 => Interpolation::Smooth,
            _ => Interpolation::Sharp,
        })
    }

    fn read_block_motion_vectors(
        &mut self,
        mi_row: usize,
        mi_column: usize,
        mode: &mut ModeInfo,
    ) -> Result<()> {
        let reference_count = 1 + usize::from(mode.has_second_reference());
        for reference_index in 0..reference_count {
            let reference = mode.references[reference_index];
            let candidates = self.find_motion_vector_references(
                mi_row,
                mi_column,
                mode.block_size,
                reference,
                mode.inter_mode,
                None,
            );
            let selected = candidates[usize::from(mode.inter_mode == InterMode::Near)]
                .lower_precision(self.header.allow_high_precision_motion_vectors);
            mode.motion_vectors[reference_index] = match mode.inter_mode {
                InterMode::Zero => MotionVector::ZERO,
                InterMode::Nearest | InterMode::Near => selected,
                InterMode::New => {
                    if reference_index == 0 {
                        self.summary.new_motion_vectors += 1;
                    }
                    read_motion_vector(
                        &mut self.bits,
                        &self.context.motion_vector,
                        selected,
                        self.header.allow_high_precision_motion_vectors,
                        self.counts
                            .as_deref_mut()
                            .map(|counts| &mut counts.motion_vector),
                    )?
                }
            };
        }
        mode.sub_motion_vectors.fill(mode.motion_vectors);
        Ok(())
    }

    #[allow(clippy::needless_range_loop)]
    fn read_sub8_motion_vectors(
        &mut self,
        mi_row: usize,
        mi_column: usize,
        mode: &mut ModeInfo,
        mode_context: usize,
    ) -> Result<()> {
        let reference_count = 1 + usize::from(mode.has_second_reference());
        let mut new_references = [MotionVector::ZERO; 2];
        let mut have_new_references = false;
        for &block in sub8_mode_indices(mode.block_size) {
            let inter_mode =
                read_inter_mode(&mut self.bits, &self.context.inter_mode[mode_context * 3..])?;
            if let Some(counts) = &mut self.counts {
                counts.inter_mode[mode_context][inter_mode_symbol(inter_mode)] += 1;
            }
            mode.inter_mode = inter_mode;

            if inter_mode == InterMode::New && !have_new_references {
                for reference_index in 0..reference_count {
                    let candidates = self.find_motion_vector_references(
                        mi_row,
                        mi_column,
                        mode.block_size,
                        mode.references[reference_index],
                        InterMode::New,
                        None,
                    );
                    new_references[reference_index] = candidates[0]
                        .lower_precision(self.header.allow_high_precision_motion_vectors);
                }
                have_new_references = true;
            }

            for reference_index in 0..reference_count {
                let reference = match inter_mode {
                    InterMode::Zero => MotionVector::ZERO,
                    InterMode::New => new_references[reference_index],
                    InterMode::Nearest | InterMode::Near => {
                        let candidates = self.find_motion_vector_references(
                            mi_row,
                            mi_column,
                            mode.block_size,
                            mode.references[reference_index],
                            inter_mode,
                            Some(block),
                        );
                        sub8_predictor(
                            inter_mode,
                            block,
                            reference_index,
                            &mode.sub_motion_vectors,
                            candidates,
                        )
                    }
                };
                mode.sub_motion_vectors[block][reference_index] = if inter_mode == InterMode::New {
                    read_motion_vector(
                        &mut self.bits,
                        &self.context.motion_vector,
                        reference,
                        self.header.allow_high_precision_motion_vectors,
                        self.counts
                            .as_deref_mut()
                            .map(|counts| &mut counts.motion_vector),
                    )?
                } else {
                    reference
                };
            }
            self.summary.new_motion_vectors += usize::from(inter_mode == InterMode::New);
            replicate_sub8_motion_vectors(mode.block_size, block, &mut mode.sub_motion_vectors);
        }
        mode.motion_vectors = mode.sub_motion_vectors[3];
        Ok(())
    }

    #[allow(clippy::collapsible_if)]
    fn find_motion_vector_references(
        &self,
        mi_row: usize,
        mi_column: usize,
        block_size: BlockSize,
        reference: ReferenceFrame,
        inter_mode: InterMode,
        sub_block: Option<usize>,
    ) -> [MotionVector; 2] {
        let offsets = motion_vector_reference_offsets(block_size);
        let early_break = inter_mode != InterMode::Near;
        let mut candidates = [MotionVector::ZERO; 2];
        let mut count = 0usize;
        let mut different_reference_found = false;
        let first_pass_start = if sub_block.is_some() { 2 } else { 0 };

        if let Some(block) = sub_block {
            for &(row_offset, column_offset) in &offsets[..2] {
                let Some(candidate) =
                    self.mode_at_offset(mi_row, mi_column, row_offset, column_offset)
                else {
                    continue;
                };
                different_reference_found = true;
                if let Some(reference_index) = candidate
                    .references
                    .iter()
                    .position(|&candidate_reference| candidate_reference == reference)
                {
                    let vector = if candidate.block_size < BlockSize::B8x8 {
                        let column_selector = usize::from(column_offset == 0);
                        let sub_index = SUBBLOCK_NEIGHBOR[block][column_selector];
                        candidate.sub_motion_vectors[sub_index][reference_index]
                    } else {
                        candidate.motion_vectors[reference_index]
                    };
                    if add_motion_vector_candidate(&mut candidates, &mut count, vector, early_break)
                    {
                        return self.clamp_motion_vector_candidates(
                            candidates, mi_row, mi_column, block_size,
                        );
                    }
                }
            }
        }

        for &(row_offset, column_offset) in &offsets[first_pass_start..] {
            let Some(candidate) = self.mode_at_offset(mi_row, mi_column, row_offset, column_offset)
            else {
                continue;
            };
            different_reference_found = true;
            if let Some(reference_index) = candidate
                .references
                .iter()
                .position(|&candidate_reference| candidate_reference == reference)
            {
                if add_motion_vector_candidate(
                    &mut candidates,
                    &mut count,
                    candidate.motion_vectors[reference_index],
                    early_break,
                ) {
                    return self
                        .clamp_motion_vector_candidates(candidates, mi_row, mi_column, block_size);
                }
            }
        }

        if let Some(candidate) = self.previous_mode(mi_row, mi_column) {
            if let Some(reference_index) = candidate
                .references
                .iter()
                .position(|&candidate_reference| candidate_reference == reference)
            {
                if add_motion_vector_candidate(
                    &mut candidates,
                    &mut count,
                    candidate.motion_vectors[reference_index],
                    early_break,
                ) {
                    return self
                        .clamp_motion_vector_candidates(candidates, mi_row, mi_column, block_size);
                }
            }
        }

        if different_reference_found {
            for &(row_offset, column_offset) in offsets {
                let Some(candidate) =
                    self.mode_at_offset(mi_row, mi_column, row_offset, column_offset)
                else {
                    continue;
                };
                if !candidate.is_inter {
                    continue;
                }
                for reference_index in 0..1 + usize::from(candidate.has_second_reference()) {
                    let source_reference = candidate.references[reference_index];
                    if source_reference == reference {
                        continue;
                    }
                    if reference_index == 1
                        && candidate.motion_vectors[1] == candidate.motion_vectors[0]
                    {
                        continue;
                    }
                    let vector = candidate.motion_vectors[reference_index].scaled_for_reference(
                        source_reference,
                        reference,
                        self.header,
                    );
                    if add_motion_vector_candidate(&mut candidates, &mut count, vector, early_break)
                    {
                        return self.clamp_motion_vector_candidates(
                            candidates, mi_row, mi_column, block_size,
                        );
                    }
                }
            }
        }

        if let Some(candidate) = self.previous_mode(mi_row, mi_column) {
            if candidate.is_inter {
                for reference_index in 0..1 + usize::from(candidate.has_second_reference()) {
                    let source_reference = candidate.references[reference_index];
                    if source_reference == reference {
                        continue;
                    }
                    if reference_index == 1
                        && candidate.motion_vectors[1] == candidate.motion_vectors[0]
                    {
                        continue;
                    }
                    let vector = candidate.motion_vectors[reference_index].scaled_for_reference(
                        source_reference,
                        reference,
                        self.header,
                    );
                    if add_motion_vector_candidate(&mut candidates, &mut count, vector, early_break)
                    {
                        return self.clamp_motion_vector_candidates(
                            candidates, mi_row, mi_column, block_size,
                        );
                    }
                }
            }
        }

        self.clamp_motion_vector_candidates(candidates, mi_row, mi_column, block_size)
    }

    fn mode_at_offset(
        &self,
        mi_row: usize,
        mi_column: usize,
        row_offset: i8,
        column_offset: i8,
    ) -> Option<ModeInfo> {
        let row = mi_row.checked_add_signed(isize::from(row_offset))?;
        let column = mi_column.checked_add_signed(isize::from(column_offset))?;
        if row >= self.mi_rows || column < self.column_start || column >= self.column_end {
            return None;
        }
        self.modes[row * self.mi_columns + column]
    }

    fn previous_mode(&self, mi_row: usize, mi_column: usize) -> Option<ModeInfo> {
        self.previous_modes
            .and_then(|modes| modes.get(mi_row * self.mi_columns + mi_column))
            .copied()
            .flatten()
    }

    fn clamp_motion_vector_candidates(
        &self,
        mut candidates: [MotionVector; 2],
        mi_row: usize,
        mi_column: usize,
        block_size: BlockSize,
    ) -> [MotionVector; 2] {
        let top = -(mi_row as i32 * 64) - 128;
        let left = -(mi_column as i32 * 64) - 128;
        let bottom = (self.mi_rows - mi_row - block_size.height_mi()) as i32 * 64 + 128;
        let right = (self.mi_columns - mi_column - block_size.width_mi()) as i32 * 64 + 128;
        for vector in &mut candidates {
            vector.row = i32::from(vector.row).clamp(top, bottom) as i16;
            vector.column = i32::from(vector.column).clamp(left, right) as i16;
        }
        candidates
    }

    fn inter_mode_context(&self, row: usize, column: usize, block_size: BlockSize) -> usize {
        let offsets = mode_context_offsets(block_size);
        let mut counter = 0usize;
        for (row_offset, column_offset) in offsets {
            let target_row = row as isize + row_offset;
            let target_column = column as isize + column_offset;
            if target_row < self.row_start as isize
                || target_row >= self.row_end as isize
                || target_column < self.column_start as isize
                || target_column >= self.column_end as isize
            {
                continue;
            }
            if let Some(mode) =
                self.modes[target_row as usize * self.mi_columns + target_column as usize]
            {
                counter += if !mode.is_inter {
                    9
                } else {
                    match mode.inter_mode {
                        InterMode::Nearest | InterMode::Near => 0,
                        InterMode::Zero => 3,
                        InterMode::New => 1,
                    }
                };
            }
        }
        [2usize, 3, 4, 1, 3, 9, 0, 9, 9, 5, 5, 9, 5, 9, 9, 9, 9, 9, 6][counter]
    }

    #[allow(clippy::collapsible_if)]
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
            let maximum = floor_transform(block_width.min(block_height));
            let transform_size = mode.transform_size.min(maximum);
            let step = transform_size.width_4x4();
            let origin_x = (mi_column * 2) >> subsampling_x;
            let origin_y = (mi_row * 2) >> subsampling_y;
            let plane_width = (self.mi_columns * 2) >> subsampling_x;
            let plane_height = (self.mi_rows * 2) >> subsampling_y;
            let usable_width = block_width.min(plane_width.saturating_sub(origin_x));
            let usable_height = block_height.min(plane_height.saturating_sub(origin_y));
            if mode.skip {
                let above_end = (origin_x + block_width).min(self.above_coefficients[plane].len());
                self.above_coefficients[plane][origin_x..above_end].fill(0);
                for row in origin_y..origin_y + block_height {
                    self.left_coefficients[plane][row & 15] = 0;
                }
                if !mode.is_inter {
                    for row in (0..usable_height).step_by(step) {
                        for column in (0..usable_width).step_by(step) {
                            self.predict_intra_transform(
                                plane,
                                origin_x,
                                origin_y,
                                row,
                                column,
                                step,
                                block_width,
                                mode,
                            );
                        }
                    }
                }
                continue;
            }
            for row in (0..usable_height).step_by(step) {
                for column in (0..usable_width).step_by(step) {
                    let x = origin_x + column;
                    let y = origin_y + row;
                    let above = self.above_coefficients[plane][x..x + step]
                        .iter()
                        .any(|&value| value != 0);
                    let left =
                        (y..y + step).any(|index| self.left_coefficients[plane][index & 15] != 0);
                    let initial_context = usize::from(above) + usize::from(left);
                    let intra_mode =
                        if !mode.is_inter && plane == 0 && mode.block_size < BlockSize::B8x8 {
                            mode.sub_intra_modes[(row << 1) + column]
                        } else {
                            mode.intra_mode
                        };
                    let transform_type = if transform_size == TransformSize::Tx32x32
                        || mode.is_inter
                        || plane != 0
                        || self
                            .header
                            .quantization
                            .expect("decoded frame has quantization")
                            .lossless()
                    {
                        TransformType::DctDct
                    } else {
                        intra_mode.transform_type()
                    };
                    if !mode.is_inter {
                        self.predict_intra_transform(
                            plane,
                            origin_x,
                            origin_y,
                            row,
                            column,
                            step,
                            block_width,
                            mode,
                        );
                    }
                    let dequant = dequant(self.header, plane, usize::from(mode.segment_id));
                    let coefficients = decode_coefficient_tokens(
                        &mut self.bits,
                        &self.context.coefficient[transform_size as usize],
                        usize::from(plane != 0),
                        usize::from(mode.is_inter),
                        initial_context,
                        scan_order(transform_size, transform_type),
                        transform_size,
                        dequant,
                        self.counts
                            .as_deref_mut()
                            .map(|counts| &mut counts.coefficient),
                    )?;
                    let nonzero = u8::from(coefficients.eob != 0);
                    let valid_width = step.min(usable_width - column);
                    let above_end = (x + step).min(self.above_coefficients[plane].len());
                    self.above_coefficients[plane][x..x + valid_width].fill(nonzero);
                    self.above_coefficients[plane][x + valid_width..above_end].fill(0);
                    let valid_height = step.min(usable_height - row);
                    for offset in 0..step {
                        self.left_coefficients[plane][(y + offset) & 15] =
                            if offset < valid_height { nonzero } else { 0 };
                    }
                    self.summary.transform_blocks += 1;
                    self.summary.nonzero_transform_blocks += usize::from(nonzero);
                    self.summary.coefficients += coefficients.eob;
                    if coefficients.eob != 0 {
                        if let Some(picture) = &mut self.picture {
                            picture.add_residual(
                                plane,
                                x * 4,
                                y * 4,
                                transform_size,
                                transform_type,
                                self.header
                                    .quantization
                                    .expect("decoded frame has quantization")
                                    .lossless(),
                                &coefficients.values[..transform_size.coefficient_count()],
                            );
                        }
                    }
                }
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn predict_intra_transform(
        &mut self,
        plane: usize,
        origin_x: usize,
        origin_y: usize,
        row: usize,
        column: usize,
        step: usize,
        block_width: usize,
        mode: ModeInfo,
    ) {
        let Some(picture) = &mut self.picture else {
            return;
        };
        let prediction_mode = if plane == 0 && mode.block_size < BlockSize::B8x8 {
            mode.sub_intra_modes[(row << 1) + column]
        } else if plane == 0 {
            mode.intra_mode
        } else {
            mode.uv_mode
        };
        let subsampling_x = usize::from(plane != 0) * self.header.chroma_subsampling().x_shift();
        let subsampling_y = usize::from(plane != 0) * self.header.chroma_subsampling().y_shift();
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

fn compound_references(header: &FrameHeader) -> (ReferenceFrame, [ReferenceFrame; 2]) {
    let last = reference_bias(header, ReferenceFrame::Last);
    let golden = reference_bias(header, ReferenceFrame::Golden);
    let alt = reference_bias(header, ReferenceFrame::Alt);
    if last == golden {
        (
            ReferenceFrame::Alt,
            [ReferenceFrame::Last, ReferenceFrame::Golden],
        )
    } else if last == alt {
        (
            ReferenceFrame::Golden,
            [ReferenceFrame::Last, ReferenceFrame::Alt],
        )
    } else {
        (
            ReferenceFrame::Last,
            [ReferenceFrame::Golden, ReferenceFrame::Alt],
        )
    }
}

fn reference_bias(header: &FrameHeader, reference: ReferenceFrame) -> bool {
    match reference {
        ReferenceFrame::Last => header.reference_sign_bias[0],
        ReferenceFrame::Golden => header.reference_sign_bias[1],
        ReferenceFrame::Alt => header.reference_sign_bias[2],
        _ => false,
    }
}

fn segment_feature(header: &FrameHeader, segment_id: u8, feature: usize) -> Option<i16> {
    let segmentation = header
        .segmentation
        .as_ref()
        .filter(|segmentation| segmentation.enabled)?;
    let feature = segmentation.features[usize::from(segment_id)][feature];
    feature.enabled.then_some(feature.value)
}

fn valid_reference_size(reference: &IntraPicture, width: usize, height: usize) -> bool {
    width.saturating_mul(2) >= reference.width()
        && height.saturating_mul(2) >= reference.height()
        && width <= reference.width().saturating_mul(16)
        && height <= reference.height().saturating_mul(16)
}

fn reference_mode_context(
    above: Option<ModeInfo>,
    left: Option<ModeInfo>,
    fixed: ReferenceFrame,
) -> usize {
    match (above, left) {
        (Some(above), Some(left)) => {
            if !above.has_second_reference() && !left.has_second_reference() {
                usize::from((above.references[0] == fixed) ^ (left.references[0] == fixed))
            } else if !above.has_second_reference() {
                2 + usize::from(above.references[0] == fixed || !above.is_inter)
            } else if !left.has_second_reference() {
                2 + usize::from(left.references[0] == fixed || !left.is_inter)
            } else {
                4
            }
        }
        (Some(mode), None) | (None, Some(mode)) => {
            if mode.has_second_reference() {
                3
            } else {
                usize::from(mode.references[0] == fixed)
            }
        }
        (None, None) => 1,
    }
}

fn compound_reference_context(
    above: Option<ModeInfo>,
    left: Option<ModeInfo>,
    fixed: ReferenceFrame,
    variables: [ReferenceFrame; 2],
    fixed_bias: bool,
) -> usize {
    let variable_index = usize::from(!fixed_bias);
    match (above, left) {
        (Some(above), Some(left)) => {
            let above_intra = !above.is_inter;
            let left_intra = !left.is_inter;
            if above_intra && left_intra {
                2
            } else if above_intra || left_intra {
                let edge = if above_intra { left } else { above };
                let reference = if edge.has_second_reference() {
                    edge.references[variable_index]
                } else {
                    edge.references[0]
                };
                1 + 2 * usize::from(reference != variables[1])
            } else {
                let above_single = !above.has_second_reference();
                let left_single = !left.has_second_reference();
                let above_variable = if above_single {
                    above.references[0]
                } else {
                    above.references[variable_index]
                };
                let left_variable = if left_single {
                    left.references[0]
                } else {
                    left.references[variable_index]
                };
                if above_variable == left_variable && above_variable == variables[1] {
                    0
                } else if above_single && left_single {
                    if (above_variable == fixed && left_variable == variables[0])
                        || (left_variable == fixed && above_variable == variables[0])
                    {
                        4
                    } else if above_variable == left_variable {
                        3
                    } else {
                        1
                    }
                } else if above_single || left_single {
                    let compound = if left_single {
                        above_variable
                    } else {
                        left_variable
                    };
                    let single = if above_single {
                        above_variable
                    } else {
                        left_variable
                    };
                    if compound == variables[1] && single != variables[1] {
                        1
                    } else if single == variables[1] && compound != variables[1] {
                        2
                    } else {
                        4
                    }
                } else if above_variable == left_variable {
                    4
                } else {
                    2
                }
            }
        }
        (Some(edge), None) | (None, Some(edge)) => {
            if !edge.is_inter {
                2
            } else if edge.has_second_reference() {
                4 * usize::from(edge.references[variable_index] != variables[1])
            } else {
                3 * usize::from(edge.references[0] != variables[1])
            }
        }
        (None, None) => 2,
    }
}

fn single_reference_context_one(above: Option<ModeInfo>, left: Option<ModeInfo>) -> usize {
    match (above, left) {
        (Some(above), Some(left)) => {
            let above_intra = !above.is_inter;
            let left_intra = !left.is_inter;
            if above_intra && left_intra {
                2
            } else if above_intra || left_intra {
                let edge = if above_intra { left } else { above };
                if edge.has_second_reference() {
                    1 + usize::from(
                        edge.references[0] == ReferenceFrame::Last
                            || edge.references[1] == ReferenceFrame::Last,
                    )
                } else {
                    4 * usize::from(edge.references[0] == ReferenceFrame::Last)
                }
            } else {
                let above_compound = above.has_second_reference();
                let left_compound = left.has_second_reference();
                if above_compound && left_compound {
                    1 + usize::from(
                        above.references.contains(&ReferenceFrame::Last)
                            || left.references.contains(&ReferenceFrame::Last),
                    )
                } else if above_compound || left_compound {
                    let single = if above_compound { left } else { above };
                    let compound = if above_compound { above } else { left };
                    if single.references[0] == ReferenceFrame::Last {
                        3 + usize::from(compound.references.contains(&ReferenceFrame::Last))
                    } else {
                        usize::from(compound.references.contains(&ReferenceFrame::Last))
                    }
                } else {
                    2 * usize::from(above.references[0] == ReferenceFrame::Last)
                        + 2 * usize::from(left.references[0] == ReferenceFrame::Last)
                }
            }
        }
        (Some(edge), None) | (None, Some(edge)) => {
            if !edge.is_inter {
                2
            } else if edge.has_second_reference() {
                1 + usize::from(edge.references.contains(&ReferenceFrame::Last))
            } else {
                4 * usize::from(edge.references[0] == ReferenceFrame::Last)
            }
        }
        (None, None) => 2,
    }
}

fn single_reference_context_two(above: Option<ModeInfo>, left: Option<ModeInfo>) -> usize {
    match (above, left) {
        (Some(above), Some(left)) => {
            let above_intra = !above.is_inter;
            let left_intra = !left.is_inter;
            if above_intra && left_intra {
                2
            } else if above_intra || left_intra {
                let edge = if above_intra { left } else { above };
                if !edge.has_second_reference() {
                    if edge.references[0] == ReferenceFrame::Last {
                        3
                    } else {
                        4 * usize::from(edge.references[0] == ReferenceFrame::Golden)
                    }
                } else {
                    1 + 2 * usize::from(edge.references.contains(&ReferenceFrame::Golden))
                }
            } else {
                let above_compound = above.has_second_reference();
                let left_compound = left.has_second_reference();
                if above_compound && left_compound {
                    if above.references == left.references {
                        3 * usize::from(above.references.contains(&ReferenceFrame::Golden))
                    } else {
                        2
                    }
                } else if above_compound || left_compound {
                    let single = if above_compound { left } else { above };
                    let compound = if above_compound { above } else { left };
                    if single.references[0] == ReferenceFrame::Golden {
                        3 + usize::from(compound.references.contains(&ReferenceFrame::Golden))
                    } else if single.references[0] == ReferenceFrame::Alt {
                        usize::from(compound.references.contains(&ReferenceFrame::Golden))
                    } else {
                        1 + 2 * usize::from(compound.references.contains(&ReferenceFrame::Golden))
                    }
                } else if above.references[0] == ReferenceFrame::Last
                    && left.references[0] == ReferenceFrame::Last
                {
                    3
                } else if above.references[0] == ReferenceFrame::Last
                    || left.references[0] == ReferenceFrame::Last
                {
                    let other = if above.references[0] == ReferenceFrame::Last {
                        left.references[0]
                    } else {
                        above.references[0]
                    };
                    4 * usize::from(other == ReferenceFrame::Golden)
                } else {
                    2 * usize::from(above.references[0] == ReferenceFrame::Golden)
                        + 2 * usize::from(left.references[0] == ReferenceFrame::Golden)
                }
            }
        }
        (Some(edge), None) | (None, Some(edge)) => {
            if !edge.is_inter
                || (edge.references[0] == ReferenceFrame::Last && !edge.has_second_reference())
            {
                2
            } else if !edge.has_second_reference() {
                4 * usize::from(edge.references[0] == ReferenceFrame::Golden)
            } else {
                3 * usize::from(edge.references.contains(&ReferenceFrame::Golden))
            }
        }
        (None, None) => 2,
    }
}

fn interpolation_from_header(filter: InterpolationFilter) -> Interpolation {
    match filter {
        InterpolationFilter::EightTap => Interpolation::EightTap,
        InterpolationFilter::EightTapSmooth => Interpolation::Smooth,
        InterpolationFilter::EightTapSharp => Interpolation::Sharp,
        InterpolationFilter::Bilinear => Interpolation::Bilinear,
        InterpolationFilter::Switchable => Interpolation::Sentinel,
    }
}

fn interpolation_kernel(interpolation: Interpolation) -> &'static [i16; 128] {
    match interpolation {
        Interpolation::EightTap => &tables::FILTER_EIGHT_TAP,
        Interpolation::Smooth => &tables::FILTER_EIGHT_TAP_SMOOTH,
        Interpolation::Sharp => &tables::FILTER_EIGHT_TAP_SHARP,
        Interpolation::Bilinear => &tables::FILTER_BILINEAR,
        Interpolation::Sentinel => &tables::FILTER_EIGHT_TAP,
    }
}

fn split_motion_vector(
    vectors: &[[MotionVector; 2]; 4],
    reference: usize,
    subsampling_x: usize,
    subsampling_y: usize,
    block: usize,
) -> MotionVector {
    let indices: &[usize] = match (subsampling_x, subsampling_y) {
        (0, 0) => &[block],
        (0, 1) => &[block, block + 2],
        (1, 0) => &[block, block + 1],
        (1, 1) => &[0, 1, 2, 3],
        _ => unreachable!("VP9 chroma subsampling shifts are at most one"),
    };
    let sum_row = indices
        .iter()
        .map(|&index| i32::from(vectors[index][reference].row))
        .sum();
    let sum_column = indices
        .iter()
        .map(|&index| i32::from(vectors[index][reference].column))
        .sum();
    let count = indices.len() as i32;
    MotionVector {
        row: round_motion_average(sum_row, count),
        column: round_motion_average(sum_column, count),
    }
}

fn round_motion_average(value: i32, divisor: i32) -> i16 {
    let adjustment = divisor / 2;
    ((if value < 0 {
        value - adjustment
    } else {
        value + adjustment
    }) / divisor) as i16
}

fn mode_context_offsets(size: BlockSize) -> [(isize, isize); 2] {
    match size {
        BlockSize::B8x16 | BlockSize::B16x32 | BlockSize::B32x64 => [(0, -1), (-1, 0)],
        BlockSize::B32x32 => [(-1, 1), (1, -1)],
        BlockSize::B64x64 => [(-1, 3), (3, -1)],
        _ => [(-1, 0), (0, -1)],
    }
}

fn intra_inter_context(above: Option<ModeInfo>, left: Option<ModeInfo>) -> usize {
    match (above, left) {
        (Some(above), Some(left)) => {
            let above_intra = !above.is_inter;
            let left_intra = !left.is_inter;
            if above_intra && left_intra {
                3
            } else {
                usize::from(above_intra || left_intra)
            }
        }
        (Some(mode), None) | (None, Some(mode)) => 2 * usize::from(!mode.is_inter),
        (None, None) => 0,
    }
}

fn size_group(size: BlockSize) -> usize {
    match size {
        BlockSize::B4x4 | BlockSize::B4x8 | BlockSize::B8x4 => 0,
        BlockSize::B8x8 | BlockSize::B8x16 | BlockSize::B16x8 => 1,
        BlockSize::B16x16 | BlockSize::B16x32 | BlockSize::B32x16 => 2,
        _ => 3,
    }
}

fn sub8_mode_indices(size: BlockSize) -> &'static [usize] {
    match size {
        BlockSize::B4x4 => &[0, 1, 2, 3],
        BlockSize::B4x8 => &[0, 1],
        BlockSize::B8x4 => &[0, 2],
        _ => &[],
    }
}

fn replicate_sub8_motion_vectors(
    size: BlockSize,
    block: usize,
    vectors: &mut [[MotionVector; 2]; 4],
) {
    match size {
        BlockSize::B4x8 => vectors[block + 2] = vectors[block],
        BlockSize::B8x4 => vectors[block + 1] = vectors[block],
        _ => {}
    }
}

fn sub8_predictor(
    mode: InterMode,
    block: usize,
    reference: usize,
    vectors: &[[MotionVector; 2]; 4],
    candidates: [MotionVector; 2],
) -> MotionVector {
    match block {
        0 => candidates[usize::from(mode == InterMode::Near)],
        1 | 2 if mode == InterMode::Nearest => vectors[0][reference],
        1 | 2 => candidates
            .into_iter()
            .find(|&candidate| candidate != vectors[0][reference])
            .unwrap_or(MotionVector::ZERO),
        3 if mode == InterMode::Nearest => vectors[2][reference],
        3 => [
            vectors[1][reference],
            vectors[0][reference],
            candidates[0],
            candidates[1],
        ]
        .into_iter()
        .find(|&candidate| candidate != vectors[2][reference])
        .unwrap_or(MotionVector::ZERO),
        _ => MotionVector::ZERO,
    }
}

fn add_motion_vector_candidate(
    candidates: &mut [MotionVector; 2],
    count: &mut usize,
    vector: MotionVector,
    early_break: bool,
) -> bool {
    if *count == 0 {
        candidates[0] = vector;
        *count = 1;
        early_break
    } else if vector != candidates[0] {
        candidates[1] = vector;
        *count = 2;
        true
    } else {
        false
    }
}

const SUBBLOCK_NEIGHBOR: [[usize; 2]; 4] = [[1, 2], [1, 3], [3, 2], [3, 3]];

fn motion_vector_reference_offsets(size: BlockSize) -> &'static [(i8, i8); 8] {
    const OFFSETS: [[(i8, i8); 8]; 13] = [
        [
            (-1, 0),
            (0, -1),
            (-1, -1),
            (-2, 0),
            (0, -2),
            (-2, -1),
            (-1, -2),
            (-2, -2),
        ],
        [
            (-1, 0),
            (0, -1),
            (-1, -1),
            (-2, 0),
            (0, -2),
            (-2, -1),
            (-1, -2),
            (-2, -2),
        ],
        [
            (-1, 0),
            (0, -1),
            (-1, -1),
            (-2, 0),
            (0, -2),
            (-2, -1),
            (-1, -2),
            (-2, -2),
        ],
        [
            (-1, 0),
            (0, -1),
            (-1, -1),
            (-2, 0),
            (0, -2),
            (-2, -1),
            (-1, -2),
            (-2, -2),
        ],
        [
            (0, -1),
            (-1, 0),
            (1, -1),
            (-1, -1),
            (0, -2),
            (-2, 0),
            (-2, -1),
            (-1, -2),
        ],
        [
            (-1, 0),
            (0, -1),
            (-1, 1),
            (-1, -1),
            (-2, 0),
            (0, -2),
            (-1, -2),
            (-2, -1),
        ],
        [
            (-1, 0),
            (0, -1),
            (-1, 1),
            (1, -1),
            (-1, -1),
            (-3, 0),
            (0, -3),
            (-3, -3),
        ],
        [
            (0, -1),
            (-1, 0),
            (2, -1),
            (-1, -1),
            (-1, 1),
            (0, -3),
            (-3, 0),
            (-3, -3),
        ],
        [
            (-1, 0),
            (0, -1),
            (-1, 2),
            (-1, -1),
            (1, -1),
            (-3, 0),
            (0, -3),
            (-3, -3),
        ],
        [
            (-1, 1),
            (1, -1),
            (-1, 2),
            (2, -1),
            (-1, -1),
            (-3, 0),
            (0, -3),
            (-3, -3),
        ],
        [
            (0, -1),
            (-1, 0),
            (4, -1),
            (-1, 2),
            (-1, -1),
            (0, -3),
            (-3, 0),
            (2, -1),
        ],
        [
            (-1, 0),
            (0, -1),
            (-1, 4),
            (2, -1),
            (-1, -1),
            (-3, 0),
            (0, -3),
            (-1, 2),
        ],
        [
            (-1, 3),
            (3, -1),
            (-1, 4),
            (4, -1),
            (-1, -1),
            (-1, 0),
            (0, -1),
            (-1, 6),
        ],
    ];
    &OFFSETS[size as usize]
}

fn read_inter_mode(bits: &mut BoolDecoder<'_>, probabilities: &[u8]) -> Result<InterMode> {
    Ok(
        match read_tree(bits, &[0, 2, -1, 4, -2, -3], probabilities)? {
            0 => InterMode::Zero,
            1 => InterMode::Nearest,
            2 => InterMode::Near,
            _ => InterMode::New,
        },
    )
}

fn inter_mode_symbol(mode: InterMode) -> usize {
    match mode {
        InterMode::Nearest => 0,
        InterMode::Near => 1,
        InterMode::Zero => 2,
        InterMode::New => 3,
    }
}

fn read_tree(bits: &mut BoolDecoder<'_>, tree: &[i16], probabilities: &[u8]) -> Result<usize> {
    let mut index = 0usize;
    loop {
        let node = index / 2;
        let branch = usize::from(
            bits.read_bool(
                *probabilities
                    .get(node)
                    .ok_or(Vp9Error::InvalidData("probability tree is truncated"))?,
            )?,
        );
        let child = *tree
            .get(index + branch)
            .ok_or(Vp9Error::InvalidData("probability tree is truncated"))?;
        if child <= 0 {
            return Ok(usize::try_from(-child).unwrap());
        }
        index = usize::try_from(child).map_err(|_| Vp9Error::IntegerOverflow)?;
    }
}

#[allow(clippy::needless_option_as_deref)]
fn read_motion_vector(
    bits: &mut BoolDecoder<'_>,
    probabilities: &[u8; 69],
    reference: MotionVector,
    allow_high_precision: bool,
    mut counts: Option<&mut MotionVectorCounts>,
) -> Result<MotionVector> {
    let joint = read_tree(bits, &[0, 2, -1, 4, -2, -3], &probabilities[..3])?;
    if let Some(counts) = &mut counts {
        counts.joints[joint] += 1;
    }
    let high_precision = allow_high_precision && reference.uses_high_precision();
    let row_delta = if joint == 2 || joint == 3 {
        read_motion_component(
            bits,
            &probabilities[3..36],
            high_precision,
            counts
                .as_deref_mut()
                .map(|counts| &mut counts.components[0]),
        )?
    } else {
        0
    };
    let column_delta = if joint == 1 || joint == 3 {
        read_motion_component(
            bits,
            &probabilities[36..69],
            high_precision,
            counts
                .as_deref_mut()
                .map(|counts| &mut counts.components[1]),
        )?
    } else {
        0
    };
    Ok(MotionVector {
        row: reference.row.saturating_add(row_delta),
        column: reference.column.saturating_add(column_delta),
    })
}

fn read_motion_component(
    bits: &mut BoolDecoder<'_>,
    probabilities: &[u8],
    high_precision: bool,
    mut counts: Option<&mut MotionVectorComponentCounts>,
) -> Result<i16> {
    let negative = bits.read_bool(probabilities[0])?;
    if let Some(counts) = &mut counts {
        counts.sign[usize::from(negative)] += 1;
    }
    let class = read_tree(
        bits,
        &[
            0, 2, -1, 4, 6, 8, -2, -3, 10, 12, -4, -5, -6, 14, 16, 18, -7, -8, -9, -10,
        ],
        &probabilities[1..11],
    )?;
    if let Some(counts) = &mut counts {
        counts.classes[class] += 1;
    }
    let class_zero = class == 0;
    let integer = if class_zero {
        let value = usize::from(bits.read_bool(probabilities[11])?);
        if let Some(counts) = &mut counts {
            counts.class_zero[value] += 1;
        }
        value
    } else {
        let mut value = 0usize;
        for index in 0..class {
            let bit = usize::from(bits.read_bool(probabilities[12 + index])?);
            if let Some(counts) = &mut counts {
                counts.bits[index][bit] += 1;
            }
            value |= bit << index;
        }
        value
    };
    let fractional = if class_zero {
        &probabilities[22 + integer * 3..][..3]
    } else {
        &probabilities[28..31]
    };
    let fraction = read_tree(bits, &[0, 2, -1, 4, -2, -3], fractional)?;
    if let Some(counts) = &mut counts {
        if class_zero {
            counts.class_zero_fractional[integer][fraction] += 1;
        } else {
            counts.fractional[fraction] += 1;
        }
    }
    let precision = if high_precision {
        usize::from(bits.read_bool(if class_zero {
            probabilities[31]
        } else {
            probabilities[32]
        })?)
    } else {
        1
    };
    // The normative MV counts include the implied high-precision value even
    // when the bit is not coded because the reference MV is outside the
    // high-precision range.
    if let Some(counts) = &mut counts {
        if class_zero {
            counts.class_zero_high_precision[precision] += 1;
        } else {
            counts.high_precision[precision] += 1;
        }
    }
    let class_base = if class_zero { 0 } else { 2usize << (class + 2) };
    let magnitude = class_base + (integer << 3) + (fraction << 1) + precision + 1;
    let magnitude = i16::try_from(magnitude).map_err(|_| Vp9Error::IntegerOverflow)?;
    Ok(if negative { -magnitude } else { magnitude })
}
