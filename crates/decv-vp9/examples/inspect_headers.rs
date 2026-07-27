use std::{env, error::Error, fs::File};

use decv_mp4::{Mp4Demuxer, TrackKind};
use decv_vp9::{
    CompressedHeader, HeaderParser, InterSyntaxSummary, InterpolationFilter, IntraSyntaxSummary,
    ReferenceMode, Superframe, TransformMode,
};

fn main() -> Result<(), Box<dyn Error>> {
    let path = env::args()
        .nth(1)
        .ok_or("usage: inspect_headers <input.mp4>")?;
    let demuxer = Mp4Demuxer::open(File::open(path)?)?;
    let track_index = demuxer
        .movie()
        .tracks()
        .iter()
        .position(|track| track.kind() == TrackKind::Video)
        .ok_or("MP4 has no video track")?;

    let mut parser = HeaderParser::new();
    let mut coded_frames = 0usize;
    let mut visible_frames = 0usize;
    let mut hidden_frames = 0usize;
    let mut superframe_packets = 0usize;
    let mut probability_updates = 0usize;
    let mut reference_modes = [0usize; 3];
    let mut transform_modes = [0usize; 5];
    let mut interpolation_filters = [0usize; 5];
    let mut segmented_frames = 0usize;
    let mut show_existing_frames = 0usize;
    let mut first_intra_syntax = None;
    let mut first_inter_syntax = None;
    for sample_index in 0..demuxer.movie().tracks()[track_index].samples().len() {
        let packet = demuxer.read_packet(track_index, sample_index)?;
        let superframe = Superframe::parse(&packet.data)?;
        superframe_packets += usize::from(superframe.len() > 1);
        for frame in superframe.frames(&packet.data) {
            let header = parser.parse(frame).map_err(|error| {
                format!("sample {sample_index}, coded frame {coded_frames}: {error}")
            })?;
            if header.show_existing_frame.is_none() {
                let compressed = CompressedHeader::parse(frame, &header).map_err(|error| {
                    format!(
                        "sample {sample_index}, coded frame {coded_frames}, compressed header: {error}"
                    )
                })?;
                probability_updates += compressed.updates.len();
                reference_modes[match compressed.reference_mode {
                    ReferenceMode::Single => 0,
                    ReferenceMode::Compound => 1,
                    ReferenceMode::Select => 2,
                }] += 1;
                transform_modes[match compressed.transform_mode {
                    TransformMode::Only4x4 => 0,
                    TransformMode::Allow8x8 => 1,
                    TransformMode::Allow16x16 => 2,
                    TransformMode::Allow32x32 => 3,
                    TransformMode::Select => 4,
                }] += 1;
                if first_intra_syntax.is_none()
                    && header.intra_only
                    && env::var_os("DECV_VP9_INSPECT_INTRA_SYNTAX").is_some()
                {
                    first_intra_syntax =
                        Some(IntraSyntaxSummary::parse(frame, &header, &compressed)?);
                }
                if first_inter_syntax.is_none()
                    && !header.intra_only
                    && env::var_os("DECV_VP9_INSPECT_INTER_SYNTAX").is_some()
                {
                    first_inter_syntax =
                        Some(InterSyntaxSummary::parse(frame, &header, &compressed)?);
                }
            }
            show_existing_frames += usize::from(header.show_existing_frame.is_some());
            segmented_frames += usize::from(
                header
                    .segmentation
                    .as_ref()
                    .is_some_and(|segmentation| segmentation.enabled),
            );
            interpolation_filters[match header.interpolation_filter {
                InterpolationFilter::EightTap => 0,
                InterpolationFilter::EightTapSmooth => 1,
                InterpolationFilter::EightTapSharp => 2,
                InterpolationFilter::Bilinear => 3,
                InterpolationFilter::Switchable => 4,
            }] += 1;
            coded_frames += 1;
            if header.is_visible() {
                visible_frames += 1;
            } else {
                hidden_frames += 1;
            }
            if coded_frames <= 4 {
                let segmentation = header.segmentation.as_ref();
                println!(
                    "frame={} sample={} type={:?} show={} existing={:?} size={:?} header={}/{} tiles={}x{} base_q={} loop={} segmentation={}/{}/{} refresh={:#04x} refs={:?} hp_mv={} ctx={}/reset{} refresh_ctx={} parallel={}",
                    coded_frames - 1,
                    sample_index,
                    header.frame_type,
                    header.show_frame,
                    header.show_existing_frame,
                    header.size,
                    header.uncompressed_header_size,
                    header.compressed_header_size,
                    1usize << header.tile_columns_log2,
                    1usize << header.tile_rows_log2,
                    header
                        .quantization
                        .map_or(0, |quantization| quantization.base_q_idx),
                    header.loop_filter.as_ref().map_or(0, |filter| filter.level),
                    segmentation.is_some_and(|segmentation| segmentation.enabled),
                    segmentation.is_some_and(|segmentation| segmentation.update_map),
                    segmentation.is_some_and(|segmentation| segmentation.temporal_update),
                    header.refresh_frame_flags,
                    header.reference_indices,
                    header.allow_high_precision_motion_vectors,
                    header.frame_context_index,
                    header.reset_frame_context,
                    header.refresh_frame_context,
                    header.frame_parallel_decoding,
                );
            }
        }
    }
    println!(
        "samples={} coded_frames={coded_frames} visible={visible_frames} hidden={hidden_frames} superframe_packets={superframe_packets} probability_updates={probability_updates}",
        demuxer.movie().tracks()[track_index].samples().len(),
    );
    println!("first_intra_syntax={first_intra_syntax:?}");
    println!("first_inter_syntax={first_inter_syntax:?}");
    println!(
        "reference_modes(single/compound/select)={reference_modes:?} transform_modes(4/8/16/32/select)={transform_modes:?} interpolation(8tap/smooth/sharp/bilinear/switchable)={interpolation_filters:?} segmented={segmented_frames} show_existing={show_existing_frames}"
    );
    Ok(())
}
