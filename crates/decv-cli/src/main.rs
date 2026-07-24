use std::{
    env,
    error::Error,
    fs::{self, File},
    io::{self, Read, Write},
    num::NonZeroUsize,
    path::Path,
};

#[cfg(not(feature = "frame-timing"))]
use decv_core::VideoDecoder;
use decv_core::{
    BitstreamFormat, DecodeInputStatus, DecodeOutput, DecodedVideoFrame, EncodedVideoPacket,
    FrameStorage, MediaTime, PixelFormat, Rect, VideoCodec, VideoDecoderConfig,
};
#[cfg(not(feature = "frame-timing"))]
use decv_h264::H264Decoder;
use decv_h264::{AnnexBReader, H264Parallelism};
use decv_mp4::{FourCc, Mp4Demuxer};

#[cfg(feature = "frame-timing")]
mod frame_timing;
#[cfg(feature = "frame-timing")]
use frame_timing::CliDecoder;
#[cfg(not(feature = "frame-timing"))]
type CliDecoder = H264Decoder;

const VIDE: FourCc = FourCc::new(*b"vide");
#[cfg(not(feature = "frame-timing"))]
const USAGE: &str = "usage: decv-cli [--seek <seconds>] [--parallelism <serial|auto|threads>] \
                     <input.h264|input.mp4> [output.nv12]";
#[cfg(feature = "frame-timing")]
const USAGE: &str = "usage: decv-cli [--seek <seconds>] [--parallelism <serial|auto|threads>] \
                     [--frame-timing] <input.h264|input.mp4> [output.nv12]";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PullState {
    NeedInput,
    EndOfStream,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CliOptions {
    input_path: String,
    output_path: Option<String>,
    seek_target: Option<MediaTime>,
    parallelism: H264Parallelism,
    #[cfg(feature = "frame-timing")]
    frame_timing: bool,
}

fn main() -> Result<(), Box<dyn Error>> {
    #[cfg(feature = "internal-profiling")]
    decv_h264::reset_inter_prediction_profile();

    let CliOptions {
        input_path: path,
        output_path,
        seek_target,
        parallelism,
        #[cfg(feature = "frame-timing")]
        frame_timing,
    } = parse_arguments(env::args().skip(1))?;
    #[cfg(feature = "frame-timing")]
    let mut decoder = CliDecoder::new(frame_timing);
    #[cfg(not(feature = "frame-timing"))]
    let mut decoder = CliDecoder::new();
    let mut raw_output = output_path.as_deref().map(File::create).transpose()?;
    decoder.set_parallelism(parallelism)?;
    let mut frame_count = 0u64;
    if is_mp4(Path::new(&path))? {
        decode_mp4(
            &path,
            &mut decoder,
            &mut raw_output,
            &mut frame_count,
            seek_target,
        )?;
    } else {
        if seek_target.is_some() {
            return Err("--seek requires an MP4 input with a sample index".into());
        }
        decode_annex_b(&path, &mut decoder, &mut raw_output, &mut frame_count)?;
    }
    decoder.drain()?;
    if receive_available(&mut decoder, &mut raw_output, &mut frame_count, seek_target)?
        != PullState::EndOfStream
    {
        return Err("decoder requested input after drain".into());
    }

    if let Some(output_path) = output_path {
        println!("wrote raw visible NV12 frames to {output_path}");
    }
    println!("decoded {frame_count} frame(s) from {path}");
    #[cfg(feature = "frame-timing")]
    if let Some(summary) = decoder.frame_timing_summary() {
        eprintln!("{summary}");
    }
    #[cfg(feature = "internal-profiling")]
    eprintln!("{}", decv_h264::inter_prediction_profile());
    Ok(())
}

fn parse_arguments(
    mut arguments: impl Iterator<Item = String>,
) -> Result<CliOptions, Box<dyn Error>> {
    let mut positional = Vec::new();
    let mut seek_target = None;
    let mut parallelism = H264Parallelism::Auto;
    let mut parallelism_specified = false;
    #[cfg(feature = "frame-timing")]
    let mut frame_timing = false;
    while let Some(argument) = arguments.next() {
        #[cfg(feature = "frame-timing")]
        if argument == "--frame-timing" {
            if frame_timing {
                return Err("--frame-timing may only be specified once".into());
            }
            frame_timing = true;
            continue;
        }
        if argument == "--seek" {
            if seek_target.is_some() {
                return Err("--seek may only be specified once".into());
            }
            seek_target = Some(parse_seconds(&arguments.next().ok_or(USAGE)?)?);
        } else if argument == "--parallelism" {
            if parallelism_specified {
                return Err("--parallelism may only be specified once".into());
            }
            parallelism = parse_parallelism(&arguments.next().ok_or(USAGE)?)?;
            parallelism_specified = true;
        } else if argument.starts_with('-') {
            return Err(format!("unknown option: {argument}\n{USAGE}").into());
        } else {
            positional.push(argument);
        }
    }
    if !(1..=2).contains(&positional.len()) {
        return Err(USAGE.into());
    }
    let output = (positional.len() == 2).then(|| positional.remove(1));
    Ok(CliOptions {
        input_path: positional.remove(0),
        output_path: output,
        seek_target,
        parallelism,
        #[cfg(feature = "frame-timing")]
        frame_timing,
    })
}

fn parse_parallelism(value: &str) -> Result<H264Parallelism, Box<dyn Error>> {
    match value {
        "serial" => Ok(H264Parallelism::Serial),
        "auto" => Ok(H264Parallelism::Auto),
        value => {
            let threads = value
                .parse::<usize>()
                .ok()
                .and_then(NonZeroUsize::new)
                .ok_or("--parallelism must be serial, auto, or a positive thread count")?;
            Ok(H264Parallelism::Threads(threads))
        }
    }
}

fn parse_seconds(seconds: &str) -> Result<MediaTime, Box<dyn Error>> {
    let (whole, fraction) = seconds.split_once('.').unwrap_or((seconds, ""));
    if whole.is_empty()
        || !whole.bytes().all(|byte| byte.is_ascii_digit())
        || !fraction.bytes().all(|byte| byte.is_ascii_digit())
        || fraction.len() > 9
    {
        return Err(
            "--seek must be a non-negative decimal number with at most 9 fractional digits".into(),
        );
    }
    let scale = 10u32
        .checked_pow(u32::try_from(fraction.len())?)
        .ok_or("seek timescale overflow")?;
    let whole = whole.parse::<i64>()?;
    let fraction = if fraction.is_empty() {
        0
    } else {
        fraction.parse::<i64>()?
    };
    let value = whole
        .checked_mul(i64::from(scale))
        .and_then(|value| value.checked_add(fraction))
        .ok_or("seek timestamp overflow")?;
    MediaTime::from_parts(value, scale).ok_or_else(|| "seek timescale is zero".into())
}

fn decode_annex_b(
    path: &str,
    decoder: &mut CliDecoder,
    raw_output: &mut Option<File>,
    frame_count: &mut u64,
) -> Result<(), Box<dyn Error>> {
    let data = fs::read(path)?;
    decoder.configure(VideoDecoderConfig::new(
        VideoCodec::H264,
        BitstreamFormat::ByteStream,
    ))?;
    for unit in AnnexBReader::new(&data) {
        let unit = unit?;
        let mut framed = Vec::with_capacity(4 + unit.bytes().len());
        framed.extend_from_slice(&[0, 0, 0, 1]);
        framed.extend_from_slice(unit.bytes());
        send_packet(
            decoder,
            EncodedVideoPacket::new(framed),
            raw_output,
            frame_count,
            None,
        )?;
    }
    Ok(())
}

fn decode_mp4(
    path: &str,
    decoder: &mut CliDecoder,
    raw_output: &mut Option<File>,
    frame_count: &mut u64,
    seek_target: Option<MediaTime>,
) -> Result<(), Box<dyn Error>> {
    let demuxer = Mp4Demuxer::open(File::open(path)?)?;
    let track_index = demuxer
        .movie()
        .tracks()
        .iter()
        .enumerate()
        .find(|(_, track)| track.handler() == VIDE && !track.samples().is_empty())
        .map(|(index, _)| index)
        .ok_or("MP4 contains no non-empty video track")?;
    let mut cursor = demuxer.packet_cursor(track_index)?;
    if let Some(target) = seek_target
        && cursor.seek_to_keyframe(target)?.is_none()
    {
        return Err("the requested time is before the first MP4 keyframe".into());
    }
    let first_description = cursor
        .track()
        .samples()
        .get(cursor.next_sample_index())
        .ok_or("seek selected no MP4 sample")?
        .description_index();
    decoder.configure(
        cursor
            .decoder_config()?
            .ok_or("seek selected no MP4 decoder configuration")?,
    )?;
    if let Some(target) = seek_target {
        decoder.flush_for_seek(target);
    }

    let mut first_packet = true;
    while let Some(mut packet) = cursor.next_packet()? {
        let sample = &cursor.track().samples()[cursor.next_sample_index() - 1];
        if sample.description_index() != first_description {
            return Err("mid-stream MP4 sample-description changes are not supported yet".into());
        }
        if first_packet && seek_target.is_some() {
            packet.discontinuity = true;
        }
        first_packet = false;
        send_packet(decoder, packet, raw_output, frame_count, seek_target)?;
    }
    Ok(())
}

fn send_packet(
    decoder: &mut CliDecoder,
    mut packet: EncodedVideoPacket,
    raw_output: &mut Option<File>,
    frame_count: &mut u64,
    minimum_pts: Option<MediaTime>,
) -> Result<(), Box<dyn Error>> {
    loop {
        match decoder.send_packet(packet)? {
            DecodeInputStatus::Accepted => break,
            DecodeInputStatus::NeedOutput(unconsumed) => {
                packet = unconsumed;
                if receive_available(decoder, raw_output, frame_count, minimum_pts)?
                    == PullState::EndOfStream
                {
                    return Err("decoder ended before all input was accepted".into());
                }
            }
            _ => return Err("decoder returned an unknown input status".into()),
        }
    }
    if receive_available(decoder, raw_output, frame_count, minimum_pts)? == PullState::EndOfStream {
        return Err("decoder ended before drain".into());
    }
    Ok(())
}

fn is_mp4(path: &Path) -> Result<bool, Box<dyn Error>> {
    if has_mp4_extension(path) {
        return Ok(true);
    }

    let mut input = File::open(path)?;
    let mut header = [0; 8];
    match input.read_exact(&mut header) {
        Ok(()) => Ok(is_mp4_box_header(header)),
        Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => Ok(false),
        Err(error) => Err(error.into()),
    }
}

fn has_mp4_extension(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| {
            extension.eq_ignore_ascii_case("mp4")
                || extension.eq_ignore_ascii_case("mov")
                || extension.eq_ignore_ascii_case("m4v")
        })
}

fn is_mp4_box_header(header: [u8; 8]) -> bool {
    matches!(&header[4..], b"ftyp" | b"moov" | b"free" | b"wide")
}

fn receive_available(
    decoder: &mut CliDecoder,
    raw_output: &mut Option<File>,
    frame_count: &mut u64,
    minimum_pts: Option<MediaTime>,
) -> Result<PullState, Box<dyn Error>> {
    loop {
        match decoder.receive_frame()? {
            DecodeOutput::FormatChanged(format) => {
                println!(
                    "format {}x{} {:?}",
                    format.coded_size.width, format.coded_size.height, format.pixel_format
                );
            }
            DecodeOutput::Frame(frame) => {
                if let Some(minimum_pts) = minimum_pts {
                    let pts = frame
                        .pts
                        .ok_or("decoded MP4 frame has no presentation timestamp")?;
                    if pts < minimum_pts {
                        continue;
                    }
                }
                let bytes = write_visible_frame(&frame, raw_output)?;
                *frame_count = frame_count
                    .checked_add(1)
                    .ok_or("decoded frame count overflow")?;
                println!("frame id={} bytes={bytes}", frame.id);
            }
            DecodeOutput::EndOfStream => return Ok(PullState::EndOfStream),
            DecodeOutput::NeedInput => return Ok(PullState::NeedInput),
            _ => return Err("decoder returned an unknown output event".into()),
        }
    }
}

fn write_visible_frame(
    frame: &DecodedVideoFrame,
    raw_output: &mut Option<File>,
) -> Result<usize, Box<dyn Error>> {
    frame.validate()?;
    match &frame.storage {
        FrameStorage::Cpu(cpu) => {
            let mut visible_bytes = 0usize;
            for (plane_index, plane) in cpu.planes.iter().enumerate() {
                let (first_row, byte_offset, row_bytes, rows) = visible_plane_layout(
                    frame.format.pixel_format,
                    frame.format.visible_rect,
                    plane_index,
                )
                .ok_or("unsupported CPU plane layout")?;
                visible_bytes = visible_bytes
                    .checked_add(
                        row_bytes
                            .checked_mul(rows)
                            .ok_or("frame byte count overflow")?,
                    )
                    .ok_or("frame byte count overflow")?;
                if let Some(output) = raw_output.as_mut() {
                    for row in 0..rows {
                        let start = plane.offset + (first_row + row) * plane.stride + byte_offset;
                        output.write_all(&plane.bytes[start..start + row_bytes])?;
                    }
                }
            }
            Ok(visible_bytes)
        }
        _ => Ok(0),
    }
}

fn visible_plane_layout(
    pixel_format: PixelFormat,
    visible_rect: Rect,
    plane_index: usize,
) -> Option<(usize, usize, usize, usize)> {
    let x = usize::try_from(visible_rect.x).ok()?;
    let y = usize::try_from(visible_rect.y).ok()?;
    let width = usize::try_from(visible_rect.width).ok()?;
    let height = usize::try_from(visible_rect.height).ok()?;
    match (pixel_format, plane_index) {
        (PixelFormat::Nv12, 0) => Some((y, x, width, height)),
        (PixelFormat::Nv12, 1) if (x | y | width | height) & 1 == 0 => {
            Some((y / 2, x, width, height / 2))
        }
        (PixelFormat::I420, 0) => Some((y, x, width, height)),
        (PixelFormat::I420, 1 | 2) if (x | y | width | height) & 1 == 0 => {
            Some((y / 2, x / 2, width / 2, height / 2))
        }
        (PixelFormat::P010, 0) => Some((y, x.checked_mul(2)?, width.checked_mul(2)?, height)),
        (PixelFormat::P010, 1) if (x | y | width | height) & 1 == 0 => {
            Some((y / 2, x.checked_mul(2)?, width.checked_mul(2)?, height / 2))
        }
        (PixelFormat::Bgra8 | PixelFormat::Rgba8, 0) => {
            Some((y, x.checked_mul(4)?, width.checked_mul(4)?, height))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::{
        has_mp4_extension, is_mp4_box_header, parse_arguments, parse_parallelism, parse_seconds,
        visible_plane_layout,
    };
    use decv_core::{MediaTime, PixelFormat, Rect};
    use decv_h264::H264Parallelism;

    #[test]
    fn derives_cropped_plane_layouts() {
        let rect = Rect::new(4, 2, 64, 32);
        assert_eq!(
            visible_plane_layout(PixelFormat::Nv12, rect, 0),
            Some((2, 4, 64, 32))
        );
        assert_eq!(
            visible_plane_layout(PixelFormat::Nv12, rect, 1),
            Some((1, 4, 64, 16))
        );
        assert_eq!(
            visible_plane_layout(PixelFormat::I420, rect, 2),
            Some((1, 2, 32, 16))
        );
        assert_eq!(
            visible_plane_layout(PixelFormat::P010, rect, 1),
            Some((1, 8, 128, 16))
        );
        assert_eq!(
            visible_plane_layout(PixelFormat::Bgra8, rect, 0),
            Some((2, 16, 256, 32))
        );
        assert_eq!(visible_plane_layout(PixelFormat::Nv12, rect, 2), None);
        assert_eq!(
            visible_plane_layout(PixelFormat::Nv12, Rect::new(1, 0, 64, 32), 1),
            None
        );
    }

    #[test]
    fn recognizes_mp4_paths_and_initial_boxes() {
        assert!(has_mp4_extension(Path::new("movie.MP4")));
        assert!(has_mp4_extension(Path::new("movie.mov")));
        assert!(!has_mp4_extension(Path::new("stream.h264")));
        assert!(is_mp4_box_header(*b"\0\0\0\x18ftyp"));
        assert!(is_mp4_box_header(*b"\0\0\0\x08moov"));
        assert!(!is_mp4_box_header(*b"\0\0\0\x01\x67\x64\0\x28"));
    }

    #[test]
    fn parses_exact_decimal_seek_times_and_options() {
        assert_eq!(
            parse_seconds("12.034").unwrap(),
            MediaTime::from_parts(12_034, 1_000).unwrap()
        );
        assert_eq!(
            parse_seconds("7").unwrap(),
            MediaTime::from_parts(7, 1).unwrap()
        );
        assert!(parse_seconds("-1").is_err());
        assert!(parse_seconds("1.1234567890").is_err());

        let options = parse_arguments(
            [
                "--seek",
                "1.25",
                "--parallelism",
                "2",
                "input.mp4",
                "output.nv12",
            ]
            .map(String::from)
            .into_iter(),
        )
        .unwrap();
        assert_eq!(options.input_path, "input.mp4");
        assert_eq!(options.output_path.as_deref(), Some("output.nv12"));
        assert_eq!(options.seek_target, MediaTime::from_parts(125, 100));
        assert!(matches!(
            options.parallelism,
            H264Parallelism::Threads(threads) if threads.get() == 2
        ));
        #[cfg(feature = "frame-timing")]
        assert!(!options.frame_timing);
        assert_eq!(
            parse_parallelism("serial").unwrap(),
            H264Parallelism::Serial
        );
        assert_eq!(parse_parallelism("auto").unwrap(), H264Parallelism::Auto);
        assert!(parse_parallelism("0").is_err());
    }

    #[cfg(feature = "frame-timing")]
    #[test]
    fn parses_frame_timing_option_once() {
        let options = parse_arguments(
            ["--frame-timing", "input.h264"]
                .map(String::from)
                .into_iter(),
        )
        .unwrap();
        assert!(options.frame_timing);
        assert!(
            parse_arguments(
                ["--frame-timing", "--frame-timing", "input.h264"]
                    .map(String::from)
                    .into_iter(),
            )
            .is_err()
        );
    }
}
