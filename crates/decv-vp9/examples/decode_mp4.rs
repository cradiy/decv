use std::{
    env,
    error::Error,
    fs::File,
    io::{self, Write},
};

use decv_core::{
    DecodeInputStatus, DecodeOutput, DecodedVideoFrame, FrameStorage, PixelFormat, VideoDecoder,
    VideoFormat,
};
use decv_mp4::{Mp4Demuxer, TrackKind};
use decv_vp9::Vp9Decoder;

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = env::args().skip(1);
    let input = arguments
        .next()
        .ok_or("usage: decode_mp4 <input.mp4> [packet-count|all] [output.yuv]")?;
    let packet_count = arguments.next();
    let output = arguments.next();
    if arguments.next().is_some() {
        return Err("too many arguments".into());
    }
    let demuxer = Mp4Demuxer::open(File::open(input)?)?;
    let track_index = demuxer
        .movie()
        .tracks()
        .iter()
        .position(|track| track.kind() == TrackKind::Video)
        .ok_or("MP4 has no video track")?;
    let track = &demuxer.movie().tracks()[track_index];
    let available = track.samples().len();
    let packet_count = match packet_count.as_deref() {
        Some("all") => available,
        Some(value) => value.parse()?,
        None => available,
    };
    if packet_count > available {
        return Err(format!(
            "requested {packet_count} packets, but the track has only {available}"
        )
        .into());
    }

    let mut decoder = Vp9Decoder::new();
    decoder.configure(track.decoder_config_for_sample(0)?)?;
    let mut output = output.map(File::create).transpose()?;
    let mut frames = 0usize;
    let mut format: Option<VideoFormat> = None;
    for packet_index in 0..packet_count {
        let packet = demuxer.read_packet(track_index, packet_index)?;
        if !matches!(decoder.send_packet(packet)?, DecodeInputStatus::Accepted) {
            return Err("decoder requested output after the previous packet was drained".into());
        }
        loop {
            match decoder.receive_frame()? {
                DecodeOutput::FormatChanged(changed) => format = Some(changed),
                DecodeOutput::Frame(frame) => {
                    frame.validate()?;
                    if let Some(output) = &mut output {
                        write_raw_frame(output, &frame)?;
                    }
                    frames += 1;
                }
                DecodeOutput::NeedInput => break,
                DecodeOutput::EndOfStream => {
                    return Err("decoder reached end of stream before drain".into());
                }
                _ => return Err("decoder produced an unknown output event".into()),
            }
        }
    }
    decoder.drain()?;
    if !matches!(decoder.receive_frame()?, DecodeOutput::EndOfStream) {
        return Err("decoder did not report end of stream after drain".into());
    }
    println!("packets={packet_count} frames={frames} format={format:?}");
    Ok(())
}

fn write_raw_frame(output: &mut File, frame: &DecodedVideoFrame) -> Result<(), Box<dyn Error>> {
    let (bytes_per_sample, subsampling_x, subsampling_y) = match frame.format.pixel_format {
        PixelFormat::I420 => (1, 1, 1),
        PixelFormat::I422 => (1, 1, 0),
        PixelFormat::I440 => (1, 0, 1),
        PixelFormat::I444 => (1, 0, 0),
        PixelFormat::PlanarYuv16 {
            subsampling_x,
            subsampling_y,
            ..
        } => (2, subsampling_x, subsampling_y),
        _ => return Err("raw output only supports planar VP9 pixel formats".into()),
    };
    let FrameStorage::Cpu(storage) = &frame.storage else {
        return Err("raw output requires CPU-backed frames".into());
    };
    for (plane_index, plane) in storage.planes.iter().enumerate() {
        let shift = if plane_index == 0 {
            (0, 0)
        } else {
            (subsampling_x, subsampling_y)
        };
        let width = frame.format.coded_size.width.div_ceil(1 << shift.0);
        let rows = frame.format.coded_size.height.div_ceil(1 << shift.1);
        let row_bytes = usize::try_from(width)?
            .checked_mul(bytes_per_sample)
            .ok_or("raw row size overflow")?;
        for row in 0..usize::try_from(rows)? {
            let start = plane
                .offset
                .checked_add(
                    row.checked_mul(plane.stride)
                        .ok_or("raw plane offset overflow")?,
                )
                .ok_or("raw plane offset overflow")?;
            let bytes = plane
                .bytes
                .get(start..start + row_bytes)
                .ok_or_else(|| io::Error::new(io::ErrorKind::UnexpectedEof, "raw plane row"))?;
            output.write_all(bytes)?;
        }
    }
    Ok(())
}
