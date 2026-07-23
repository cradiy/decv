use std::{
    env,
    error::Error,
    fs::{self, File},
    io::Write,
};

use decv_core::{
    BitstreamFormat, DecodeInputStatus, DecodeOutput, EncodedVideoPacket, FrameStorage,
    PixelFormat, VideoCodec, VideoDecoder, VideoDecoderConfig,
};
use decv_h264::H264Decoder;

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = env::args().skip(1);
    let path = arguments
        .next()
        .ok_or("usage: decv-cli <annex-b.h264> [output.nv12]")?;
    let output_path = arguments.next();
    if arguments.next().is_some() {
        return Err("usage: decv-cli <annex-b.h264> [output.nv12]".into());
    }
    let mut raw_output = output_path.as_deref().map(File::create).transpose()?;
    let data = fs::read(&path)?;
    let mut decoder = H264Decoder::new();
    decoder.configure(VideoDecoderConfig {
        codec: VideoCodec::H264,
        bitstream_format: BitstreamFormat::ByteStream,
        codec_data: None,
    })?;
    match decoder.send_packet(EncodedVideoPacket::new(data))? {
        DecodeInputStatus::Accepted => {}
        DecodeInputStatus::NeedOutput(_) => {
            return Err("decoder unexpectedly required output before its first packet".into());
        }
    }
    decoder.drain()?;

    let mut frame_count = 0u64;
    loop {
        match decoder.receive_frame()? {
            DecodeOutput::FormatChanged(format) => {
                println!(
                    "format {}x{} {:?}",
                    format.coded_size.width, format.coded_size.height, format.pixel_format
                );
            }
            DecodeOutput::Frame(frame) => {
                frame.validate()?;
                frame_count += 1;
                let bytes = match &frame.storage {
                    FrameStorage::Cpu(cpu) => {
                        let mut visible_bytes = 0usize;
                        for (plane_index, plane) in cpu.planes.iter().enumerate() {
                            let row_bytes = visible_plane_row_bytes(
                                frame.format.pixel_format,
                                frame.format.coded_size.width,
                                plane_index,
                            )
                            .ok_or("unsupported CPU plane layout")?;
                            visible_bytes = visible_bytes
                                .checked_add(
                                    row_bytes
                                        .checked_mul(plane.rows)
                                        .ok_or("frame byte count overflow")?,
                                )
                                .ok_or("frame byte count overflow")?;
                            if let Some(output) = raw_output.as_mut() {
                                for row in 0..plane.rows {
                                    let start = plane.offset + row * plane.stride;
                                    output.write_all(&plane.bytes[start..start + row_bytes])?;
                                }
                            }
                        }
                        visible_bytes
                    }
                    _ => 0,
                };
                println!("frame id={} bytes={bytes}", frame.id);
            }
            DecodeOutput::EndOfStream => break,
            DecodeOutput::NeedInput => {
                return Err("decoder requested input after drain".into());
            }
        }
    }
    if let Some(output_path) = output_path {
        println!("wrote raw NV12 frames to {output_path}");
    }
    println!("decoded {frame_count} frame(s) from {path}");
    Ok(())
}

fn visible_plane_row_bytes(
    pixel_format: PixelFormat,
    coded_width: u32,
    plane_index: usize,
) -> Option<usize> {
    let width = usize::try_from(coded_width).ok()?;
    match (pixel_format, plane_index) {
        (PixelFormat::Nv12, 0 | 1) => Some(width),
        (PixelFormat::I420, 0) => Some(width),
        (PixelFormat::I420, 1 | 2) => Some(width / 2),
        (PixelFormat::P010, 0 | 1) => width.checked_mul(2),
        (PixelFormat::Bgra8 | PixelFormat::Rgba8, 0) => width.checked_mul(4),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::visible_plane_row_bytes;
    use decv_core::PixelFormat;

    #[test]
    fn derives_tightly_packed_row_widths() {
        assert_eq!(visible_plane_row_bytes(PixelFormat::Nv12, 64, 0), Some(64));
        assert_eq!(visible_plane_row_bytes(PixelFormat::Nv12, 64, 1), Some(64));
        assert_eq!(visible_plane_row_bytes(PixelFormat::I420, 64, 2), Some(32));
        assert_eq!(visible_plane_row_bytes(PixelFormat::P010, 64, 1), Some(128));
        assert_eq!(
            visible_plane_row_bytes(PixelFormat::Bgra8, 64, 0),
            Some(256)
        );
        assert_eq!(visible_plane_row_bytes(PixelFormat::Nv12, 64, 2), None);
    }
}
