use std::{
    env,
    error::Error,
    fs::{self, File},
    io::Write,
};

use decv_core::{
    BitstreamFormat, DecodeInputStatus, DecodeOutput, EncodedVideoPacket, FrameStorage,
    PixelFormat, Rect, VideoCodec, VideoDecoder, VideoDecoderConfig,
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
                                    let start = plane.offset
                                        + (first_row + row) * plane.stride
                                        + byte_offset;
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
    use super::visible_plane_layout;
    use decv_core::{PixelFormat, Rect};

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
}
