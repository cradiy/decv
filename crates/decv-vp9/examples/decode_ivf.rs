use std::{env, error::Error, fs, io::Write};

use decv_vp9::Vp9Decoder;

fn main() -> Result<(), Box<dyn Error>> {
    let input = env::args()
        .nth(1)
        .ok_or("usage: decode_ivf <input.ivf> <output.yuv> [frame-count|all]")?;
    let output = env::args()
        .nth(2)
        .ok_or("usage: decode_ivf <input.ivf> <output.yuv> [frame-count|all]")?;
    let data = fs::read(input)?;
    if data.len() < 32 || &data[..4] != b"DKIF" || &data[8..12] != b"VP90" {
        return Err("input is not a VP9 IVF stream".into());
    }
    let declared_frames = u32::from_le_bytes(data[24..28].try_into()?) as usize;
    let requested = match env::args().nth(3).as_deref() {
        Some("all") | None => declared_frames,
        Some(value) => value.parse()?,
    };

    let mut decoder = Vp9Decoder::new();
    let mut offset = 32usize;
    let mut picture = None;
    for _ in 0..requested {
        let header = data
            .get(offset..offset + 12)
            .ok_or("truncated IVF frame header")?;
        let size = u32::from_le_bytes(header[..4].try_into()?) as usize;
        offset += 12;
        let frame = data
            .get(offset..offset + size)
            .ok_or("truncated IVF frame payload")?;
        offset += size;
        if let Some(decoded) = decoder.decode_packet(frame)?.pop() {
            picture = Some(decoded);
        }
    }

    let picture = picture.ok_or("decoder produced no visible picture")?;
    let mut file = fs::File::create(output)?;
    for plane in 0..3 {
        file.write_all(picture.plane(plane))?;
    }
    Ok(())
}
