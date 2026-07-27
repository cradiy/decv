use std::{
    env,
    error::Error,
    fs::{File, write},
};

use decv_mp4::{Mp4Demuxer, TrackKind};
use decv_vp9::Vp9Decoder;

fn main() -> Result<(), Box<dyn Error>> {
    let input = env::args()
        .nth(1)
        .ok_or("usage: decode_first_visible_inter <input.mp4> <output.yuv>")?;
    let output = env::args()
        .nth(2)
        .ok_or("usage: decode_first_visible_inter <input.mp4> <output.yuv>")?;
    let demuxer = Mp4Demuxer::open(File::open(input)?)?;
    let track_index = demuxer
        .movie()
        .tracks()
        .iter()
        .position(|track| track.kind() == TrackKind::Video)
        .ok_or("MP4 has no video track")?;
    let available_packets = demuxer.movie().tracks()[track_index].samples().len();
    let packet_count = match env::args().nth(3).as_deref() {
        Some("all") => available_packets,
        Some(value) => value.parse()?,
        None => 2,
    };
    if packet_count > available_packets {
        return Err(format!(
            "requested {packet_count} packets, but the track has only {available_packets}"
        )
        .into());
    }
    let mut decoder = Vp9Decoder::new();
    let mut picture = None;
    for packet_index in 0..packet_count {
        let packet = demuxer.read_packet(track_index, packet_index)?;
        if let Some(decoded) = decoder.decode_packet(&packet.data)?.pop() {
            picture = Some(decoded);
        }
    }
    let picture = picture.ok_or("decoder produced no visible picture")?;
    let mut bytes = Vec::with_capacity(
        picture.plane(0).len() + picture.plane(1).len() + picture.plane(2).len(),
    );
    bytes.extend_from_slice(picture.plane(0));
    bytes.extend_from_slice(picture.plane(1));
    bytes.extend_from_slice(picture.plane(2));
    write(output, bytes)?;
    Ok(())
}
