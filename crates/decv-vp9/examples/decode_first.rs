use std::{
    env,
    error::Error,
    fs::{File, write},
};

use decv_mp4::{Mp4Demuxer, TrackKind};
use decv_vp9::{CompressedHeader, HeaderParser, Superframe, decode_intra_picture};

fn main() -> Result<(), Box<dyn Error>> {
    let input = env::args()
        .nth(1)
        .ok_or("usage: decode_first <input.mp4> <output.yuv>")?;
    let output = env::args()
        .nth(2)
        .ok_or("usage: decode_first <input.mp4> <output.yuv>")?;
    let demuxer = Mp4Demuxer::open(File::open(input)?)?;
    let track_index = demuxer
        .movie()
        .tracks()
        .iter()
        .position(|track| track.kind() == TrackKind::Video)
        .ok_or("MP4 has no video track")?;
    let packet = demuxer.read_packet(track_index, 0)?;
    let superframe = Superframe::parse(&packet.data)?;
    let frame = superframe
        .frames(&packet.data)
        .next()
        .ok_or("empty packet")?;
    let header = HeaderParser::new().parse(frame)?;
    let compressed = CompressedHeader::parse(frame, &header)?;
    let picture = decode_intra_picture(frame, &header, &compressed)?;
    let mut bytes = Vec::with_capacity(
        picture.plane(0).len() + picture.plane(1).len() + picture.plane(2).len(),
    );
    bytes.extend_from_slice(picture.plane(0));
    bytes.extend_from_slice(picture.plane(1));
    bytes.extend_from_slice(picture.plane(2));
    write(output, bytes)?;
    Ok(())
}
