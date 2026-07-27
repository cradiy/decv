use std::{
    env,
    error::Error,
    fs::{File, write},
};

use decv_mp4::{Mp4Demuxer, TrackKind};
use decv_vp9::{
    CompressedHeader, HeaderParser, Superframe, decode_inter_picture, decode_intra_picture,
};

fn main() -> Result<(), Box<dyn Error>> {
    let input = env::args()
        .nth(1)
        .ok_or("usage: decode_first_inter <input.mp4> <output.yuv>")?;
    let output = env::args()
        .nth(2)
        .ok_or("usage: decode_first_inter <input.mp4> <output.yuv>")?;
    let demuxer = Mp4Demuxer::open(File::open(input)?)?;
    let track_index = demuxer
        .movie()
        .tracks()
        .iter()
        .position(|track| track.kind() == TrackKind::Video)
        .ok_or("MP4 has no video track")?;
    let mut parser = HeaderParser::new();

    let key_packet = demuxer.read_packet(track_index, 0)?;
    let key_superframe = Superframe::parse(&key_packet.data)?;
    let key_frame = key_superframe
        .frames(&key_packet.data)
        .next()
        .ok_or("empty keyframe packet")?;
    let key_header = parser.parse(key_frame)?;
    let key_compressed = CompressedHeader::parse(key_frame, &key_header)?;
    let key_picture = decode_intra_picture(key_frame, &key_header, &key_compressed)?;

    let inter_packet = demuxer.read_packet(track_index, 1)?;
    let inter_superframe = Superframe::parse(&inter_packet.data)?;
    let inter_frame = inter_superframe
        .frames(&inter_packet.data)
        .next()
        .ok_or("empty inter-frame packet")?;
    let inter_header = parser.parse(inter_frame)?;
    let inter_compressed = CompressedHeader::parse(inter_frame, &inter_header)?;
    let picture = decode_inter_picture(
        inter_frame,
        &inter_header,
        &inter_compressed,
        [&key_picture, &key_picture, &key_picture],
    )?;

    let mut bytes = Vec::with_capacity(
        picture.plane(0).len() + picture.plane(1).len() + picture.plane(2).len(),
    );
    bytes.extend_from_slice(picture.plane(0));
    bytes.extend_from_slice(picture.plane(1));
    bytes.extend_from_slice(picture.plane(2));
    write(output, bytes)?;
    Ok(())
}
