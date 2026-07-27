use std::{env, error::Error, fs::File};

use decv_core::{DecodeInputStatus, DecodeOutput, VideoDecoder, VideoFormat};
use decv_mp4::{Mp4Demuxer, TrackKind};
use decv_vp9::Vp9Decoder;

fn main() -> Result<(), Box<dyn Error>> {
    let input = env::args()
        .nth(1)
        .ok_or("usage: decode_mp4 <input.mp4> [packet-count|all]")?;
    let demuxer = Mp4Demuxer::open(File::open(input)?)?;
    let track_index = demuxer
        .movie()
        .tracks()
        .iter()
        .position(|track| track.kind() == TrackKind::Video)
        .ok_or("MP4 has no video track")?;
    let track = &demuxer.movie().tracks()[track_index];
    let available = track.samples().len();
    let packet_count = match env::args().nth(2).as_deref() {
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
