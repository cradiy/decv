use std::{env, error::Error, fs::File};

use decv::{
    DecodeInputStatus, DecodeOutput, FourCc, Mp4Demuxer, SoftwareVideoDecoder, VideoDecoder,
};

const VIDEO_HANDLER: FourCc = FourCc::new(*b"vide");

fn main() -> Result<(), Box<dyn Error>> {
    let path = env::args()
        .nth(1)
        .ok_or("usage: decode_video_mp4 <input.mp4> [maximum-packets]")?;
    let maximum_packets = env::args()
        .nth(2)
        .map(|value| value.parse::<usize>())
        .transpose()?;
    let demuxer = Mp4Demuxer::open(File::open(&path)?)?;
    let track_index = demuxer
        .movie()
        .tracks()
        .iter()
        .position(|track| track.handler() == VIDEO_HANDLER && !track.samples().is_empty())
        .ok_or("MP4 contains no non-empty video track")?;
    let mut cursor = demuxer.packet_cursor(track_index)?;
    let config = cursor
        .decoder_config()?
        .ok_or("video track contains no decoder configuration")?;
    let codec = config.codec;
    let mut decoder = SoftwareVideoDecoder::new();
    decoder.configure(config)?;

    let mut packets = 0usize;
    let mut frames = 0usize;
    let mut coded_size = None;
    while maximum_packets.is_none_or(|maximum| packets < maximum) {
        let Some(mut packet) = cursor.next_packet()? else {
            break;
        };
        loop {
            match decoder.send_packet(packet)? {
                DecodeInputStatus::Accepted => break,
                DecodeInputStatus::NeedOutput(unconsumed) => {
                    packet = unconsumed;
                    pull_outputs(&mut decoder, &mut frames, &mut coded_size)?;
                }
                _ => return Err("decoder returned an unknown input status".into()),
            }
        }
        packets += 1;
        pull_outputs(&mut decoder, &mut frames, &mut coded_size)?;
    }

    decoder.drain()?;
    loop {
        match decoder.receive_frame()? {
            DecodeOutput::FormatChanged(format) => coded_size = Some(format.coded_size),
            DecodeOutput::Frame(frame) => {
                frame.validate()?;
                frames += 1;
            }
            DecodeOutput::EndOfStream => break,
            DecodeOutput::NeedInput => {
                return Err("decoder requested input after drain".into());
            }
            _ => return Err("decoder returned an unknown output".into()),
        }
    }
    println!(
        "decoded {frames} frame(s) from {packets} packet(s), codec={codec:?}, size={coded_size:?}"
    );
    Ok(())
}

fn pull_outputs(
    decoder: &mut SoftwareVideoDecoder,
    frames: &mut usize,
    coded_size: &mut Option<decv::Size>,
) -> Result<(), Box<dyn Error>> {
    loop {
        match decoder.receive_frame()? {
            DecodeOutput::FormatChanged(format) => *coded_size = Some(format.coded_size),
            DecodeOutput::Frame(frame) => {
                frame.validate()?;
                *frames += 1;
            }
            DecodeOutput::NeedInput => return Ok(()),
            DecodeOutput::EndOfStream => {
                return Err("decoder ended before drain".into());
            }
            _ => return Err("decoder returned an unknown output".into()),
        }
    }
}
