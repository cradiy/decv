use std::{
    env,
    error::Error,
    fs::File,
    io::{self, Write},
};

use decv_audio::SoftwareAudioDecoder;
use decv_core::{
    AudioDecodeInputStatus, AudioDecodeOutput, AudioDecoder, DecodedAudioFrame, EncodedAudioPacket,
};
use decv_mp4::{Mp4Demuxer, TrackKind};

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = env::args().skip(1);
    let input_path = arguments
        .next()
        .ok_or("usage: decode_mp4 <input.mp4> [output.f32le]")?;
    let output_path = arguments.next();
    if arguments.next().is_some() {
        return Err("usage: decode_mp4 <input.mp4> [output.f32le]".into());
    }

    let demuxer = Mp4Demuxer::open(File::open(&input_path)?)?;
    let track_index = demuxer
        .movie()
        .tracks()
        .iter()
        .position(|track| track.kind() == TrackKind::Audio && !track.samples().is_empty())
        .ok_or("MP4 contains no non-empty audio track")?;
    let mut cursor = demuxer.audio_packet_cursor(track_index)?;
    let config = cursor
        .decoder_config()?
        .ok_or("audio track has no decoder configuration")?;
    let mut decoder = SoftwareAudioDecoder::new();
    decoder.configure(config)?;
    let mut output = output_path.map(File::create).transpose()?;
    let mut frames = 0u64;
    let mut sample_frames = 0u64;

    while let Some(packet) = cursor.next_packet()? {
        send_packet(
            &mut decoder,
            packet,
            &mut output,
            &mut frames,
            &mut sample_frames,
        )?;
    }
    decoder.drain()?;
    loop {
        match receive_one(&mut decoder, &mut output, &mut frames, &mut sample_frames)? {
            ReceiveState::More => {}
            ReceiveState::NeedInput => {
                return Err("AAC decoder requested input after drain".into());
            }
            ReceiveState::End => break,
        }
    }

    let track = &demuxer.movie().tracks()[track_index];
    println!(
        "decoded {frames} AAC frame(s), {sample_frames} sample frame(s), \
         channels={}, rate={} Hz",
        track
            .audio_decoder_config_for_sample(0)?
            .channel_layout
            .channels(),
        track.audio_decoder_config_for_sample(0)?.sample_rate,
    );
    Ok(())
}

fn send_packet(
    decoder: &mut SoftwareAudioDecoder,
    mut packet: EncodedAudioPacket,
    output: &mut Option<File>,
    frames: &mut u64,
    sample_frames: &mut u64,
) -> Result<(), Box<dyn Error>> {
    loop {
        match decoder.send_packet(packet)? {
            AudioDecodeInputStatus::Accepted => break,
            AudioDecodeInputStatus::NeedOutput(unconsumed) => {
                packet = unconsumed;
                if receive_one(decoder, output, frames, sample_frames)? != ReceiveState::More {
                    return Err("AAC decoder returned NeedOutput without output".into());
                }
            }
            _ => return Err("AAC decoder returned an unknown input status".into()),
        }
    }
    while receive_one(decoder, output, frames, sample_frames)? == ReceiveState::More {}
    Ok(())
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ReceiveState {
    More,
    NeedInput,
    End,
}

fn receive_one(
    decoder: &mut SoftwareAudioDecoder,
    output: &mut Option<File>,
    frames: &mut u64,
    sample_frames: &mut u64,
) -> Result<ReceiveState, Box<dyn Error>> {
    match decoder.receive_frame()? {
        AudioDecodeOutput::FormatChanged(format) => {
            eprintln!(
                "format {} Hz, {} channels, {:?}",
                format.sample_rate,
                format.channel_layout.channels(),
                format.sample_format
            );
            Ok(ReceiveState::More)
        }
        AudioDecodeOutput::Frame(frame) => {
            write_frame(&frame, output)?;
            *frames = frames.checked_add(1).ok_or("AAC frame count overflow")?;
            *sample_frames = sample_frames
                .checked_add(
                    u64::try_from(frame.sample_frames())
                        .map_err(|_| "AAC sample-frame count exceeds u64")?,
                )
                .ok_or("AAC sample-frame count overflow")?;
            Ok(ReceiveState::More)
        }
        AudioDecodeOutput::NeedInput => Ok(ReceiveState::NeedInput),
        AudioDecodeOutput::EndOfStream => Ok(ReceiveState::End),
        _ => Err("AAC decoder returned an unknown output event".into()),
    }
}

fn write_frame(frame: &DecodedAudioFrame, output: &mut Option<File>) -> io::Result<()> {
    let Some(output) = output.as_mut() else {
        return Ok(());
    };
    for sample in frame.samples.iter().copied() {
        output.write_all(&sample.to_le_bytes())?;
    }
    Ok(())
}
