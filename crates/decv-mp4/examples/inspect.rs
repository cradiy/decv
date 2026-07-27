use std::{env, error::Error, fs::File};

use decv_mp4::{Mp4Demuxer, SampleDescription, TrackKind};

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = env::args().skip(1);
    let path = arguments
        .next()
        .ok_or("usage: cargo run -p decv-mp4 --example inspect -- <file.mp4> [--samples]")?;
    let dump_samples = arguments.any(|argument| argument == "--samples");
    let input = File::open(&path)?;
    let demuxer = Mp4Demuxer::open(input)?;
    let movie = demuxer.movie();

    println!(
        "movie timescale={} duration={:?} tracks={}",
        movie.timescale(),
        movie.duration(),
        movie.tracks().len()
    );
    for (track_index, track) in movie.tracks().iter().enumerate() {
        println!(
            "track id={} handler={} timescale={} duration={:?} samples={}",
            track.id(),
            track.handler(),
            track.media_timescale(),
            track.media_duration(),
            track.samples().len()
        );
        match track.presentation_time_offset() {
            Ok(offset) => println!(
                "  presentation offset={}/{} edits={}",
                offset.value,
                offset.timescale,
                track.edits().len()
            ),
            Err(error) => println!(
                "  unsupported presentation mapping: {error} edits={}",
                track.edits().len()
            ),
        }
        for edit in track.edits() {
            println!(
                "  edit duration={} media_time={:?} rate={}.{}",
                edit.segment_duration(),
                edit.media_time(),
                edit.media_rate_integer(),
                edit.media_rate_fraction()
            );
        }
        for description in track.sample_descriptions() {
            match description {
                SampleDescription::Avc(entry) => println!(
                    "  {} {}x{} avcC={} bytes",
                    entry.format(),
                    entry.width(),
                    entry.height(),
                    entry.codec_configuration().len()
                ),
                SampleDescription::Vp9(entry) => {
                    let configuration = entry.codec_configuration();
                    println!(
                        "  {} {}x{} profile={} level={} bit-depth={} chroma={} full-range={}",
                        entry.format(),
                        entry.width(),
                        entry.height(),
                        configuration.profile(),
                        configuration.level(),
                        configuration.bit_depth(),
                        configuration.chroma_subsampling(),
                        configuration.full_range()
                    )
                }
                SampleDescription::Aac(entry) => println!(
                    "  {} {} Hz {} channels AudioSpecificConfig={} bytes",
                    entry.format(),
                    entry.sample_rate(),
                    entry.channel_count(),
                    entry.audio_specific_config().len()
                ),
                SampleDescription::Unsupported { format } => {
                    println!("  unsupported {format}")
                }
                _ => println!("  unknown sample description"),
            }
        }
        if dump_samples {
            for (index, sample) in track.samples().iter().enumerate() {
                let (dts, pts) = if track.kind() == TrackKind::Audio {
                    let packet = demuxer.read_audio_packet(track_index, index)?;
                    (packet.dts, packet.pts)
                } else {
                    let packet = demuxer.read_packet(track_index, index)?;
                    (packet.dts, packet.pts)
                };
                println!(
                    "  sample={index} offset={} size={} raw_dts={} raw_pts={} dts={} pts={} duration={} description={} sync={}",
                    sample.offset(),
                    sample.size(),
                    sample.decode_time(),
                    sample.presentation_time(),
                    dts.map(|time| time.value).unwrap_or_default(),
                    pts.map(|time| time.value).unwrap_or_default(),
                    sample.duration(),
                    sample.description_index(),
                    sample.is_sync()
                );
            }
        }
    }
    Ok(())
}
