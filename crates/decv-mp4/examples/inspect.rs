use std::{env, error::Error, fs::File};

use decv_mp4::{Movie, Mp4File, SampleDescription};

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = env::args().skip(1);
    let path = arguments
        .next()
        .ok_or("usage: cargo run -p decv-mp4 --example inspect -- <file.mp4> [--samples]")?;
    let dump_samples = arguments.any(|argument| argument == "--samples");
    let input = File::open(&path)?;
    let file = Mp4File::open(&input)?;
    let movie = Movie::parse(file)?;

    println!(
        "movie timescale={} duration={:?} tracks={}",
        movie.timescale(),
        movie.duration(),
        movie.tracks().len()
    );
    for track in movie.tracks() {
        println!(
            "track id={} handler={} timescale={} duration={:?} samples={}",
            track.id(),
            track.handler(),
            track.media_timescale(),
            track.media_duration(),
            track.samples().len()
        );
        for description in track.sample_descriptions() {
            match description {
                SampleDescription::Avc(entry) => println!(
                    "  {} {}x{} avcC={} bytes",
                    entry.format(),
                    entry.width(),
                    entry.height(),
                    entry.codec_configuration().len()
                ),
                SampleDescription::Unsupported { format } => {
                    println!("  unsupported {format}")
                }
            }
        }
        if dump_samples {
            for (index, sample) in track.samples().iter().enumerate() {
                println!(
                    "  sample={index} offset={} size={} dts={} pts={} duration={} description={} sync={}",
                    sample.offset(),
                    sample.size(),
                    sample.decode_time(),
                    sample.presentation_time(),
                    sample.duration(),
                    sample.description_index(),
                    sample.is_sync()
                );
            }
        }
    }
    Ok(())
}
