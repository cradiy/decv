use std::{env, error::Error, fs::File};

use decv_mp4::{Movie, Mp4File, SampleDescription};

fn main() -> Result<(), Box<dyn Error>> {
    let path = env::args()
        .nth(1)
        .ok_or("usage: cargo run -p decv-mp4 --example inspect -- <file.mp4>")?;
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
            "track id={} handler={} timescale={} duration={:?}",
            track.id(),
            track.handler(),
            track.media_timescale(),
            track.media_duration()
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
    }
    Ok(())
}
