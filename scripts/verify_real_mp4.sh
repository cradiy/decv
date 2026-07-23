#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
work_dir="$(mktemp -d)"
trap 'rm -rf -- "$work_dir"' EXIT

command -v ffmpeg >/dev/null || {
    echo "ffmpeg is required" >&2
    exit 1
}
ffmpeg -hide_banner -encoders 2>/dev/null | grep -q 'libx264' || {
    echo "ffmpeg must provide the libx264 encoder" >&2
    exit 1
}

input="$work_dir/input.mp4"
reference="$work_dir/ffmpeg.nv12"
actual="$work_dir/decv.nv12"
seek_reference="$work_dir/ffmpeg-seek.nv12"
seek_actual="$work_dir/decv-seek.nv12"

ffmpeg -hide_banner -loglevel error \
    -f lavfi -i "testsrc2=size=320x180:rate=24" \
    -frames:v 48 -pix_fmt yuv420p -c:v libx264 -profile:v high \
    -x264-params \
    "cabac=1:bframes=3:ref=3:weightp=2:weightb=1:8x8dct=1:direct=auto:keyint=12:min-keyint=12:scenecut=0" \
    -movflags +faststart -y "$input"

ffmpeg -hide_banner -loglevel error \
    -i "$input" -pix_fmt nv12 -f rawvideo -y "$reference"
cargo build --quiet --manifest-path "$repo_dir/Cargo.toml" \
    -p decv-cli --release
"$repo_dir/target/release/decv-cli" "$input" "$actual" >/dev/null

cmp "$reference" "$actual"
echo "mp4-avc: 48-frame byte-exact NV12 match ($(wc -c <"$actual") bytes)"

ffmpeg -hide_banner -loglevel error \
    -ss 1.37 -i "$input" -pix_fmt nv12 -f rawvideo -y "$seek_reference"
"$repo_dir/target/release/decv-cli" \
    --seek 1.37 "$input" "$seek_actual" >/dev/null

cmp "$seek_reference" "$seek_actual"
echo "mp4-seek: byte-exact NV12 suffix match ($(wc -c <"$seek_actual") bytes)"
