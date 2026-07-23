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

ffmpeg -hide_banner -loglevel error \
    -f lavfi -i "testsrc2=size=320x180:rate=24" \
    -frames:v 24 -pix_fmt yuv420p -c:v libx264 -profile:v high \
    -x264-params \
    "cabac=1:bframes=3:ref=3:weightp=2:weightb=1:8x8dct=1:direct=auto:keyint=24:min-keyint=24:scenecut=0" \
    -movflags +faststart -y "$input"

ffmpeg -hide_banner -loglevel error \
    -i "$input" -pix_fmt nv12 -f rawvideo -y "$reference"
cargo build --quiet --manifest-path "$repo_dir/Cargo.toml" \
    -p decv-cli --release
"$repo_dir/target/release/decv-cli" "$input" "$actual" >/dev/null

cmp "$reference" "$actual"
echo "mp4-avc: 24-frame byte-exact NV12 match ($(wc -c <"$actual") bytes)"
