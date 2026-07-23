#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
frame_count="${1:-60}"
work_dir="$(mktemp -d)"
trap 'rm -rf -- "$work_dir"' EXIT

if ! [[ "$frame_count" =~ ^[1-9][0-9]*$ ]]; then
    echo "usage: $0 [positive-frame-count]" >&2
    exit 2
fi
command -v ffmpeg >/dev/null || {
    echo "ffmpeg is required" >&2
    exit 1
}
ffmpeg -hide_banner -encoders 2>/dev/null | grep -q 'libx264' || {
    echo "ffmpeg must provide the libx264 encoder" >&2
    exit 1
}
test -x /usr/bin/time || {
    echo "/usr/bin/time is required" >&2
    exit 1
}

ffmpeg -hide_banner -loglevel error \
    -f lavfi -i "testsrc2=size=1920x1080:rate=60" \
    -frames:v "$frame_count" -pix_fmt yuv420p \
    -c:v libx264 -preset medium -profile:v high \
    -x264-params \
    "cabac=1:bframes=3:ref=3:weightp=2:weightb=1:8x8dct=1:direct=auto:keyint=60:min-keyint=60:scenecut=0" \
    -f h264 -y "$work_dir/input.h264"

cargo build --quiet --manifest-path "$repo_dir/Cargo.toml" \
    -p decv-cli --release

echo "1920x1080 progressive High Profile software decode"
echo "frames=$frame_count input_bytes=$(wc -c <"$work_dir/input.h264")"
/usr/bin/time \
    -f "elapsed_seconds=%e user_seconds=%U system_seconds=%S max_rss_kb=%M" \
    "$repo_dir/target/release/decv-cli" \
    "$work_dir/input.h264" /dev/null >/dev/null
