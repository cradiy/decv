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

verify_stream() {
    local name="$1"
    local input="$work_dir/$name.h264"
    local reference="$work_dir/$name.ffmpeg.nv12"
    local actual="$work_dir/$name.decv.nv12"

    ffmpeg -hide_banner -loglevel error \
        -i "$input" -pix_fmt nv12 -f rawvideo -y "$reference"
    cargo run --quiet --manifest-path "$repo_dir/Cargo.toml" -p decv-cli -- \
        "$input" "$actual" >/dev/null
    cmp "$reference" "$actual"
    echo "$name: byte-exact NV12 match"
}

ffmpeg -hide_banner -loglevel error \
    -f lavfi \
    -i "nullsrc=size=16x16:rate=1,geq=lum='50+10*N':cb=100:cr=150" \
    -frames:v 4 -pix_fmt yuv420p -c:v libx264 -profile:v baseline \
    -x264-params \
    "cabac=0:bframes=0:keyint=30:min-keyint=30:scenecut=0:ref=1:weightp=0:8x8dct=0:deblock=0:no-fast-pskip=1" \
    -f h264 -y "$work_dir/constant-step.h264"
verify_stream constant-step

ffmpeg -hide_banner -loglevel error \
    -f lavfi -i "testsrc2=size=64x48:rate=24" \
    -frames:v 12 -pix_fmt yuv420p -c:v libx264 -profile:v baseline \
    -x264-params \
    "cabac=0:bframes=0:keyint=30:min-keyint=30:scenecut=0:ref=1:weightp=0:8x8dct=0" \
    -f h264 -y "$work_dir/testsrc2.h264"
verify_stream testsrc2

ffmpeg -hide_banner -loglevel error \
    -f lavfi \
    -i "nullsrc=size=32x16:rate=1,geq=lum='50+10*N+X':cb=100:cr=150" \
    -frames:v 2 -pix_fmt yuv420p -c:v libx264 -profile:v high \
    -x264-params \
    "cabac=1:bframes=0:keyint=30:min-keyint=30:scenecut=0:ref=1:weightp=0:8x8dct=1:deblock=0:no-fast-pskip=1" \
    -f h264 -y "$work_dir/cabac-p.h264"
verify_stream cabac-p

ffmpeg -hide_banner -loglevel error \
    -f lavfi \
    -i "nullsrc=size=32x16:rate=2,geq=lum='40+8*N+X':cb='90+N':cr='150-N'" \
    -frames:v 6 -pix_fmt yuv420p -c:v libx264 -profile:v high \
    -x264-params \
    "cabac=1:bframes=2:b-adapt=0:keyint=30:min-keyint=30:scenecut=0:ref=2:weightp=0:weightb=0:8x8dct=1:deblock=0:direct=spatial:no-fast-pskip=1" \
    -f h264 -y "$work_dir/cabac-b.h264"
verify_stream cabac-b

ffmpeg -hide_banner -loglevel error \
    -f lavfi -i "testsrc2=size=64x48:rate=24" \
    -frames:v 16 -pix_fmt yuv420p -c:v libx264 -profile:v main \
    -x264-params \
    "cabac=0:bframes=2:b-adapt=0:keyint=30:min-keyint=30:scenecut=0:ref=1:weightp=0:weightb=0:8x8dct=0:direct=spatial" \
    -f h264 -y "$work_dir/main-b.h264"
verify_stream main-b

ffmpeg -hide_banner -loglevel error \
    -f lavfi -i "testsrc2=size=64x48:rate=24" \
    -frames:v 24 -pix_fmt yuv420p -c:v libx264 -profile:v high \
    -x264-params \
    "cabac=0:bframes=3:b-adapt=0:keyint=48:min-keyint=48:scenecut=0:ref=2:weightp=2:weightb=1:8x8dct=1:direct=auto" \
    -f h264 -y "$work_dir/high-b.h264"
verify_stream high-b
