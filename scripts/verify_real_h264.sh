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
    -i "nullsrc=size=32x32:rate=1,geq=lum='40+X+Y':cb='90+X':cr='150-Y'" \
    -frames:v 1 -pix_fmt yuv420p -c:v libx264 -profile:v high -qp 30 \
    -x264-params \
    "cabac=1:bframes=0:keyint=1:scenecut=0:8x8dct=1:deblock=0" \
    -f h264 -y "$work_dir/cabac-i-neighbours.h264"
verify_stream cabac-i-neighbours

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
    -f lavfi -i "testsrc2=size=128x72:rate=24" \
    -frames:v 24 -pix_fmt yuv420p -c:v libx264 -profile:v high \
    -x264-params \
    "cabac=1:bframes=3:ref=3:weightp=2:weightb=1:8x8dct=1:direct=auto" \
    -f h264 -y "$work_dir/cabac-realistic.h264"
verify_stream cabac-realistic

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

ffmpeg -hide_banner -loglevel error \
    -f lavfi -i "testsrc2=size=64x48:rate=4" \
    -frames:v 4 -pix_fmt yuv420p -c:v libx264 -profile:v baseline \
    -x264-params \
    "cabac=0:bframes=0:keyint=4:min-keyint=4:scenecut=0" \
    -f h264 -y "$work_dir/size-a.h264"
ffmpeg -hide_banner -loglevel error \
    -f lavfi -i "testsrc2=size=96x64:rate=4" \
    -frames:v 4 -pix_fmt yuv420p -c:v libx264 -profile:v baseline \
    -x264-params \
    "cabac=0:bframes=0:keyint=4:min-keyint=4:scenecut=0" \
    -f h264 -y "$work_dir/size-b.h264"
ffmpeg -hide_banner -loglevel error -y \
    -i "concat:$work_dir/size-a.h264|$work_dir/size-b.h264" \
    -c copy -f h264 "$work_dir/size-change.h264"
ffmpeg -hide_banner -loglevel error \
    -i "$work_dir/size-a.h264" -pix_fmt nv12 -f rawvideo \
    -y "$work_dir/size-a.ffmpeg.nv12"
ffmpeg -hide_banner -loglevel error \
    -i "$work_dir/size-b.h264" -pix_fmt nv12 -f rawvideo \
    -y "$work_dir/size-b.ffmpeg.nv12"
cargo run --quiet --manifest-path "$repo_dir/Cargo.toml" -p decv-cli -- \
    "$work_dir/size-change.h264" "$work_dir/size-change.decv.nv12" \
    >"$work_dir/size-change.log"

first_size="$(wc -c <"$work_dir/size-a.ffmpeg.nv12")"
dd if="$work_dir/size-change.decv.nv12" bs=1 count="$first_size" status=none \
    | cmp - "$work_dir/size-a.ffmpeg.nv12"
dd if="$work_dir/size-change.decv.nv12" bs=1 skip="$first_size" status=none \
    | cmp - "$work_dir/size-b.ffmpeg.nv12"
rg -q '^format 64x48 Nv12$' "$work_dir/size-change.log"
rg -q '^format 96x64 Nv12$' "$work_dir/size-change.log"
echo "size-change: both FormatChanged events and byte-exact NV12 segments match"
