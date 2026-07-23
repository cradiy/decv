#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
frame_count="${1:-60}"
run_count="${2:-5}"
work_dir="$(mktemp -d)"
trap 'rm -rf -- "$work_dir"' EXIT

if ! [[ "$frame_count" =~ ^[1-9][0-9]*$ && "$run_count" =~ ^[1-9][0-9]*$ ]]; then
    echo "usage: $0 [positive-frame-count] [positive-run-count]" >&2
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

median() {
    sort -n "$1" | awk '
        { values[NR] = $1 }
        END {
            middle = int((NR + 1) / 2)
            if (NR % 2) {
                print values[middle]
            } else {
                printf "%.3f\n", (values[middle] + values[middle + 1]) / 2
            }
        }
    '
}

benchmark() {
    local key="$1"
    local label="$2"
    shift 2
    local elapsed_file="$work_dir/$key.elapsed"
    local user_file="$work_dir/$key.user"
    local rss_file="$work_dir/$key.rss"
    local timing_file="$work_dir/$key.time"
    : >"$elapsed_file"
    : >"$user_file"
    : >"$rss_file"

    "$@" >/dev/null 2>/dev/null
    for ((run = 1; run <= run_count; run++)); do
        LC_ALL=C /usr/bin/time -f "%e %U %S %M" -o "$timing_file" \
            "$@" >/dev/null 2>/dev/null
        read -r elapsed user system rss <"$timing_file"
        echo "$elapsed" >>"$elapsed_file"
        echo "$user" >>"$user_file"
        echo "$rss" >>"$rss_file"
        printf "%-30s run=%d elapsed=%ss user=%ss system=%ss rss=%sKiB\n" \
            "$label" "$run" "$elapsed" "$user" "$system" "$rss"
    done

    printf "%-30s median_elapsed=%ss median_user=%ss median_rss=%sKiB\n" \
        "$label" "$(median "$elapsed_file")" "$(median "$user_file")" "$(median "$rss_file")"
}

input="$work_dir/input.h264"
decv="$repo_dir/target/release/decv-cli"

echo "1920x1080 60fps High Profile CABAC software decode"
echo "frames=$frame_count runs=$run_count input_bytes=$(wc -c <"$input")"
ffmpeg -version | sed -n '1p'
echo "NV12 cases include pixel-format conversion/packing and writes to /dev/null."

benchmark decv_serial "decv serial -> NV12" \
    "$decv" --parallelism serial "$input" /dev/null
benchmark decv_auto "decv auto -> NV12" \
    "$decv" --parallelism auto "$input" /dev/null
benchmark ffmpeg_1_nv12 "FFmpeg 1 thread -> NV12" \
    ffmpeg -hide_banner -loglevel error -threads 1 -filter_threads 1 -i "$input" \
    -map 0:v:0 -an -sn -pix_fmt nv12 -f rawvideo -y /dev/null
benchmark ffmpeg_auto_nv12 "FFmpeg auto -> NV12" \
    ffmpeg -hide_banner -loglevel error -threads 0 -i "$input" \
    -map 0:v:0 -an -sn -pix_fmt nv12 -f rawvideo -y /dev/null
benchmark ffmpeg_1_null "FFmpeg 1 thread decode-only" \
    ffmpeg -hide_banner -loglevel error -threads 1 -filter_threads 1 -i "$input" \
    -map 0:v:0 -an -sn -f null -y /dev/null
benchmark ffmpeg_auto_null "FFmpeg auto decode-only" \
    ffmpeg -hide_banner -loglevel error -threads 0 -i "$input" \
    -map 0:v:0 -an -sn -f null -y /dev/null
