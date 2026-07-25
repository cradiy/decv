#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
work_dir="$(mktemp -d)"
trap 'rm -rf -- "$work_dir"' EXIT

command -v ffmpeg >/dev/null || {
    echo "ffmpeg is required" >&2
    exit 1
}
command -v python3 >/dev/null || {
    echo "python3 is required for the floating-point comparison" >&2
    exit 1
}
ffmpeg -hide_banner -encoders 2>/dev/null | grep -q ' aac ' || {
    echo "ffmpeg must provide the native AAC encoder" >&2
    exit 1
}

input="$work_dir/input.m4a"
reference="$work_dir/ffmpeg.f32le"
actual="$work_dir/decv.f32le"

ffmpeg -hide_banner -loglevel error \
    -f lavfi -i "sine=frequency=997:sample_rate=48000:duration=2" \
    -af "pan=stereo|c0=c0|c1=0.375*c0" \
    -c:a aac -profile:a aac_low -b:a 192k \
    -movflags +faststart -y "$input"

ffmpeg -hide_banner -loglevel error -ignore_editlist 1 \
    -i "$input" -map 0:a:0 -acodec pcm_f32le -f f32le -y "$reference"

cargo build --quiet --manifest-path "$repo_dir/Cargo.toml" \
    -p decv-aac --example decode_mp4 --release
"$repo_dir/target/release/examples/decode_mp4" \
    "$input" "$actual" >/dev/null

python3 - "$reference" "$actual" <<'PY'
import math
import struct
import sys

reference_path, actual_path = sys.argv[1:]
with open(reference_path, "rb") as reference_file:
    reference = reference_file.read()
with open(actual_path, "rb") as actual_file:
    actual = actual_file.read()

if len(reference) != len(actual):
    raise SystemExit(
        f"AAC PCM length mismatch: ffmpeg={len(reference)} decv={len(actual)}"
    )
if len(actual) % 4:
    raise SystemExit("AAC PCM output is not aligned to f32 samples")

maximum = 0.0
sum_absolute = 0.0
sum_squared = 0.0
sample_count = len(actual) // 4
for (expected,), (decoded,) in zip(
    struct.iter_unpack("<f", reference),
    struct.iter_unpack("<f", actual),
):
    if not math.isfinite(decoded):
        raise SystemExit("decv produced a non-finite AAC PCM sample")
    error = abs(expected - decoded)
    maximum = max(maximum, error)
    sum_absolute += error
    sum_squared += error * error

tolerance = 1.0e-4
if maximum > tolerance:
    raise SystemExit(
        f"AAC PCM error exceeds tolerance: max={maximum:.9g} limit={tolerance:.9g}"
    )

mean = sum_absolute / sample_count
rms = math.sqrt(sum_squared / sample_count)
print(
    f"aac-lc: {sample_count} interleaved f32 samples, "
    f"max_abs={maximum:.9g}, mean_abs={mean:.9g}, rms={rms:.9g}"
)
PY
