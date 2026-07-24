#!/usr/bin/env bash
set -euo pipefail

usage() {
    printf 'usage: %s <representative .h264 or .mp4 input>...\n' "$0" >&2
}

if (($# == 0)); then
    usage
    exit 2
fi

for input in "$@"; do
    if [[ ! -f "$input" ]]; then
        printf 'training input does not exist: %s\n' "$input" >&2
        exit 2
    fi
done

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
host="$(rustc -vV | sed -n 's/^host: //p')"
sysroot="$(rustc --print sysroot)"
llvm_profdata="$sysroot/lib/rustlib/$host/bin/llvm-profdata"
if [[ ! -x "$llvm_profdata" ]]; then
    printf 'llvm-profdata for the active Rust toolchain is missing.\n' >&2
    printf 'install it with: rustup component add llvm-tools-preview\n' >&2
    exit 1
fi
if ! command -v sha256sum >/dev/null; then
    printf 'sha256sum is required to fingerprint PGO training inputs\n' >&2
    exit 1
fi

pgo_target_dir="${DECV_PGO_TARGET_DIR:-"$repo_dir/target/pgo"}"
instrumented_target_dir="$pgo_target_dir-instrumented"
profile_dir="$pgo_target_dir-data"
merged_profile="$profile_dir/merged.profdata"
training_manifest="$profile_dir/training-manifest.tsv"

mkdir -p "$profile_dir"
find "$profile_dir" -maxdepth 1 -type f \
    \( -name '*.profraw' -o -name '*.profdata' \) -delete

native_rustflags="${RUSTFLAGS:-}"
if [[ -n "$native_rustflags" ]]; then
    native_rustflags+=" "
fi
native_rustflags+="-C target-cpu=native"

{
    printf 'format\tdecv-pgo-training-v1\n'
    printf 'rustc\t%s\n' "$(rustc -V)"
    printf 'rustflags\t%s\n' "$native_rustflags"
    printf 'parallelism\tserial,auto\n'
    for input in "$@"; do
        canonical_input="$(realpath -- "$input")"
        if [[ "$canonical_input" == *$'\t'* || "$canonical_input" == *$'\n'* ]]; then
            printf 'training input path contains a tab or newline: %q\n' "$canonical_input" >&2
            exit 2
        fi
        input_size="$(wc -c <"$input")"
        input_checksum="$(sha256sum -- "$input")"
        input_checksum="${input_checksum%% *}"
        printf 'input\t%s\t%s\t%s\n' \
            "$input_checksum" "$input_size" "$canonical_input"
    done
} >"$training_manifest"

printf 'building instrumented decoder...\n'
env \
    RUSTFLAGS="$native_rustflags -C profile-generate=$profile_dir" \
    CARGO_TARGET_DIR="$instrumented_target_dir" \
    cargo build \
        --manifest-path "$repo_dir/Cargo.toml" \
        --release \
        -p decv-cli

instrumented_decoder="$instrumented_target_dir/release/decv-cli"
for input in "$@"; do
    for parallelism in serial auto; do
        printf 'training %-6s %s\n' "$parallelism" "$input"
        LLVM_PROFILE_FILE="$profile_dir/decv-%m-%p.profraw" \
            "$instrumented_decoder" \
            --parallelism "$parallelism" \
            "$input" \
            /dev/null \
            >/dev/null
    done
done

shopt -s nullglob
raw_profiles=("$profile_dir"/decv-*.profraw)
if ((${#raw_profiles[@]} == 0)); then
    printf 'instrumented decoder produced no profile data\n' >&2
    exit 1
fi
"$llvm_profdata" merge -o "$merged_profile" "${raw_profiles[@]}"

printf 'building profile-guided decoder...\n'
env \
    RUSTFLAGS="$native_rustflags -C profile-use=$merged_profile -C llvm-args=-pgo-warn-missing-function" \
    CARGO_TARGET_DIR="$pgo_target_dir" \
    cargo build \
        --manifest-path "$repo_dir/Cargo.toml" \
        --release \
        -p decv-cli

printf 'PGO release artifact: %s/release/decv-cli\n' "$pgo_target_dir"
printf 'PGO training manifest: %s\n' "$training_manifest"
