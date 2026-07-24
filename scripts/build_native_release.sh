#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
native_target_dir="${DECV_NATIVE_TARGET_DIR:-"$repo_dir/target/native"}"
native_rustflags="${RUSTFLAGS:-}"
if [[ -n "$native_rustflags" ]]; then
    native_rustflags+=" "
fi
native_rustflags+="-C target-cpu=native"

if (($# == 0)); then
    set -- -p decv-cli
fi

env RUSTFLAGS="$native_rustflags" cargo build \
    --manifest-path "$repo_dir/Cargo.toml" \
    --target-dir "$native_target_dir" \
    --release \
    "$@"

printf 'native release artifacts: %s/release\n' "$native_target_dir"
