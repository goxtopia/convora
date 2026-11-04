#!/usr/bin/env bash
set -euo pipefail
: "${CXX:?Set CXX to the Luckfox ARM uClibc hard-float g++ compiler}"
release_root="$(cd "$(dirname "$0")/../.." && pwd)"
out="$release_root/build/board"
runtime_libdir="${RKNN_LIBDIR:-$release_root/runtime/pico-chat/board}"
test -f "$runtime_libdir/librknnmrt.so" || { echo 'Supply matching SDK librknnmrt.so via RKNN_LIBDIR' >&2; exit 1; }
mkdir -p "$out"
for pair in 'bench.cpp:delta-bench' 'dump.cpp:delta-dump' 'runner.cpp:runner'; do
  source_file="${pair%%:*}"
  output_file="${pair##*:}"
  "$CXX" -O3 -std=c++17 -mcpu=cortex-a7 -mfpu=neon-vfpv4 -mfloat-abi=hard \
    -static-libstdc++ -static-libgcc "$release_root/tests/board/$source_file" \
    -I "$release_root/models/rknn" -I "$release_root/runtime/pico-chat/board" \
    -L "$runtime_libdir" -lrknnmrt \
    -Wl,-rpath,/oem/usr/lib -lpthread -ldl -lm -o "$out/$output_file"
done
