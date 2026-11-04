#!/usr/bin/env bash
set -euo pipefail
cd "${1:?build directory required}"
: "${CXX:?Set CXX to the Luckfox ARM uClibc hard-float g++ compiler}"
runtime_libdir="${RKNN_LIBDIR:-.}"
"$CXX" -O3 -std=c++17 -mcpu=cortex-a7 -mfpu=neon-vfpv4 -mfloat-abi=hard -static-libstdc++ -static-libgcc chatd.cpp -L "$runtime_libdir" -lrknnmrt -Wl,-rpath,/oem/usr/lib -lpthread -ldl -lm -o pico-chatd
"${STRIP:-${CXX%g++}strip}" pico-chatd
