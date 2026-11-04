#!/bin/sh
set -u
cd "${1:?independent variant directory}"
mkdir -p results
cpu=/sys/devices/system/cpu/cpu0/cpufreq
saved=$(cat "$cpu/scaling_governor")
restore() { echo "$saved" > "$cpu/scaling_governor"; }
trap restore EXIT INT TERM
echo performance > "$cpu/scaling_governor"
cat /proc/cmdline > results/bootargs.txt
cat /proc/meminfo > results/meminfo-before.txt
cat /proc/rk_dma_heap/alloc_bitmap > results/dma-before.txt
cat "$cpu/scaling_cur_freq" > results/cpu-khz.txt
if [ -f /sys/kernel/debug/rknpu/freq ]; then cat /sys/kernel/debug/rknpu/freq > results/npu-frequency.txt; fi
rm -f results/metrics.json results/init-memory.json results/logits.f32 results/states.f32
./delta-bench "${2:-2048}" > results/board.log 2>&1
rc=$?
echo "$rc" > results/exit-code.txt
cat /proc/meminfo > results/meminfo-after.txt
cat /proc/rk_dma_heap/alloc_bitmap > results/dma-after.txt
restore
trap - EXIT INT TERM
echo "EXIT_CODE=$rc GOVERNOR_RESTORED=$(cat "$cpu/scaling_governor")"
tail -n 3 results/board.log
exit "$rc"
