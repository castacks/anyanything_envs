#!/bin/bash
# GPU render timing on airstation03's 5090, queued through sched5090 from gs2 (anyanything/bin/sched5090): scenes
# rendered with the Starling front camera by bl52_frame.py, Blender 5.2, OptiX, 64 and 32 samples, 4 renders each.
# Each scene is staged from gs2 into /dev/shm (airstation03's root disk is nearly full), rendered, and removed; the
# Blender process's peak GPU memory is sampled once a second.
#
#   sched5090 add --repo anyanything_envs --vram 16 --cls oneoff --min-disk 2 --timeout 1h --note "..." -- \
#       bash infinigen/tools/gpu_timing.sh NAME=GS2_BLEND[,GS2_VIEWS_JSON] ...
#
# Writes out/gpu_timing/: <name>.log (its BL52_FRAME line), <name>_<spp>spp_0.png, gpu.txt, summary.txt. Fetch with
#   airstation fetch anyanything_envs out/gpu_timing <dest>
set -u
B=${BLENDER:-$HOME/opt/blender-5.2.2-linux-x64/blender}
SRC=${GPU_TIMING_SRC:-gs2}
OUT=out/gpu_timing
STAGE=/dev/shm/gpu_timing_${SCHED5090_ID:-$$}
mkdir -p $OUT $STAGE
trap 'rm -rf $STAGE' EXIT
export TMPDIR=$STAGE
nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv,noheader > $OUT/gpu.txt
for spec in "$@"; do
  name=${spec%%=*}; rest=${spec#*=}; blend=${rest%%,*}; views=""
  [[ $rest == *,* ]] && views=${rest#*,}
  t0=$(date +%s)
  scp -q "$SRC:$blend" $STAGE/$name.blend || { echo "$name copy failed" | tee -a $OUT/summary.txt; continue; }
  vw=()
  [ -n "$views" ] && scp -q "$SRC:$views" $STAGE/$name.views.json && vw=(--views $STAGE/$name.views.json --view 0)
  echo "$name staged $(du -h $STAGE/$name.blend | cut -f1) in $(( $(date +%s) - t0 )) s" | tee -a $OUT/summary.txt
  $B -b --factory-startup --python infinigen/tools/bl52_frame.py -- --scene $STAGE/$name.blend --out $OUT/$name.png \
      "${vw[@]}" --samples 64 32 --renders 4 --device OPTIX > $OUT/$name.log 2>&1 &
  pid=$!; peak=0
  while kill -0 $pid 2>/dev/null; do
    m=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -F', ' -v p=$pid '$1 == p {print $2}')
    [ -n "$m" ] && [ "$m" -gt "$peak" ] && peak=$m
    sleep 1
  done
  wait $pid
  if grep -q BL52_FRAME $OUT/$name.log; then
    grep BL52_FRAME $OUT/$name.log | sed "s/^/$name /" | tee -a $OUT/summary.txt
  else
    echo "$name render failed: $(grep -i -m1 'error' $OUT/$name.log)" | tee -a $OUT/summary.txt
  fi
  echo "$name vram_peak_mib $peak" | tee -a $OUT/summary.txt
  rm -f $STAGE/$name.blend
done
echo GPU_TIMING_DONE | tee -a $OUT/summary.txt
