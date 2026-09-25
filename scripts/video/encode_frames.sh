#!/usr/bin/env bash
# Assemble a directory of PNG frames into an mp4.
#   bash scripts/video/encode_frames.sh <frames dir> <fps> <out.mp4>
# Accepts the runner's frame_NNNNNN.png, Isaac Movie Capture's name.NNNN.png,
# or any other PNG naming that sorts in frame order. If <frames dir> holds no
# PNGs but has subdirectories that do (a VIDEO_FRAMES_DIR root of timestamped
# runs), the newest one is used. FPS sets playback speed: the sim writes
# camera.fps frames per sim second (30 in the dual-arm scene), so 30 plays at
# true speed, 20 at ~0.67x. Uses libx264 when this ffmpeg has it, else NVENC.
set -euo pipefail
DIR="${1:?frames directory}"
FPS="${2:?playback fps}"
OUT="${3:?output mp4}"

[ -d "$DIR" ] || { echo "ERROR: not a directory: $DIR" >&2; exit 1; }
count_png() { find "$1" -maxdepth 1 -name '*.png' | wc -l; }
if [ "$(count_png "$DIR")" -eq 0 ]; then
  newest=""
  for sub in "$DIR"/*/; do
    [ -d "$sub" ] && [ "$(count_png "$sub")" -gt 0 ] && newest="$sub"
  done
  [ -n "$newest" ] || { echo "ERROR: no .png frames in $DIR or its subdirectories" >&2; exit 1; }
  DIR="${newest%/}"
  echo "using newest run: $DIR"
fi
N="$(count_png "$DIR")"

# A distinct file across every .png name: the suffix that changes between the
# first and last frame is the frame index; the rest is the pattern.
names="$(find "$DIR" -maxdepth 1 -name '*.png' -printf '%f\n' | sort)"
first="$(sed -n '1p' <<<"$names")"
last="$(sed -n '$p' <<<"$names")"
if [ "$N" -gt 1 ] && [ "${#first}" -ne "${#last}" ]; then
  echo "ERROR: frame names are not fixed-width numbers ($first .. $last); ffmpeg needs zero-padded names" >&2
  exit 1
fi
# Locate the trailing digit run before .png: prefix + digits + .png.
if [[ "$first" =~ ^(.*[^0-9])?([0-9]+)\.png$ ]]; then
  prefix="${BASH_REMATCH[1]}"; digits="${BASH_REMATCH[2]}"
else
  echo "ERROR: cannot find a frame number in '$first'" >&2; exit 1
fi
PATTERN="$DIR/${prefix}%0${#digits}d.png"
START="$((10#$digits))"

if ffmpeg -hide_banner -encoders 2>/dev/null | grep -q " libx264 "; then
  CODEC=(-c:v libx264 -crf 18)
else
  CODEC=(-c:v h264_nvenc -preset p7 -rc vbr -cq 19 -b:v 0)
fi
ffmpeg -y -hide_banner -framerate "$FPS" -start_number "$START" -i "$PATTERN" "${CODEC[@]}" -pix_fmt yuv420p "$OUT"
echo "wrote $OUT ($N frames at $FPS fps, pattern $(basename "$PATTERN") from $START)"
