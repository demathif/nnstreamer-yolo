#!/usr/bin/env bash
#
# coco128_yolo.sh - run the COCO128 -> nnstreamer -> YOLO detection pipeline.
#
# The pipeline reads the COCO128 JPEGs, runs a YOLO ONNX model through the
# nnstreamer tensor_filter element, draws the detected bounding boxes with the
# tensor_decoder element, composites them over the source frame and writes the
# annotated images to out/ (or shows them in a window).
#
# Usage:
#   scripts/coco128_yolo.sh [OPTIONS]
#
# Options:
#   --limit N     process only the first N images (default: all)
#   --display     show a live window instead of writing files
#   --to-file     write annotated JPEGs (default)
#   --dry-run     print the gst-launch command without running it
#   -h, --help    show this help

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
COCO_IMAGES="datasets/coco128/images/train2017"   # COCO128 images (prepare.sh)
MODEL="assets/models/yolov8n.onnx"                 # ONNX YOLO model (prepare.sh)
LABELS="assets/models/coco_labels.txt"             # class names (prepare.sh)
OUT_DIR="out"                                       # annotated images
FRAMEWORK="onnxruntime"                             # tensor_filter backend
MODEL_WIDTH=640                                     # YOLO input width
MODEL_HEIGHT=640                                    # YOLO input height
NORMALIZE_DIV="255.0"                               # scale pixels to 0..1
CONF_IOU="1:0.25:0.45"                              # scaled:confidence:IOU (1 = boxes already in pixels)
LIMIT=0                                             # 0 = all images
SINK_TYPE="file"                                    # "file" or "display"

log() { printf '==> %s\n' "$*"; }

usage() { sed -n '2,19p' "$0"; }

# --- command line ----------------------------------------------------------
DRY_RUN=0
while [ $# -gt 0 ]; do
    case "$1" in
        --limit) LIMIT="${2:-}"; shift 2 ;;
        --display) SINK_TYPE="display"; shift ;;
        --to-file) SINK_TYPE="file"; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            echo "error: unknown option: $1 (try --help)" >&2
            exit 2
            ;;
    esac
done

# --- resolve configured paths ---------------------------------------------
COCO_IMAGES_PATH="$PROJECT_ROOT/$COCO_IMAGES"
MODEL_PATH="$PROJECT_ROOT/$MODEL"
LABELS_PATH="$PROJECT_ROOT/$LABELS"
OUT_PATH="$PROJECT_ROOT/$OUT_DIR"

# --- preflight checks ------------------------------------------------------
command -v gst-launch-1.0 >/dev/null 2>&1 || { echo "error: gst-launch-1.0 not found" >&2; exit 1; }
[ -d "$COCO_IMAGES_PATH" ] || { echo "error: images not found: $COCO_IMAGES_PATH (run scripts/prepare.sh)" >&2; exit 1; }
[ -f "$MODEL_PATH" ] || { echo "error: model not found: $MODEL_PATH (run scripts/prepare.sh)" >&2; exit 1; }
[ -f "$LABELS_PATH" ] || { echo "error: labels not found: $LABELS_PATH (run scripts/prepare.sh)" >&2; exit 1; }
for e in tensor_filter tensor_converter tensor_decoder; do
    gst-inspect-1.0 "$e" >/dev/null 2>&1 || { echo "error: nnstreamer element '$e' missing (run scripts/setup.sh)" >&2; exit 1; }
done

mkdir -p "$OUT_PATH"

# Pick a video mixer; compositor is preferred, videomixer is the fallback.
MIXER="compositor"
if ! gst-inspect-1.0 compositor >/dev/null 2>&1; then
    if gst-inspect-1.0 videomixer >/dev/null 2>&1; then
        MIXER="videomixer"
    else
        echo "error: neither 'compositor' nor 'videomixer' is available" >&2
        exit 1
    fi
fi

# --- source -----------------------------------------------------------------
# multifilesrc does not expand globs: it needs a printf-style pattern. So the
# sorted COCO128 images are staged as symlinks named img_%04d.jpg and
# multifilesrc walks them via start-index/stop-index. --limit caps how many
# images are staged.
STAGE_DIR="$OUT_PATH/.stage_$$"
cleanup() { rm -rf "$STAGE_DIR"; }
trap cleanup EXIT
mkdir -p "$STAGE_DIR"

total=0
while IFS= read -r img; do
    if [ "${LIMIT:-0}" -gt 0 ] 2>/dev/null && [ "$total" -ge "$LIMIT" ]; then
        break
    fi
    ln -sf "$img" "$STAGE_DIR/$(printf 'img_%04d.jpg' "$total")"
    total=$((total + 1))
done < <(find "$COCO_IMAGES_PATH" -maxdepth 1 -name '*.jpg' | sort)

if [ "$total" -eq 0 ]; then
    echo "error: no JPEG images found in $COCO_IMAGES_PATH" >&2
    exit 1
fi
SRCLOC="$STAGE_DIR/img_%04d.jpg"
SRC_PROPS=("start-index=0" "stop-index=$((total - 1))")
if [ "${LIMIT:-0}" -gt 0 ] 2>/dev/null; then
    log "limited run: $total image(s)"
else
    log "processing $total image(s)"
fi

# --- sink ------------------------------------------------------------------
# Each sink starts with a videoconvert so the sink can pick its own format
# (forcing RGB right before autovideosink breaks negotiation on Wayland).
if [ "$SINK_TYPE" = "display" ]; then
    SINK=('!' videoconvert '!' autovideosink sync=false)
else
    SINK=('!' videoconvert '!' jpegenc '!' multifilesink "location=$OUT_PATH/detected_%04d.jpg")
fi

# --- assemble the gst-launch argv ------------------------------------------
# Each P+=(...) appends one token (an element, a property, a pad reference or
# "!"). Building an array avoids shell-quoting problems. The comments below are
# the line-by-line description of the pipeline; run with --dry-run to print it.
P=(gst-launch-1.0 -e)                              # gst-launch; -e sends EOS on exit so files are finalised
P+=(multifilesrc "location=$SRCLOC")               # source: read the staged img_%04d.jpg files in order
if [ "${#SRC_PROPS[@]}" -gt 0 ]; then              # bounded range (used by --limit)
    P+=("${SRC_PROPS[@]}")                         #   start-index / stop-index for multifilesrc
fi
P+=("caps=image/jpeg,framerate=1/1")               # tell multifilesrc the files are JPEG at 1 frame/s
P+=('!' jpegdec)                                   # decode each JPEG into raw video frames
P+=('!' videoconvert)                              # convert to a format videoscale can resize
P+=('!' videoscale)                                # resize element (target given by the next caps)
P+=('!' "video/x-raw,format=RGB,width=$MODEL_WIDTH,height=$MODEL_HEIGHT")  # force RGB 640x640 (model input)
P+=('!' tee name=t)                                # tee = duplicate the stream into two branches

# Branch A: the plain frame is the bottom layer of the mixer.
P+=(t. '!' queue '!' videoconvert '!' mix.sink_0)  # tee src -> queue -> compositor bottom layer (original image)

# Branch B: inference then the decoded overlay as the top layer.
#   tensor_converter yields [channels][width][height][frames]; nnstreamer
#   reverses the ONNX model dimensions at negotiation, so the filter expects
#   [height][width][channels][batch] -> transpose with option=1:2:0:3, then
#   cast to float32 and scale to 0..1.
P+=(t. '!' queue '!' tensor_converter)             # tee src -> raw video converted into a tensor
P+=('!' tensor_transform mode=transpose option=1:2:0:3)               # reorder to CHW (NCHW)
P+=('!' tensor_transform mode=arithmetic "option=typecast:float32,div:$NORMALIZE_DIV")  # uint8 0..255 -> float32 0..1
P+=('!' tensor_filter "framework=$FRAMEWORK" "model=$MODEL_PATH")     # nnstreamer inference (onnxruntime)
# The model output (1,84,8400) becomes [8400][84][1]; the bounding-box decoder
# wants dim[0]=84 and dim[1]=8400, so swap the first two dimensions.
P+=('!' tensor_transform mode=transpose option=1:0:2:3)               # output back to box-major [84][8400]
P+=('!' tensor_decoder mode=bounding_boxes option1=yolov8 "option2=$LABELS_PATH" \
    "option3=$CONF_IOU" "option4=$MODEL_WIDTH:$MODEL_HEIGHT" \
    "option5=$MODEL_WIDTH:$MODEL_HEIGHT")          # draw boxes + labels -> RGBA overlay (transparent background)
P+=('!' queue '!' mix.sink_1)                      # tee src -> compositor top layer (the overlay)

# Mixer and sink.
P+=("$MIXER" name=mix)                             # compositor: blend the overlay over the source frame
P+=('!' videoconvert)                              # convert to a format the encoder can handle
P+=('!' "video/x-raw,format=RGB,width=$MODEL_WIDTH,height=$MODEL_HEIGHT")  # fix the output size (compositor defaults to 1x1)
P+=("${SINK[@]}")                                  # output: '! jpegenc ! multifilesink' or '! autovideosink'

# --- run -------------------------------------------------------------------
log "running pipeline"
if [ "$DRY_RUN" -eq 1 ]; then
    printf ' %q' "${P[@]}"
    printf '\n'
    exit 0
fi

"${P[@]}"

log "done"
if [ "$SINK_TYPE" = "file" ]; then
    log "annotated images: $OUT_PATH"
fi
