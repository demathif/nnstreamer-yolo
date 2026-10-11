#!/usr/bin/env bash
#
# camera_yolo.sh - run the webcam -> nnstreamer -> YOLO detection pipeline.
#
# The pipeline reads frames from the laptop camera, runs a YOLO ONNX model
# through the nnstreamer tensor_filter element, draws the detected bounding
# boxes with the tensor_decoder element, composites them over the source frame
# and shows the result in a window (or writes JPEG snapshots).
#
# Usage:
#   scripts/camera_yolo.sh [OPTIONS]
#
# Options:
#   --device PATH     V4L2 camera device (default: /dev/video0)
#   --caps CAPS       pin the camera format (default: MJPG 640x480@30)
#   --fps N           rate fed to the model (default: 3)
#   --record-dir DIR  write annotated JPEG snapshots instead of a window
#   --dry-run         print the gst-launch command without running it
#   -h, --help        show this help

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEVICE="/dev/video0"                                # V4L2 camera device
CAM_CAPS="image/jpeg,width=640,height=480,framerate=30/1"  # pinned camera format (MJPG)
CAM_FPS=3                                           # rate fed downstream; must be <= what the model can do
MODEL="assets/models/yolov8n.onnx"                 # ONNX YOLO model (prepare.sh)
LABELS="assets/models/coco_labels.txt"             # class names (prepare.sh)
FRAMEWORK="onnxruntime"                             # tensor_filter backend
MODEL_WIDTH=640                                     # YOLO input width
MODEL_HEIGHT=640                                    # YOLO input height
NORMALIZE_DIV="255.0"                               # scale pixels to 0..1
CONF_IOU="1:0.25:0.45"                              # scaled:confidence:IOU (1 = boxes already in pixels)
RECORD_DIR=""                                       # empty = live window; else JPEG snapshots

log() { printf '==> %s\n' "$*"; }

usage() { sed -n '2,19p' "$0"; }

# --- command line ----------------------------------------------------------
DRY_RUN=0
while [ $# -gt 0 ]; do
    case "$1" in
        --device) DEVICE="${2:-}"; shift 2 ;;
        --caps) CAM_CAPS="${2:-}"; shift 2 ;;
        --fps) CAM_FPS="${2:-}"; shift 2 ;;
        --record-dir) RECORD_DIR="${2:-}"; shift 2 ;;
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
MODEL_PATH="$PROJECT_ROOT/$MODEL"
LABELS_PATH="$PROJECT_ROOT/$LABELS"

# --- preflight checks ------------------------------------------------------
command -v gst-launch-1.0 >/dev/null 2>&1 || { echo "error: gst-launch-1.0 not found" >&2; exit 1; }
[ -f "$MODEL_PATH" ] || { echo "error: model not found: $MODEL_PATH (run scripts/prepare.sh)" >&2; exit 1; }
[ -f "$LABELS_PATH" ] || { echo "error: labels not found: $LABELS_PATH (run scripts/prepare.sh)" >&2; exit 1; }
for e in tensor_filter tensor_converter tensor_decoder v4l2src; do
    gst-inspect-1.0 "$e" >/dev/null 2>&1 || { echo "error: GStreamer element '$e' missing (run scripts/setup.sh)" >&2; exit 1; }
done
if [ "$DRY_RUN" -eq 0 ] && [ ! -e "$DEVICE" ]; then
    echo "error: camera device not found: $DEVICE" >&2
    echo "       pick one with --device (available: $(ls /dev/video* 2>/dev/null | tr '\n' ' '))" >&2
    exit 1
fi

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

# --- decoder -----------------------------------------------------------------
# The camera format is pinned, so pick the decoder explicitly instead of relying
# on decodebin autoplugging (which fails delayed linking on live sources).
# MJPG frames need jpegdec; raw formats (e.g. YUY2) need none.
DECODER=()
case "$CAM_CAPS" in
    image/jpeg*) DECODER=('!' jpegdec) ;;
esac

# --- sink ------------------------------------------------------------------
# Each sink starts with a videoconvert so the sink can pick its own format
# (forcing RGB right before autovideosink breaks negotiation on Wayland).
if [ -n "$RECORD_DIR" ]; then
    mkdir -p "$RECORD_DIR"
    SINK=('!' videoconvert '!' jpegenc '!' multifilesink "location=$RECORD_DIR/frame_%06d.jpg")
else
    SINK=('!' videoconvert '!' autovideosink sync=false)
fi

# --- assemble the gst-launch argv ------------------------------------------
# Each P+=(...) appends one token (an element, a property, a pad reference or
# "!"). Building an array avoids shell-quoting problems. The comments below are
# the line-by-line description of the pipeline; run with --dry-run to print it.
P=(gst-launch-1.0 -e)                              # gst-launch; -e sends EOS on exit so files are finalised
P+=(v4l2src "device=$DEVICE")                      # source: capture frames from the camera
P+=('!' "$CAM_CAPS")                               # pin the camera format (avoids live renegotiation)
P+=("${DECODER[@]}")                               # decode MJPG into raw video frames (no-op for raw formats)
P+=('!' videoconvert)                              # convert to a format videoscale can resize
P+=('!' videoscale)                                # resize element (target given by the next caps)
P+=('!' "video/x-raw,format=RGB,width=$MODEL_WIDTH,height=$MODEL_HEIGHT")  # force RGB 640x640 (model input)
P+=('!' videorate)                                 # throttle the camera to the model rate
P+=('!' "video/x-raw,framerate=$CAM_FPS/1")        # so the overlay keeps up and is actually composited
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

if ! "${P[@]}"; then
    echo >&2
    echo "error: the camera pipeline failed." >&2
    echo "  - 'V4l2Src ... not-negotiated' right away usually means the device" >&2
    echo "    is busy or the format is unsupported: close other apps (or another" >&2
    echo "    copy of this script) using it, then retry." >&2
    echo "  - list cameras with 'ls /dev/video*' and select one with --device." >&2
    exit 1
fi

log "done"
if [ -n "$RECORD_DIR" ]; then
    log "annotated images: $RECORD_DIR"
fi
