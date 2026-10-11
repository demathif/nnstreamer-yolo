#!/usr/bin/env bash
#
# camera_yolo.sh - run the camera -> nnstreamer -> YOLO detection pipeline.
#
# The pipeline reads frames from the MIPI CSI camera (libcamera, nxp/neo), runs
# a YOLO int8 TFLite model on the Neutron NPU through the nnstreamer
# tensor_filter element, draws the detected bounding boxes with the
# tensor_decoder element, composites them over the source frame and shows the
# result with waylandsink (or writes JPEG snapshots).
#
# Usage:
#   imx95-scripts/camera_yolo.sh [OPTIONS]
#
# Options:
#   --cam-name NAME   libcamera camera-name (default: first camera)
#   --caps CAPS       pin the camera format (default: YUY2 1920x1080@60)
#   --record-dir DIR  write annotated JPEG snapshots instead of a window
#   --dry-run         print the gst-launch command without running it
#   -h, --help        show this help

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CAM_NAME=""                                         # libcamera camera-name ("" = first camera)
CAM_CAPS="video/x-raw,format=YUY2,width=1920,height=1080,framerate=60/1"  # pinned camera format (MIPI CSI)
MODEL="/root/yolov8_neutron.tflite"                 # int8 TFLite YOLO model (on the board)
LABELS="/root/coco_labels.txt"                      # class names (on the board)
FRAMEWORK="tensorflow-lite"                         # tensor_filter backend
MODEL_WIDTH=640                                     # YOLO input width
MODEL_HEIGHT=640                                    # YOLO input height
DELEGATE="custom=UseDefaultDelegates:true,Delegate:External,ExtDelegateLib:libneutron_delegate.so"  # Neutron NPU delegate
CONF_IOU="1:0.25:0.45"                              # scaled:confidence:IOU (1 = boxes already in pixels)
RECORD_DIR=""                                       # empty = waylandsink; else JPEG snapshots

log() { printf '==> %s\n' "$*"; }

usage() { sed -n '2,18p' "$0"; }

# libcamera must select the "nxp/neo" pipeline handler on i.MX95.
: "${LIBCAMERA_PIPELINES_MATCH_LIST:=nxp/neo,imx8-isi,uvcvideo}"
export LIBCAMERA_PIPELINES_MATCH_LIST

# --- command line ----------------------------------------------------------
DRY_RUN=0
while [ $# -gt 0 ]; do
    case "$1" in
        --cam-name) CAM_NAME="${2:-}"; shift 2 ;;
        --caps) CAM_CAPS="${2:-}"; shift 2 ;;
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
MODEL_PATH="$MODEL"
LABELS_PATH="$LABELS"

# --- preflight checks ------------------------------------------------------
command -v gst-launch-1.0 >/dev/null 2>&1 || { echo "error: gst-launch-1.0 not found" >&2; exit 1; }
if [ "$DRY_RUN" -eq 0 ]; then
    [ -f "$MODEL_PATH" ] || { echo "error: model not found: $MODEL_PATH" >&2; exit 1; }
    [ -f "$LABELS_PATH" ] || { echo "error: labels not found: $LABELS_PATH" >&2; exit 1; }
    for e in tensor_filter tensor_converter tensor_decoder libcamerasrc waylandsink; do
        gst-inspect-1.0 "$e" >/dev/null 2>&1 || { echo "error: GStreamer element '$e' missing" >&2; exit 1; }
    done
fi

# Pick a video mixer; the G2D compositor is preferred on i.MX95.
MIXER="compositor"
if gst-inspect-1.0 imxcompositor_g2d >/dev/null 2>&1; then
    MIXER="imxcompositor_g2d"
elif ! gst-inspect-1.0 compositor >/dev/null 2>&1; then
    if gst-inspect-1.0 videomixer >/dev/null 2>&1; then
        MIXER="videomixer"
    else
        echo "error: neither 'imxcompositor_g2d', 'compositor' nor 'videomixer' is available" >&2
        exit 1
    fi
fi

# --- decoder -----------------------------------------------------------------
# RAW camera frames (YUY2) need no decoder.
DECODER=()
case "$CAM_CAPS" in
    image/jpeg*) DECODER=('!' jpegdec) ;;
esac

# --- sink ------------------------------------------------------------------
# Each sink starts with a videoconvert so the sink can pick its own format.
if [ -n "$RECORD_DIR" ]; then
    mkdir -p "$RECORD_DIR"
    SINK=('!' videoconvert '!' jpegenc '!' multifilesink "location=$RECORD_DIR/frame_%06d.jpg")
else
    SINK=('!' videoconvert '!' waylandsink sync=false)
fi

# --- assemble the gst-launch argv ------------------------------------------
# Each P+=(...) appends one token (an element, a property, a pad reference or
# "!"). Building an array avoids shell-quoting problems. The comments below are
# the line-by-line description of the pipeline; run with --dry-run to print it.
P=(gst-launch-1.0 -e)                              # gst-launch; -e sends EOS on exit so files are finalised
P+=(libcamerasrc)                                  # source: MIPI CSI camera through libcamera
if [ -n "$CAM_NAME" ]; then                        # select a camera by name (default: first)
    P+=("camera-name=$CAM_NAME")
fi
P+=('!' queue)                                     # decouple the live camera
P+=('!' "$CAM_CAPS")                               # pin the camera format (avoids live renegotiation)
P+=("${DECODER[@]}")                               # decode MJPG into raw video frames (no-op for raw formats)
P+=('!' videoconvert)                              # convert to a format videoscale can resize
P+=('!' videoscale)                                # resize element (target given by the next caps)
P+=('!' "video/x-raw,format=RGB,width=$MODEL_WIDTH,height=$MODEL_HEIGHT")  # force RGB 640x640 (model input)
P+=('!' tee name=t)                                # tee = duplicate the stream into two branches

# Branch A: the plain frame is the bottom layer of the mixer.
P+=(t. '!' queue '!' videoconvert '!' mix.sink_0)  # tee src -> queue -> compositor bottom layer (original image)

# Branch B: inference then the decoded overlay as the top layer.
#   tensor_converter yields [channels][width][height][frames]; TFLite wants NHWC
#   and nnstreamer reverses the model dimensions, so the two match: no input
#   transpose is needed and the uint8 tensor is fed as is.
P+=(t. '!' queue '!' tensor_converter)             # tee src -> raw video converted into a tensor
P+=('!' tensor_filter "framework=$FRAMEWORK" "model=$MODEL_PATH")     # nnstreamer inference (Neutron NPU)
P+=("$DELEGATE")                                   # external TFLite delegate (Neutron NPU)
# The model output (1,84,8400) becomes [8400][84][1]; the bounding-box decoder
# wants dim[0]=84 and dim[1]=8400, so swap the first two dimensions.
P+=('!' tensor_transform mode=transpose option=1:0:2:3)               # output back to box-major [84][8400]
P+=('!' tensor_decoder mode=bounding_boxes option1=yolov8 "option2=$LABELS_PATH" \
    "option3=$CONF_IOU" "option4=$MODEL_WIDTH:$MODEL_HEIGHT" \
    "option5=$MODEL_WIDTH:$MODEL_HEIGHT")          # draw boxes + labels -> RGBA overlay (transparent background)
P+=('!' queue '!' mix.sink_1)                      # tee src -> compositor top layer (the overlay)

# Mixer and sink.
P+=("$MIXER" name=mix)                             # compositor: blend the overlay over the source frame
P+=('!' videoconvert)                              # convert to a format the sink can handle
P+=('!' "video/x-raw,format=RGB,width=$MODEL_WIDTH,height=$MODEL_HEIGHT")  # fix the output size (compositor defaults to 1x1)
P+=("${SINK[@]}")                                  # output: '! jpegenc ! multifilesink' or '! waylandsink'

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
    echo "  - list cameras with 'cam -l' and pass --cam-name if needed" >&2
    echo "  - check LIBCAMERA_PIPELINES_MATCH_LIST includes nxp/neo" >&2
    echo "  - check the model input dtype (uint8 vs int8)" >&2
    exit 1
fi

log "done"
if [ -n "$RECORD_DIR" ]; then
    log "annotated images: $RECORD_DIR"
fi
