#!/usr/bin/env bash
#
# prepare.sh - fetch the dataset and the YOLO model used by the pipeline.
#
# Every step is skipped when its artifact already exists, so this script is
# idempotent and safe to re-run. It only needs network access, curl/unzip and
# python3-venv; no sudo and no nnstreamer are required.
#
# Steps:
#   1. Download and extract the COCO128 dataset into datasets/coco128.
#   2. Export the YOLOv8n model to ONNX into assets/models.
#   3. Write the COCO class-name label file used by the bounding-box decoder.
#
# Usage:
#   scripts/prepare.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

VENV_DIR="$PROJECT_ROOT/.venv"
REQ_EXPORT="$SCRIPT_DIR/requirements-export.txt"

DATASETS_DIR="$PROJECT_ROOT/datasets"
IMAGES_DIR="$DATASETS_DIR/coco128/images/train2017"

MODELS_DIR="$PROJECT_ROOT/assets/models"
MODEL_NAME="yolov8n"
MODEL_FILE="$MODELS_DIR/$MODEL_NAME.onnx"
LABELS_FILE="$MODELS_DIR/coco_labels.txt"

# COCO128 archives, tried in order.
COCO_URLS="
https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip
https://ultralytics.com/assets/coco128.zip
"

# COCO128 has 128 images; use it as the completeness threshold.
EXPECTED_IMAGES=128

log() { printf '==> %s\n' "$*"; }

# count_images -> number of JPEG files currently extracted.
count_images() {
    find "$IMAGES_DIR" -maxdepth 1 -name '*.jpg' 2>/dev/null | wc -l
}

# images_present -> succeed when the dataset looks complete.
images_present() {
    [ -d "$IMAGES_DIR" ] && [ "$(count_images)" -ge "$EXPECTED_IMAGES" ]
}

# download_coco128 -> download and extract the dataset when needed.
download_coco128() {
    if images_present; then
        echo "COCO128 already present at $IMAGES_DIR"
        return 0
    fi

    mkdir -p "$DATASETS_DIR"
    local archive="$DATASETS_DIR/coco128.zip"
    local url ok=0

    for url in $COCO_URLS; do
        log "downloading COCO128 from $url"
        if curl -fL --retry 3 -o "$archive" "$url"; then
            ok=1
            break
        fi
        echo "download failed, trying the next mirror" >&2
    done

    if [ "$ok" -ne 1 ]; then
        echo "error: could not download the COCO128 archive" >&2
        exit 1
    fi

    log "extracting $archive"
    unzip -q -o "$archive" -d "$DATASETS_DIR"
    rm -f "$archive"

    if ! images_present; then
        echo "error: COCO128 extraction looks incomplete ($IMAGES_DIR)" >&2
        exit 1
    fi
    echo "COCO128 images: $(count_images)"
}

# export_model -> export YOLOv8n to ONNX when the file is missing.
export_model() {
    if [ -f "$MODEL_FILE" ]; then
        echo "model already present at $MODEL_FILE"
        return 0
    fi

    mkdir -p "$MODELS_DIR"

    # The virtual environment only holds the export tooling and can be created
    # without root, because python3-venv is part of the base packages.
    if [ ! -x "$VENV_DIR/bin/python" ]; then
        log "creating virtual environment at $VENV_DIR"
        python3 -m venv "$VENV_DIR"
        "$VENV_DIR/bin/pip" install --upgrade pip
    fi

    # The default PyPI index pulls in large CUDA wheels; on a CPU host the
    # CPU-only wheels are much smaller and sufficient for the ONNX export.
    log "installing a CPU-only PyTorch build"
    "$VENV_DIR/bin/pip" install --index-url https://download.pytorch.org/whl/cpu \
        torch torchvision || echo "CPU torch install failed; using the default index" >&2

    log "installing export dependencies"
    "$VENV_DIR/bin/pip" install -r "$REQ_EXPORT"

    log "exporting $MODEL_NAME to ONNX"
    # ultralytics downloads yolov8n.pt automatically and writes yolov8n.onnx
    # next to it, so run from the models directory.
    ( cd "$MODELS_DIR" && "$VENV_DIR/bin/yolo" export "model=$MODEL_NAME.pt" format=onnx )

    if [ ! -f "$MODEL_FILE" ]; then
        echo "error: ONNX export did not produce $MODEL_FILE" >&2
        exit 1
    fi
}

# write_labels -> write the COCO class-name list required by the decoder.
write_labels() {
    if [ -s "$LABELS_FILE" ]; then
        echo "label file already present at $LABELS_FILE"
        return 0
    fi

    mkdir -p "$MODELS_DIR"
    log "writing the COCO class-name label file"
    # The order must match the YOLO / COCO class indices (0..79).
    cat > "$LABELS_FILE" <<'EOF'
person
bicycle
car
motorcycle
airplane
bus
train
truck
boat
traffic light
fire hydrant
stop sign
parking meter
bench
bird
cat
dog
horse
sheep
cow
elephant
bear
zebra
giraffe
backpack
umbrella
handbag
tie
suitcase
frisbee
skis
snowboard
sports ball
kite
baseball bat
baseball glove
skateboard
surfboard
tennis racket
bottle
wine glass
cup
fork
knife
spoon
bowl
banana
apple
sandwich
orange
broccoli
carrot
hot dog
pizza
donut
cake
chair
couch
potted plant
bed
dining table
toilet
tv
laptop
mouse
remote
keyboard
cell phone
microwave
oven
toaster
sink
refrigerator
book
clock
vase
scissors
teddy bear
hair drier
toothbrush
EOF
}

# ---------------------------------------------------------------------------
log "preparing dataset"
download_coco128

log "preparing model"
export_model

log "preparing labels"
write_labels

log "summary"
echo "images : $IMAGES_DIR ($(count_images) files)"
echo "model  : $MODEL_FILE"
echo "labels : $LABELS_FILE"
