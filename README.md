# nnstreamer → YOLO detection (GStreamer demo)

A small GStreamer project that runs a **YOLO** model inside an **nnstreamer**
inference step and draws the detected **bounding boxes**. Two inputs are
provided:

- [`scripts/coco128_yolo.sh`](scripts/coco128_yolo.sh) — reads the **COCO128**
  dataset images and writes annotated JPEGs to `out/`.
- [`scripts/camera_yolo.sh`](scripts/camera_yolo.sh) — reads the **laptop camera**
  (or any V4L2 device) and shows the annotated video in a window (or writes JPEG
  snapshots).

Target: a normal **x86_64 Ubuntu** host. nnstreamer is installed from the
`nnstreamer/ppa` (no source builds, no repository cloning).

## Layout

```text
scripts/setup.sh             install GStreamer + nnstreamer + Python venv (idempotent)
scripts/prepare.sh           download COCO128 + export YOLOv8n to ONNX + labels
scripts/coco128_yolo.sh      detection pipeline, COCO128 image input
scripts/camera_yolo.sh       detection pipeline, camera input (same chain)
scripts/requirements.txt     runtime Python deps
scripts/requirements-export.txt  model-export Python deps
assets/models/               yolov8n.onnx + coco_labels.txt (from prepare.sh)
datasets/coco128/            dataset images (from prepare.sh)
out/                         annotated images (from coco128_yolo.sh)
plans/plan.md                design notes
```

Both pipeline scripts share the same header, configuration block, section
banners and per-element comments; only the source (and its options) differ.

## Quick start

```sh
# 1. System packages: GStreamer, nnstreamer + ONNX Runtime backend (from the
#    nnstreamer PPA) and the project virtual environment. Idempotent; asks for
#    your sudo password. Re-running is safe.
scripts/setup.sh

# 2. Download COCO128, export YOLOv8n to ONNX and write the COCO label list.
#    Needs network access but no sudo.
scripts/prepare.sh

# 3. Run over all 128 images into out/, or show a live window.
scripts/coco128_yolo.sh
scripts/coco128_yolo.sh --limit 5
scripts/coco128_yolo.sh --display

# 4. Or run on the laptop camera.
scripts/camera_yolo.sh
```

## Options

`scripts/coco128_yolo.sh`:

| Option | Effect |
|--------|--------|
| `--limit N` | process only the first N images (default: all) |
| `--display` | show a live window instead of writing files |
| `--to-file` | write annotated JPEGs (default) |
| `--dry-run` | print the `gst-launch` command without running it |
| `-h`, `--help` | show the help |

`scripts/camera_yolo.sh`:

| Option | Effect |
|--------|--------|
| `--device PATH` | V4L2 camera device (default: `/dev/video0`) |
| `--caps CAPS` | pin the camera format (default: MJPG 640x480@30) |
| `--fps N` | rate fed to the model (default: 3) |
| `--record-dir DIR` | write annotated JPEG snapshots instead of a window |
| `--dry-run` | print the `gst-launch` command without running it |
| `-h`, `--help` | show the help |

```sh
scripts/camera_yolo.sh                       # live window, /dev/video0
scripts/camera_yolo.sh --device /dev/video1  # pick another camera
scripts/camera_yolo.sh --fps 5               # faster rate (if the CPU keeps up)
scripts/camera_yolo.sh --caps "video/x-raw,format=YUY2,width=640,height=480,framerate=30/1"
scripts/camera_yolo.sh --record-dir out/cam  # JPEG snapshots instead of a window
```

## Pipeline — images (COCO128)

```text
multifilesrc (staged coco128/*.jpg) ! jpegdec ! videoconvert ! videoscale
  ! video/x-raw,format=RGB,width=640,height=640
  ! tee name=t
      t. ! queue ! videoconvert                                      ! mix.sink_0
      t. ! queue ! tensor_converter
                 ! tensor_transform mode=transpose option=1:2:0:3
                 ! tensor_transform mode=arithmetic option=typecast:float32,div:255.0
                 ! tensor_filter framework=onnxruntime model=assets/models/yolov8n.onnx
                 ! tensor_transform mode=transpose option=1:0:2:3
                 ! tensor_decoder mode=bounding_boxes option1=yolov8 ...
                 ! queue                                             ! mix.sink_1
  ! compositor name=mix ! videoconvert
  ! video/x-raw,format=RGB,width=640,height=640
  ! videoconvert ! jpegenc ! multifilesink location=out/detected_%04d.jpg
```

## Pipeline — camera

```text
v4l2src device=/dev/video0
  ! image/jpeg,width=640,height=480,framerate=30/1 ! jpegdec
  ! videoconvert ! videoscale ! video/x-raw,format=RGB,width=640,height=640
  ! videorate ! video/x-raw,framerate=3/1
  ! tee name=t
      (same two branches and the same inference chain as above)
  ! compositor name=mix ! videoconvert
  ! video/x-raw,format=RGB,width=640,height=640
  ! videoconvert ! autovideosink sync=false
```

`tensor_decoder` produces an RGBA overlay with a transparent background, so the
boxes are composited over the original frame with `compositor`.

## Why the two transposes

nnstreamer stores tensor dimensions in the reverse order of the ONNX model, and
`tensor_converter` emits a `[channels][width][height][frames]` tensor whose
memory is interleaved **HWC**:

- **Input**: the filter expects the model input `[1,3,640,640]` as a
  `[height][width][channels][batch]` tensor, so
  `tensor_transform mode=transpose option=1:2:0:3` reorders it to CHW (NCHW).
  The arithmetic transform then casts to float32 and scales to 0..1.
- **Output**: the model returns `[1,84,8400]`. The bounding-box decoder reads a
  box-major buffer (`buffer[box*84 + class]`), so
  `tensor_transform mode=transpose option=1:0:2:3` reorders the tensor to match.

`tensor_decoder` options used:

| option | value | meaning |
|--------|-------|---------|
| `option1` | `yolov8` | box decoding scheme |
| `option2` | `assets/models/coco_labels.txt` | the 80 COCO class names |
| `option3` | `1:0.25:0.45` | boxes are in pixels; confidence 0.25; IoU 0.45 |
| `option4` | `640:640` | overlay video size |
| `option5` | `640:640` | model input size (mandatory for YOLO) |

## Configuration

The knobs live at the top of each script: model path, `tensor_filter` framework,
input size, normalization divisor, decoder options and the output. The camera
script adds `DEVICE`, `CAM_CAPS`, `CAM_FPS` and `RECORD_DIR`.

## Verified

- **COCO128**: all 128 images processed, `out/detected_0000.jpg ..
  out/detected_0127.jpg`, labelled boxes such as `zebra`, `bowl`, `broccoli`,
  `cow`, `laptop`, `person`.
- **Camera** (`/dev/video0`, HP HD Camera): live window runs with no errors;
  recording produced frames with drawn boxes (`person`, `chair`).

## Troubleshooting

- **`tensor_filter: no such element` or `framework=onnxruntime` unknown** —
  nnstreamer/ONNX Runtime is not installed; run `scripts/setup.sh`, then check
  `gst-inspect-1.0 tensor_filter`.
- **`not-negotiated` around `tensor_filter`** — the tensor shape does not match
  the model; check the input transpose (`option=1:2:0:3`) and the model input.
- **No boxes drawn (images)** — the boxes are filtered out or drawn off-frame;
  adjust `option3` (first field and the confidence threshold).
- **Camera: `V4l2Src ... not-negotiated` at startup** — the device is busy or the
  format is unsupported: close other apps using the camera, or pick another
  device/format with `--device`/`--caps`.
- **Camera: `decodebin: Delayed linking failed`** — not used anymore (the decoder
  is selected explicitly from `--caps`).
- **Camera: window shows no boxes** — the model must keep up with the source;
  lower `--fps`. The compositor only blends the overlay when both branches run at
  the same rate.
- **Live window fails to negotiate** — each sink starts with a `videoconvert` so
  the sink can pick its own format; forcing RGB right before `autovideosink`
  breaks negotiation on Wayland.
- **`compositor` missing** — the scripts fall back to `videomixer`.
