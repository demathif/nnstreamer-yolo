# i.MX95 scripts (Neutron NPU, libcamera, Wayland)

Target-specific variants of the pipelines for an **i.MX95** board running Yocto.
The x86 versions stay in `../scripts/`.

## Differences vs ../scripts (x86)

- inference: `tensor_filter framework=tensorflow-lite` with the Neutron delegate
  `custom=UseDefaultDelegates:true,Delegate:External,ExtDelegateLib:libneutron_delegate.so`
- model: `/root/yolov8_neutron.tflite` (int8) and labels `/root/coco_labels.txt`
- camera: `libcamerasrc` (MIPI CSI) with `LIBCAMERA_PIPELINES_MATCH_LIST='nxp/neo,imx8-isi,uvcvideo'`
- sink: `waylandsink`
- no input transpose (TFLite is NHWC); the output transpose `1:0:2:3` is kept
- `imxcompositor_g2d` is used automatically when available

## Prerequisites (expected to be already present on the Yocto image)

- nnstreamer with the tensorflow-lite backend
- `libneutron_delegate.so` (eIQ Neutron)
- `libcamerasrc` (gst-plugin-libcamera) and `waylandsink` (gstreamer wayland plugin)
- the model at `/root/yolov8_neutron.tflite` and the labels at `/root/coco_labels.txt`

## Usage

```sh
imx95-scripts/coco128_yolo.sh                  # images -> waylandsink
imx95-scripts/camera_yolo.sh                   # libcamera -> waylandsink
imx95-scripts/camera_yolo.sh --record-dir out  # JPEG snapshots instead of a window
imx95-scripts/camera_yolo.sh --dry-run         # print the gst-launch command
```

## Options

Both scripts: `--model PATH`, `--normalize OPT`, `--delegate STR`, `--dry-run`, `-h`.
`camera_yolo.sh` also: `--cam-name NAME`, `--format FMT`, `--size WxH`, `--fps N`, `--record-dir DIR`.

## Notes

- `--normalize` is empty by default (the model is assumed to take `uint8`). If the
  tflite expects int8, use `--normalize typecast:int8,add:-128`.
- CPU fallback: `--delegate custom=Delegate:XNNPACK,NumThreads:4`.
- To validate without the board: `--dry-run` prints the exact gst-launch pipeline.
