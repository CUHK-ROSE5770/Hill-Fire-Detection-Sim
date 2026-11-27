# Model files

| File | Experiment |
| --- | --- |
| `v8n_base.pt` | YOLOv8n, baseline augmentations |
| `v8n_base.onnx` | Baseline YOLOv8n in FP32 for the edge deployment trial |
| `v8s_base.pt` | YOLOv8s, baseline augmentations |
| `v8n_aug.pt` | YOLOv8n with mosaic, scale jitter and vertical flip |
| `v8n_aug.onnx` | Augmented YOLOv8n in FP32 for comparison |

The `.pt` files are the best checkpoints by validation fitness. The ONNX graphs expose detection head output with fixed `1 × 3 × 256 × 256` RGB input. Apply Ultralytics-compatible letterboxing, normalization and downstream NMS. The TensorRT FP16 engine can be built on the target device using the [deployment instructions](../README.md#perception-and-deployment). See the [benchmark table](../results/benchmark.csv) and [full metrics](../results/benchmark.json) for selection and calibration details.

Transfer learning started from the Ultralytics v8.3.0 assets. Initial SHA-256 values are provided for reproduction.

| Initial weights | SHA-256 |
| --- | --- |
| `yolov8n.pt` | `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36` |
| `yolov8s.pt` | `1f47a78bf100391c2a140b7ac73a1caae18c32779be7d310658112f7ac9aa78a` |

The checkpoints and ONNX export are covered by the [Section 4 AGPL-3.0 license](../LICENSE).
