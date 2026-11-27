# Model files

| File | Resolution | Role / Experiment |
| --- | ---: | --- |
| `v8s_recall_reference.pt` | 640 px | Production single-pass YOLOv8s checkpoint (`fire_s.pt` from Alim Tleuliyev, locally calibrated with per-class peak gating & NMS 0.50; passes ≥85% P & ≥90% R overall, smoke & fire) |
| `v8n_base_640.pt` | 640 px | Lightweight edge YOLOv8n checkpoint (`fire_n.pt` from Alim Tleuliyev, 3.01M params; locally calibrated: 93.8% P / 93.8% R overall, 2.9% negative alarms) |
| `v8n_base_640.onnx` | 640 px | FP32 ONNX export of `v8n_base_640.pt` (`1 × 3 × 640 × 640`, 14.58 CPU FPS on Ryzen 5 5625U) |
| `v8n_aerial_aug.pt` | 640 px | Locally fine-tuned YOLOv8n (aerial small-target + hard-negative curriculum + `close_mosaic=2` + WiSE-FT α=0.35; 74.4% smoke box precision) |
| `v8n_aerial_aug.onnx` | 640 px | FP32 ONNX export of `v8n_aerial_aug.pt` (`1 × 3 × 640 × 640`) |
| `v8n_base.pt` / `v8n_base.onnx` | 256 px | Local 5-epoch from-COCO YOLOv8n baseline (reference ablation in `results/benchmark.json`) |
| `v8s_base.pt` | 256 px | Local 5-epoch from-COCO YOLOv8s baseline |
| `v8n_aug.pt` / `v8n_aug.onnx` | 256 px | Local 5-epoch from-COCO augmented YOLOv8n baseline |

All ONNX files expose raw detection-head outputs (`batch=1`, fixed square input) requiring Ultralytics-compatible letterboxing, normalization, and downstream class-aware NMS (`IoU = 0.50`) + per-class peak gating. See [production_benchmark.json](../results/production_benchmark.json) and [production_quality.json](../results/production_quality.json) for full SHA-256 digests and held-out metrics.

`v8n_base_640.pt` and `v8s_recall_reference.pt` are unmodified `fire-models/fire_n.pt` and `fire-models/fire_s.pt` from [Alim Tleuliyev's repository at commit `a44e1ec`](https://github.com/AlimTleuliyev/wildfire-detection/tree/a44e1ecdb84b06bda145b1c3ad53854046fe14dd/fire-models), redistributed with the preserved [upstream MIT notice](LICENSE-upstream-fire-s.txt). Locally trained/fine-tuned checkpoints and ONNX exports are governed by the [Section 4 AGPL-3.0 license](../LICENSE).
