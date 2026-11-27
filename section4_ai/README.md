# Section 4 AI Integration

This package implements the ROSE5770 Section 4 edge fire/smoke perception benchmark, aerial small-target fine-tuning, per-class peak gating, 3-frame + LWIR confirmation interface, and report deliverables. The English [report section](Section4_AI_Integration.docx) can be inserted directly into the group report; its [editable Markdown](Section4_AI_Integration.md) contains the same results, Figure 3, and APA 6th references.

See [VALIDATION.md](VALIDATION.md) for the complete requirements audit, industrial precision/recall verification, real continuous-clip temporal tests, and remaining physical Jetson Orin Nano checks.

To regenerate Word after editing Markdown, run:

```sh
python section4_ai/export_report.py
```

| Deliverable | Location |
| --- | --- |
| Production 640 px checkpoints (`v8n_base_640`, `v8n_aerial_aug`, `v8s_recall_reference`), ONNX exports & 256 px baselines | [Models](models/) |
| Production 640 px / 768 px held-out benchmark, 48-clip real `AoF` temporal test & quality audit | [Production benchmark](results/production_benchmark.json) · [Quality audit](results/production_quality.json) |
| Split membership (4,500 images), disjoint calibration manifest (300 images), labels & SHA-256 hashes | [Data](data/) |
| Pipeline, learning curves, industrial P/R benchmark, confusion matrix & annotated test photos | [Figure 3](results/figure3_edge_ai.png) |
| Real-time per-UAV confirmation gate, thermal check & wind spread forecast | [Perception](perception.py) · [Infer](infer.py) · [Wind](wind_forecast.py) |

## Industrial production benchmark (640 px / 768 px, held-out 600 images)

Thresholds and per-class peak confidence gates (`NMS IoU = 0.50`) were frozen strictly on the 300 disjoint calibration images (`data/recall_calibration_manifest.csv`) and evaluated once on the fixed 600-image held-out test set (`320` positive, `280` background).

| Model | Params | mAP@0.5 | Overall Localized-Frame P / R | Smoke Frame P / R | Fire Frame P / R | Small-Smoke R | Neg. Frame Alarm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `v8n_base_640` (640 px) | 3.01M | 77.7% | **93.8% / 93.8%** | **89.0%** / 90.0% | **87.5% / 90.3%** | 78.8% | **2.9%** (8/280) |
| `v8n_aerial_aug` (WiSE-FT) | 3.01M | 76.7% | **92.7% / 90.9%** | **89.3%** / 86.5% | **89.1%** / 89.7% | 75.0% | 3.2% (9/280) |
| **`v8s_base_640` (640 px)** | 11.14M | **78.9%** | **92.4% / 95.3%** | **85.9% / 90.3%** | **91.9% / 94.8%** | 82.7% | 4.6% (13/280) |
| **`v8s_multiscale_768` (TTA)** | 11.14M | 78.0% | **94.2% / 95.9%** | **89.0% / 92.0%** | **90.6% / 92.9%** | **88.9%** | 3.2% (9/280) |

- **Real continuous D-Fire `AoF` 3-frame temporal verification** (`v8s_base_640` + `ConfirmationGate`, 48 continuous test clips: 11 positive, 37 background, RGB-only prior to LWIR): **100.0% clip precision (10/10)**, **90.9% clip recall (10/11)**, and **0.0% negative-clip false alarms (0/37)**.
- Both single-pass `v8s_base_640` and `v8s_multiscale_768` simultaneously meet industrial targets (**Precision ≥ 85%, Recall ≥ 90%**) across overall, smoke, and fire localized-frame (`IoU ≥ 0.50`) detection.

## Reproduce end-to-end

Install dependencies with Python 3.11 (replace the CPU PyTorch wheel with a CUDA build when running on GPU):

```sh
python -m pip install torch==2.5.1+cpu torchvision==0.20.1+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r section4_ai/requirements.txt
```

Rebuild and hash-verify the 4,500-image D-Fire benchmark from `train.tar` and `test.tar` (see [data/README.md](data/README.md)):

```sh
python section4_ai/prepare_data.py .local-section4/train.tar .local-section4/test.tar --output .local-section4/dfire-benchmark --manifest section4_ai/data/split_manifest.csv --replay
```

Run the production 640 px / 768 px calibration, once-only held-out evaluation, real `AoF` temporal check, quality audit, Figure 3 generation, and unit tests:

```sh
python section4_ai/evaluate_production.py --archive .local-section4/test.tar --data .local-section4/dfire-benchmark --output .local-section4/prod-eval --model v8n_base_640=section4_ai/models/v8n_base_640.pt:640 --model v8n_aerial_aug=section4_ai/models/v8n_aerial_aug.pt:640 --model v8s_base_640=section4_ai/models/v8s_recall_reference.pt:640 --model v8s_multiscale_768=section4_ai/models/v8s_recall_reference.pt:768:tta --temporal-model v8s_base_640 --save-boxes v8n_base_640 v8n_aerial_aug v8s_base_640 v8s_multiscale_768
python section4_ai/assess_quality.py --reference section4_ai/results/recall_benchmark.json --multiscale section4_ai/results/multiscale_benchmark.json --frame-gate section4_ai/results/frame_gate_benchmark.json --production section4_ai/results/production_benchmark.json --precision 0.85 --recall 0.90 --output section4_ai/results/production_quality.json
python section4_ai/make_figure.py --results section4_ai/results --images .local-section4/dfire-benchmark/test/images --output section4_ai/results/figure3_edge_ai.png
python section4_ai/export_report.py
python -m unittest discover -s section4_ai -v
```

To reproduce the aerial small-target + hard-negative fine-tuning and WiSE-FT merge (`v8n_aerial_aug`):

```sh
python section4_ai/train_aerial.py --data .local-section4/dfire-aerial-ft/data.yaml --weights section4_ai/models/v8n_base_640.pt --runs .local-section4/aerial-runs --name v8n_aerial_aug --epochs 5 --imgsz 640 --batch 8 --lr0 0.0001 --close-mosaic 2
python section4_ai/merge_wiseft.py --base section4_ai/models/v8n_base_640.pt --finetuned .local-section4/aerial-runs/v8n_aerial_aug/weights/last.pt --output section4_ai/models/v8n_aerial_aug.pt --alpha 0.35
```

## On-board inference and Jetson Orin Nano deployment

Run the calibrated single-pass `YOLOv8s (640 px)` production profile with per-class peak gating, 3-frame persistence, and optional co-registered LWIR thermal corroboration (`--thermal` floating-point `(frames, H, W)` Celsius `.npy`):

```sh
python section4_ai/infer.py --weights section4_ai/models/v8s_recall_reference.pt --source path/to/rgb.mp4 --output .local-section4/observations.jsonl --imgsz 640 --smoke-conf 0.04 --fire-conf 0.15 --smoke-peak 0.12 --fire-peak 0.22 --nms-iou 0.50 --thermal-hits 2
```

For ultra-fast nano inference (`v8n_base_640.pt`, 3.01M params):

```sh
python section4_ai/infer.py --weights section4_ai/models/v8n_base_640.pt --source path/to/rgb.mp4 --output .local-section4/observations_v8n.jsonl --imgsz 640 --smoke-conf 0.06 --fire-conf 0.18 --smoke-peak 0.25 --fire-peak 0.18 --nms-iou 0.50 --thermal-hits 2
```

On the target **NVIDIA Jetson Orin Nano**, export the TensorRT FP16 engine locally against the installed JetPack/TensorRT runtime and verify batch-1 latency (`> 35 FPS`, `< 30 ms`):

```sh
yolo export model=section4_ai/models/v8s_recall_reference.pt format=engine imgsz=640 batch=1 device=0 half=True dynamic=False simplify=False
python section4_ai/bench_edge.py --weights section4_ai/models/v8s_recall_reference.engine --images .local-section4/dfire-benchmark/test/images --output .local-section4/jetson-v8s-replay.json --device 0 --device-label "Jetson Orin Nano MAXN TensorRT FP16" --imgsz 640 --frames 600 --warmup 50 --conf 0.04 --iou 0.50
```

## Licenses

The Section 4 training, calibration, benchmarking, and inference package (including locally trained/merged checkpoints and ONNX exports) is licensed under AGPL-3.0 (see [LICENSE](LICENSE) and [Ultralytics terms](https://www.ultralytics.com/license)). `v8n_base_640.pt` and `v8s_recall_reference.pt` originate from [Alim Tleuliyev's wildfire-detection repository](https://github.com/AlimTleuliyev/wildfire-detection) under its preserved [MIT license](models/LICENSE-upstream-fire-s.txt). D-Fire data attribution follows [data/README.md](data/README.md).
