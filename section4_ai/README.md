# Section 4 AI Integration

This package contains the ROSE5770 D-Fire benchmark and RGB-to-alert interface. The English [report section](Section4_AI_Integration.docx) can be inserted into the group report. Its [editable Markdown](Section4_AI_Integration.md) includes the same results, Figure 3 and APA 6 references.

The [requirements audit](VALIDATION.md) maps each Section 4 requirement to the available evidence, including the simulated timing and LWIR tests and the remaining physical Jetson checks.

To regenerate Word after editing the Markdown, install `python-docx` and run `python section4_ai/export_report.py`.

| Deliverable | Location |
| --- | --- |
| Three fine-tuned PyTorch checkpoints and both nano variants in ONNX | [Models](models/) |
| Published YOLOv8s checkpoint with a locally calibrated high-recall profile | [Recall benchmark](results/recall_benchmark.json) |
| Split membership, label archive, hashes and source information | [Data](data/) |
| Held-out metrics, calibration sweep, training histories and test box coordinates | [Results](results/) |
| Pipeline, learning curves, confusion matrix and annotated test photographs | [Figure 3](results/figure3_edge_ai.png) |
| Simulation observation contract and Jetson validation procedure | [Perception and deployment](#perception-and-deployment) |

## Reproduce the benchmark

Run from the repository root with Python 3.11. The archived trial used PyTorch 2.5.1+cpu, torchvision 0.20.1+cpu and the dependencies listed below. Install a device-appropriate PyTorch build when repeating on a GPU.

```sh
python -m pip install torch==2.5.1+cpu torchvision==0.20.1+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r section4_ai/requirements.txt
```

Download the two D-Fire archives listed in the [data instructions](data/README.md) into `.local-section4`, then rebuild the 4,500 split members and verify each image and annotation hash. The preparation script also accepts the extracted archives as input.

```sh
python section4_ai/prepare_data.py .local-section4/train.tar .local-section4/test.tar --output .local-section4/dfire-4500 --manifest section4_ai/data/split_manifest.csv --replay
```

Download the official [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt) and [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt) COCO weights into `.local-section4/init`. Train and evaluate into a fresh local run directory.

```sh
python section4_ai/train.py --data .local-section4/dfire-4500/data.yaml --runs .local-section4/runs --init .local-section4/init
python section4_ai/evaluate.py --data .local-section4/dfire-4500/data.yaml --runs .local-section4/runs --results .local-section4/results --models .local-section4/models
python section4_ai/make_figure.py --results .local-section4/results --images .local-section4/dfire-4500/test/images --output .local-section4/results/figure3_edge_ai.png
python -m unittest discover -s section4_ai -v
```

All arms use the same seed, 256 px input, freeze policy, optimizer, epoch budget, train/validation data, and test data. The baseline retains HSV jitter and horizontal flipping. The augmented nano arm adds mosaic (1.0), random scale (0.35), and vertical flipping (0.10). Compare the two baseline arms for model size and the two nano arms for augmentation. The ablation estimates the effect of the combined augmentation settings at one seed.

Test mAP@0.5 and mAP@0.5:0.95 use Ultralytics' AP calculation with a confidence floor of 0.001 and class-aware NMS at IoU 0.70. Operating-point recall and precision use greedy, confidence-ordered, one-to-one same-class matching at IoU ≥ 0.50. A small instance covers fewer than 1,024 pixels² after resizing and letterboxing. Validation selects maximum macro precision among grid points with both class recalls above 90%, falling back to maximum macro F2 when no grid point qualifies. The JSON also provides common-threshold metrics at 0.10. The confusion matrix preserves these true-positive matches before pairing remaining wrong-class predictions with unmatched labels at IoU ≥ 0.50. Its background row and column count unmatched boxes, with no true-negative box count. Negative-frame alarm rate is the share of images with empty annotations and at least one retained RGB box. These image-level measures precede temporal and thermal filtering.

CPU speed is reciprocal mean batch-one RGB preprocessing, inference and NMS time on the same test images at a confidence floor of 0.001, after excluding the first 10 frames. It excludes file decode, thermal alignment, tracking and network transport. For equipment comparisons, compare the `pipeline_mean_ms` and `pipeline_p95_ms` fields using the same input and evaluation settings.

Figure 3 overlays selected dataset and model boxes on four fixed [D-Fire](https://github.com/gaia-solutions-on-demand/DFireDataset) test photographs. Other boxes on those frames remain available in the per-image JSON records. The photos retain their original aspect ratios and camera markings.

## High-recall profile

The high-recall profile uses the [published YOLOv8s weights](https://github.com/AlimTleuliyev/wildfire-detection/tree/a44e1ecdb84b06bda145b1c3ad53854046fe14dd), which the author reports training on the official D-Fire train split. We calibrated smoke and fire confidence separately on [300 calibration images](data/recall_calibration_manifest.csv) drawn from the official test archive and excluded the 600 benchmark images by filename and SHA-256. The fixed 600-image test set was evaluated with the calibrated thresholds at 640 px.

| Class | Confidence | Matched / annotated boxes | Recall at IoU ≥ 0.50 | Box precision |
| --- | ---: | ---: | ---: | ---: |
| Smoke | 0.030 | 294 / 321 | 91.6% | 33.1% |
| Fire | 0.002 | 367 / 398 | 92.2% | 9.5% |

The RGB detector flagged 41 of 280 negative images (14.6%). This profile supplies high-recall candidates to the confirmation gate. Jetson timing and thermal confirmation require device and sensor measurements. The 640 px profile was evaluated separately from the three 256 px local training arms printed in Figure 3.

For the same 640 px checkpoint, the standard Ultralytics benchmark measured 78.9% mAP@0.5 and 45.9% mAP@0.5:0.95. Smoke AP@0.5 was 83.9% and fire AP@0.5 was 73.9%. At the frozen high-recall operating point, 314 of 320 positive frames had at least one same-class IoU match. Across all 360 frames with any prediction, that gives 98.1% localized-frame recall and 87.2% localized-candidate precision. These metrics are available alongside the box results in the [JSON replay](results/recall_benchmark.json).

After rebuilding `.local-section4/dfire-4500`, reproduce the calibration and holdout result with the same D-Fire `test.tar` and the packaged reference weights.

```sh
python section4_ai/calibrate_recall.py --archive .local-section4/test.tar --data .local-section4/dfire-4500 --weights section4_ai/models/v8s_recall_reference.pt --output .local-section4/recall-check --report-ap
```

To replay the same RGB candidate policy on a recording, pass the two calibrated thresholds to the existing confirmation pipeline.

```sh
python section4_ai/infer.py --weights section4_ai/models/v8s_recall_reference.pt --source path/to/rgb.mp4 --output .local-section4/high-recall.jsonl --imgsz 640 --smoke-conf .03 --fire-conf .002 --nms-iou .80 --max-det 300
```

For frame-level RGB candidate screening, the calibration-selected peak gate `--frame-conf .125` lowered the number of negative frames flagged to 13 of 280. The fixed test yielded 312 localized frames out of 320 positives and 330 candidate frames, for 97.5% localized-frame recall and 94.5% localized-candidate precision. This test still includes low box precision on frames with multiple proposals. Reproduce both settings with [the frame-gate evaluator](evaluate_frame_gate.py) and see the [results](results/frame_gate_benchmark.json).

## Perception and deployment

`infer.py` accepts one RGB image or video and emits newline-delimited observations. The baseline nano checkpoint is the starting point for this trial because it combines the stronger small-smoke recall of the two nano variants with batch-one CPU throughput of 37.7 FPS. Use the validation-selected threshold in `--conf` for the checkpoint under evaluation. The library path into a simulator is `ConfirmationGate.step(frame_id, detections)` in [perception.py](perception.py), with a separate gate per UAV.

```sh
python section4_ai/infer.py --weights section4_ai/models/v8n_base.pt --conf 0.10 --source path/to/rgb.mp4 --output .local-section4/observations.jsonl
```

An optional `--thermal path/to/celsius_frames.npy` accepts a floating-point `(frames, height, width)` array. Register LWIR to the RGB camera coordinates and synchronize timestamps before calling the script. Invalid temperatures are represented by `NaN`. The script compares the RGB box's 90th percentile temperature to the surrounding ring median. A same-class track linked at IoU ≥ 0.30 must persist for three consecutive frames. A rise of at least 5 °C in the current registered frame yields `confirmed` and emits `new_alert` once per track. A persistent RGB cue without sufficient heat yields `rgb_candidate` for cross-view or operator verification. The association and temperature thresholds are prototype settings.

Use `--thermal-hits 2` to require at least two warm observations in the latest three RGB hits, including the current frame. The [simulated replay](results/edge_simulation.json) compares it with a one-hit corroboration policy and includes persistent fog, low cloud, a brief heat spike and a warm-background confuser. [Wind extrapolation](wind_forecast.py) uses a capped spread rate and flags downwind observation priorities. The [software trace](results/edge_simulation.json) records the assumed wind and simulated perimeter.

Each observation supplies class, confidence, box, pixel centroid, consecutive frame count, thermal delta, state and the `new_alert` flag. Pixel coordinates refer to the original RGB frame. Attach UAV ID, timestamp, pose, camera intrinsics and pose uncertainty in the simulator or flight integration layer before Ray–DEM geolocation. Smoke aloft can drift away from the ignition source, so retain uncertainty or verify from another viewpoint. APF/EGO-Swarm and the behavior tree consume localized evidence while retaining flight and separation constraints.

On the Jetson Orin Nano, build the TensorRT FP16 engine locally against the installed JetPack/TensorRT stack. Use the same preprocessing and record the device power profile, input size, batch, warm-up and sensor pipeline scope.

```sh
yolo export model=section4_ai/models/v8n_base.pt format=engine imgsz=256 batch=1 device=0 half=True dynamic=False simplify=False
python section4_ai/bench_edge.py --weights section4_ai/models/v8n_base.engine --images .local-section4/dfire-4500/test/images --output .local-section4/jetson-replay.json --device 0 --device-label "Jetson Orin Nano and recorded power mode" --imgsz 256 --frames 600 --warmup 50
```

For the course targets, replay synchronized RGB/LWIR sequences and measure sustained RGB rate, capture-to-confirmation latency, fog/cloud alarm rate and per-class recall. Record the single-frame model latency and the three-frame alert latency separately. The target is >35 RGB FPS, <30 ms single-frame processing and >90% recall. The laptop CPU trial and the unit tests establish the offline benchmark and the confirmation logic. Jetson throughput, thermal performance, and Hong Kong aerial recall require the corresponding hardware and synchronized footage.

## Licenses

The Section 4 Python training, benchmarking and inference package, including its trained Ultralytics weights and ONNX export, is distributed under AGPL-3.0. See the [Section 4 license](LICENSE) and the [Ultralytics licensing terms](https://www.ultralytics.com/license). Use the data source and attribution terms in [data/README.md](data/README.md) for the D-Fire data. The repository's top-level Apache-2.0 license governs the other repository content it covers.
