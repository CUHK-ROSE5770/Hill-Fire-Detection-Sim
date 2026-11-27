# Section 4 requirements and validation

This audit follows sections 4.1–4.4 of the project brief. The RGB benchmark uses the fixed D-Fire split. The timing and thermal simulations use declared inputs and assumptions. Physical Jetson and field acceptance require device and flight data.

## Production quality review

The project's offline review target is precision and recall above 90% simultaneously for each class on the same held-out operating point. [The quality check](results/production_quality.json) recomputes both measures from matched, predicted and annotated box counts. Image-level candidate metrics count a frame as localized only when at least one box has a correct same-class IoU match. The box metrics also capture duplicate and poorly localized proposals. Incident recall requires event-level labels on continuous live or recorded streams.

For field acceptance, collect independent sequences grouped by flight and incident, with day and night coverage, morning fog, low cloud, glare, canopy occlusion and 70 m views. Measure event-level precision and recall, false confirmations per flight hour, detection delay and uptime. Agree the operational thresholds with the operator. Record the device and power mode and repeat the tests after changing the model, camera or thresholds. [NIST AI RMF Measure guidance](https://airc.nist.gov/airmf-resources/airmf/5-sec-core/) · [Operational wildfire camera study](https://www.mdpi.com/2072-4292/12/1/166)

### Fixed D-Fire holdout

| Same 600 held-out images | Smoke box precision / recall | Fire box precision / recall | Localized frame precision / recall | Negative frames flagged |
| --- | ---: | ---: | ---: | ---: |
| YOLOv8s high-recall profile | 33.1% / 91.6% | 9.5% / 92.2% | 87.2% / 98.1% | 41/280 (14.6%) |
| Same checkpoint with 0.125 peak confidence gate | 33.6% / 91.0% | 9.6% / 92.2% | 94.5% / 97.5% | 13/280 (4.6%) |
| Dual-scale YOLOv8s with per-class duplicate suppression | 49.0% / 90.03% | 19.3% / 91.5% | 86.1% / 96.9% | 41/280 (14.6%) |

The peak confidence gate was selected from the 300-image calibration set. It reduced the held-out negative-frame alert count by 28 and retained at least one correctly localized detection in 312 of 320 positive frames. The frame gate passed 90% for both localized-frame metrics. Box precision remains below the offline production review target for both classes. [High-recall results](results/recall_benchmark.json) · [Peak-gate replay](results/frame_gate_benchmark.json)

The dual-scale trial adds an augmented 768 px fire pass to a 640 px detector and applies a 0.60 IoU NMS separately to each class. The held-out fire box precision approximately doubled, while both box recalls remained above 90%. Smoke recall corresponds to 289 of 321 boxes and is close to the target. The extra inference work and low box precision limit this profile to offline comparison. [Dual-scale replay](results/multiscale_benchmark.json)

```sh
python section4_ai/assess_quality.py --reference section4_ai/results/recall_benchmark.json --multiscale section4_ai/results/multiscale_benchmark.json --frame-gate section4_ai/results/frame_gate_benchmark.json --output .local-section4/production_quality.json
```

## Requirement mapping

| Requirement | Evidence | Status |
| --- | --- | --- |
| Onboard smoke/fire perception, Ray–DEM localization, distributed Mesh alert and explainable planning | Report §§4.1–4.2, [observation interface](README.md#perception-and-deployment) and Figure 3. The planner receives localized evidence and retains its APF/EGO-Swarm constraints. | Design and perception interface implemented. Flight integration is documented. |
| Short-horizon wind extrapolation and downwind observation priority | The [metric XY forecast](wind_forecast.py) projects an assumed spread rate along the current perimeter and flags downwind vertices for verification. The [software trace](results/edge_simulation.json) includes the output for a 15 s scenario. | Synthetic forecast verified. Spread coefficients require local calibration. |
| Swarm Deep RL and attention-based PPO | Report §4.1 discusses simulated task allocation, transfer across terrain and wind, and emergency constraints. | Design discussion in the report. |
| Lightweight detector, accuracy–speed trade-off and high recall | The [local benchmark](results/benchmark.csv) compares YOLOv8n and YOLOv8s. The [high-recall benchmark](results/recall_benchmark.json) evaluates RGB detections on 600 fixed images using same-class IoU ≥ 0.50. | Both RGB class recalls exceed 90% in the published-reference profile. Box precision and negative-frame alerts are reported alongside recall. |
| Small-target/viewpoint augmentation | Same seed, split and training budget for YOLOv8n baseline and Mosaic/scale/flip-augmented YOLOv8n. | Ablation completed. This augmentation recipe reduced the fixed-threshold small-smoke recall, which is reported in §4.4. |
| Three-frame agreement and synchronized LWIR corroboration | Input alignment contract, per-UAV confirmation, thermal persistence option, tests and the [software simulation](results/edge_simulation.json). | Software paths verified with synthetic RGB detections and temperature arrays. Field false-alarm rates remain open. |
| TensorRT FP16 on Jetson Orin Nano at >35 FPS and <30 ms | [Deployment instructions](README.md#perception-and-deployment), [device benchmark](bench_edge.py) and the timing simulation below. | A feasible simulated budget was demonstrated. Jetson throughput and latency await the board. |
| Local fog, low clouds, glare, canopy occlusion, viewpoint shift, power and sim-to-real limitations | Report §4.3 and the synthetic negative scenarios. | Covered, with explicit validation limits. |
| D-Fire benchmark and Figure 3 | The checked [manifest](data/split_manifest.csv) contains 4,500 images. Figure 3 contains authored pipeline and metrics panels and credited D-Fire photos with program-generated overlays. | Reproducible. Report also includes APA references. |

## Simulated timing and corroboration

The simulator feeds an RGB track and aligned Celsius samples into the real confirmation code. A 40 FPS source has a 25 ms frame interval. The queue keeps the latest waiting frame, and skipped frames interrupt consecutive-hit tracking. Service times are assumed. The two latency columns measure different endpoints.

| Assumed processing time | Simulated processing FPS | Capture-to-result p95 | Fire confirmation after onset |
| ---: | ---: | ---: | ---: |
| 18 ms | 40.00 | 18 ms | 68 ms |
| 24 ms | 40.00 | 24 ms | 74 ms |
| 29 ms | 34.48 | 53 ms | 87 ms |
| 33 ms | 30.30 | 57 ms | 99 ms |

In the controlled thermal cases, requiring two warm observations in the latest three RGB hits reduced negative confirmations from two to one across five negative scenarios. Both thermally corroborated positive scenarios still confirmed. The positive scene without thermal evidence remained a visual candidate. A persistent warm confuser still triggered a false confirmation, so the simulation identifies a field validation case as well as the short heat spike that the extra check rejects.

```sh
python section4_ai/simulate_edge.py --output .local-section4/edge_simulation.json
python -m unittest discover -s section4_ai -v
```

## Physical benchmark protocol

The same 180 held-out frames were replayed on an AMD Ryzen 5 5625U after 25 warm-up frames. Batch-one wall time includes RGB preprocessing, inference, NMS and the predictor call. The timing starts after image decoding.

| Local CPU configuration | RGB FPS | p95 processing latency | Evidence |
| --- | ---: | ---: | --- |
| YOLOv8n baseline, 256 px, ONNX Runtime FP32 | 84.59 | 14.153 ms | [CPU ONNX replay](results/cpu_edge_v8n_onnx.json) |
| Same YOLOv8n baseline, 256 px, PyTorch | 43.72 | 26.773 ms | [CPU PyTorch replay](results/cpu_edge_v8n_pt.json) |
| YOLOv8s high-recall reference, 640 px, PyTorch | 5.74 | 197.880 ms | [CPU reference replay](results/cpu_edge_v8s_reference.json) |

The fast replay uses the compact local model. The stronger RGB recall results use the published 640 px checkpoint. The available evidence therefore leaves combined precision, recall and Jetson timing on a single deployed configuration open.

Build the FP16 engine on the target Jetson with its JetPack and TensorRT runtime. Run the fixed RGB replay in batch one after warm-up and record the module, power mode and runtime. Compare throughput and p95 capture-processing wall time against the project targets. A separate time-synchronized camera-and-LWIR replay measures capture-to-confirmation and verifies the negative scenarios with real sensor data. The detector thresholds and TensorRT precision should be checked on the benchmark before field use. [NVIDIA TensorRT benchmarking guide](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/benchmarking.html)
