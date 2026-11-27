# Section 4 AI Integration

## 4.1 Role of AI in the Swarm

Each of the four UAVs runs onboard smoke and fire perception. Visible RGB imagery provides early cues of weak smoke or open flame, while registered long-wave infrared (LWIR) verifies elevated surface temperature. Confirmed detections report class, confidence, and RGB centroid; combined with synchronized EKF pose and camera intrinsics, Section 3's Ray–DEM module projects coordinates onto Hong Kong country-park terrain for Mesh mapping. Because smoke drifts downwind, uncorroborated cues trigger cross-view or operator verification rather than immediate flame localization.

At the decision layer, measured wind and detected perimeters support short-horizon downwind spread extrapolation to prioritize high-risk sectors. Flight paths, inter-UAV spacing (≥ 25 m), and emergency helicopter avoidance remain governed by the analytical behavior tree and APF/EGO-Swarm planner for verifiable safety (Zhou et al., 2021). Multi-agent deep reinforcement learning (e.g., attention-based PPO) offers adaptive task-allocation potential, but field adoption requires sim-to-real verification across unseen terrain, wind shear, and link dropouts.

## 4.2 Detection and Confirmation

We evaluate single-stage YOLOv8n and YOLOv8s detectors at 640 × 640 resolution to balance edge speed and small-target sensitivity (Ultralytics, 2023). To curb mountain fog and cloud false alarms while preserving recall, we pair class-aware NMS (IoU 0.50) with per-class confidence and peak-frame gating frozen on disjoint calibration frames. Aerial small-target fine-tuning (mosaic, scale jitter, vertical flip, copy-paste, hard-negative mining) merged via WiSE-FT weight interpolation sharpens box precision (Wortsman et al., 2022).

At runtime, three consecutive overlapping RGB tracks (IoU ≥ 0.30) establish persistence before LWIR checks whether the box's 90th-percentile temperature exceeds its surrounding ring median by ≥ 5 °C across ≥ 2 hits. On 48 continuous real D-Fire test clips (11 incident, 37 background), this 3-frame RGB gate reached 100.0% clip precision, 90.9% recall, and 0.0% negative-clip alarms before thermal filtering. In 40 FPS timing simulation (24 ms budget), confirmation took 74 ms while rejecting brief heat spikes. On Jetson Orin Nano (TensorRT FP16, 640 px), YOLOv8n (~4.5 ms) and YOLOv8s (~9.8 ms) satisfy > 35 FPS and < 30 ms targets, pending board verification.

## 4.3 Limitations in Country Parks

Hong Kong morning valley fog, bright cloud edges, and solar glare mimic thin white smoke. At 70 m altitude, incipient flames occupy few pixels and dense canopy occludes RGB and LWIR views, while misalignment weakens thermal confirmation. D-Fire mixes ground and oblique views without aligned LWIR or Hong Kong aerial footage, and multi-flame box fragmentation limits published single-box IoU ≥ 0.50 mAP@0.5 near 79–81%. Field deployment therefore requires local aerial RGB/LWIR collection, Jetson power profiling against the 40 min battery budget, and sim-to-real validation before adopting deep RL planners.

## 4.4 D-Fire Benchmark Trial

Using a fixed 4,500-image stratified D-Fire split (3,300 train, 600 validation, 600 held-out test; de Venâncio, Lisboa, & Barbosa, 2022), we compared a 5-epoch 256 px baseline against converged 640 px YOLOv8n and YOLOv8s models (Tleuliyev, n.d.), a lightweight YOLOv8n aerial-augmentation ablation fine-tuned for 5 epochs on 1,200 images to prevent overfitting/catastrophic forgetting (v8n aug WiSE-FT), and 768 px multi-scale TTA (3.1× latency, offline upper bound only). Thresholds and peak gates were frozen on 300 disjoint calibration images before once-only testing.

Deployed single-pass YOLOv8s (640 px, conf 0.04/0.15) and offline YOLOv8s (768 TTA) met industrial targets across overall, smoke, and fire localized-frame detection (IoU ≥ 0.50: Precision ≥ 85%, Recall ≥ 90%), reaching 92.4%/95.3% and 94.2%/95.9% overall with 82.7%–88.9% small-smoke recall. Of deployed v8s's 373 unmatched boxes, 357 (95.7%) fell on active fire/smoke positive frames (multi-box fragmentation), and only 16 fell on empty background frames (13/280, 4.6%). In ablation, v8n aug WiSE-FT raised smoke box precision +8.0 pp (to 74.4%), but strong geometric augmentation and weight interpolation slightly diluted faint low-contrast small-smoke features (small-smoke recall 78.8% → 75.0%).

**Held-out D-Fire benchmark results (600 fixed test images)**

| Model | Params | mAP50 | Frame P / R¹ | Smoke P / R¹ | Fire P / R¹ | Small R² | Neg.³ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLOv8n 256 px (5 ep) | 3.01M | 39.3% | 69.7% / 64.1% | 67.7% / 59.5% | 69.1% / 72.3% | 21.4% | 17.5% |
| YOLOv8n 640 px base | 3.01M | 77.7% | 93.8% / 93.8% | 89.0% / 90.0% | 87.5% / 90.3% | 78.8% | 2.9% |
| YOLOv8n 640 px aerial-aug | 3.01M | 76.7% | 92.7% / 90.9% | 89.3% / 86.5% | 89.1% / 89.7% | 75.0% | 3.2% |
| YOLOv8s 640 px base | 11.14M | 78.9% | 92.4% / 95.3% | 85.9% / 90.3% | 91.9% / 94.8% | 82.7% | 4.6% |
| YOLOv8s 768 px TTA | 11.14M | 78.0% | 94.2% / 95.9% | 89.0% / 92.0% | 90.6% / 92.9% | 88.9% | 3.2% |

*Note.* ¹ Localized-frame precision and recall requiring ≥ 1 same-class IoU ≥ 0.50 match; thresholds frozen on 300 disjoint calibration images. ² Small-smoke box recall (< 1,024 px² at input). ³ Single-frame negative-image alarm rate before 3-frame temporal (0.0% on 37 real clips) and LWIR filtering.

![Original pipeline, learning curves, industrial precision-recall comparison, confusion matrix and annotated D-Fire test photographs](results/figure3_edge_ai.png)

**Figure 3.** Edge AI fire/smoke detection pipeline and held-out D-Fire benchmark evaluation. (a) On-board dual-sensor perception and alert confirmation architecture; (b) lightweight YOLOv8n (640 px) aerial-augmentation ablation trajectories fine-tuned for 5 epochs on 1,200 images (to prevent overfitting/catastrophic forgetting) with mosaic closure; (c) held-out localized-frame precision/recall and small-smoke recall across 600 test images with relative latency anchors (deployed v8s 640 px at 1× vs. offline 768 TTA upper bound at 3.1×); (d) deployed v8s 640 px box confusion counts (conf 0.04/0.15; 357/373 [95.7%] unmatched boxes fall on active fire/smoke positive frames via multi-box fragmentation, and only 16 boxes fall on 13/280 [4.6%] empty background frames); (e) qualitative held-out D-Fire detections (de Venâncio et al., 2022).

## References

de Venâncio, P. V. A. B., Lisboa, A. C., & Barbosa, A. V. (2022). An automatic fire detection system based on deep convolutional neural networks for low-power, resource-constrained devices. *Neural Computing and Applications, 34*, 15349–15368. doi:10.1007/s00521-022-07467-z

Tleuliyev, A. (n.d.). *Wildfire detection* [Model weights]. Retrieved from https://github.com/AlimTleuliyev/wildfire-detection

Ultralytics. (2023). *YOLOv8* [Computer software]. Retrieved from https://docs.ultralytics.com/models/yolov8/

Wortsman, M., Ilharco, G., Gadre, S. Y., Roelofs, R., Gontijo-Lopes, R., Morcos, A. S., Namkoong, H., Farhadi, A., Carmon, Y., Kornblith, S., & Schmidt, L. (2022). Model soups: Averaging weights of multiple fine-tuned models improves accuracy without increasing inference time. In *Proceedings of the 39th International Conference on Machine Learning* (pp. 23965–23998).

Zhou, X., Zhu, J., Zhou, H., Xu, C., & Gao, F. (2021). EGO-Swarm: A fully autonomous and decentralized quadrotor swarm system in cluttered environments. In *2021 IEEE International Conference on Robotics and Automation* (pp. 4101–4107). doi:10.1109/ICRA48506.2021.9561902
