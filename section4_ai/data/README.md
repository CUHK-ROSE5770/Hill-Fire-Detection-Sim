# D-Fire benchmark data

The [split manifest](split_manifest.csv) freezes the 4,500-image experiment at seed `5770`. Each row records split, upstream split, source path, class stratum, number of boxes by class, and SHA-256 for the exact original image and label bytes. [Selected annotations](selected_yolo_labels.zip) contains the 4,500 original YOLO text labels in the experiment's split layout, including empty files for background examples. The class order is `0=smoke, 1=fire`.

## Rebuild the images

Use the pinned D-Fire raw mirror below. The full files need approximately 3.24 GB on disk. Their bytes were checked against the mirror's Git LFS SHA-256 values before use.

| File | SHA-256 |
| --- | --- |
| [train.tar](https://huggingface.co/datasets/hanvithSai/gavin-dfire-raw/resolve/fee27ef6934d5bb569ae6271c123673f10a877c8/train.tar) | `48d8586d1fa4444d1bb97a693ed6f023155ef7d5aa26fa545bedac10c90633c9` |
| [test.tar](https://huggingface.co/datasets/hanvithSai/gavin-dfire-raw/resolve/fee27ef6934d5bb569ae6271c123673f10a877c8/test.tar) | `593cb2d08cf8ec1176eaad6d0f682ce2eaf7cc9513beacf58823a9ba16429d30` |

The source [D-Fire repository](https://github.com/gaia-solutions-on-demand/DFireDataset) also provides the original download links. This repository distributes the manifest, selected annotations and derived results. Obtain the original image bytes from the upstream download or mirror. `prepare_data.py --replay` validates every selected image and label against the checked-in hashes.

## Selection and provenance

The release's train split supplies both the 3,300 training images and 600 validation images. Its held-out test split supplies 600 test images. Sampling follows four image strata: smoke only, fire only, both, and background. Fourteen training-source and twelve test-source images had invalid normalized label values and were excluded from the sampling population. The subset builder also skips unreadable JPEGs and prevents exact image-byte duplicates across the selected splits. D-Fire lacks consistent event and camera IDs, so near-duplicates and scene overlap may remain. Background examples are unlabeled for both target classes, with no separate fog/cloud labels.

The [original D-Fire license](https://github.com/gaia-solutions-on-demand/DFireDataset/blob/master/LICENSE) applies CC0 1.0 to rights held in the collection. Individual third-party images retain their providers' terms. The [upstream source notes](https://github.com/gaia-solutions-on-demand/DFireDataset#globe_with_meridians-public-dataset-images) identify HPWREN/FIgLib, the Center for Wildfire Research at the University of Split, and FireNet. The raw mirror's metadata describes the mirror as CC BY 4.0. Preserve both the original source credits and mirror attribution.

de Venâncio, P. V. A. B., Lisboa, A. C., & Barbosa, A. V. (2022). An automatic fire detection system based on deep convolutional neural networks for low-power, resource-constrained devices. *Neural Computing and Applications, 34*, 15349–15368. doi:10.1007/s00521-022-07467-z
