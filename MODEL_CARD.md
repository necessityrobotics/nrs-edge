# Model card: nrs-edge 0.1.0

**Summary.** `nrs-edge` is a small stereo depth network by Necessity Robotics. It takes a rectified stereo pair and gives the disparity of the left image. With the focal length and the baseline of the rig, the disparity gives metric depth. Our aim is to run it on the robot's own computer (a Jetson Orin Nano class board). We have not measured it on such a board.

| | |
|---|---|
| Maker | Necessity Robotics (a robotics company), India |
| Network | LiteAnyStereo v2, size S (LAS2-S): FasterNet-T0 feature network, correlation volume at 1/4 resolution, cost aggregation, softmax disparity regression, 4x upsampling. One pass, no iterations. |
| Code | The authors' MIT code, github.com/TomTomTommi/LiteAnyStereo, commit 8c97bd4 (`third_party/LiteAnyStereo` in the code repository) |
| Parameters | 6,040,420 (6.04 M) |
| Weights | **Ours.** No weights of another group were used. No ImageNet weights were used. |
| Weights in this card | candidate B (fine-tune of recipe A, fixed statistics); source file SHA-256 `13abd7b3981f9786dd398198ed25e3d1336f9d382b586e9c5cd1d0551b4f24b1`; weights fingerprint `b8b8402ca978ec29f2a0ebd31a5ec46b083aa7234974fd2b34efb98749e6aede` |
| Licence | **Apache-2.0** |

## What the model is and is not

| It is | It is not |
|---|---|
| A stereo network designed for a small computer: 6.04 M parameters, one forward pass | A monocular depth network |
| Input: a rectified left/right pair (rows aligned) | A rectifier: rectify the pair first |
| Output: disparity of the left image, 0 to 192 px | A model for disparities above 192 px |
| Disparity for every pixel | A confidence or validity output: there is none |
| Aimed at Jetson-class boards with TensorRT FP16 | A measured Jetson result: there is no device log |

## Inputs and outputs

- **Inputs:** `left`, `right`: RGB, values 0 to 255. Grayscale images are read as three equal channels. The network normalises the values itself.
- **Output:** disparity of the left image in pixels, float32. The maximum is 192 px.
- **Depth:** depth [m] = fx [px] x baseline [m] / disparity [px]. Use the focal length of the rectified image.
- **Size:** the PyTorch path pads the pair to a multiple of 32 (edge replication, centred) and crops the output back. The ONNX graphs have fixed sizes; `infer.py` pads a smaller pair to the graph size and resizes a larger pair (resizing loses detail).
- **Precision:** the accuracy numbers below are FP32. The FP16 ONNX graphs keep FP32 islands (the Softmax output tail and InstanceNorm).

**Release files**

| File | Bytes | SHA-256 |
|---|---|---|
| `nrs-edge.safetensors` | 24,282,720 | `aff7dfb82cbdba0cbfaa943bcbebc8a0583d276a3477317401e14a054054a254` |
| `nrs-edge_w640h480_fp32.onnx` | 24,378,438 | `4b52db5579e981bd0edfb3a43f7e7e1a4cb5c535b2ce6ba5d3337def41c1d2b6` |
| `nrs-edge_w640h480_fp16.onnx` | 12,417,308 | `3875321e1ac87d3e588d1fd3af20558d453b8e4565fc778f6be6c33d5dd0c541` |
| `nrs-edge_w640h400_fp32.onnx` | 24,408,310 | `4a59dee6859f8e7499f2d88347e597bb448b9cc4995918fe837bfb346ab07fcf` |
| `nrs-edge_w640h400_fp16.onnx` | 12,466,974 | `2d26e7f9840c9615f989a7294fde2d8a8cdc33b8d79b59c5c10f1cd4c122f7d4` |

## Accuracy (zero-shot benchmark sets; see the KITTI note)

Our own evaluation harness; mask `wmg` (the WMG protocol: Middlebury 2014 non-occluded, every other set all valid ground-truth pixels); aggregation `image_mean` (mean over images); IEEE FP32 (TF32 off, mixed precision off); input padded to a multiple of 32; maximum disparity 192 px. Sets: Middlebury-H bad-2 (noc, 15 images); Middlebury 2021 bad-2 (all, 24 images); ETH3D bad-1 (all, 27 images); KITTI-12 bad-3 (all, 194 images); KITTI-15 bad-3 (all, 200 images); Booster-Q bad-2 (all, 228 images). Lower is better.

| Model | Middlebury-H bad-2 | Middlebury 2021 bad-2 | ETH3D bad-1 | KITTI-12 bad-3 | KITTI-15 bad-3 | Booster-Q bad-2 | Far band (px) | Floor | Near | Scored |
|---|---|---|---|---|---|---|---|---|---|---|
| first attempt (stage 3) | 11.66 | 27.50 | 22.27 | 5.22 | 5.82 | 26.89 | +0.628 | FAIL (1 of 4) | FAIL | 2026-11-19 |
| recipe A | 13.68 | 31.33 | 5.15 | 4.98 | 6.11 | 30.70 | +0.004 | FAIL (2 of 4) | FAIL | 2026-11-22 |
| candidate A-bn (recipe A + large-crop normalisation statistics) | 13.68 | 31.22 | 5.19 | 4.88 | 6.05 | 30.00 | +0.021 | FAIL (2 of 4) | FAIL | 2026-11-22 |
| **candidate B (fine-tune of recipe A, fixed statistics)** | 12.71 | 30.07 | 5.09 | 4.62 | 5.57 | 28.63 | +0.011 | FAIL (2 of 4) | PASS | 2026-11-24 |
| teacher line T0, diagnostic scoring at step 100,000 | 7.37 | — | 2.61 | — | 5.17 | 13.89 | -0.034 | not gated | not gated | 2026-11-07 |
| first attempt, stage 1 (stage gate only, not a release candidate) | 14.76 | 35.36 | 21.99 | 6.99 | 6.69 | 32.49 | +0.405 | FAIL (0 of 4) | FAIL | 2026-11-16 |
| Teacher: nrs-large 0.1.0 (our own DLNR weights, trained from random initialisation on WMGStereo-150k only) | 3.91 | 8.44 | 2.04 | 3.38 | 4.59 | 9.22 | +0.017 | — | — | 2026-11-12 |
| RT-IGEV, Scene Flow weights, 8 iterations (Floor bar model) | 11.51 | 19.34 | 6.48 | 6.11 | 6.29 | 23.48 | -0.005 | — | — | 2026-11-14 |
| LAS2-S, authors' weights (Stretch bar model) | 11.29 | 20.54 | 4.11 | 3.52 | 4.01 | 20.60 | -0.014 | — | — | 2026-11-14 |
| RAFT-Stereo, public Scene Flow weights (harness reference row, P8) | 8.67 | 10.26 | 2.61 | 4.76 | 5.48 | 17.75 | -0.018 | — | — | 2026-11-05 |

- **Bold** = the weights of this card. Each row comes from one result file of our harness; the numbers are not rounded before the verdict.
- **Floor gate** (pre-registered): at least 3 of 4 at or below the bars Middlebury-H 11.51, ETH3D 6.49, KITTI-15 6.29, Booster-Q 23.47 (RT-IGEV, Scene Flow weights, 8 iterations, first scoring).
- **Near gate** (pre-registered): ETH3D at or below 5.14 (mandatory) and 2 of 3 of Middlebury-H 14.11, KITTI-15 6.02, Booster-Q 25.76.
- **Stretch** (pre-registered): ETH3D 4.11, KITTI-15 4.01.
- **Far band:** the mean over the ETH3D images of the median signed error at a true disparity of 1 to 9 px (only images with such pixels). Gate: at most 0.1 px.
- The Floor bars were fixed from the first RT-IGEV scoring (2026-11-05, TF32 allowed). The FP32 re-scoring in the table differs from the bars by at most 0.011.
- The RAFT-Stereo reference row was scored before the FP32 default (TF32 convolutions allowed) and was not scored again.
- **KITTI note:** the training data has DrivingStereo images (driving scenes, labels from our teacher). The KITTI-12 and KITTI-15 columns are therefore **domain-exposed, not zero-shot**. No image of any benchmark set was used in training.

**How often the benchmark sets were scored.** Our own weights were scored on these sets **7 times** in total; 5 of these are edge (LAS2-S) weights. Each candidate was scored once. Every scoring:

| Result file | Date | Network | Weights SHA-256 (first 12) | What |
|---|---|---|---|---|
| `DIAG_T0_w100000.json` | 2026-11-07 22:37 | dlnr | `9ba1e259df51` | teacher line T0, diagnostic scoring at step 100,000 |
| `G_TEACH_w200000.json` | 2026-11-12 12:21 | dlnr | `17b7e60be3c8` | teacher nrs-large 0.1.0 gate |
| `G_S1_w200000.json` | 2026-11-16 20:12 | las2 | `a8d64865efbf` | first attempt, stage 1 (stage gate only, not a release candidate) |
| `G_S3_w200000.json` | 2026-11-19 14:56 | las2 | `fe54ff0837e6` | first attempt (stage 3) |
| `G_EDGE_A_w200000.json` | 2026-11-22 11:42 | las2 | `e2cc920aa30f` | recipe A |
| `G_EDGE_Abn_w200000.json` | 2026-11-22 14:09 | las2 | `f1dbcf6389ae` | candidate A-bn (recipe A + large-crop normalisation statistics) |
| `G_EDGE_B_w200000.json` | 2026-11-24 15:04 | las2 | `13abd7b3981f` | candidate B (fine-tune of recipe A, fixed statistics) |

**How these weights were chosen.** The rule was fixed before any of the repair results existed (our pre-registration, amendment A10 note 28 item 6): the candidate with the higher gate level is the release candidate (Floor above Near above none); at the same level, the lower mean of (value / Floor bar) over Middlebury-H, ETH3D, KITTI-15 and Booster-Q wins. No candidate at Near or above means no release as `nrs-edge` without a new owner decision. This choice between pre-registered candidates uses benchmark numbers. That is a known departure from our rule "benchmarks label, they never select", and we state it here.

| Rank | Candidate | Level | Mean of value / Floor bar |
|---|---|---|---|
| 1 | candidate B (fine-tune of recipe A, fixed statistics) | Near | 0.9986 |
| 2 | candidate A-bn (recipe A + large-crop normalisation statistics) | none | 1.0573 |
| 3 | recipe A | none | 1.0655 |
| 4 | first attempt (stage 3) | none | 1.6285 |

Result of the rule: **candidate B (fine-tune of recipe A, fixed statistics)**. It reaches the Near level. It does not reach the Floor level (2 of 4 sets at or below the Floor bars; 3 are needed).

## Speed

- **No Jetson speed was measured.** Our aim is real-time depth on a Jetson Orin Nano class board with TensorRT FP16. This is an aim, not a result.
- A desktop TensorRT timing (RTX 5090) is measured 2026-11-24: TensorRT 11.4 FP16 (strongly typed), RTX 5090, 400x640, end to end with copies p50 3.18 ms, p99 4.07 ms (Python, desktop; not an edge number). A desktop number is not an edge number.

## Training

**Stages of these weights** (from the `run_config.json` of each stage, oldest first):

| Stage (run) | Start weights | Steps | One step | Large-crop samples | Small-crop samples | Peak learning rate | Feature network frozen (steps) | Seed |
|---|---|---|---|---|---|---|---|---|
| `S1a_las2s_recipeA_s20261119` | `fd_pre_s1/w_init.pth` | 20,000 | 64 crops 128x256 only (large crops not used in this stage) | 0 | 1,280,000 | 0.0002 | 2000 | 20261119 |
| `S1b_las2s_recipeA_s20261119` | `S1a_las2s_recipeA_s20261119/LAS2_S_final.pth` | 40,000 | 4 x 16 crops 384x768 + 64 crops 128x256 (small part weight 0.5) | 2,560,000 | 2,560,000 | 0.0002 | 0 | 20261119 |
| `S3_las2s_recipeA_s20261119` | `S1b_las2s_recipeA_s20261119/LAS2_S_final.pth` | 200,000 | 4 x 16 crops 384x768 + 64 crops 128x256 (small part weight 0.5) | 12,800,000 | 12,800,000 | 0.0002 | 0 | 20261119 |
| `B_las2s_recipeA_ft_s20261122` | `S3_las2s_recipeA_s20261119/LAS2_S_final.pth` | 200,000 | 6 x 16 crops 384x768 + 64 crops 128x256 (small part weight 0.5); BatchNorm statistics fixed (eval mode) from `bn_large_stats.pt` | 19,200,000 | 12,800,000 | 0.0001 | 0 | 20261122 |

**Power cuts.** The last stage (`B_las2s_recipeA_ft_s20261122`) was stopped by 2 power cuts on 2026-11-23. It resumed from its last saved state with the same command each time, at steps 128,372 and 129,594. About 1,530 steps were run again. No setting was changed. (Record: our pre-registration, A10 note 30 item 1.)

**Training data**

| Source | Pairs | Licence | Licence evidence | Role | Share in `S1a_las2s_recipeA_s20261119` | Share in `S1b_las2s_recipeA_s20261119` | Share in `S3_las2s_recipeA_s20261119` | Share in `B_las2s_recipeA_ft_s20261122` |
|---|---|---|---|---|---|---|---|---|
| StereoGenBench (train/IPD_Gaussian and train/Uniform; captions not downloaded) | 81,891 | CC BY 4.0 | licence text read: card and paper appendix K (copy stored by Necessity Robotics); the paper says that third-party source assets keep their original licence terms | synthetic; true labels | 0.350 | 0.350 | 0.175 | 0.175 |
| WMGStereo-150k, indoor | 71,800 | BSD-3-Clause | tag only (Hugging Face card, read again 2026-11-24: bsd-3-clause); generator repository BSD-3 text read | synthetic; true labels | 0.220 | 0.220 | 0.110 | 0.110 |
| TartanAir V1 (hold-out environments excluded) | 45,903 | CC BY 4.0 | licence text read (copy stored by Necessity Robotics) | synthetic; true labels | 0.200 | 0.200 | 0.100 | 0.100 |
| TartanAir V2, indoor environments | 43,208 | CC BY 4.0 | licence text read on the site tartanair.org (CC BY 4.0); the Hugging Face card is tagged bsd-3-clause; we follow the site | synthetic; true labels | 0.150 | 0.150 | 0.075 | 0.075 |
| WMGStereo-150k, nature_filtered | 13,669 | BSD-3-Clause | tag only (Hugging Face card, read again 2026-11-24: bsd-3-clause); generator repository BSD-3 text read | synthetic; true labels | 0.080 | 0.080 | 0.040 | 0.040 |
| Own OAK-D Lite recordings (Necessity Robotics) | 10,065 | own data | own recordings | real images; labels from our own teacher nrs-large | — | — | 0.053 | 0.053 |
| TUM-VI (rectified, 512 px, room and corridor) | 14,199 | CC BY 4.0 | licence text read (copy stored by Necessity Robotics) | real images; labels from our own teacher nrs-large | — | — | 0.074 | 0.074 |
| Rosario v2 | 3,456 | CC BY 4.0 | licence text read (copy stored by Necessity Robotics) | real images; labels from our own teacher nrs-large | — | — | 0.018 | 0.018 |
| DrivingStereo (left/right images only) | 19,120 | MIT | licence text read: 'This dataset is released under the MIT license.' (copy stored by Necessity Robotics) | real images; labels from our own teacher nrs-large | — | — | 0.100 | 0.100 |
| TartanRGBT (ZED left/right images only; its FoundationStereo depth was not used) | 24,545 | BSD-3-Clause-Clear | tag only (Hugging Face card, read again 2026-11-24: bsd-3-clause-clear) | real images; labels from our own teacher nrs-large | — | — | 0.129 | 0.129 |
| DROID (raw ZED stereo) | 16,421 | CC BY 4.0 | licence text read (paper; copy stored by Necessity Robotics) | real images; labels from our own teacher nrs-large | — | — | 0.086 | 0.086 |
| Unitree G1 head stereo (Hugging Face unitreerobotics/*, rectified streams) | 3,751 | Apache-2.0 | tag only (Hugging Face cards; all 33 repositories read again 2026-11-24: apache-2.0) | real images; labels from our own teacher nrs-large | — | — | 0.020 | 0.020 |
| FoMo (boreal forest, ZED X rectified) | 3,759 | CC BY 4.0 | LICENSE.txt in the data bucket (read by us; our pre-registration, A9 note 6) | real images; labels from our own teacher nrs-large | — | — | 0.020 | 0.020 |

- Synthetic pairs: 256,471, with true labels. Real pairs: 95,316, with labels from our own teacher.
- **Teacher:** nrs-large 0.1.0 (our own DLNR weights, trained from random initialisation on WMGStereo-150k only). Its own card states its training and its scores.
- **Start of the feature network:** distilled from our own teacher (label-free), from random weights. The run configuration states: "S1b start weights from --init (stage 1: feature network distilled from the own teacher T-W, no outside weights; stage 3: the stage-1 weights)".
- **Feature pre-stage:** 4,000 steps, batch 64, crops 128x256, teacher `TW_dlnr_wmg_s20261105/w_200000.pth`.
- **No weights of another group** and **no ImageNet weights**: the network is built with `fnet_pretrained=False`. The authors' LAS2 weights and the RT-IGEV weights were used only for evaluation.
- **Never trained on:** Middlebury, ETH3D, KITTI, Booster (the benchmark sets).
- Licence evidence: "text read" means we read the licence sentence; "tag only" means the only statement is a licence tag on a dataset card. Details: `ATTRIBUTION.md`.

## Known limits

- **Failed attempts, stated plainly.** These scored candidates did not reach the Floor or the Near gate: first attempt (stage 3) (Floor 1 of 4, Near FAIL); recipe A (Floor 2 of 4, Near FAIL); candidate A-bn (recipe A + large-crop normalisation statistics) (Floor 2 of 4, Near FAIL).
- **Glass, mirrors, screens (Booster-Q, 228 images): bad-2 28.63, above the Floor bar 23.47.** Worst images: TV/im2 86.8, TV/im0 81.0, TV/im1 81.0, TV/im3 80.7.
- **Large disparities (Middlebury-H, 15 images): bad-2 12.71, above the Floor bar 11.51.** Worst images: Jadeplant 39.1, Vintage 33.1, Shelves 21.6, Playtable 17.6.
- Pixels with a true disparity above 192 px are always wrong: the network has a 192 px range. On Middlebury-H they are 2.9 % of the scored pixels. (Record: our pre-registration, A10 note 27 item 2.)
- Large disparities are weak. Recipe A against the first attempt, pooled bad-2: Middlebury-H 50-100 px 11.9 against 8.6; Booster-Q 100-192 px 35.3 against 31.2. Measured: 15.4 M large-crop samples against 51.2 M. Our reading: this is the main cause (not proven). (Record: our pre-registration, A10 note 27 items 2 and 3.)
- The stored normalisation statistics of recipe A did not fit full-size images (own test: 1.45 px stored, 0.93 px with statistics from large crops). Candidate A-bn uses the large-crop statistics; its gate values are in the table above. (Record: our pre-registration, A10 notes 27 and 29.)
- The real training pairs keep a vertical offset of 0.3 to 0.6 px after the label repair (OAK, DrivingStereo, TUM-VI, Rosario). Pairs above 1.0 px were left out. (Record: internal audit report of 2026-11-19, section 19.4.)
- Our own synthetic large-disparity test does not predict the real indoor sets: the authors' LAS2-S scores 2.87 px on it (recipe A with large-crop statistics 0.86 px; the released weights 0.74 px) and is better on Middlebury and Booster. (Record: our pre-registration, A10 note 27 item 5; note 30 item 2.)
- **No confidence output.** Every pixel gets a disparity, also where the answer is a guess (sky, glass, blank walls).
- **Rectification and vertical offset.** The output is only as good as the rectification of the rig. A vertical offset between the two images makes the result worse.
- **FP16:** the FP16 graphs were checked on ONNX Runtime CPU only. No GPU or device FP16 accuracy result is in this card; the desktop timing above is a timing only.
- Not for safety-critical use without an independent safety layer.

## How the 0 to 3 px fault was cured

The first attempt had a fault at true disparities of 0 to 3 px: the output stayed near a constant value (ETH3D bad-1 22.27). Recipe A cured it with two standard steps. (1) A label-free pre-stage: the feature network of the student learns the feature map of our own teacher (cosine loss), as in the hint stage of FitNets (Romero et al., ICLR 2015) and the backbone distillation of Fast-FoundationStereo (Wen, Dewan, Birchfield, arXiv 2512.11130). (2) The feature network is frozen for the first 2,000 training steps and then trained with the rest, the order that LP-FT (Kumar et al., ICLR 2022) recommends. Recipe A then trains on small crops first and adds large crops after that. ETH3D bad-1 fell from 22.27 to 5.15; the far-band value fell from +0.628 px to +0.004 px. We do not claim this as a new method. The record of the fault and the cure is in our audit report (sections 15 and 16).

## Intended use

- Depth for a robot's own navigation and mapping, from a calibrated, rectified stereo camera.
- Research and benchmarking, with the numbers above.

**Out of scope:** safety-critical use without an independent safety layer; unrectified or monocular input; objects nearer than the 192 px disparity limit; measurement or survey work.

## Citation and credits

The network design and code are by Jing et al.: *Lite Any Stereo: Efficient Zero-Shot Stereo Matching*, CVPR 2026; *Lite Any Stereo V2*, arXiv 2606.24457. Please cite their work with this release. The data credits are in `ATTRIBUTION.md` and `NOTICE`.

Contact: necessityrobotics@gmail.com
