# Attribution: nrs-edge 0.1.0

This file credits every work that `nrs-edge` is built from. The licence texts are in `LICENSES/`. The code repository also holds the authors' code with its licence in `third_party/LiteAnyStereo/`. The short notice is `NOTICE`.

**Licence of the nrs-edge weights and scripts: Apache-2.0.**

## Code and network design

| Work | Used for | Licence | Copyright |
|---|---|---|---|
| LiteAnyStereo, github.com/TomTomTommi/LiteAnyStereo, commit 8c97bd4 | network design and code (unmodified; in `third_party/LiteAnyStereo` of the code repository) | MIT | (c) 2025 Junpeng Jing |
| PyTorch Image Models (timm), a dependency, not bundled | FasterNet-T0 definition used by LAS2 | Apache-2.0 (`LICENSES/timm_Apache-2.0.txt`) | Ross Wightman and contributors |
| FasterNet | design of the feature network | MIT (as stated in the timm source) | Microsoft Corporation |
| DLNR code (our teacher's network), not bundled | the teacher that made the labels of the real pairs and the feature start | MIT | (c) 2021 Princeton Vision & Learning Lab |

Citations for the code and the network design:

- J. Jing, W. Luo, Y. Mao, K. Mikolajczyk. Lite Any Stereo: Efficient Zero-Shot Stereo Matching. CVPR 2026, pp. 21725-21735. arXiv:2511.16555.
- J. Jing, R. Zuo, Z. Shen, S. Zhou, R. A. Potamias, S. Zafeiriou, K. Mikolajczyk, J. Deng. Lite Any Stereo V2: Faster and Stronger Efficient Zero-Shot Stereo Matching. arXiv:2606.24457, 2026.
- H. Zhao, H. Zhou, Y. Zhang, J. Chen, Y. Yang, Y. Zhao. High-Frequency Stereo Matching Network. CVPR 2023, pp. 1327-1336 (the network of our teacher).
- L. Lipson, Z. Teed, J. Deng. RAFT-Stereo: Multilevel Recurrent Field Transforms for Stereo Matching. 3DV 2021 (the code base that the teacher's network is built on).

## Training data

| Source | Pairs | Licence | Evidence | Role | Credit |
|---|---|---|---|---|---|
| StereoGenBench (train/IPD_Gaussian and train/Uniform; captions not downloaded) | 81,891 | CC BY 4.0 | licence text read: card and paper appendix K (copy stored by Necessity Robotics); the paper says that third-party source assets keep their original licence terms | synthetic; true labels | Y. Cui, F. Qiao, N. Jacobs. StereoGenBench: A Synthetic Multi-Camera Benchmark for Stereo Generation under Controlled Baseline Regimes. arXiv:2605.23237, 2026 |
| WMGStereo-150k, indoor | 71,800 | BSD-3-Clause | tag only (Hugging Face card, read again 2026-11-24: bsd-3-clause); generator repository BSD-3 text read | synthetic; true labels | D. Yan, A. Raistrick, J. Deng. What Makes Good Synthetic Training Data for Zero-Shot Stereo Matching? CVPR 2026, pp. 34171-34180. arXiv:2504.16930. Generator: Copyright (c) 2025, Princeton Vision & Learning Lab, BSD 3-Clause |
| TartanAir V1 (hold-out environments excluded) | 45,903 | CC BY 4.0 | licence text read (copy stored by Necessity Robotics) | synthetic; true labels | W. Wang, D. Zhu, X. Wang, Y. Hu, Y. Qiu, C. Wang, Y. Hu, A. Kapoor, S. Scherer. TartanAir: A Dataset to Push the Limits of Visual SLAM. IEEE/RSJ IROS 2020 |
| TartanAir V2, indoor environments | 43,208 | CC BY 4.0 | licence text read on the site tartanair.org (CC BY 4.0); the Hugging Face card is tagged bsd-3-clause; we follow the site | synthetic; true labels | TartanAir V2, Carnegie Mellon University (AirLab); W. Wang, Y. Hu, Y. Qiu, S. Shen, Y. Shaoul. https://tartanair.org/ |
| WMGStereo-150k, nature_filtered | 13,669 | BSD-3-Clause | tag only (Hugging Face card, read again 2026-11-24: bsd-3-clause); generator repository BSD-3 text read | synthetic; true labels | D. Yan, A. Raistrick, J. Deng. What Makes Good Synthetic Training Data for Zero-Shot Stereo Matching? CVPR 2026, pp. 34171-34180. arXiv:2504.16930. Generator: Copyright (c) 2025, Princeton Vision & Learning Lab, BSD 3-Clause |
| Own OAK-D Lite recordings (Necessity Robotics) | 10,065 | own data | own recordings | real images; labels from our own teacher nrs-large | Necessity Robotics. Recording oak_20260329_161834 is held out and was never used. |
| TUM-VI (rectified, 512 px, room and corridor) | 14,199 | CC BY 4.0 | licence text read (copy stored by Necessity Robotics) | real images; labels from our own teacher nrs-large | D. Schubert, T. Goll, N. Demmel, V. Usenko, J. Stueckler, D. Cremers. The TUM VI Benchmark for Evaluating Visual-Inertial Odometry. IROS 2018. arXiv:1804.06120 |
| Rosario v2 | 3,456 | CC BY 4.0 | licence text read (copy stored by Necessity Robotics) | real images; labels from our own teacher nrs-large | N. Soncini, J. Cremona, E. Vidal, M. García, G. Castro, T. Pire. The Rosario dataset v2: Multi-modal dataset for agricultural robotics. The International Journal of Robotics Research, 2025. doi:10.1177/02783649251368909. CIFASIS |
| DrivingStereo (left/right images only) | 19,120 | MIT | licence text read: 'This dataset is released under the MIT license.' (copy stored by Necessity Robotics) | real images; labels from our own teacher nrs-large | G. Yang, X. Song, C. Huang, Z. Deng, J. Shi, B. Zhou. DrivingStereo: A Large-Scale Dataset for Stereo Matching in Autonomous Driving Scenarios. CVPR 2019. MIT License, Copyright (c) 2019 drivingstereo-dataset |
| TartanRGBT (ZED left/right images only; its FoundationStereo depth was not used) | 24,545 | BSD-3-Clause-Clear | tag only (Hugging Face card, read again 2026-11-24: bsd-3-clause-clear) | real images; labels from our own teacher nrs-large | TartanRGBT dataset (huggingface.co/datasets/theairlabcmu/TartanRGBT). Cite: P. Maheshwari, J. Karhade, Y. Chawla, I. Adu, F. Heisen, A. Porco, A. Jong, Y. Liu, S. Pitla, S. Scherer, W. Wang. AnyThermal: Towards Learning Universal Representations for Thermal Perception. IEEE ICRA 2026. arXiv:2602.06203 |
| DROID (raw ZED stereo) | 16,421 | CC BY 4.0 | licence text read (paper; copy stored by Necessity Robotics) | real images; labels from our own teacher nrs-large | A. Khazatsky, K. Pertsch, S. Nair, et al. DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset. Robotics: Science and Systems (RSS) 2024. doi:10.15607/RSS.2024.XX.120. arXiv:2403.12945 |
| Unitree G1 head stereo (Hugging Face unitreerobotics/*, rectified streams) | 3,751 | Apache-2.0 | tag only (Hugging Face cards; all 33 repositories read again 2026-11-24: apache-2.0) | real images; labels from our own teacher nrs-large | Unitree Robotics, Unitree G1 Dex1 / WBT teleoperation datasets on Hugging Face (huggingface.co/unitreerobotics; the 33 repositories are listed in ATTRIBUTION.md) |
| FoMo (boreal forest, ZED X rectified) | 3,759 | CC BY 4.0 | LICENSE.txt in the data bucket (read by us; our pre-registration, A9 note 6) | real images; labels from our own teacher nrs-large | M. Boxan, G. Jeanson, A. Krawciw, E. Daum, X. Qiao, S. Lilge, T. D. Barfoot, F. Pomerleau. FoMo: A Multi-Season Dataset for Robot Navigation in Forêt Montmorency. arXiv:2603.08433, 2026. Copyright (c) 2026 Matej Boxan, Gabriel Jeanson, Alexander Krawciw, Effie Daum, Xinyuan Qiao, Sven Lilge, Timothy D. Barfoot, and Francois Pomerleau. https://registry.opendata.aws/fomo-norlab/ |

Licence texts read: 8 sources. Tag only: 4 sources. Own data: 1 source.

Unitree repositories used (Hugging Face, `unitreerobotics/<name>`, 33 in total, each tagged apache-2.0): G1_Dex1_Arrange_Flowers, G1_Dex1_Arrange_Surgical_Kits, G1_Dex1_Assemble_Mouse, G1_Dex1_Assemble_The_Watch_Band, G1_Dex1_Beading, G1_Dex1_Cosmetic_Storage, G1_Dex1_Fold_Clothes_Piece_By_Piece, G1_Dex1_Fold_Socks, G1_Dex1_Guess_Ball, G1_Dex1_Hang_Pants, G1_Dex1_Hang_To_Dry_Socks, G1_Dex1_Inflate_Basketball, G1_Dex1_Insert_Electronic_Components, G1_Dex1_Jewelry_Box_Storage, G1_Dex1_Jigsaw_Puzzle, G1_Dex1_Light_The_Birthday_Candles, G1_Dex1_Mix_Drinks, G1_Dex1_Notebook_Storage, G1_Dex1_Organize_Desk, G1_Dex1_Organize_The_Tweezer_Set, G1_Dex1_Packaged_Bread, G1_Dex1_Pink_Tool_Storage_Hard, G1_Dex1_Replace_Mosquito_Liquid, G1_Dex1_Serve_Pizza, G1_Dex1_Shoe_Rack_Placement, G1_Dex1_Soap_Storage, G1_Dex1_Sort_The_Pills, G1_Dex1_Stock_Beverages_On_Shelves, G1_Dex1_Storage_For_Emergency_Kit, G1_Dex1_Store_Earphones, G1_Dex1_Take_Out_The_Fan_And_Turn_It_On, G1_Dex1_Wash_Cherries, G1_WBT_Brainco_Pick_Up_Clothes.

The teacher (nrs-large) was trained on WMGStereo-150k only; that source is credited above.

**Not training data:** Middlebury, ETH3D, KITTI 2012/2015 and Booster were used for evaluation only. The RT-IGEV (Scene Flow) and LAS2-S (authors') weights were used for evaluation only. The example pairs are from our own OAK-D Lite.

DrivingStereo images may show faces and number plates without blur. No DrivingStereo image is in this release.
