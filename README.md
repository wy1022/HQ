# HQ
New Traffic Sign Detector.

This repository contains the source code and experimental data for the proposed model. It includes the implementations of the three modules, model configurations, ablation study results, comparative experiment results, and test results in practical applications.

Publication Information:
@article{ZHANG2026134355,
title = {HQ-YOLO: Robust traffic sign detection via high-dimensional feature mapping and quad-stream wavelet transform},
journal = {Neurocomputing},
pages = {134355},
year = {2026},
issn = {0925-2312},
doi = {https://doi.org/10.1016/j.neucom.2026.134355},
url = {https://www.sciencedirect.com/science/article/pii/S0925231226017534},
author = {Jianming Zhang and Yuan Wang and Zikang Liu and Lei Wang},
keywords = {YOLOv12s, CrossStar, Quad-stream wavelet transform, High-dimensional feature space},
abstract = {With the vigorous development of urban road transportation, traffic sign detection technology plays an increasingly important role in Intelligent Transportation Systems (ITS). However, adverse weather conditions (such as rain and fog), lighting changes, long-distance shooting, and background clutter continue to pose significant challenges. In order to achieve high detection accuracy for small object in traffic scenarios, we improve YOLOv12s to get a new traffic sign detector named HQ-YOLO. First, the CrossStar module is carefully designed. This module can effectively map input features to a high-dimensional feature space, thereby enriching feature representations. Second, the Quad-stream Wavelet Transform module is creatively proposed. The traditional multi-level wavelet transform only continuously decomposes the approximate components, while we decompose all four components of each level. By doing so, it effectively preserves global features while emphasizing the target fine-grained features, thereby further improving detection accuracy. Third, a Multi-scale Efficient Large Kernel convolution module is designed. This module integrates multi-scale features and global contextual information, effectively adapting to different sizes of traffic signs, and expanding the limited receptive field of vanilla convolution. Finally, the experimental results on the GTSDB, CCTSDB 2021, and GLARE datasets show that HQ-YOLO significantly improves detection accuracy. Compared with YOLOv12s, HQ-YOLO improves mAP0.5 by 3.1%, 3.2%, and 3.5%, respectively. The code and datasets are available at https://github.com/wy1022/HQ.}
}
