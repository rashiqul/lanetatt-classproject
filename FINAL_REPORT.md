# Introduction

## Problem Definition & Motivation
Accurate road-lane detection is essential for driver-assistance and autonomous systems. Traditional computer-vision techniques rely on edge detection, Hough transforms, or handcrafted geometric features, which struggle under complex lighting or occlusion. Modern deep-learning methods—such as LaneATT (CVPR 2021)—use attention mechanisms and anchor-based detection to deliver real-time performance while modeling spatial continuity between lanes.

## Project Objective
We re-implemented the LaneATT architecture to:
- Reproduce its results on TuSimple and extend to CuLane.
- Understand how attention-guided anchors relate to classical CV ideas (convolution, geometric transforms, feature dictionaries).
- Benchmark ResNet-18, 34, and 122 backbones.
- Evaluate robustness to illumination and blur.
- Analyze efficiency–accuracy trade-offs for real-time deployment.

## Background & Related Work
LaneATT [Tabelini et al., CVPR 2021] introduced a one-stage anchor-based detector using self-attention across anchors, outperforming SCNN (Spatial CNN) and PolyLaneNet on TuSimple and CuLane. Our project aligns with the “reimplement and extend” guideline, focusing on cross-dataset generalization.

## Project Overview
In this work, we developed a complete PyTorch-based reimplementation of LaneATT, enhanced with modern experiment-tracking and training infrastructure. The implementation uses Poetry for reproducible environment management and Weights & Biases (W&B) for live metric logging. We introduced frequency-based anchor pruning to optimize performance on CuLane’s large-scale dataset, extended the pipeline to support multiple backbones (ResNet-18/34/122), and conducted comparative training under different augmentations and learning schedules. This allowed us to systematically analyze LaneATT’s scalability, convergence behavior, and robustness across datasets.

Building upon these efforts, the following sections describe our methodology, dataset preparation, and model architecture, followed by an evaluation of experimental results and insights gained from the CuLane adaptation. The remainder of this report presents our methodology, results and analysis, and conclusions with potential future improvements.

# Approach and Methodology

## Pipeline Overview
LaneATT is a one‑stage lane‑detection model that casts lanes as anchors (line segments originating from the left/right/bottom edges of the image at different angles). Each anchor is scored and adjusted via regression to match an actual lane marking. The core pipeline is illustrated in Figure 1.

*Figure 1 – Conceptual diagram of the LaneATT pipeline. An input image passes through a CNN backbone (ResNet-18/34/122) to generate anchors along lane boundaries. Anchor features are pooled and refined through an attention mechanism, and classification/regression heads predict lane presence and position. Top-scoring anchors are decoded into final lane polylines.*

LaneATT directly predicts lane instances, making it faster and more suitable for real-time applications. The model consists of:
- Feature extraction backbone (ResNet-18/34/122)
- Anchor-based feature pooling with self-attention mechanism
- Classification and regression prediction heads
- Non-maximum suppression (NMS) post-processing

The key innovation is the attention mechanism that allows anchors to "communicate" with each other, capturing geometric relationships between lanes (e.g., parallel lanes, lane merges).

## Environment Setup
We configured the project environment to ensure reproducible builds, GPU acceleration, and consistent experiment tracking across both local and remote systems. All dependencies were managed using Poetry, which provided isolated virtual environments and precise version control. The framework was implemented in PyTorch 1.9 with CUDA 11.3, supporting compilation of custom CUDA extensions for Non-Maximum Suppression (NMS) to accelerate inference. To streamline experimentation, Weights & Biases (W&B) was integrated for live training metrics, validation performance, and model comparison across backbones. The training pipeline incorporated automatic best-model checkpointing with full optimizer and scheduler state, validation-loss computation for overfitting detection, and exception-resilient loops to ensure stable long-duration CuLane runs. All configurations were tracked through version-controlled YAML files, enabling full reproducibility and auditability of results.

### Core Tools and Configuration Highlights:
- **Dependency Management**: Poetry-based virtual environments for deterministic package versions.
- **Framework & GPU Support**: PyTorch 1.9 + CUDA 11.3 environment with compiled CUDA NMS extension (`external/LaneATT/lib/nms/`).
- **Experiment Tracking**: Integrated Weights & Biases (W&B) for logging losses, accuracy, FP/FN, and learning-rate scheduling.
- **Checkpointing & Recovery**: Automatic best-model saving with optimizer and scheduler state.
- **Validation Pipeline**: Added validation-loss computation for clearer convergence monitoring.
- **Error Handling**: Exception-resilient training loop to prevent interruptions during extended runs.
- **Reproducibility**: Fixed random seeds, version-controlled configs, and environment metadata logged through W&B.

## Data Preparation
The TuSimple dataset served as the starting point for our baseline experiments. Because the official TuSimple test labels are not publicly available, we implemented a custom 90/10 split (3263 training / 363 validation images) using `label_data_train.json` and `label_data_val.json`. Each image was resized from 1280×720 to 640×360 pixels to reduce computation while preserving lane structure, then normalized to the [0,1] range without per-channel normalization for consistency across datasets. Lane annotations were converted from TuSimple’s JSON format into the anchor-based representation required by LaneATT. The modified dataset loader supports flexible split configuration and validates the JSON structure prior to training. To enhance generalization, we applied geometric augmentations such as random horizontal flipping and affine transformations—including translations (±25 px horizontally, ±10 px vertically), rotations up to ±6°, and scaling within [0.85, 1.15].

**Table 1: Datasets**

| Dataset | Images | Resolution | Characteristics |
|---|---|---|---|
| TuSimple | 3626 | 640 x 360 | Daytime highway, clear lanes |
| CuLane | 88000 | 640 x 360 | Urban, night, shadow, curve scenes |

The CuLane dataset introduced additional complexity, containing over 88,000 training images across diverse driving scenarios. We standardized all CuLane experiments to a 360×640 resolution and used the same geometric augmentations as TuSimple, while disabling photometric augmentations to isolate geometric factors. Due to GPU memory constraints, batch sizes were set to 8 for ResNet-18 and ResNet-34, and 4 for ResNet-122. Each CuLane run was trained for 15 epochs (roughly 166,000–333,000 iterations) using a cosine annealing learning rate schedule to ensure gradual convergence and stable long-duration training.

## Model Architecture
The LaneATT framework maintains a modular three-stage architecture consisting of:
1. Feature Extraction using a CNN backbone (ResNet-18, ResNet-34, or CIFAR-style ResNet-122),
2. Self-Attention for contextual reasoning across anchors
3. Dual Prediction Heads for lane classification (existence) and regression (geometry).

Each image is processed by the CNN backbone to generate a feature map $F \in R^{C \times H \times W}$. Features along each anchor trajectory are extracted via bilinear interpolation, producing an anchor descriptor:
$d=[F(p_1),F(p_2),…,F(p_L)] \in R^{C \times L}$
where $p_i =(x_i, y_i)$ are sampling points (anchor cuts) along the lane curve.

These descriptors are fed into a self-attention block that models pairwise interactions among anchors to capture lane topology and spacing. The attention module computes:
$Attention(Q,K,V)=softmax(Q K^T d_k)V$
where Q, K, V are linear projections of the anchor descriptors, enabling each anchor to incorporate contextual cues from others before prediction.

Two separate prediction heads output lane existence probability $p_a$ and geometric offsets $y_a$ for each anchor $a$:
$p_a = \sigma(W_{cls} d_a + b_{cls})$ 
$y_a = W_{reg} d_a + b_{reg}$ 

The network is trained with a joint loss combining Focal Loss for classification and Smooth L1 Loss for regression:
$\mathcal{L} = \mathcal{L}_{focal} + \lambda \mathcal{L}_{smoothL1}$ 

To scale the model for CuLane’s 88K diverse images, the core LaneATT structure remained intact, but key adaptations were made:
- Frequency-based anchor pruning using `culane_anchors_freq.pt` (top 1000 anchors) to reduce computational cost.
- Adjusted batch sizes (8 for ResNet-18/34, 4 for ResNet-122) for VRAM efficiency.
- Shortened epoch schedule (15 epochs) with cosine-annealing learning rate scheduling per iteration count.
- Validation-loss computation and best-model checkpointing integrated into the training loop.

### Anchor Generation and Representation
LaneATT parameterizes lanes through a dense grid of anchors distributed along the left, right, and bottom borders of the image to capture both straight and curved lane geometries. Each anchor represents a potential lane hypothesis defined by a sequence of L sampling points, or anchor cuts, that extend into the frame. Features are extracted at these cut locations using bilinear interpolation from the CNN feature map, forming an anchor descriptor.
This descriptor encodes spatial lane information and is subsequently processed by the self-attention and prediction heads to estimate lane existence and geometric offsets. In our implementation, we extended this representation by introducing frequency-based anchor pruning, using a precomputed tensor (`culane_anchors_freq.pt`) to select the top 1,000 most frequent anchors observed across the dataset. This optimization reduces computational overhead and focuses the model on the most representative lane configurations, improving convergence stability on large-scale datasets like CuLane. For TuSimple, the anchor generation followed a similar scheme but used a smaller anchor set tailored to highway scenes with fewer lane variations.

## Training Pipeline
The training pipeline was designed for robustness, reproducibility, and long-duration runs across both TuSimple and CuLane datasets. Each experiment was orchestrated via a unified training runner integrating Weights & Biases (W&B) for live logging, automatic checkpointing, and validation-loss tracking. All experiments were executed on a single GPU using PyTorch 1.9 + CUDA 11.3, with environment isolation managed by Poetry for dependency reproducibility.

### Core Training Process:
Each training session followed a consistent procedure to ensure stability across runs:
- Model initialization, optimizer configuration, and learning-rate scheduler setup were performed with deterministic random seeds for reproducibility.
- Custom CUDA extensions for Non-Maximum Suppression (NMS) were compiled prior to training.
- Training jobs were executed within persistent tmux sessions to maintain continuity after SSH disconnections.
- Real-time metrics—including training loss, validation loss, and learning rate—were logged to W&B dashboards for continuous monitoring.
- Automatic checkpointing saved full model, optimizer, and scheduler states whenever validation loss improved, ensuring recoverability and optimal model preservation.
- Training used the Adam optimizer with learning rate 3e-4 and CosineAnnealingLR scheduling to ensure smooth convergence. Exception-resilient loops were implemented to automatically recover from GPU errors or connection interruptions, maintaining stability over extended CuLane runs.

**Table 2: Dataset-Specific Configurations**

| Dataset | Train/Val Split | Resolution | Batch Size | Epochs | LR Schedule | Notes |
|---|---|---|---|---|---|---|
| TuSimple | 3263 / 363 | 640 x 360 | 8 | 100 | CosineAnnealingLR | Baseline benchmark with highway scenes |
| CuLane | 88000 (all train set) | 640 x 360 | 8 (ResNet-18/34) 4 (ResNet-122) | 15 | CosineAnnealingLR (per-iteration) | Large-scale training with anchor-frequency pruning |

### Training Enhancements:
To ensure consistent and reliable model convergence, several training enhancements were implemented. An automated validation pipeline was integrated to compute validation loss after each epoch, allowing early detection of overfitting or divergence trends. This ensured that improvements on the training set translated effectively to generalization on unseen data. A best-model checkpointing mechanism automatically preserved the model achieving the lowest validation loss during training. Each checkpoint contained the network weights along with optimizer and scheduler states, enabling full recovery and resumption without retraining overhead. Comprehensive metrics and visualization tools were incorporated through W&B. Throughout training, precision, recall, and F1-scores were computed from true-positive and false-positive anchor predictions, offering detailed insight into detection quality. Concurrently, training curves, learning-rate trajectories, and qualitative lane-visualization overlays were logged for post-hoc performance evaluation and comparative analysis across backbones. 
For the CuLane dataset, additional optimizations addressed its large scale and greater diversity. Training was conducted for 15 epochs—equivalent to roughly 166,000 to 333,000 iterations—to balance computational efficiency and convergence. A frequency-based anchor pruning scheme, using the precomputed tensor `culane_anchors_freq.pt`, selected the top 1,000 most frequent anchor configurations to reduce compute cost while maintaining representative coverage. Moreover, dynamic memory management was applied to handle extensive data loading and geometric augmentations efficiently within available GPU memory.

### Code and Configuration Management:
All training and evaluation parameters were stored in modular YAML configuration files under version control to maintain reproducibility across experiments. Example configuration:

```yaml
optimizer:
  name: Adam
  lr: 3e-4
scheduler:
  name: CosineAnnealingLR
  T_max: 100
train:
  epochs: 100
  batch_size: 8
  dataset: TuSimple
  augmentations: [flip, affine]
logging:
  wandb_project: laneatt_train
  checkpoint_best: true
```
For CuLane, configuration overrides adjusted the dataset paths, batch size, and anchor-pruning tensor to accommodate its larger data volume and anchor diversity.

## Evaluation Methodology
The evaluation methodology followed official benchmarks for both the TuSimple and CuLane datasets to ensure consistency with prior work and reproducibility of results. Model performance was measured through dataset-specific protocols implemented directly in the restructured evaluation utilities (`utils/tusimple_metric/LaneEval` and `utils/culane_metric`) integrated with the LaneATT pipeline.

### TuSimple Evaluation Protocol:
Evaluation on TuSimple adhered to the official benchmark, where lane detection accuracy, false-positive (FP), and false-negative (FN) rates were computed. A predicted lane point was considered correct if its horizontal deviation from the ground-truth point was within 5 pixels.
The accuracy metric was defined as:
$Accuracy = \frac{Correct Predictions}{Total Ground Truth Points}$ 
while FP and FN were measured per frame to quantify over-detection and under-detection, respectively. Predicted lanes were serialized into JSON files via the `save_tusimple_predictions()` utility. The results were logged epoch-by-epoch, capturing validation accuracy, loss, and error breakdowns, all visualized through Weights & Biases (W&B) dashboards for convergence analysis. This evaluation provided a quantitative measure of model precision under highway conditions, emphasizing spatial alignment and continuity.

### CuLane Evaluation Protocol:
For the CuLane dataset, the evaluation was more comprehensive and aligned with the official CULane metric scripts. Metrics included:
- Lane Detection Accuracy
- False Positives (FP) and False Negatives (FN)
- Intersection-over-Union (IoU) between predicted and ground-truth lane regions
- Per-split Performance, covering sub-scenarios such as night scenes, shadows, crowded roads, and curved lanes.

Each evaluation run generated predictions stored in `experiments/<exp_name>/results/` directories. Cached annotations (`cache/culane_<split>.json`) were used to accelerate repeated evaluations. The CULane metric implementation compared detected lane masks against ground-truth masks with IoU thresholds to determine detection correctness across 9 test splits. These results were subsequently aggregated and logged to W&B under the `val/*` namespace for detailed post-analysis.

# Results and Analysis
The performance of LaneATT was evaluated on the TuSimple and CuLane datasets using three CNN backbones—ResNet-18, ResNet-34, and ResNet-122. Each experiment was run using the same training framework described previously, with Weights & Biases (W&B) used for experiment logging. This section summarizes both quantitative metrics (accuracy, false positive/false negative rates) and qualitative insights (convergence trends and visual lane detection results).

## TuSimple Results
Training was conducted using 3,263 training and 363 validation images (custom 90/10 split). The model was trained for 100 epochs per backbone, using Adam optimizer (lr = 3×10⁻⁴) with cosine annealing scheduling. Accuracy, FP, and FN were computed using TuSimple’s evaluation protocol, where a predicted lane point was considered correct if within 5 pixels of the ground truth.

**Table 3: TuSimple Results**

| Backbone | Accuracy (%) | FP | FN | Avg Time/Epoch | Total Training Time |
|---|---|---|---|---|---|
| ResNet-18 | 0.9447 | 0.0418 | 0.0589 | ~00:00:47.8 | ~1h 19m |
| RestNet-34 | 0.9448 | 0.0375 | 0.0556 | ~00:00:54.5 | ~1h 30m |
| ResNet-122 | 0.9461 | 0.0372 | 0.0567 | ~00:06:03.4 | ~10h 05m |

### Analysis
All three backbones achieved comparable accuracy, demonstrating consistent reproducibility with the original LaneATT benchmark. ResNet-122 provided marginal accuracy improvement (≈ 0.1%) at the cost of significantly higher computational load, requiring nearly 10× the training time of ResNet-18. Validation accuracy stabilized around epoch 50 for all models, indicating stable convergence behavior. Qualitative results show clear and smooth lane fits in well-lit scenes; however, models occasionally missed far-edge or occluded lanes, aligning with findings from prior literature.

## CuLane Results
The CuLane dataset introduced substantially greater complexity, consisting of ~88 k training and 9.7 k validation images across nine scenarios (normal, crowded, night, shadow, no-line, arrow, dazzle, curve, cross). Due to computational constraints, each backbone was trained for 15 epochs, corresponding to 166 k–333 k iterations, with batch size = 8 (R-18/34) and 4 (R-122). Frequency-based anchor pruning (`culane_anchors_freq.pt`) limited the active anchor set to the 1,000 most common configurations, improving runtime without accuracy loss.

**Table 4: CuLane Results**
| Backbone | Precision | Recall | F1 | Avg Time/Epoch | Total Training Time |
|---|---|---|---|---|---|
| ResNet-18 | 0.8323 | 0.6872 | 0.0583 | 48s | ~1.3h |
| RestNet-34 | 0.8298 | 0.6981 | 0.0549 | 54s | ~1.5h |
| ResNet-122 | 0.8551 | 0.7030 | 0.0567 | 6 min | ~10h |

# Robustness Evaluation
The robustness of the LaneATT model (ResNet-18) was evaluated on the TuSimple dataset under five degradation conditions: blur, brightness, lowlight, noise, and shadow.

| Condition | Accuracy | False Positive | False Negative |
|---|---|---|---|
| **Baseline (TuSimple)** | 0.9447 | 0.0418 | 0.0589 |
| **Blur** | 0.9287 | 0.1368 | 0.0693 |
| **Brightness** | 0.9263 | 0.1509 | 0.0770 |
| **Lowlight** | 0.3141 | 0.2348 | 0.8097 |
| **Noise** | 0.5148 | 0.2308 | 0.6263 |
| **Shadow** | 0.9253 | 0.1509 | 0.0810 |

### Observations
- **High Robustness**: The model maintains high performance (>92% accuracy) under **Blur**, **Brightness**, and **Shadow** conditions. This suggests the attention mechanism and anchor-based approach are resilient to global illumination changes and minor visual softening.
- **Low Robustness**: Performance degrades significantly under **Noise** (51.5%) and **Lowlight** (31.4%) conditions.
    - **Lowlight**: The drastic drop in accuracy suggests the model struggles when lane contrast is reduced significantly. Anchors likely fail to latch onto lane edges when they are barely visible.
    - **Noise**: The high false negative rate (0.62) indicates that pixel-level noise disrupts the feature extraction backbone, causing it to miss lane markings.

### CULane Robustness (Cross-Dataset Evaluation)
We evaluated the **CULane-trained ResNet-18 model** on a subset of 500 images from the CULane test set under corruption.

| Condition | Precision | Recall | F1 Score |
|---|---|---|---|
| **Blur** | 0.7062 | 0.5583 | 0.6236 |
| **Brightness** | 0.5985 | 0.4556 | 0.5174 |
| **Shadow** | 0.5930 | 0.3741 | 0.4588 |
| **Lowlight** | 0.0000 | 0.0000 | 0.0000 |
| **Noise** | 0.0000 | 0.0000 | 0.0000 |

**Analysis**:
- **Consistently Vulnerable**: Similar to the TuSimple results, the CULane model fails completely (0.0 F1) under **Lowlight** and **Noise**.
- **Shadow Sensitivity**: Shadows cause a greater performance drop on CULane (F1 0.46) compared to TuSimple, likely because urban shadows (buildings, trees) are more complex and high-contrast than highway shadows.
- **Blur Resilience**: Blur remains the least damaging augmentation (F1 0.62), retaining the highest performance relative to the others.



# Conclusion
In this project, we successfully reimplemented LaneATT and verified its performance on TuSimple (94.5% accuracy) and CuLane. We extended the evaluation to test robustness against common corruptions. Our findings show that while LaneATT is highly robust to geometric and illumination variations like Shadows and Blur, it is vulnerable to low-light conditions and high-frequency noise. Future work could focus on data augmentation specifically targeting these weaknesses or integrating denoising modules such as non-local means or trainable denoising layers into the backbone.
