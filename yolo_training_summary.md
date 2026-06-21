# 🧠 YOLO Handcart Detection — Experiment Tracking Report

This document summarizes multiple experiments conducted to train a YOLOv8 model for handcart/trolley detection using a custom dataset and Roboflow data.

---

## 📊 Overview

We evaluated different training configurations using YOLOv8s with variations in:
- data augmentation
- learning rate schedules
- training duration
- optimization settings

All models were evaluated using:
- mAP@50
- mAP@50–95
- Precision
- Recall

---

# Experiment Results

## v1 — Baseline (No Augmentation)

### Configuration
- Model: YOLOv8s (pretrained)
- Epochs: 10
- Image size: 640
- Batch size: 16
- Optimizer: AdamW
- LR: 0.001
- Augmentation: None

### Results
- mAP@50: 0.63455  
- mAP@50–95: 0.26977  
- Precision: 0.66356  
- Recall: 0.58333  

---

## v2 — Strong Augmentation

### Configuration
- Epochs: 50
- LR: 0.01
- Optimizer: AdamW

### Augmentations
- HSV jitter
- Horizontal flip
- Mosaic
- MixUp

### Results
- mAP@50: 0.5515  
- mAP@50–95: 0.2126  
- Precision: 0.6561  
- Recall: 0.5417  

---

## v3 — Best Balanced Run

### Configuration
- Epochs: 50
- LR: 0.01
- Optimizer: AdamW

### Results
- mAP@50: 0.6040  
- mAP@50–95: 0.2645  
- Precision: 0.5674  
- Recall: 0.6285  

---

# Summary Comparison

| Model | mAP@50 | mAP@50–95 | Precision | Recall |
|------|--------|------------|-----------|--------|
| v1 baseline | **0.6345** | **0.2697** | **0.6635** | 0.5833 |
| v2 augmented | 0.5515 | 0.2126 | 0.6561 | 0.5417 |
| v3 best | 0.6040 | 0.2645 | 0.5674 | **0.6285** |

---

# Key Insights

- Best overall detection: v1 baseline
- Best recall: v3 balanced model
- Augmentation did not improve performance
- Main bottleneck likely dataset size + label noise
- Model is stable but not fully optimized

---


## **NEW** Experiment: Large-Scale Trolley Dataset

To improve model robustness, an additional dataset containing approximately 5,000 annotated images was introduced. Unlike the previous datasets, this collection contains two object categories:

- shopping trolley
- plastic shopping trolley

For training purposes, both categories can be merged into a single class (handcart) since the long-term objective is to detect trolley-like objects rather than distinguish between their specific subtypes.

Training Performance (In-Domain Evaluation)

When evaluated on its own validation/test split, the model achieved excellent performance:

Metric	Value
mAP@50	0.9027
mAP@50–95	0.7324
Precision	0.9100
Recall	0.7976

These results indicate that the model learned the characteristics of this dataset effectively.

Cross-Dataset Evaluation

The same model was then evaluated on a different trolley dataset that contains different environments, viewpoints, object styles, and annotation conventions.

Metric	Value
mAP@50	0.2439
mAP@50–95	0.1243
Precision	0.4246
Recall	0.2292

The significant performance drop suggests poor cross-dataset generalization despite strong in-domain results.

Key Observations
1. Domain shift: The training and evaluation datasets differ substantially in image characteristics, backgrounds, camera angles, and trolley appearances, making generalization challenging.
2. Label inconsistency: The large dataset distinguishes between shopping trolley and plastic shopping trolley, whereas earlier datasets use a single trolley class. Mapping both labels to a unified handcart class is expected to simplify learning and improve consistency.
3. High in-domain performance does not guarantee robustness: Although the model achieved over 90% mAP@50 on its own dataset, its performance degraded considerably on unseen datasets, indicating overfitting to dataset-specific features.
Next Steps

To improve real-world performance, the planned approach is to:

Merge multiple public trolley and handcart datasets into a unified training corpus.
Normalize all trolley-related annotations into a single class (handcart).
Increase visual diversity by incorporating airport luggage trolleys, warehouse trolleys, platform carts, utility carts, shopping carts, and other similar objects.
Use this merged dataset as the initial foundation model.
Subsequently expand the training data through the automated Bronze → Silver → Gold pipeline, which scrapes new images, filters them using CLIP, labels them with GroundingDINO, and periodically retrains the YOLO model.

This strategy is expected to produce a detector that generalizes better across unseen environments and trolley variants while supporting continuous improvement through data-centric updates.