# YOLO Handcart Detection

## Overview

This project explores how to adapt a pretrained YOLO object detection model to recognize a custom object class: **handcarts**, that is not included in the standard COCO dataset.

Rather than training a detector from scratch, the project focuses on leveraging transfer learning and building a reproducible machine learning pipeline for data collection, model fine-tuning, evaluation, and deployment.

## Objectives

The goals of this project are to:

- Build a reproducible **Bronze → Silver → Gold data pipeline**
- Collect diverse real-world images from the internet (scrape data from the internet)
- Filter noisy data using embedding similarity (CLIP)
- Auto-label images using foundation models (GroundingDINO: ref: https://medium.com/data-science/automatic-labeling-of-object-detection-datasets-using-groundingdino-b66c486656fe)
- Construct a YOLO-compatible dataset with train/val/test split
- Fine-tune a pretrained YOLO model (YOLOv8)
- Evaluate using standard object detection metrics
- Track experiments using Weights & Biases
- Enable real-time inference via Streamlit/API

## Proposed Workflow

Instead of manually labeling everything from scratch, the system uses a **data-centric AI approach**:

- Build a reproducible **Bronze → Silver → Gold data pipeline**
- Collect diverse real-world images from the internet
- Filter noisy data using embedding similarity (CLIP)
- Auto-label images using foundation models (GroundingDINO)
- Construct a YOLO-compatible dataset with train/val/test split
- Fine-tune a pretrained YOLO model (YOLOv8)
- Evaluate using standard object detection metrics
- Track experiments using Weights & Biases
- Enable real-time inference via Streamlit/API

## Overall System Architecture

Internet Images
      ↓
Bronze Layer (Web scraping + metadata logging)
      ↓
Silver Layer (CLIP-based filtering + cleanup)
      ↓
Gold Layer (GroundingDINO auto-labeling + dataset construction)
      ↓
Dataset Versioning (batch_id)
      ↓
YOLO Training (Ultralytics)
      ↓
Evaluation (mAP / Precision / Recall)
      ↓
Inference (Streamlit / API)

## Bronze Layer (Data Collection)
Scrapes images from Google or web sources
Stores raw, unfiltered images
Logs metadata (query, timestamp, source)
Organizes data into batch-based folders for reproducibility

Output: data/bronze/{batch_id}/

## Silver Layer (Data Cleaning)
Filters irrelevant images using CLIP similarity scoring
Removes low-quality or noisy samples
Optionally retains a small fraction of low-score samples for robustness
Prepares dataset for labeling

Output: data/silver/{batch_id}/

## Gold Layer (Dataset Construction)

- Uses GroundingDINO for auto labeling.
- Generates bounding boxes for handcarts
- Converts predictions into YOLO annotation format
- Filters low-confidence detections
- Splits dataset into:
   - Train (70%)
   - Validation (20%)
   - Test (10%)
- Generates data.yaml automatically for YOLO training

Output structure:
```text
data/gold/{batch_id}/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml
```

## Model Training

Choosen model: Ultralytics YOLOv8

Fine-tunes a pretrained YOLO model (yolov8s.pt) taht will train on the Gold dataset. This stage will also log all the  experiments with batch-based naming (batch_id). it also saves both best.pt and last.pt checkpoints

## Evaluation

Model performance is evaluated using:
- mAP@50 (mean Average Precision @ IoU=0.5): haviong the object ground truth and the predicted bounding boxed to be overlapped by 50% threshold
- mAP@50-95: average mAP accriss multiple IoU thresholds 
- Precision: how many predicted handcarts are correct 
- Recall: how many did we detct handcarts

Evaluation is performed using:
- validation set (during training)
- test set (final unbiased evaluation)
- 
Only the best checkpoint (best.pt) is used for final evaluation.

## Inference & Deployment (WIP)
- Loads trained best.pt model
Supports:
- Image upload inference (Streamlit)
- Real-time inference (future extension)
- API-based inference (future extension)

## Experiment Tracking

Each experiment is fully reproducible using: batch_id = YYYYMMDD_HHMMSS

This ensures:
- Dataset versioning
- Model versioning
- Full training traceability

Integrated with Weights & Biases (W&B) for:
- Loss tracking
- Metric visualization
- Run comparison

Future Improvement:
- Active learning loop (model selects uncertain samples for labeling)
- Human-in-the-loop annotation refinement
- Dataset quality scoring system
- Model drift detection in production
- Streamlit real-time inference dashboard
- Airflow orchestration for scalable pipelines
