from datetime import datetime
from pathlib import Path
import os

from ultralytics import YOLO


def train_yolo(
    batch_id: str,
    data_yaml: str,
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
):
    # Load pretrained model
    model = YOLO("yolov8s.pt")

    # Unique experiment name
    run_name = "yolo_model_" + batch_id

    # Train
    train_results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project= os.getenv("WANDB_PROJECT_NAME"),
        name=run_name,
    )

    # Evaluate using the best checkpoint from training
    metrics = train_results.val(data=data_yaml)

    # Location where YOLO saved this run
    run_dir = Path(train_results.save_dir)
    best_model_path = run_dir / "weights" / "best.pt"

    return {
        "model": model,
        "metrics": metrics,
        "run_name": run_name,
        "run_dir": str(run_dir),
        "best_model_path": str(best_model_path),
    }