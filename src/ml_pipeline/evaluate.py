from ultralytics import YOLO


def evaluate_model(model_path: str, data_yaml: str):
    """
    Evaluate a trained YOLO model on a labeled validation dataset.

    Args:
        model_path: Path to trained weights (e.g. best.pt)
        data_yaml: Path to data.yaml
    """
    model = YOLO(model_path)

    metrics = model.val(data=data_yaml)

    print("\n===== Evaluation Results =====")
    print(f"mAP50      : {metrics.box.map50:.4f}")
    print(f"mAP50-95   : {metrics.box.map:.4f}")
    print(f"Precision  : {metrics.box.mp:.4f}")
    print(f"Recall     : {metrics.box.mr:.4f}")

    return metrics