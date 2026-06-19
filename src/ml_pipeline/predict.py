from ultralytics import YOLO


def predict_image(model_path: str, image_path: str):
    model = YOLO(model_path)

    results = model.predict(
        source=image_path,
        conf=0.25,
        save=True
    )

    return results