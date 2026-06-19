import os
import cv2
import random
import shutil
from groundingdino.util.inference import load_model, load_image, predict, annotate


def run_gold(silver_root: str, batch_id: str):

    config_path = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    weights_path = "GroundingDINO/weights/groundingdino_swint_ogc.pth"

    model = load_model(config_path, weights_path)

    TEXT_PROMPT = "handcart ."
    BOX_THRESHOLD = 0.35

    base_dir = f"data/gold/{batch_id}"

    img_dirs = {
        "train": os.path.join(base_dir, "images/train"),
        "val": os.path.join(base_dir, "images/val"),
        "test": os.path.join(base_dir, "images/test"),
    }

    label_dirs = {
        "train": os.path.join(base_dir, "labels/train"),
        "val": os.path.join(base_dir, "labels/val"),
        "test": os.path.join(base_dir, "labels/test"),
    }

    for d in list(img_dirs.values()) + list(label_dirs.values()):
        os.makedirs(d, exist_ok=True)

    images = os.listdir(silver_root)
    random.shuffle(images)

    n = len(images)
    train_split = int(0.7 * n)
    val_split = int(0.9 * n)

    splits = {
        "train": images[:train_split],
        "val": images[train_split:val_split],
        "test": images[val_split:]
    }

    for split, img_list in splits.items():

        for img_name in img_list:

            img_path = os.path.join(silver_root, img_name)
            image_source, image = load_image(img_path)

            boxes, logits, phrases = predict(
                model=model,
                image=image,
                caption=TEXT_PROMPT,
                box_threshold=BOX_THRESHOLD,
                text_threshold=0.25
            )

            # save image to split folder
            dst_img_path = os.path.join(img_dirs[split], img_name)
            shutil.copy(img_path, dst_img_path)

            label_path = os.path.join(
                label_dirs[split],
                img_name.replace(".jpg", ".txt")
            )

            with open(label_path, "w") as f:

                for box, score in zip(boxes, logits):

                    if float(score) < 0.30:
                        continue

                    cx, cy, bw, bh = box.tolist()

                    cx = max(0, min(1, cx))
                    cy = max(0, min(1, cy))
                    bw = max(0, min(1, bw))
                    bh = max(0, min(1, bh))

                    f.write(f"0 {cx} {cy} {bw} {bh}\n")

    # create data.yaml
    data_yaml_path = f"{base_dir}/data.yaml"

    with open(data_yaml_path, "w") as f:
        f.write(f"""path: {base_dir}
        train: images/train
        val: images/val
        test: images/test

        names:
        0: handcart
        """)

    print("Gold dataset built with train/val/test split")

    return {
        "data_yaml": data_yaml_path,
        "base_dir": base_dir
    }