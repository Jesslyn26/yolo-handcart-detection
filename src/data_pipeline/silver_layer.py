import os
from PIL import Image
from tqdm import tqdm

from utils.silver_utils import clip_score

def run_silver(bronze_root: str, batch_id: str):

    silver_root = f"data/silver/{batch_id}"

    os.makedirs(silver_root, exist_ok=True)

    threshold = 0.25

    kept = 0
    total = 0

    for root, _, files in os.walk(bronze_root):
        for img_name in tqdm(files):
            total += 1

            img_path = os.path.join(root, img_name)

            try:
                score = clip_score(img_path)

                # keep good + small % of bad (important for robustness)
                if score > threshold or (total % 20 == 0):
                    img = Image.open(img_path).convert("RGB")
                    save_path = os.path.join(silver_root, img_name)
                    img.save(save_path)
                    kept += 1

            except Exception as e:
                print(f"skip {img_name}: {e}")

    print(f" Silver done | kept {kept}/{total}")
    return silver_root, batch_id