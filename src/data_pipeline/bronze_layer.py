import os
import json
from datetime import datetime
from icrawler.builtin import GoogleImageCrawler
from utils.bronze_utils import scrape_bronze
from datetime import datetime

def run_bronze(batch_id=None):
    if batch_id is None:
        batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    bronze_root = os.path.join("data", "bronze", batch_id)
    os.makedirs(bronze_root, exist_ok=True)

    queries = [
        ("handcart", f"{bronze_root}/handcart"),
        ("warehouse hand trolley", f"{bronze_root}/trolley"),
        ("shopping cart supermarket", f"{bronze_root}/shopping_cart"),
        ("wheelbarrow construction", f"{bronze_root}/wheelbarrow"),
    ]

    logs = []

    for query, path in queries:
        log = scrape_bronze(query, path)
        logs.append(log)

    with open(f"{bronze_root}/metadata.json", "w") as f:
        json.dump(logs, f, indent=4)

    print("Bronze collection complete")

    return bronze_root, batch_id