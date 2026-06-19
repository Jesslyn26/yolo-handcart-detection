import os
from icrawler.builtin import GoogleImageCrawler
import json
from datetime import datetime

def scrape_bronze(query, save_dir, max_num=150):
    os.makedirs(save_dir, exist_ok=True)

    crawler = GoogleImageCrawler(storage={"root_dir": save_dir})
    crawler.crawl(keyword=query, max_num=max_num)

    return {
        "query": query,
        "save_dir": save_dir,
        "max_num": max_num,
        "timestamp": datetime.utcnow().isoformat()
    }