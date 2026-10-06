import csv
import os
import time

import requests

URLS = [
    "https://www.kwsp.gov.my/en/others/resource-centre/references",
    "https://www.kwsp.gov.my/en/others/resource-centre/references/epf-act-1991",
    "https://www.kwsp.gov.my/en/others/resource-centre/easy-guides/employer",
    "https://www.kwsp.gov.my/en/others/resource-centre/easy-guides/member",
    "https://www.kwsp.gov.my/en/others/resource-centre/dividend"
]

RAW_DIR = os.path.join("data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

manifest_path = os.path.join(RAW_DIR, "sources.csv")

with open(manifest_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["file_name", "url", "status_code"])

    for i, url in enumerate(URLS, start=1):
        file_name = f"page_{i:02d}.html"
        response = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (portfolio project)"},
            timeout=30,
        )
        print(file_name, response.status_code, url)

        if response.status_code == 200:
            with open(os.path.join(RAW_DIR, file_name), "w", encoding="utf-8") as page:
                page.write(response.text)

        writer.writerow([file_name, url, response.status_code])
        time.sleep(2)