import pandas as pd
import requests
import hashlib

df = pd.read_csv('curated_datasets/combined_catalogs/v4/accessory_detect_v4_dataset.csv', nrows=5)
for _, row in df.iterrows():
    url = row['original_url']
    im_name = row['im_name']
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            md5_hash = hashlib.md5(resp.content).hexdigest()
            print(f"URL: {url}")
            print(f"im_name: {im_name}")
            print(f"MD5:     {md5_hash}")
            print(f"Match:   {im_name == md5_hash}\n")
    except Exception as e:
        print(f"Failed {url}: {e}")
