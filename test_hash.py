import requests
import hashlib
url = "https://images.asos-media.com/products/asos-design-curve-textured-button-through-long-sleeve-top-in-black/209471251-3?etag=%2295b51ffcb19069a3bee6194af759a8f6%22&wid=1024"
im_name = "95b51ffcb19069a3bee6194af759a8f6"
resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
if resp.status_code == 200:
    content = resp.content
    md5_hash = hashlib.md5(content).hexdigest()
    print(f"MD5: {md5_hash}")
    print(f"im_name: {im_name}")
    print(f"Match: {md5_hash == im_name}")
else:
    print(f"Failed to download: {resp.status_code}")
