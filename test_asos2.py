import requests
url = "https://images.asos-media.com/products/prohibited-pegasus-contrast-peak-cap-in-cream-camo/210751425-2?etag=%22391ac126bb2cdfa1ea5513947ec5782a%22&wid=1024"
headers = {
    "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "accept-language": "en-US,en;q=0.5",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}
try:
    resp = requests.get(url, headers=headers, timeout=5)
    print("STATUS:", resp.status_code)
except Exception as e:
    print("ERROR:", e)
