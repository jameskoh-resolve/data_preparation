import requests
url = "https://images.asos-media.com/products/prohibited-pegasus-contrast-peak-cap-in-cream-camo/210751425-2?etag=%22391ac126bb2cdfa1ea5513947ec5782a%22&wid=1024"
headers = {
    "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "accept-language": "en-GB,en-US;q=0.9,en;q=0.8,zh-CN;q=0.7,zh-TW;q=0.6,zh;q=0.5",
    "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36"
}
resp = requests.get(url, headers=headers)
print(resp.status_code)
