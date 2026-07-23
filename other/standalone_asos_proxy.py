import http.server
import socketserver
import urllib.parse
import urllib.request

class ProxyHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith('/image-proxy/'):
            query = urllib.parse.urlparse(self.path).query
            params = urllib.parse.parse_qs(query)
            if 'url' in params:
                url = params['url'][0]
                try:
                    req = urllib.request.Request(
                        url,
                        headers={
                            "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                            "accept-language": "en-GB,en-US;q=0.9,en;q=0.8",
                            "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36"
                        }
                    )
                    with urllib.request.urlopen(req, timeout=10) as resp:
                        content = resp.read()
                    self.send_response(200)
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.end_headers()
                    self.wfile.write(content)
                    return
                except Exception as e:
                    print(f"Error fetching {url}: {e}")
            self.send_error(404, "Image fetch failed")
            return
        return super().do_GET()

if __name__ == '__main__':
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer(("", 8000), ProxyHandler) as httpd:
        print("Standalone proxy running on http://127.0.0.1:8000")
        httpd.serve_forever()
