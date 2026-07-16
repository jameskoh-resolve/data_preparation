#!/usr/bin/env python
"""
Simple HTTP Server that also proxies ASOS images to bypass browser CORS/header restrictions.
Usage: python scripts/curation/serve.py 8000
"""

import http.server
import socketserver
import urllib.parse
import sys
import os

# Ensure data_preparation root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.vis_image import get_image_content

class ProxyHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith('/image-proxy/'):
            query = urllib.parse.urlparse(self.path).query
            params = urllib.parse.parse_qs(query)
            if 'url' in params:
                url = params['url'][0]
                content = get_image_content(url)
                if content:
                    self.send_response(200)
                    # Add CORS headers just in case
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.end_headers()
                    self.wfile.write(content)
                    return
            self.send_error(404, "Image fetch failed")
            return
        
        # Default behavior for other files
        return super().do_GET()

if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    # allow address reuse to prevent "address already in use" errors if restarting quickly
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", port), ProxyHandler) as httpd:
        print(f"Serving at port {port} with ASOS image proxy at /image-proxy/?url=...")
        httpd.serve_forever()
