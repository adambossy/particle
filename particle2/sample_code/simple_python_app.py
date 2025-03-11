import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer

# Keep track of visits (Note: This resets when server restarts)
visit_count = 0

# HTML template without styling
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Simple Python Web App</title>
</head>
<body>
    <h1>Welcome to My Simple Web App!</h1>
    <div>
        Button clicked: {count} times
    </div>
    <form method="get">
        <button type="submit">Click Me!</button>
    </form>
</body>
</html>
"""


class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global visit_count

        # Parse the URL
        parsed_path = urllib.parse.urlparse(self.path)

        # Increment counter for any GET request except favicon
        if parsed_path.path != "/favicon.ico":
            visit_count += 1

        # Send response
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()

        # Send the HTML content
        self.wfile.write(HTML_TEMPLATE.format(count=visit_count).encode())


def run(server_class=HTTPServer, handler_class=SimpleHTTPRequestHandler, port=8001):
    server_address = ("", port)
    httpd = server_class(server_address, handler_class)
    print(f"Server running at http://localhost:{port}")
    httpd.serve_forever()


if __name__ == "__main__":
    run()
