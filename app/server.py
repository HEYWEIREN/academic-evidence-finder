from __future__ import annotations

import json
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

try:
    from .search_engine import ENGINE, SearchConfig
except ImportError:  # pragma: no cover
    from search_engine import ENGINE, SearchConfig

try:
    from evaluate import evaluate_all
except ImportError:  # pragma: no cover
    evaluate_all = None


ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = ROOT / "web"
HOST = "127.0.0.1"
PORT = 8000


class AcademicEvidenceHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

    def do_GET(self):  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/"):
            self.handle_api(parsed.path, parse_qs(parsed.query))
            return
        if parsed.path == "/":
            self.path = "/index.html"
        return super().do_GET()

    def handle_api(self, path: str, params: dict[str, list[str]]) -> None:
        if path == "/api/search":
            query = self._param(params, "q", "")
            mode = self._param(params, "mode", "hybrid")
            topic = self._param(params, "topic", "")
            year_value = self._param(params, "year", "")
            limit_value = self._param(params, "limit", "12")
            year = int(year_value) if year_value.isdigit() else None
            limit = int(limit_value) if limit_value.isdigit() else 12
            payload = ENGINE.search(
                query,
                SearchConfig(
                    mode=mode if mode in {"hybrid", "bm25", "semantic"} else "hybrid",
                    year=year,
                    topic=topic or None,
                    limit=limit,
                ),
            )
            self.send_json(payload)
            return

        if path.startswith("/api/papers/"):
            paper_id = unquote(path.removeprefix("/api/papers/"))
            paper = ENGINE.get_paper(paper_id)
            if paper is None:
                self.send_json({"error": "paper not found"}, HTTPStatus.NOT_FOUND)
                return
            self.send_json(paper)
            return

        if path == "/api/topics":
            self.send_json({"topics": ENGINE.topics(), "years": ENGINE.years(), "paper_count": ENGINE.paper_count()})
            return

        if path == "/api/evaluate":
            if evaluate_all is None:
                self.send_json({"error": "evaluation module unavailable"}, HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self.send_json(evaluate_all())
            return

        self.send_json({"error": "unknown api route"}, HTTPStatus.NOT_FOUND)

    def send_json(self, payload, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    @staticmethod
    def _param(params: dict[str, list[str]], key: str, default: str) -> str:
        values = params.get(key)
        return values[0] if values else default

    def log_message(self, format, *args):  # noqa: A003
        print("[%s] %s" % (self.log_date_time_string(), format % args))


def run() -> None:
    server = ThreadingHTTPServer((HOST, PORT), AcademicEvidenceHandler)
    print(f"Academic Evidence Finder running at http://{HOST}:{PORT}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    run()
