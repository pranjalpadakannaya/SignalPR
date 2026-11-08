"""A tiny OpenAI-compatible server for tests (stdlib only). Replies echo the settings they received."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class FakeLLM:
    def __init__(self, fail_first: int = 0, status: int = 429):
        self.requests: list[dict] = []
        self.fail_first, self.status = fail_first, status
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, obj, headers=None):
                body = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                self._send(200, {"data": [{"id": "fake-coder"}]})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append(body)
                if self.headers.get("Authorization") == "Bearer bad":
                    return self._send(401, {"error": "bad key"})
                if outer.fail_first > 0:
                    outer.fail_first -= 1
                    return self._send(outer.status, {"error": "busy"}, {"Retry-After": "0"})
                user = body["messages"][-1]["content"]
                content = json.dumps(
                    {"echo": user[:40], "seed": body.get("seed"), "temperature": body.get("temperature")}
                )
                if "fenced" in user:
                    content = "<think>hmm</think>Sure:\n```json\n" + content + "\n```"
                self._send(
                    200,
                    {
                        "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": len(user.split()), "completion_tokens": 12},
                    },
                )

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def __enter__(self):
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *a):
        self.server.shutdown()
