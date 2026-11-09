"""python -m app [--host 127.0.0.1] [--port 8000]  ->  SignalPR dashboard, API and GitHub webhook."""

import argparse

import uvicorn


def main():
    ap = argparse.ArgumentParser(prog="python -m app", description="SignalPR web app")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reload", action="store_true", help="restart on code changes (development)")
    a = ap.parse_args()
    print(f"SignalPR on http://{a.host}:{a.port}")
    uvicorn.run("app.server:app", host=a.host, port=a.port, reload=a.reload, log_level="info")


if __name__ == "__main__":
    main()
