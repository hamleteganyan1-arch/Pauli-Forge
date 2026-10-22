"""Start and exercise the local Streamlit server, then stop only that process.

This checks real HTTP readiness alongside the separately tested Python AppTest
interactions. It does not publish a web preview or assert scientific accuracy.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path


def validate_console(port=0):
    import pauliforge.ui

    if not port:
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(Path(pauliforge.ui.__file__)),
        "--server.headless=true",
        "--server.address=127.0.0.1",
        f"--server.port={port}",
        "--browser.gatherUsageStats=false",
    ]
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with tempfile.TemporaryFile() as log:
        process = subprocess.Popen(command, stdout=log, stderr=log, env=env)
        try:
            deadline = time.monotonic() + 30
            ready = False
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError("Streamlit exited before HTTP readiness")
                try:
                    with opener.open(f"http://127.0.0.1:{port}/_stcore/health", timeout=1) as reply:
                        ready = reply.status == 200 and reply.read().strip() == b"ok"
                    if ready:
                        break
                except OSError:
                    time.sleep(0.2)
            if not ready:
                raise RuntimeError("Streamlit did not become healthy within 30 seconds")
            with opener.open(f"http://127.0.0.1:{port}/", timeout=5) as reply:
                document = reply.read()
                if reply.status != 200 or b"<html" not in document.lower():
                    raise RuntimeError("Streamlit frontend document is missing")
            return {
                "health_http_status": 200,
                "frontend_http_status": 200,
                "status": "NUMERICAL_HTTP_SMOKE_PASS",
                "scientific_certificate": False,
            }
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate_console(args.port)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
