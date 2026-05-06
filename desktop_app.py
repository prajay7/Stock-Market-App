from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import requests
import webview


PROJECT_ROOT = Path(__file__).resolve().parent
STREAMLIT_APP = PROJECT_ROOT / "dashboard" / "streamlit_app.py"


def _wait_for_service(url: str, timeout_seconds: int = 60) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            response = requests.get(url, timeout=2)
            if response.status_code < 500:
                return
        except Exception:
            pass
        time.sleep(0.5)
    raise TimeoutError(f"Streamlit did not start within {timeout_seconds} seconds")


def _start_streamlit(port: int) -> subprocess.Popen[str]:
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(STREAMLIT_APP),
        "--server.port",
        str(port),
        "--server.address",
        "127.0.0.1",
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]
    return subprocess.Popen(command, cwd=str(PROJECT_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch the stock dashboard in a native webview window.")
    parser.add_argument("--port", type=int, default=8501, help="Local Streamlit port")
    args = parser.parse_args()

    url = f"http://127.0.0.1:{args.port}"
    process = _start_streamlit(args.port)
    try:
        _wait_for_service(f"{url}/_stcore/health")
        window = webview.create_window("Stock AI Dashboard", url, width=1400, height=900)
        webview.start(debug=False, gui=None)
        return 0 if window is not None else 1
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()


if __name__ == "__main__":
    raise SystemExit(main())