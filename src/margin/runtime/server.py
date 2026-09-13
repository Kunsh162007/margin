"""Run llama-server as a child process bound to loopback.

One process serves one model. It binds 127.0.0.1 on a free port so nothing
leaves the machine and two Margin instances never collide.
"""

from __future__ import annotations

import socket
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import psutil

READY_TIMEOUT_S = 300.0
POLL_S = 0.5


class ServerError(RuntimeError):
    pass


@dataclass(frozen=True)
class ServerConfig:
    exe: Path
    model: Path
    gpu: bool
    threads: int
    ctx: int = 8192
    mmproj: Path | None = None
    chat_template_file: Path | None = None
    extra_args: tuple[str, ...] = field(default_factory=tuple)


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def build_args(cfg: ServerConfig, port: int) -> list[str]:
    return [
        str(cfg.exe),
        "-m", str(cfg.model),
        "--host", "127.0.0.1",
        "--port", str(port),
        "-c", str(cfg.ctx),
        "-ngl", "999" if cfg.gpu else "0",
        "-t", str(cfg.threads),
        "-np", "1",
        "--jinja",
        "--no-webui",
        *(["--mmproj", str(cfg.mmproj)] if cfg.mmproj else []),
        *(["--chat-template-file", str(cfg.chat_template_file)] if cfg.chat_template_file else []),
        *cfg.extra_args,
    ]


class LlamaServer:
    def __init__(self, cfg: ServerConfig, log_path: Path):
        self.cfg = cfg
        self.log_path = log_path
        self.port = free_port()
        self._proc: subprocess.Popen[bytes] | None = None
        self.load_seconds: float | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def pid(self) -> int:
        if self._proc is None:
            raise ServerError("server not started")
        return self._proc.pid

    def start(self) -> "LlamaServer":
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        with self.log_path.open("wb") as log:
            self._proc = subprocess.Popen(build_args(self.cfg, self.port), stdout=log, stderr=subprocess.STDOUT)
        self._wait_ready()
        self.load_seconds = time.perf_counter() - started
        return self

    def _wait_ready(self) -> None:
        deadline = time.monotonic() + READY_TIMEOUT_S
        while time.monotonic() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                raise ServerError(f"llama-server exited with {self._proc.returncode}; see {self.log_path}")
            try:
                if httpx.get(f"{self.base_url}/health", timeout=2.0).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(POLL_S)
        self.stop()
        raise ServerError(f"llama-server not ready after {READY_TIMEOUT_S:.0f}s; see {self.log_path}")

    def rss_gb(self) -> float:
        """Resident memory of the server process. On GPU this excludes VRAM."""
        try:
            return psutil.Process(self.pid).memory_info().rss / 1024**3
        except psutil.Error:
            return 0.0

    def stop(self) -> None:
        if self._proc is None:
            return
        self._proc.terminate()
        try:
            self._proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait(timeout=10)
        self._proc = None

    def __enter__(self) -> "LlamaServer":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()
