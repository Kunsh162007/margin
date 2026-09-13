"""The model for a whole session: located once, started on first use, stopped at exit.

The CLI starts a server per command. The study interface keeps one alive, because
loading the model is the slowest step a student waits for and should happen once.
"""

from __future__ import annotations

import threading
from pathlib import Path

from margin import config, hardware
from margin.runtime import models
from margin.runtime.binaries import LLAMA_CPP_BUILD, find_server
from margin.runtime.client import LlamaClient
from margin.runtime.server import LlamaServer, ServerConfig

CONTEXT_TOKENS = 16384


class NotInstalled(RuntimeError):
    pass


def server_config(backend: str | None = None, model: str | None = None) -> tuple[models.ModelSpec, str, ServerConfig]:
    hw = hardware.detect()
    chosen = hardware.choose_backend(hw, backend)
    spec = models.get(model) if model else models.recommended(chosen, hw.ram_gb, hw.gpu.vram_gb if hw.gpu else None)
    paths = config.paths().ensure()
    server_dir = paths.bin_dir / f"{LLAMA_CPP_BUILD}-{chosen}"
    exe = find_server(server_dir) if server_dir.exists() else None
    if exe is None or not spec.path(paths.models_dir).exists():
        raise NotInstalled("The model is not installed yet. Run: margin setup")
    cfg = ServerConfig(exe=exe, model=spec.path(paths.models_dir), gpu=chosen != "cpu", threads=hardware.default_threads(hw),
                       ctx=CONTEXT_TOKENS, chat_template_file=spec.chat_template_path())
    return spec, chosen, cfg


class ModelSession:
    def __init__(self, spec: models.ModelSpec, backend: str, cfg: ServerConfig, log_path: Path):
        self.spec = spec
        self.backend = backend
        self._cfg = cfg
        self._log_path = log_path
        self._lock = threading.Lock()
        self._server: LlamaServer | None = None
        self._client: LlamaClient | None = None

    @property
    def started(self) -> bool:
        return self._client is not None

    def client(self) -> LlamaClient:
        with self._lock:
            if self._client is None:
                self._server = LlamaServer(self._cfg, self._log_path).start()
                self._client = LlamaClient(self._server.base_url, tools_template_kwarg=self.spec.tools_template_kwarg)
            return self._client

    def close(self) -> None:
        with self._lock:
            if self._client is not None:
                self._client.close()
                self._client = None
            if self._server is not None:
                self._server.stop()
                self._server = None
