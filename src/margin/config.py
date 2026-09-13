"""Where Margin keeps its files.

Everything lives under one directory so that uninstalling is deleting a folder.
``MARGIN_HOME`` overrides the default, which matters on machines whose system
drive is small: model files are several gigabytes each.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import platformdirs

ENV_HOME = "MARGIN_HOME"


@dataclass(frozen=True)
class Paths:
    home: Path

    @property
    def bin_dir(self) -> Path:
        return self.home / "bin"

    @property
    def models_dir(self) -> Path:
        return self.home / "models"

    @property
    def workspaces_dir(self) -> Path:
        return self.home / "workspaces"

    @property
    def logs_dir(self) -> Path:
        return self.home / "logs"

    def ensure(self) -> "Paths":
        for d in (self.bin_dir, self.models_dir, self.workspaces_dir, self.logs_dir):
            d.mkdir(parents=True, exist_ok=True)
        return self


def paths() -> Paths:
    override = os.environ.get(ENV_HOME)
    home = Path(override) if override else platformdirs.user_data_path("margin", appauthor=False)
    return Paths(home=home.expanduser().resolve())
