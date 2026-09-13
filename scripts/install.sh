#!/bin/sh
# Margin installer for macOS and Linux.
#   curl -LsSf https://raw.githubusercontent.com/Kunsh162007/margin/main/scripts/install.sh | sh
#
# Installs uv if it is missing, installs Margin as an isolated tool with its own
# Python, then downloads the runtime and the model that suit this machine.
set -eu
REPO="git+https://github.com/Kunsh162007/margin"

if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv (Python package manager)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

echo "Installing Margin..."
uv tool install --python 3.12 --force "$REPO"
uv tool update-shell >/dev/null 2>&1 || true

if ! margin setup; then
    echo "margin setup failed; run it again to resume the download." >&2
    exit 1
fi
echo ""
echo "Margin is installed. Start it with:  margin"
