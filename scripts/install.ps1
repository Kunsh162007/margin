# Margin installer for Windows.
#   irm https://raw.githubusercontent.com/Kunsh162007/margin/main/scripts/install.ps1 | iex
#
# Installs uv if it is missing, installs Margin as an isolated tool with its own
# Python, then downloads the runtime and the model that suit this machine.
$ErrorActionPreference = 'Stop'
$Repo = 'git+https://github.com/Kunsh162007/margin'

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host 'Installing uv (Python package manager)...'
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}

Write-Host 'Installing Margin...'
uv tool install --python 3.12 --force $Repo
uv tool update-shell | Out-Null
$env:Path = "$env:USERPROFILE\.local\bin;$env:Path"

margin setup
if ($LASTEXITCODE -ne 0) { throw 'margin setup failed; run it again to resume the download.' }
Write-Host ''
Write-Host 'Margin is installed. Start it with:  margin'
