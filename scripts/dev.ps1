param(
    [int]$ApiPort = 8000,
    [int]$WebPort = 5173
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $root ".venv\Scripts\python.exe"
$python = if (Test-Path $venvPython) { $venvPython } else { "python" }

if (-not (Get-Command $python -ErrorAction SilentlyContinue)) {
    throw "Python was not found. Install Python 3.10+ or create .venv first."
}
if (-not (Get-Command "npm.cmd" -ErrorAction SilentlyContinue)) {
    throw "npm was not found. Install Node.js 20+ from https://nodejs.org/."
}

if (-not (Test-Path (Join-Path $root "web\node_modules"))) {
    Push-Location (Join-Path $root "web")
    try { npm.cmd ci } finally { Pop-Location }
}

$apiCommand = "& '$python' -m uvicorn cr_rl.server.app:app --reload --port $ApiPort"
$webCommand = "npm.cmd run dev -- --port $WebPort"

Start-Process powershell.exe -WorkingDirectory $root -ArgumentList "-NoExit", "-Command", $apiCommand
Start-Process powershell.exe -WorkingDirectory (Join-Path $root "web") -ArgumentList "-NoExit", "-Command", $webCommand

Write-Host "API:     http://localhost:$ApiPort"
Write-Host "Website: http://localhost:$WebPort"
