<#
.SYNOPSIS
    Build and run the full Clash Royale RL Coach app in Docker.

.DESCRIPTION
    Builds the multi-stage image (React/PixiJS frontend + FastAPI backend) and runs
    it as a single container. The app is served at http://localhost:<Port>
    (frontend, REST API and WebSocket on the same port).

.EXAMPLE
    .\scripts\docker.ps1                # build (if needed) and run
    .\scripts\docker.ps1 -Rebuild       # force a fresh image build
    .\scripts\docker.ps1 -Stop          # stop and remove the container
#>
param(
    [switch]$Rebuild,
    [switch]$Stop,
    [int]$Port = 8000,
    [string]$Image = "clash-royale-rl-coach",
    [string]$Container = "cr-rl-coach"
)

# Native docker commands write progress to stderr; failures are caught via $LASTEXITCODE.
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "Docker was not found. Install Docker Desktop: https://www.docker.com/products/docker-desktop/"
}

$existing = docker ps -aq -f "name=^/$Container$"
if ($existing) {
    docker rm -f $Container | Out-Null
}

if ($Stop) {
    Write-Host "Stopped container '$Container'."
    return
}

$imageExists = [bool](docker images -q $Image)
if ($Rebuild -or -not $imageExists) {
    Write-Host "Building image '$Image'..."
    docker build -t $Image $root
    if ($LASTEXITCODE -ne 0) { throw "docker build failed" }
}

Write-Host "Starting container '$Container' on port $Port..."
docker run --rm -d --name $Container -e PORT=8000 -p "${Port}:8000" $Image | Out-Null

Write-Host ""
Write-Host "App:     http://localhost:$Port"
Write-Host "Health:  http://localhost:$Port/api/health"
Write-Host "Logs:    docker logs -f $Container"
Write-Host "Stop:    .\scripts\docker.ps1 -Stop"
