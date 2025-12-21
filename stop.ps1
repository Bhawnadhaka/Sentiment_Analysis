#!/usr/bin/env powershell
# Stop All Services Script

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "    Stopping MLOps Sentiment Analysis System" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "[1/2] Stopping Docker containers..." -ForegroundColor Yellow
Set-Location docker
docker-compose down
Set-Location ..
Write-Host "      ✓ Docker containers stopped" -ForegroundColor Green

Write-Host ""
Write-Host "[2/2] Stopping Python processes..." -ForegroundColor Yellow
Get-Process | Where-Object {$_.ProcessName -like "*python*" -or $_.ProcessName -like "*uvicorn*" -or $_.ProcessName -like "*streamlit*"} | Stop-Process -Force -ErrorAction SilentlyContinue
Write-Host "      ✓ Python processes stopped" -ForegroundColor Green

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "    ✓ System Stopped Successfully!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
