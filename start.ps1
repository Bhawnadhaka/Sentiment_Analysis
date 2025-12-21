#!/usr/bin/env powershell
# Complete System Startup Script

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "    Starting MLOps Sentiment Analysis System" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Check if Docker is running
Write-Host "[1/5] Checking Docker..." -ForegroundColor Yellow
try {
    docker ps | Out-Null
    Write-Host "      ✓ Docker is running" -ForegroundColor Green
} catch {
    Write-Host "      ✗ Docker is not running. Please start Docker Desktop" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "[2/5] Starting infrastructure services..." -ForegroundColor Yellow
Set-Location docker
docker-compose up -d zookeeper kafka postgres redis prometheus grafana
Start-Sleep -Seconds 5
Write-Host "      ✓ Infrastructure started" -ForegroundColor Green

Write-Host ""
Write-Host "[3/5] Checking services health..." -ForegroundColor Yellow
$services = @(
    @{Name="Zookeeper"; Port=2181},
    @{Name="Kafka"; Port=9092},
    @{Name="Redis"; Port=6379},
    @{Name="Prometheus"; Port=9090},
    @{Name="Grafana"; Port=3000}
)

foreach ($svc in $services) {
    $result = Test-NetConnection -ComputerName localhost -Port $svc.Port -WarningAction SilentlyContinue
    if ($result.TcpTestSucceeded) {
        Write-Host "      ✓ $($svc.Name) is ready" -ForegroundColor Green
    } else {
        Write-Host "      ⚠ $($svc.Name) is starting..." -ForegroundColor Yellow
    }
}

Set-Location ..
Write-Host ""
Write-Host "[4/5] Starting API server..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "& {Write-Host 'FastAPI Server' -ForegroundColor Cyan; python -m uvicorn src.api.main:app --reload --port 8000}"
Start-Sleep -Seconds 3
Write-Host "      ✓ API server starting..." -ForegroundColor Green

Write-Host ""
Write-Host "[5/5] Starting Streamlit Dashboard..." -ForegroundColor Yellow
Start-Sleep -Seconds 2
Start-Process powershell -ArgumentList "-NoExit", "-Command", "& {Write-Host 'Streamlit Dashboard' -ForegroundColor Magenta; streamlit run app.py}"
Write-Host "      ✓ Dashboard starting..." -ForegroundColor Green

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "    ✓ System Started Successfully!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "📊 Access your dashboards:" -ForegroundColor Yellow
Write-Host "   • Streamlit UI:     http://localhost:8501" -ForegroundColor White
Write-Host "   • FastAPI Docs:     http://localhost:8000/docs" -ForegroundColor White
Write-Host "   • Prometheus:       http://localhost:9090" -ForegroundColor White
Write-Host "   • Grafana:          http://localhost:3000 (admin/admin)" -ForegroundColor White
Write-Host ""
Write-Host "🚀 Next steps:" -ForegroundColor Yellow
Write-Host "   1. Open Streamlit UI in your browser" -ForegroundColor White
Write-Host "   2. Try live predictions" -ForegroundColor White
Write-Host "   3. Start streaming: python run_pipeline.py producer" -ForegroundColor White
Write-Host ""
Write-Host "Press Ctrl+C to stop all services" -ForegroundColor Gray
Write-Host ""

# Keep script running
while ($true) {
    Start-Sleep -Seconds 10
}
