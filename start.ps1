# Launches the AeroOpti backend (FastAPI) and frontend (React/Vite) each in their
# own PowerShell window, waits for the frontend to come up, then opens the browser.
#
# Usage:  .\start.ps1
# Stop:   close the two opened windows (or Ctrl+C in each).

$root = $PSScriptRoot

Write-Host "Starting backend (FastAPI on :8000)..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList @(
    "-NoExit", "-Command",
    "Set-Location '$root'; python -m uvicorn backend.main:app --port 8000 --reload"
)

Write-Host "Starting frontend (Vite on :5173)..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList @(
    "-NoExit", "-Command",
    "Set-Location '$root\frontend'; npm run dev"
)

Write-Host "Waiting for the frontend to come up..." -ForegroundColor Cyan
$ready = $false
for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep -Seconds 1
    try {
        $resp = Invoke-WebRequest -Uri "http://localhost:5173" -UseBasicParsing -TimeoutSec 1
        if ($resp.StatusCode -eq 200) { $ready = $true; break }
    } catch {}
}

if ($ready) {
    Start-Process "http://localhost:5173"
} else {
    Write-Host "Frontend didn't respond in time -- check the two opened windows for errors." -ForegroundColor Yellow
}
