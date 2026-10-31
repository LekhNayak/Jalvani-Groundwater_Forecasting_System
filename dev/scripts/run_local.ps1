# Start JalVaani locally.
#   default : backend on http://localhost:8000 + frontend dev server on http://localhost:5173 (open this one)
#   -Single : build the frontend once and serve everything from http://localhost:8000
param([switch]$Single)
$dev = Split-Path -Parent $PSScriptRoot
if ($Single) {
    Push-Location "$dev\frontend"; npm run build; Pop-Location
    $env:SERVE_FRONTEND = "1"
    Push-Location "$dev\backend"
    Write-Host "JalVaani: http://localhost:8000   (Ctrl+C to stop)"
    python -m uvicorn app.main:app --port 8000
    Pop-Location
} else {
    $api = Start-Process -PassThru -NoNewWindow python -ArgumentList "-m", "uvicorn", "app.main:app", "--reload", "--port", "8000" -WorkingDirectory "$dev\backend"
    try {
        Push-Location "$dev\frontend"
        Write-Host "JalVaani: http://localhost:5173   (Ctrl+C to stop; the API runs on :8000)"
        npm run dev
    } finally {
        Pop-Location
        Stop-Process -Id $api.Id -ErrorAction SilentlyContinue
    }
}
