# One-time setup: Python packages for the backend + build script, and the frontend's npm packages.
# Run from anywhere:   powershell -ExecutionPolicy Bypass -File dev\scripts\setup.ps1
$dev = Split-Path -Parent $PSScriptRoot
python -m pip install -r "$dev\backend\requirements.txt"
Push-Location "$dev\frontend"
npm install --no-audit --no-fund
Pop-Location
if (-not (Test-Path "$dev\backend\.env")) { Copy-Item "$dev\backend\.env.example" "$dev\backend\.env" }
& "$PSScriptRoot\build_data.ps1"
Write-Host "Setup done. Start the app with: powershell -ExecutionPolicy Bypass -File dev\scripts\run_local.ps1"
