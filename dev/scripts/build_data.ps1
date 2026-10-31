# Rebuild the dashboard data (dev/backend/data) from data/ and 04_results/ (read-only; nothing outside dev/ changes).
# Run after new data or a new model run (e.g. the v4.3 results unzipped into 04_results\tft_v4_3).
$dev = Split-Path -Parent $PSScriptRoot
python "$dev\pipeline\build_dashboard_data.py"
