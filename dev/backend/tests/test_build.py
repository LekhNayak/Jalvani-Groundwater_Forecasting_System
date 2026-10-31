"""The build reads the repo's data but never changes it, and writes only inside dev/backend/data."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
WATCH = [ROOT / "data", ROOT / "04_results", ROOT / "03_models"]


def snapshot():
    out = {}
    for base in WATCH:
        for p in base.rglob("*"):
            if p.is_file():
                s = p.stat()
                out[str(p)] = (s.st_size, s.st_mtime_ns)
    return out


def test_build_never_touches_project_data():
    before = snapshot()
    res = subprocess.run([sys.executable, str(ROOT / "dev" / "pipeline" / "build_dashboard_data.py")],
                         capture_output=True, text=True, cwd=ROOT)
    assert res.returncode == 0, res.stderr[-2000:]
    after = snapshot()
    assert before == after, "the build changed files under data/, 04_results/ or 03_models/"
    out = ROOT / "dev" / "backend" / "data"
    assert (out / "manifest.json").exists() and len(list((out / "history").glob("*.json"))) == 36
