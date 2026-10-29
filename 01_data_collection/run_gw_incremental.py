"""Incremental CGWB groundwater pull: only the window after the last collected date.
Reuses groundwater_data.py; own progress file so the original run history is untouched."""
import groundwater_data as g
import os as _os
_os.chdir(_os.path.dirname(_os.path.abspath(__file__)))  # relative paths below are from this script's folder
g.START_DATE = "2026-06-01"
g.END_DATE   = "2026-10-01"
g.PROGRESS_F = "progress/progress_2026_10_extend.txt"
g.main()
