# archive/

Nothing in this folder is part of the normal pipeline. Two optional tools read one backup each (see the table). The folder exists so that the 2026-10-29 restructure deleted nothing.

| Item | What it is |
|---|---|
| `MANIFEST.csv` | one row per archived file: size, md5, original path, why it is here, and the live file with identical content (if any) |
| `MOVES_2026-10-29.csv` | every file moved in the restructure: old path → new path |
| `MANIFEST_before_md5.txt` | md5 of every project file before the restructure; all of them are still present |
| `backups/2026-04_pre_extend/` | raw files and progress files saved before the April 2026 extension (originally `*.bak`) |
| `backups/2026-10_pre_extend/` | raw, clean and master files saved before the October 2026 extension. `01_data_collection/chirps_extend_assemble.py` reads one file from here |
| `backups/2026-10_v3_target/` | the v3-target groundwater series, kept for `02_preprocessing/audit_groundwater_target.py` |
| `duplicates/` | byte-identical extra copies (suffix `__dup_<origin>`); the live copy is named in `MANIFEST.csv` |
| `notebook_checkpoints/` | Jupyter autosave copies of notebooks |

Folder layout and naming rules: `docs/STRUCTURE.md`.
