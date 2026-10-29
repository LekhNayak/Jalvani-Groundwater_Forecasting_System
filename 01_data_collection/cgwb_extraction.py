"""
Extracts district-level groundwater extraction, recharge, and aquifer stress
category from CGWB "Dynamic Ground Water Resources" PDF reports for Maharashtra.

These are the variables the TFT model is currently missing — they capture
*human water demand*, which dominates GWL variation in agricultural districts
like Jalgaon, Nashik, Aurangabad.

────────────────────────────────────────────────────────────────────────────
WHERE TO DOWNLOAD THE PDFS
────────────────────────────────────────────────────────────────────────────
URLs change yearly. Start at the official landing page:
    https://cgwb.gov.in/en/dynamic-ground-water-resources

Look for the Maharashtra state-level GEC (Groundwater Estimation Committee)
reports. Approximate naming/URL patterns:
    https://cgwb.gov.in/cgwbpnm/public/uploads/documents/{year}/{...}MAHARASHTRA{...}.pdf

You typically need:
    Dynamic Ground Water Resources of Maharashtra - 2023
    Dynamic Ground Water Resources of Maharashtra - 2024
    Dynamic Ground Water Resources of Maharashtra - 2025  (often a year late)

Download them manually and save to:
    data/raw/cgwb_pdfs/maharashtra_2023.pdf
    data/raw/cgwb_pdfs/maharashtra_2024.pdf
    data/raw/cgwb_pdfs/maharashtra_2025.pdf

If 2025 isn't published yet, that's expected — re-use 2024 values for the
2025 test period (standard practice in groundwater modelling).

────────────────────────────────────────────────────────────────────────────
WHAT THIS SCRIPT EXTRACTS PER DISTRICT, PER YEAR
────────────────────────────────────────────────────────────────────────────
    annual_recharge_bcm        natural input (rain + irrigation return)
    annual_extraction_bcm      what farmers / industry / cities pumped
    extractable_resource_bcm   physically pumpable upper bound
    stage_of_extraction_pct    extraction / extractable × 100
    category                   Safe / Semi-Critical / Critical / Over-Exploited

Output: datasets/cgwb_extraction_maharashtra.csv

────────────────────────────────────────────────────────────────────────────
HOW THE PARSER WORKS
────────────────────────────────────────────────────────────────────────────
1. Tries pdfplumber's table extractor on each page
2. Finds rows whose first cell matches a Maharashtra district name
3. Pulls numeric columns by header keyword matching
4. Falls back to a raw-text regex parse if table extraction fails

You'll see a quality report at the end. Districts that failed parsing
need to be filled by hand using the auto-generated template:
    datasets/cgwb_extraction_TEMPLATE.csv
"""

import sys
import re
import pandas as pd
from pathlib import Path
import os as _os
_os.chdir(_os.path.dirname(_os.path.abspath(__file__)))  # relative paths below are from this script's folder

try:
    import pdfplumber
except ImportError:
    print("ERROR: pip install pdfplumber")
    sys.exit(1)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PDF_DIR = Path("../data/raw/cgwb_pdfs")
OUT     = Path("../data/raw/cgwb_extraction_maharashtra.csv")
TPL     = Path("../data/raw/cgwb_extraction_TEMPLATE.csv")
PDF_DIR.mkdir(exist_ok=True)
OUT.parent.mkdir(exist_ok=True)

# Canonical district list — matches your master_dataset.csv
DISTRICTS = [
    "Ahmednagar", "Akola", "Amravati", "Aurangabad", "Beed",
    "Bhandara", "Buldhana", "Chandrapur", "Dhule", "Gadchiroli",
    "Gondia", "Hingoli", "Jalgaon", "Jalna", "Kolhapur",
    "Latur", "Mumbai City", "Mumbai Suburban", "Nagpur", "Nanded",
    "Nandurbar", "Nashik", "Osmanabad", "Palghar", "Parbhani",
    "Pune", "Raigad", "Ratnagiri", "Sangli", "Satara",
    "Sindhudurg", "Solapur", "Thane", "Wardha", "Washim", "Yavatmal",
]

# Maharashtra renamed two districts officially in 2023 — CGWB reports use new
# names from 2023 onwards. Map them back to our canonical list.
NAME_ALIASES = {
    "chhatrapati sambhajinagar": "Aurangabad",
    "sambhajinagar":             "Aurangabad",
    "dharashiv":                 "Osmanabad",
    "ahilyanagar":                "Ahmednagar",
    "ahmadnagar":                 "Ahmednagar",
    "buldana":                    "Buldhana",
    "raigarh":                    "Raigad",
    "gondiya":                    "Gondia",
    "bid":                        "Beed",
    "mumbai (suburban)":          "Mumbai Suburban",
}

CATEGORY_KEYWORDS = {
    "safe":           "Safe",
    "semi-critical":  "Semi-Critical",
    "semi critical":  "Semi-Critical",
    "critical":       "Critical",
    "over-exploited": "Over-Exploited",
    "over exploited": "Over-Exploited",
    "saline":         "Saline",
}

# Header keywords for column detection (case-insensitive substring match)
HEADER_PATTERNS = {
    "annual_recharge_bcm":      ["total annual ground water recharge", "annual recharge", "total recharge"],
    "extractable_resource_bcm": ["annual extractable", "extractable ground water", "net annual"],
    "annual_extraction_bcm":    ["current annual ground water extraction", "annual extraction", "total extraction"],
    "stage_of_extraction_pct":  ["stage of ground water extraction", "stage of extraction", "stage of development"],
}


def normalise_district(name: str) -> str | None:
    """Map any district variant in a PDF to our canonical name."""
    if not name:
        return None
    raw = re.sub(r"[^a-z\s]", "", name.lower()).strip()
    if not raw:
        return None
    if raw in NAME_ALIASES:
        return NAME_ALIASES[raw]
    for canonical in DISTRICTS:
        if canonical.lower() == raw:
            return canonical
        if canonical.lower() in raw or raw in canonical.lower():
            return canonical
    return None


def parse_number(s: str):
    """Pull a number out of a cell, ignoring units/commas/asterisks."""
    if s is None:
        return None
    s = str(s).replace(",", "").replace("*", "").replace("%", "").strip()
    m = re.search(r"-?\d+\.?\d*", s)
    return float(m.group()) if m else None


def detect_category(row_text: str) -> str | None:
    low = row_text.lower()
    for key, label in CATEGORY_KEYWORDS.items():
        if key in low:
            return label
    return None


def map_columns(header_row):
    """Given a header row, return {our_col_name: column_index}."""
    mapping = {}
    header_text = [str(c).lower() if c else "" for c in header_row]
    for our_col, patterns in HEADER_PATTERNS.items():
        for i, cell in enumerate(header_text):
            if any(p in cell for p in patterns):
                mapping[our_col] = i
                break
    return mapping


def parse_pdf_tables(pdf_path: Path, year: int) -> list[dict]:
    """Try pdfplumber's table extractor first — works for well-structured PDFs."""
    rows_out = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            tables = page.extract_tables() or []
            for table in tables:
                if not table or len(table) < 2:
                    continue

                # Find a header row (one of the first 3 rows) that has any of our keywords
                col_map = {}
                header_idx = -1
                for r in range(min(3, len(table))):
                    candidate = map_columns(table[r])
                    if len(candidate) >= 2:   # need at least 2 of our columns
                        col_map  = candidate
                        header_idx = r
                        break
                if header_idx < 0:
                    continue

                # Walk data rows
                for r in range(header_idx + 1, len(table)):
                    row = table[r]
                    if not row or not row[0]:
                        continue
                    dist = normalise_district(row[0])
                    if dist is None:
                        continue
                    record = {"district": dist, "year": year, "source_page": page_num}
                    row_text = " ".join(str(c) for c in row if c)
                    for col, idx in col_map.items():
                        if idx < len(row):
                            record[col] = parse_number(row[idx])
                    cat = detect_category(row_text)
                    if cat:
                        record["category"] = cat
                    # Only keep if we got at least one numeric field
                    if any(record.get(c) is not None for c in HEADER_PATTERNS):
                        rows_out.append(record)
    return rows_out


def parse_pdf_text_fallback(pdf_path: Path, year: int) -> list[dict]:
    """If table extraction missed a district, scan raw text for lines that
    start with a district name and contain numbers."""
    rows_out = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            for line in text.split("\n"):
                # Try to match: "<District Name>  num  num  num  num ... <Category>"
                parts = re.split(r"\s{2,}|\t", line.strip())
                if len(parts) < 4:
                    continue
                dist = normalise_district(parts[0])
                if dist is None:
                    continue
                nums = [parse_number(p) for p in parts[1:] if parse_number(p) is not None]
                if len(nums) < 3:
                    continue
                cat = detect_category(line)
                rows_out.append({
                    "district": dist,
                    "year":     year,
                    "_text_nums": nums,
                    "category":  cat,
                })
    return rows_out


def fill_template(parsed: pd.DataFrame, template_path: Path):
    """Build a full (district × year) template CSV pre-filled with whatever
    we parsed. Empty cells = needs manual entry from the PDF."""
    full = pd.DataFrame(
        [(d, y) for d in DISTRICTS for y in [2023, 2024, 2025]],
        columns=["district", "year"],
    )
    if not parsed.empty:
        full = full.merge(parsed, on=["district", "year"], how="left")
    for col in ["annual_recharge_bcm", "extractable_resource_bcm",
                "annual_extraction_bcm", "stage_of_extraction_pct", "category"]:
        if col not in full.columns:
            full[col] = None
    full.to_csv(template_path, index=False)


# ─── MAIN ─────────────────────────────────────────────────────────────────────
pdfs = sorted(PDF_DIR.glob("maharashtra_*.pdf"))
if not pdfs:
    print(f"No PDFs found in {PDF_DIR}/")
    print("Download CGWB Maharashtra reports first — see header of this script.")
    print(f"Generating empty template at {TPL} for manual fill ...")
    fill_template(pd.DataFrame(), TPL)
    print(f"  Edit {TPL} by hand from the PDFs and save as {OUT}")
    sys.exit(0)

all_rows = []
for pdf_path in pdfs:
    m = re.search(r"(\d{4})", pdf_path.name)
    if not m:
        print(f"  Skipping {pdf_path.name} — can't extract year from filename")
        continue
    year = int(m.group(1))
    print(f"\nParsing {pdf_path.name}  (year={year})")

    rows = parse_pdf_tables(pdf_path, year)
    found = {r["district"] for r in rows}
    print(f"  Table parser: {len(found):>2} / {len(DISTRICTS)} districts found")

    if len(found) < len(DISTRICTS):
        # Try text-based fallback for missing districts
        fb = parse_pdf_text_fallback(pdf_path, year)
        added = 0
        for r in fb:
            if r["district"] in found:
                continue
            # Heuristic: in the standard CGWB table, the columns appear in this order:
            #   recharge, extractable, extraction, stage_of_extraction_pct
            nums = r.get("_text_nums", [])
            if len(nums) >= 4:
                rows.append({
                    "district": r["district"],
                    "year": year,
                    "annual_recharge_bcm":      nums[0],
                    "extractable_resource_bcm": nums[1],
                    "annual_extraction_bcm":    nums[2],
                    "stage_of_extraction_pct":  nums[3],
                    "category": r.get("category"),
                    "source_page": "text-fallback",
                })
                found.add(r["district"])
                added += 1
        if added:
            print(f"  Text fallback: recovered {added} more districts")

    all_rows.extend(rows)

if not all_rows:
    print("\nNothing parsed. Generating blank template for manual fill.")
    fill_template(pd.DataFrame(), TPL)
    sys.exit(0)

# Build dataframe; if same (district, year) appears twice, keep the most complete one
df = pd.DataFrame(all_rows)
numeric_cols = ["annual_recharge_bcm", "extractable_resource_bcm",
                "annual_extraction_bcm", "stage_of_extraction_pct"]
df["_completeness"] = df[numeric_cols].notna().sum(axis=1)
df = (df.sort_values(["district", "year", "_completeness"], ascending=[True, True, False])
        .drop_duplicates(subset=["district", "year"], keep="first")
        .drop(columns=["_completeness"]))

# Save both the parsed result and a template covering every (district, year)
df.to_csv(OUT, index=False)
fill_template(df, TPL)

# ─── QUALITY REPORT ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PARSE QUALITY REPORT")
print("=" * 60)
expected = {(d, y) for d in DISTRICTS for y in [2023, 2024, 2025]}
have     = set(zip(df["district"], df["year"]))
missing  = expected - have
print(f"Rows parsed       : {len(df):>3} / {len(expected)}")
print(f"District-years OK : {len(have):>3}")
print(f"Missing           : {len(missing):>3}")
if missing:
    print("\nMissing (need manual entry in template):")
    by_year = {}
    for d, y in sorted(missing):
        by_year.setdefault(y, []).append(d)
    for y, ds in sorted(by_year.items()):
        print(f"  {y}: {ds}")

print(f"\nSaved -> {OUT}")
print(f"Template -> {TPL}  (edit blanks manually, then overwrite {OUT.name})")
print("\nNext step: python merge_cgwb_into_master.py")
