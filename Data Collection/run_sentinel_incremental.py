"""Fill Sentinel months Jul-Sep 2026 (Oct is a partial month, excluded)."""
from datetime import date
import sentinel_indices as s
s.END_DATE = date(2026, 9, 30)
s.main()
