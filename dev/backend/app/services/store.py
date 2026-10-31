"""Read-only access to the built dashboard data (dev/backend/data, made by dev/pipeline/build_dashboard_data.py)."""
import json
from functools import lru_cache
from pathlib import Path

from ..config import settings


class DataMissing(RuntimeError):
    pass


def _load(path: Path):
    if not path.exists():
        raise DataMissing(f"{path.name} not found - run dev/pipeline/build_dashboard_data.py first")
    return json.loads(path.read_text(encoding="utf-8"))


class Store:
    def __init__(self, data_dir: Path):
        self.dir = Path(data_dir)
        self.manifest = _load(self.dir / "manifest.json")
        self.districts = _load(self.dir / "districts.json")["districts"]
        self.by_slug = {d["slug"]: d for d in self.districts}
        self.geo = _load(self.dir / "geo_districts.json")
        self.stations = _load(self.dir / "stations.json")

    def district(self, slug: str):
        return self.by_slug.get(slug.lower())

    @lru_cache(maxsize=64)
    def history(self, slug: str):
        p = self.dir / "history" / f"{slug.lower()}.json"
        return _load(p) if p.exists() else None


@lru_cache(maxsize=1)
def get_store() -> Store:
    return Store(settings.data_dir)
