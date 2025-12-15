from pathlib import Path
from typing import Iterable
import sys

from pathlib import Path
from typing import Iterable

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from preprocessing.osm_parser import extract_coordinates
from preprocessing.elevation_fetcher import ElevationFetcher, HeightmapGenerator


def process_osm_directory(
    osm_dir: str | Path = "data/osm",
    out_dir: str | Path = "data/elevation_processed",
    size: int = 64,
) -> None:
    # Подготовка .osm файлов и создание heightmap по координатам
    osm_path = Path(osm_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    fetcher = ElevationFetcher()
    generator = HeightmapGenerator(size=size)

    osm_files: Iterable[Path] = sorted(osm_path.glob("*.osm"))
    for osm_file in osm_files:
        coords = extract_coordinates(osm_file)
        heights = fetcher.fetch(coords)
        heightmap = generator.from_heights(heights)
        target = out_path / f"{osm_file.stem}_heightmap.npy"
        generator.save_npy(heightmap, target)


if __name__ == "__main__":
    process_osm_directory()


