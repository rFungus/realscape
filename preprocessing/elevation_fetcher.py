from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import requests


class ElevationFetcher:
    # Получение реальных высот по координатам через OpenTopoData (SRTM)

    def __init__(self, base_url: str = "https://api.opentopodata.org/v1/srtm90m"):
        self.base_url = base_url

    def fetch(self, coords: Iterable[Tuple[float, float]]) -> List[float]:
        coords = list(coords)
        if not coords:
            return []

        # OpenTopoData принимает до ~100 точек за запрос; здесь отправляем все сразу
        locations = "|".join(f"{lat},{lon}" for lat, lon in coords)

        try:
            resp = requests.get(self.base_url, params={"locations": locations}, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            heights: List[float] = []
            for item in results:
                elev = item.get("elevation")
                if elev is None:
                    elev = 0.0
                heights.append(float(elev))
            # Если по каким-то причинам длина не совпала, подрежем/дополняем нулями
            if len(heights) < len(coords):
                heights.extend([0.0] * (len(coords) - len(heights)))
            return heights[: len(coords)]
        except Exception:
            # Фолбэк: при ошибке API вернём плоский рельеф
            return [0.0 for _ in coords]


class HeightmapGenerator:
    # Создание простой heightmap из высот

    def __init__(self, size: int = 64):
        self.size = size

    def from_heights(self, heights: List[float]) -> np.ndarray:
        if not heights:
            return np.zeros((self.size, self.size), dtype=np.float32)
        base = np.linspace(min(heights), max(heights), self.size, dtype=np.float32)
        grid = np.tile(base[None, :], (self.size, 1))
        # Нормализация с использованием np.ptp для совместимости с NumPy 2.x
        ptp = float(np.ptp(grid))
        grid = (grid - grid.min()) / max(ptp, 1e-6)
        return grid

    def save_npy(self, heightmap: np.ndarray, out_path: str | Path) -> None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, heightmap)


