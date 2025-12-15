from pathlib import Path
from typing import Iterable, List, Tuple

import xml.etree.ElementTree as ET


def extract_coordinates(osm_path: str | Path) -> List[Tuple[float, float]]:
    # Реальный разбор .osm: координаты нод, входящих в линии/дороги
    osm_path = Path(osm_path)
    if not osm_path.is_file():
        return []

    nodes: dict[str, Tuple[float, float]] = {}
    used_node_ids: set[str] = set()

    # Потоковый парсинг, чтобы не держать весь файл в памяти
    for event, elem in ET.iterparse(osm_path, events=("start", "end")):
        if event == "start" and elem.tag == "node":
            node_id = elem.get("id")
            lat = elem.get("lat")
            lon = elem.get("lon")
            if node_id is not None and lat is not None and lon is not None:
                try:
                    nodes[node_id] = (float(lat), float(lon))
                except ValueError:
                    pass

        if event == "end" and elem.tag == "way":
            has_highway = False
            for tag in elem.findall("tag"):
                k = tag.get("k")
                if k == "highway":
                    has_highway = True
                    break

            if has_highway:
                for nd in elem.findall("nd"):
                    ref = nd.get("ref")
                    if ref is not None:
                        used_node_ids.add(ref)

            elem.clear()

    coords: List[Tuple[float, float]] = []
    for node_id in used_node_ids:
        coord = nodes.get(node_id)
        if coord is not None:
            coords.append(coord)

    # Если ни одной линии/дороги не нашли, возвращаем все ноды
    if not coords:
        coords = list(nodes.values())

    return coords


def save_coordinates_json(coords: Iterable[Tuple[float, float]], out_path: str | Path) -> None:
    # Сохранение координат в JSON
    import json

    data = [{"lat": lat, "lon": lon} for lat, lon in coords]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)