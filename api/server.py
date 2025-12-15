from typing import Any, Dict

from fastapi import FastAPI
from pydantic import BaseModel

from inference import TerrainGenerator


class GenerateRequest(BaseModel):
    seed: int = 42
    resolution: int = 64
    world_scale: float = 100.0
    temperature: float = 0.7


app = FastAPI(title="RealScape API")
generator = TerrainGenerator()


@app.post("/api/generate")
def generate(req: GenerateRequest) -> Dict[str, Any]:
    heightmap = generator.generate(image_size=req.resolution, seed=req.seed)
    data = heightmap.squeeze(0).squeeze(0).numpy()
    return {
        "heightmap": data.tolist(),
        "metadata": {
            "resolution": req.resolution,
            "world_scale": req.world_scale,
        },
    }


