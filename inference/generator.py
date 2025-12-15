from pathlib import Path
from typing import Optional

import torch

from models import DiffusionModel


class TerrainGenerator:
    # Простой генератор высотных карт

    def __init__(self, checkpoint_path: Optional[str] = None, device: str = "cuda"):
        dev = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
        self.device = dev
        self.model = DiffusionModel()
        self.model.to(self.device)
        self.model.eval()

        if checkpoint_path is not None and Path(checkpoint_path).is_file():
            state = torch.load(checkpoint_path, map_location=self.device)
            if "model_state" in state:
                self.model.load_state_dict(state["model_state"], strict=False)
            else:
                self.model.load_state_dict(state, strict=False)

    @torch.no_grad()
    def generate(self, image_size: int = 64, num_steps: int = 10, seed: int = 42) -> torch.Tensor:
        torch.manual_seed(seed)
        x = torch.randn(1, 1, image_size, image_size, device=self.device)
        timesteps = torch.linspace(1.0, 0.0, num_steps, device=self.device)
        for t in timesteps:
            t_batch = torch.full((1,), t, device=self.device)
            noise_pred = self.model(x, t_batch)
            x = x - noise_pred / max(num_steps, 1)
        return x.clamp(0.0, 1.0).cpu()


