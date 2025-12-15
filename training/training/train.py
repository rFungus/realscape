import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

# Добавляем корень проекта в sys.path, чтобы импортировать models при запуске как скрипт
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models import DiffusionModel


class TerrainNPYDataset(Dataset):
    # Датасет, читающий heightmap из .npy файлов

    def __init__(self, files: list[Path], target_size: int = 64):
        self.files = files
        self.target_size = target_size

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        arr = np.load(path).astype(np.float32)
        if arr.ndim == 2:
            arr = arr[None, ...]  # (1, H, W)
        elif arr.ndim == 3:
            # берём первый канал, если (H, W, C)
            if arr.shape[0] != 1 and arr.shape[-1] == 1:
                arr = np.transpose(arr, (2, 0, 1))
            elif arr.shape[0] != 1:
                arr = arr[:1, ...]
        tensor = torch.from_numpy(arr)
        # Приводим к target_size через bilinear
        if tensor.shape[-1] != self.target_size or tensor.shape[-2] != self.target_size:
            tensor = F.interpolate(
                tensor[None, ...],
                size=(self.target_size, self.target_size),
                mode="bilinear",
                align_corners=False,
            )[0]
        # Нормализация в [0,1]
        tensor = torch.clamp(tensor, 0.0, 1.0)
        return tensor


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train(config_path: str, device: str = "cuda"):
    cfg = load_config(config_path)

    train_cfg = cfg["training"]
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    device_cfg = cfg["device"]

    torch.manual_seed(device_cfg.get("seed", 42))

    dev = torch.device(device if torch.cuda.is_available() and device_cfg["backend"] == "cuda" else "cpu")

    model = DiffusionModel(
        in_channels=model_cfg["in_channels"],
        out_channels=model_cfg["out_channels"],
        base_channels=model_cfg["channels"][0],
    ).to(dev)

    train_dir = Path(data_cfg.get("train_path", "data/elevation_processed"))
    train_files = sorted(train_dir.glob("*.npy"))
    if not train_files:
        raise RuntimeError(f"Не найдено .npy в {train_dir} — запустите preprocessing/data_processor.py")

    dataset = TerrainNPYDataset(train_files, target_size=data_cfg["image_size"])
    loader = DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 0),
    )

    lr = float(train_cfg["learning_rate"])
    betas = tuple(float(b) for b in train_cfg["optimizer"]["betas"])
    eps = float(train_cfg["optimizer"]["eps"])
    weight_decay = float(train_cfg["optimizer"]["weight_decay"])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
    )

    num_epochs = train_cfg["num_epochs"]
    num_timesteps = train_cfg["num_timesteps"]
    checkpoint_dir = Path(train_cfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        model.train()
        for step, x0 in enumerate(loader, start=1):
            x0 = x0.to(dev)
            t = torch.randint(0, num_timesteps, (x0.size(0),), device=dev, dtype=torch.long)

            noise = torch.randn_like(x0)
            x_noisy = x0 + noise

            pred_noise = model(x_noisy, t.float())
            loss = torch.nn.functional.mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % train_cfg["save_interval"] == 0:
            ckpt_path = checkpoint_dir / f"model_epoch_{epoch}.pt"
            torch.save({"epoch": epoch, "model_state": model.state_dict()}, ckpt_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RealScape training loop")
    parser.add_argument(
        "--config",
        type=str,
        default="training/config.yaml",
        help="Путь к YAML конфигурации",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cuda или cpu",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.config, device=args.device)


