import torch
import torch.nn as nn


class TimeEmbedding(nn.Module):
    # Простое синусоидальное time-embedding

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.proj = nn.Linear(dim, dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        device = t.device
        emb = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.proj(emb)


class SelfAttention(nn.Module):
    # Одноголовое self-attention по пространству

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        h_in = self.norm(x)
        q = self.q(h_in).reshape(b, c, h * w).transpose(1, 2)
        k = self.k(h_in).reshape(b, c, h * w)
        v = self.v(h_in).reshape(b, c, h * w).transpose(1, 2)
        attn = torch.softmax(q @ k / (c ** 0.5), dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).reshape(b, c, h, w)
        return x + self.proj(out)


class ResBlock(nn.Module):
    # Простой residual-блок для UNet

    def __init__(self, in_ch: int, out_ch: int, time_dim: int | None = None):
        super().__init__()
        self.time_dim = time_dim
        self.norm1 = nn.GroupNorm(1, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(1, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.time_mlp = None
        if time_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_dim, out_ch),
            )

        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor | None = None) -> torch.Tensor:
        h = self.conv1(torch.relu(self.norm1(x)))
        if self.time_mlp is not None and t_emb is not None:
            time = self.time_mlp(t_emb)[:, :, None, None]
            h = h + time
        h = self.conv2(torch.relu(self.norm2(h)))
        return h + self.skip(x)


class UNetDiffusion(nn.Module):
    # Упрощённый UNet для диффузионной модели высот

    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 64):
        super().__init__()
        time_dim = base_channels * 4
        self.time_embedding = TimeEmbedding(time_dim)

        self.in_block = ResBlock(in_channels, base_channels, time_dim)
        self.down1 = ResBlock(base_channels, base_channels * 2, time_dim)
        self.down2 = ResBlock(base_channels * 2, base_channels * 4, time_dim)

        self.mid = ResBlock(base_channels * 4, base_channels * 4, time_dim)

        # При подъёме используем скипы с предыдущих уровней по пространству
        self.up2 = ResBlock(base_channels * 4 + base_channels * 2, base_channels * 2, time_dim)
        self.up1 = ResBlock(base_channels * 2 + base_channels, base_channels, time_dim)

        self.out_block = nn.Sequential(
            nn.GroupNorm(1, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, out_channels, 3, padding=1),
        )

        self.downsample = nn.AvgPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embedding(t)

        x1 = self.in_block(x, t_emb)
        x2 = self.down1(self.downsample(x1), t_emb)
        x3 = self.down2(self.downsample(x2), t_emb)

        mid = self.mid(x3, t_emb)

        u2 = self.upsample(mid)
        # Конкат с x2 (совпадает по пространственному размеру)
        u2 = self.up2(torch.cat([u2, x2], dim=1), t_emb)

        u1 = self.upsample(u2)
        # Конкат с x1
        u1 = self.up1(torch.cat([u1, x1], dim=1), t_emb)

        out = self.out_block(u1)
        return out


class DiffusionModel(nn.Module):
    # Обёртка над UNet для шага диффузии

    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 64):
        super().__init__()
        self.unet = UNetDiffusion(in_channels, out_channels, base_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.unet(x, t)


