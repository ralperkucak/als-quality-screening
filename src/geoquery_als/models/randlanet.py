"""RandLA-Net (Hu et al., CVPR 2020) in plain PyTorch.

The architecture follows the original description: random point sampling for
downsampling, a local spatial encoding that explicitly injects relative point
geometry, attentive pooling instead of max pooling, and dilated residual blocks
that widen the effective receptive field to compensate for the aggressive
sampling. Neighbourhood indices for every resolution level are precomputed on
CPU by the dataset, exactly as in the reference implementation.

The penultimate feature map is exposed because the acquisition functions need a
per-patch embedding.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gather_neighbours(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather per-point neighbour features.

    x   : (B, C, N, 1)
    idx : (B, M, K) indices into N
    out : (B, C, M, K)
    """
    b, c, n, _ = x.shape
    m, k = idx.shape[1], idx.shape[2]
    flat = idx.reshape(b, 1, m * k).expand(-1, c, -1)
    return torch.gather(x.squeeze(-1), 2, flat).reshape(b, c, m, k)


class SharedMLP(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, bn: bool = True,
                 activation: nn.Module | None = None) -> None:
        layers: list[nn.Module] = [nn.Conv2d(in_ch, out_ch, 1, bias=not bn)]
        if bn:
            layers.append(nn.BatchNorm2d(out_ch, eps=1e-6, momentum=0.01))
        if activation is not None:
            layers.append(activation)
        super().__init__(*layers)


class LocalSpatialEncoding(nn.Module):
    """Encode relative position of each neighbour and concatenate to features."""

    def __init__(self, out_ch: int) -> None:
        super().__init__()
        self.mlp = SharedMLP(10, out_ch, activation=nn.LeakyReLU(0.2))

    def forward(self, xyz: torch.Tensor, feat: torch.Tensor,
                neigh_idx: torch.Tensor) -> torch.Tensor:
        # xyz (B, 3, N, 1), feat (B, C, N, 1), neigh_idx (B, N, K)
        nxyz = _gather_neighbours(xyz, neigh_idx)             # (B, 3, N, K)
        cxyz = xyz.expand(-1, -1, -1, neigh_idx.shape[-1])
        rel = cxyz - nxyz
        dist = torch.sqrt(torch.sum(rel ** 2, dim=1, keepdim=True) + 1e-12)
        enc = self.mlp(torch.cat([cxyz, nxyz, rel, dist], dim=1))
        return torch.cat([enc, _gather_neighbours(feat, neigh_idx)], dim=1)


class AttentivePooling(nn.Module):
    """Learned softmax weighting over the K neighbours."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.score = nn.Conv2d(in_ch, in_ch, 1, bias=False)
        self.mlp = SharedMLP(in_ch, out_ch, activation=nn.LeakyReLU(0.2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.score(x), dim=-1)
        return self.mlp(torch.sum(x * w, dim=-1, keepdim=True))


class DilatedResidualBlock(nn.Module):
    """Two stacked local-spatial-encoding + attentive-pooling units, plus a shortcut."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        # Widths follow the reference implementation: the input projection and
        # both position encodings produce out_ch/2 channels, so each attentive
        # pooling sees out_ch channels and the second one widens to out_ch.
        d = out_ch // 2
        self.mlp_in = SharedMLP(in_ch, d, activation=nn.LeakyReLU(0.2))
        self.lse1 = LocalSpatialEncoding(d)
        self.pool1 = AttentivePooling(2 * d, d)
        self.lse2 = LocalSpatialEncoding(d)
        self.pool2 = AttentivePooling(2 * d, out_ch)
        self.mlp_out = SharedMLP(out_ch, out_ch * 2, bn=True, activation=None)
        self.shortcut = SharedMLP(in_ch, out_ch * 2, bn=True, activation=None)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, xyz: torch.Tensor, feat: torch.Tensor,
                neigh_idx: torch.Tensor) -> torch.Tensor:
        h = self.mlp_in(feat)
        h = self.pool1(self.lse1(xyz, h, neigh_idx))
        h = self.pool2(self.lse2(xyz, h, neigh_idx))
        return self.act(self.mlp_out(h) + self.shortcut(feat))


def random_sample(feat: torch.Tensor, pool_idx: torch.Tensor) -> torch.Tensor:
    """Downsample by max-pooling each new point over its K source neighbours."""
    return torch.max(_gather_neighbours(feat, pool_idx), dim=-1, keepdim=True).values


def nearest_interpolate(feat: torch.Tensor, up_idx: torch.Tensor) -> torch.Tensor:
    """Upsample by copying the feature of the nearest coarse point."""
    b, c = feat.shape[0], feat.shape[1]
    idx = up_idx.reshape(b, 1, -1).expand(-1, c, -1)
    return torch.gather(feat.squeeze(-1), 2, idx).unsqueeze(-1)


@dataclass
class RandLANetConfig:
    num_classes: int = 7
    in_features: int = 8          # extra per-point features, excluding xyz
    num_layers: int = 4
    d_out: tuple[int, ...] = (16, 64, 128, 256)
    dropout: float = 0.5


class RandLANet(nn.Module):
    def __init__(self, cfg: RandLANetConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.fc_in = SharedMLP(cfg.in_features + 3, 8, activation=nn.LeakyReLU(0.2))

        enc, dec = [], []
        c = 8
        widths = []
        for d in cfg.d_out:
            enc.append(DilatedResidualBlock(c, d))
            c = d * 2
            widths.append(c)
        self.encoder = nn.ModuleList(enc)

        self.bridge = SharedMLP(c, c, activation=nn.LeakyReLU(0.2))
        for i in range(cfg.num_layers):
            skip = widths[-i - 2] if i < cfg.num_layers - 1 else 8
            dec.append(SharedMLP(c + skip, skip, activation=nn.LeakyReLU(0.2)))
            c = skip
        self.decoder = nn.ModuleList(dec)

        self.head = nn.Sequential(
            SharedMLP(c, 64, activation=nn.LeakyReLU(0.2)),
            SharedMLP(64, 32, activation=nn.LeakyReLU(0.2)),
        )
        self.drop = nn.Dropout(cfg.dropout)
        self.classifier = nn.Conv2d(32, cfg.num_classes, 1, bias=True)

    def forward(self, batch: dict[str, torch.Tensor], return_embedding: bool = False):
        """batch keys: features (B, F, N), xyz[i], neigh_idx[i], pool_idx[i], up_idx[i]."""
        x = batch["features"].unsqueeze(-1)              # (B, F, N, 1)
        x = self.fc_in(x)

        skips = [x]
        for i, blk in enumerate(self.encoder):
            x = blk(batch["xyz"][i].unsqueeze(-1), x, batch["neigh_idx"][i])
            x = random_sample(x, batch["pool_idx"][i])
            skips.append(x)

        x = self.bridge(x)
        for i, mlp in enumerate(self.decoder):
            x = nearest_interpolate(x, batch["up_idx"][-i - 1])
            x = mlp(torch.cat([x, skips[-i - 2]], dim=1))

        emb = self.head(x)                               # (B, 32, N, 1)
        logits = self.classifier(self.drop(emb))         # (B, C, N, 1)
        if return_embedding:
            return logits.squeeze(-1), emb.squeeze(-1)
        return logits.squeeze(-1)

    def adaptation_parameters(self, mode: str):
        """Parameter groups used by the adaptation stage.

        The same choice is applied to every acquisition method so that the
        comparison isolates the query strategy.
        """
        if mode == "full":
            return list(self.parameters())
        if mode == "head":
            return list(self.head.parameters()) + list(self.classifier.parameters())
        if mode == "decoder_head":
            return (list(self.bridge.parameters()) + list(self.decoder.parameters())
                    + list(self.head.parameters()) + list(self.classifier.parameters()))
        if mode == "last_encoder_decoder_head":
            return (list(self.encoder[-1].parameters()) + list(self.bridge.parameters())
                    + list(self.decoder.parameters()) + list(self.head.parameters())
                    + list(self.classifier.parameters()))
        raise ValueError(f"unknown adaptation mode: {mode}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
