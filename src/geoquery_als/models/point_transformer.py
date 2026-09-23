"""Point Transformer (Zhao et al., ICCV 2021) on the sampling skeleton of RandLA-Net.

The purpose of this backbone is an architecture-generality check. Everything that
is not the local aggregation operator is shared verbatim with
`geoquery_als.models.randlanet`: the same input projection, the same
random-sampling downsampling by max-pooling over the precomputed pooling
neighbourhood, the same nearest-neighbour upsampling, the same decoder,
the same 32-dimensional penultimate embedding and the same classifier. The
`_gather_neighbours`, `random_sample` and `nearest_interpolate` helpers are
imported rather than reimplemented so that the two models provably see the same
tensors.

What changes is how a point aggregates its k neighbours. RandLA-Net concatenates
an explicit relative-geometry encoding to the gathered neighbour features and
reduces them with a learned scalar softmax weight per neighbour. This backbone
uses the vector self-attention of Zhao et al.: the attention weight is produced
by a mapping network applied to the *difference* of a query and a key projection,
a learned relative-position encoding is added both to that difference and to the
value, and the resulting weight is a vector, so each feature channel is pooled
with its own coefficient.

Both operators read `batch["xyz"][i]`, `batch["neigh_idx"][i]`,
`batch["pool_idx"][i]` and `batch["up_idx"][i]` exactly as produced by
`geoquery_als.data.dataset`, so the data pipeline is untouched.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .randlanet import (SharedMLP, _gather_neighbours, nearest_interpolate,
                        random_sample)

__all__ = ["PointTransformerConfig", "PointTransformerSeg",
           "PointTransformerLayer", "PointTransformerBlock"]


class PointTransformerLayer(nn.Module):
    """Vector self-attention over a precomputed k-nearest-neighbour set.

    y_i = sum_j softmax_j( gamma( phi(x_i) - psi(x_j) + delta_ij ) ) * ( alpha(x_j) + delta_ij )

    with delta_ij = theta(p_i - p_j) the learned relative-position encoding,
    gamma and theta two-layer mapping networks, and the softmax taken over the k
    neighbours. The weight is a vector of length `dim`, so the product with the
    value is element-wise (Eq. 3 of Zhao et al., ICCV 2021).

    The first layer of the position encoding is batch-normalised. Coordinates
    reach this module divided by the patch half-width, so a neighbour offset is
    of order 0.02 in normalised units; without that normalisation the position
    encoding starts at a magnitude the attention cannot see.
    """

    def __init__(self, dim: int, pos_hidden: int | None = None,
                 attn_hidden: int | None = None, fp32_softmax: bool = True) -> None:
        super().__init__()
        ph = pos_hidden if pos_hidden is not None else dim
        ah = attn_hidden if attn_hidden is not None else dim
        self.phi = nn.Conv2d(dim, dim, 1, bias=False)
        self.psi = nn.Conv2d(dim, dim, 1, bias=False)
        self.alpha = nn.Conv2d(dim, dim, 1, bias=False)
        self.delta = nn.Sequential(
            nn.Conv2d(3, ph, 1, bias=False),
            nn.BatchNorm2d(ph, eps=1e-6, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.Conv2d(ph, dim, 1, bias=True),
        )
        self.gamma = nn.Sequential(
            nn.Conv2d(dim, ah, 1, bias=False),
            nn.BatchNorm2d(ah, eps=1e-6, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.Conv2d(ah, dim, 1, bias=True),
        )
        self.fp32_softmax = fp32_softmax

    def forward(self, xyz: torch.Tensor, feat: torch.Tensor,
                neigh_idx: torch.Tensor) -> torch.Tensor:
        # xyz (B, 3, N, 1), feat (B, C, N, 1), neigh_idx (B, N, K)
        k = neigh_idx.shape[-1]
        # The attention runs outside the autocast region. Half precision is safe
        # at initialisation, where the logits reach about six, but the mapping
        # network that produces them is trained without any bound on its output,
        # and a single overflow turns the softmax into a row of missing values
        # from which the weights never recover.
        with torch.amp.autocast("cuda", enabled=False):
            xyz = xyz.float()
            feat = feat.float()
            nxyz = _gather_neighbours(xyz, neigh_idx)             # (B, 3, N, K)
            rel = xyz.expand(-1, -1, -1, k) - nxyz
            pos = self.delta(rel)                                 # (B, C, N, K)
            q = self.phi(feat)                                    # (B, C, N, 1)
            key = _gather_neighbours(self.psi(feat), neigh_idx)   # (B, C, N, K)
            val = _gather_neighbours(self.alpha(feat), neigh_idx)  # (B, C, N, K)
            logits = self.gamma(q - key + pos)
            w = torch.softmax(logits, dim=-1)
            return torch.sum(w * (val + pos), dim=-1, keepdim=True)


class PointTransformerBlock(nn.Module):
    """Project in, attend over the neighbourhood, project out, residual.

    The residual wrapping mirrors `randlanet.DilatedResidualBlock` so that the
    two encoders differ only in the aggregation in the middle.
    """

    def __init__(self, in_ch: int, dim: int, out_ch: int,
                 fp32_softmax: bool = True) -> None:
        super().__init__()
        self.linear_in = SharedMLP(in_ch, dim, activation=nn.LeakyReLU(0.2))
        self.attn = PointTransformerLayer(dim, fp32_softmax=fp32_softmax)
        self.norm = nn.Sequential(nn.BatchNorm2d(dim, eps=1e-6, momentum=0.01),
                                  nn.LeakyReLU(0.2))
        self.linear_out = SharedMLP(dim, out_ch, bn=True, activation=None)
        self.shortcut = SharedMLP(in_ch, out_ch, bn=True, activation=None)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, xyz: torch.Tensor, feat: torch.Tensor,
                neigh_idx: torch.Tensor) -> torch.Tensor:
        h = self.linear_in(feat)
        h = self.norm(self.attn(xyz, h, neigh_idx))
        return self.act(self.linear_out(h) + self.shortcut(feat))


@dataclass
class PointTransformerConfig:
    num_classes: int = 7
    in_features: int = 8          # extra per-point features, excluding xyz
    num_layers: int = 4
    d_out: tuple[int, ...] = (16, 64, 128, 256)
    # Width of the attention itself. Kept below the stage width so that the
    # parameter count matches RandLA-Net to within a per cent; see
    # scripts/bench_backbone.py.
    attn_dim: tuple[int, ...] = (16, 48, 96, 200)
    dropout: float = 0.5
    fp32_softmax: bool = True


class PointTransformerSeg(nn.Module):
    """Same interface as `randlanet.RandLANet`.

    forward(batch, return_embedding) -> logits (B, C, N) or (logits, embedding)
    with a 32-channel embedding, and the same four `adaptation_parameters`
    modes over identically named submodules.
    """

    def __init__(self, cfg: PointTransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.fc_in = SharedMLP(cfg.in_features + 3, 8, activation=nn.LeakyReLU(0.2))

        enc, widths = [], []
        c = 8
        for i, d in enumerate(cfg.d_out):
            enc.append(PointTransformerBlock(c, cfg.attn_dim[i], d * 2,
                                             fp32_softmax=cfg.fp32_softmax))
            c = d * 2
            widths.append(c)
        self.encoder = nn.ModuleList(enc)

        self.bridge = SharedMLP(c, c, activation=nn.LeakyReLU(0.2))
        dec = []
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
