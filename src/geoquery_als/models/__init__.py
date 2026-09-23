"""Segmentation backbones.

Both backbones read the same batch produced by `geoquery_als.data.dataset` and
expose the same interface, so a script selects one by name and changes nothing
else.
"""
from __future__ import annotations

import re

import torch.nn as nn

from .point_transformer import PointTransformerConfig, PointTransformerSeg
from .randlanet import RandLANet, RandLANetConfig, count_parameters

__all__ = ["RandLANet", "RandLANetConfig", "PointTransformerSeg",
           "PointTransformerConfig", "count_parameters", "BACKBONES",
           "build_backbone", "backbone_suffix", "base_backbone",
           "variant_index"]

BACKBONES = ("randlanet", "point_transformer")

# A name of the form "<backbone>_s<N>" is the same architecture trained from a
# different initialisation. It builds the same network, keeps its artefacts
# under its own suffix, and shifts the training seed by N, so a seed
# sensitivity run needs no change anywhere downstream.
_VARIANT = re.compile(r"^(?P<base>[a-z_]+?)_s(?P<index>[0-9]+)$")


def base_backbone(name: str) -> str:
    """The architecture a possibly seed-suffixed name refers to."""
    m = _VARIANT.match(name)
    return m.group("base") if m else name


def variant_index(name: str) -> int:
    """Seed offset carried by the name, zero for the original run."""
    m = _VARIANT.match(name)
    return int(m.group("index")) if m else 0


def build_backbone(name: str, model_conf: dict) -> nn.Module:
    """Instantiate a backbone from the `model` block of the experiment config."""
    name = base_backbone(name)
    common = dict(num_classes=model_conf["num_classes"],
                  in_features=model_conf["in_features"],
                  num_layers=model_conf["num_layers"],
                  d_out=tuple(model_conf["d_out"]),
                  dropout=model_conf["dropout"])
    if name == "randlanet":
        return RandLANet(RandLANetConfig(**common))
    if name == "point_transformer":
        return PointTransformerSeg(PointTransformerConfig(
            attn_dim=tuple(model_conf.get("attn_dim", (16, 48, 96, 200))),
            fp32_softmax=bool(model_conf.get("fp32_softmax", True)), **common))
    raise ValueError(f"unknown backbone: {name!r}, expected one of {BACKBONES}")


def backbone_suffix(name: str) -> str:
    """Filename suffix so a second backbone never overwrites the first's results."""
    return "" if name == "randlanet" else f"_{name}"
