"""Frozen pretrained tactile encoders for the student's tactile input.

Each maps fingerpad TacMap images (B, H, W) in [0, 1] (the sharpa-tacmap 8-bit deformation
scale / 255) to one embedding per image (B, D):

- ``ftp1``: FTP1's SharpaWave tactile tokenizer (T3-style ViT encoder + FTP1's shared
  chunk trunk), CLS token, 768-d. Weights: HF ``MJJJJ1064/ftp1_v0426_50kstep``
  ``hpt_tokenizer/{SharpaWave_image_224_224_3,shared_image_chunk_encoder}.safetensors``.
  Input as FTP1 trains on Sharpa: x*2-1, then z-scored with its stored mean/std.
- ``t3``: T3 medium (Zhao et al.) GelSight Mini encoder + shared trunk, CLS token, 768-d.
  Weights: HF dataset ``alanz-mit/FoundationTactile`` ``models/t3_medium``. ImageNet norm.
- ``wan``: Wan2.2 VAE encoder (single frame, x*2-1), latent (48, H/16, W/16)
  average-pooled to (48, 4, 4) = 768-d.

Images are resized to ``size`` (224) and repeated to 3 channels. All-zero images (no
contact) map to one cached embedding, so only pads in contact are run through the encoder.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

PRETRAINED = Path("/weka/robots-default/jasonl/FoundationTouch/physics_ood/pretrained")
WAN_VAE = Path("/weka/robots-default/jasonl/checkpoints/Wan2.2-TI2V-5B/Wan2.2_VAE.pth")
WAN_CODE = Path("/weka/robots-default/jasonl/wan_tactile/wan22/wan/modules/vae2_2.py")


class _Attention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        return self.proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, C))


class _Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class _Block(nn.Module):
    """timm ``Block`` (pre-norm, LayerNorm eps 1e-6), same parameter names."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = _Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = _Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class _PatchEmbed(nn.Module):
    def __init__(self, dim: int, patch: int):
        super().__init__()
        self.proj = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class T3StyleViT(nn.Module):
    """T3 ``ViTEncoder`` (patch embed + CLS + pos + encoder blocks) -> shared trunk -> norm -> CLS."""

    def __init__(self, dim=768, heads=12, enc_depth=3, trunk_depth=9, patch=16, img=224):
        super().__init__()
        self.patch_embed = _PatchEmbed(dim, patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, (img // patch) ** 2 + 1, dim))
        self.blocks = nn.ModuleList(_Block(dim, heads) for _ in range(enc_depth))
        self.trunk = nn.ModuleList(_Block(dim, heads) for _ in range(trunk_depth))
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def load(self, encoder_sd: dict, trunk_sd: dict) -> None:
        sd = {k: v for k, v in encoder_sd.items() if k.split(".")[0] in ("patch_embed", "cls_token", "pos_embed", "blocks")}
        sd.update({("trunk." + k[len("blocks."):]) if k.startswith("blocks.") else k: v
                   for k, v in trunk_sd.items() if k.startswith(("blocks.", "norm."))})
        self.load_state_dict(sd, strict=True)

    def forward(self, x):
        x = self.patch_embed(x)
        x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], 1) + self.pos_embed
        for b in self.blocks:
            x = b(x)
        for b in self.trunk:
            x = b(x)
        return self.norm(x)[:, 0]


def _strip(sd: dict, prefix: str) -> dict:
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


class TactileFoundationEncoder(nn.Module):
    """(B, H, W) TacMap images in [0, 1] -> (B, D) frozen embeddings."""

    def __init__(self, kind: str, device, size: int = 224):
        super().__init__()
        self.kind, self.size = kind, size
        if kind in ("ftp1", "t3"):
            self.net = T3StyleViT(img=size)
            if kind == "ftp1":
                from safetensors.torch import load_file

                d = PRETRAINED / "ftp1_v0426_50kstep" / "hpt_tokenizer"
                self.net.load(_strip(load_file(str(d / "SharpaWave_image_224_224_3.safetensors")), "vit_encoder."),
                              _strip(load_file(str(d / "shared_image_chunk_encoder.safetensors")), "shared_chunk_encoder."))
                # normalization/sharpa/independent_norm_stats_all_t0_zscore.json (same value on all 3 channels)
                mean, std = torch.full((3,), -0.8676983714103699), torch.full((3,), 0.19994118809700012)
                self._prep = lambda x: (x * 2 - 1 - mean.view(1, 3, 1, 1).to(x)) / std.view(1, 3, 1, 1).to(x)
            else:
                d = PRETRAINED / "t3" / "models" / "t3_medium"
                self.net.load(torch.load(d / "encoders" / "mini.pth", map_location="cpu"),
                              torch.load(d / "trunk.pth", map_location="cpu"))
                m, s = torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])
                self._prep = lambda x: (x - m.view(1, 3, 1, 1).to(x)) / s.view(1, 3, 1, 1).to(x)
            self.dim = 768
        elif kind == "wan":
            spec = importlib.util.spec_from_file_location("wan_vae2_2", str(WAN_CODE))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self.vae = mod.Wan2_2_VAE(vae_pth=str(WAN_VAE), dtype=torch.float32, device=str(device))
            self.net = self.vae.model
            self._prep = lambda x: x * 2 - 1
            self.dim = 48 * 4 * 4
        else:
            raise ValueError(f"unknown tactile encoder {kind!r} (ftp1 | t3 | wan)")
        self.net.to(device).eval().requires_grad_(False)
        self.device = device
        self._zero_feat = None

    @torch.no_grad()
    def _encode(self, img: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(img.unsqueeze(1).float(), size=(self.size, self.size), mode="bilinear", align_corners=False)
        x = self._prep(x.expand(-1, 3, -1, -1))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if self.kind == "wan":
                z = self.net.encode(x.unsqueeze(2), self.vae.scale)            # (B, 48, 1, h, w)
                out = F.adaptive_avg_pool2d(z[:, :, 0].float(), 4).flatten(1)
            else:
                out = self.net(x)
        return out.float()

    def _zero_feat_or_none(self, num_envs: int) -> torch.Tensor:
        """(num_envs, 5*D) placeholder before the first step: the no-contact embedding if known, else zeros."""
        if self._zero_feat is None:
            return torch.zeros(num_envs, 5 * self.dim, device=self.device)
        return self._zero_feat.repeat(5).expand(num_envs, -1).clone()

    @torch.no_grad()
    def forward(self, img: torch.Tensor, chunk: int | None = None) -> torch.Tensor:
        chunk = chunk or (256 if self.kind == "wan" else 1024)
        B = img.shape[0]
        if self._zero_feat is None:
            self._zero_feat = self._encode(torch.zeros(1, *img.shape[1:], device=img.device))[0]
        out = self._zero_feat.expand(B, -1).clone()
        hit = img.flatten(1).amax(1) > 0
        idx = hit.nonzero(as_tuple=False).squeeze(-1)
        for i in range(0, idx.numel(), chunk):
            j = idx[i:i + chunk]
            out[j] = self._encode(img[j])
        return out
