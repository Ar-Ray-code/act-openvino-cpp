"""
Standalone ACT Model Definition for ONNX Export

This is a minimal implementation of the ACT model that doesn't depend on lerobot library.
It replicates the essential components needed for inference.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d
import einops


class ACTModel(nn.Module):
    """Standalone ACT model for inference"""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        # Vision backbone
        self.vision_backbone = config.get("vision_backbone", "resnet18")
        self.dim_model = config.get("dim_model", 512)
        self.chunk_size = config.get("chunk_size", 100)
        self.latent_dim = config.get("latent_dim", 32)
        self.n_encoder_layers = config.get("n_encoder_layers", 4)
        self.n_decoder_layers = config.get("n_decoder_layers", 1)
        self.n_heads = config.get("n_heads", 8)
        self.dim_feedforward = config.get("dim_feedforward", 3200)
        self.dropout = config.get("dropout", 0.1)
        self.pre_norm = config.get("pre_norm", False)

        # Get input/output dimensions
        self.state_dim = config["input_features"]["observation.state"]["shape"][0]
        self.action_dim = config["output_features"]["action"]["shape"][0]

        # Build backbone for image feature extraction
        if "observation.images.front" in config["input_features"]:
            backbone_model = getattr(torchvision.models, self.vision_backbone)(
                replace_stride_with_dilation=[False, False, False],
                weights="IMAGENET1K_V1",
                norm_layer=FrozenBatchNorm2d,
            )
            self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})
            backbone_out_channels = backbone_model.fc.in_features

        # Transformer encoder input projections
        self.encoder_robot_state_input_proj = nn.Linear(self.state_dim, self.dim_model)
        self.encoder_latent_input_proj = nn.Linear(self.latent_dim, self.dim_model)
        self.encoder_img_feat_input_proj = nn.Conv2d(backbone_out_channels, self.dim_model, kernel_size=1)

        # Positional embeddings
        self.encoder_1d_feature_pos_embed = nn.Embedding(2, self.dim_model)  # latent + state
        self.encoder_cam_feat_pos_embed = SinusoidalPositionEmbedding2d(self.dim_model // 2)

        # Transformer encoder
        self.encoder = TransformerEncoder(
            dim_model=self.dim_model,
            n_heads=self.n_heads,
            dim_feedforward=self.dim_feedforward,
            n_layers=self.n_encoder_layers,
            dropout=self.dropout,
            pre_norm=self.pre_norm
        )

        # Transformer decoder
        self.decoder = TransformerDecoder(
            dim_model=self.dim_model,
            n_heads=self.n_heads,
            dim_feedforward=self.dim_feedforward,
            n_layers=self.n_decoder_layers,
            dropout=self.dropout,
            pre_norm=self.pre_norm
        )

        self.decoder_pos_embed = nn.Embedding(self.chunk_size, self.dim_model)

        # Action head
        self.action_head = nn.Linear(self.dim_model, self.action_dim)

    def forward(self, state: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (B, state_dim) robot state
            image: (B, 3, H, W) camera image

        Returns:
            actions: (B, chunk_size, action_dim) predicted actions
        """
        batch_size = state.shape[0]
        device = state.device
        dtype = self.encoder_latent_input_proj.weight.dtype

        # Latent (zero for inference without VAE)
        latent = torch.zeros(batch_size, self.latent_dim, device=device, dtype=dtype)

        # Prepare encoder inputs
        encoder_tokens = []
        encoder_pos_embeds = []

        # Latent and state tokens
        encoder_tokens.append(self.encoder_latent_input_proj(latent))
        encoder_tokens.append(self.encoder_robot_state_input_proj(state))
        encoder_pos_embeds.extend(list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1)))

        # Image features
        cam_features = self.backbone(image)["feature_map"]
        cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
        cam_features = self.encoder_img_feat_input_proj(cam_features)

        # Rearrange to (sequence, batch, dim)
        cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
        cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")

        # Extend tokens
        encoder_tokens.extend(list(cam_features))
        encoder_pos_embeds.extend(list(cam_pos_embed))

        # Stack all tokens
        encoder_tokens = torch.stack(encoder_tokens, axis=0)
        encoder_pos_embeds = torch.stack(encoder_pos_embeds, axis=0)

        # Forward through encoder
        encoder_out = self.encoder(encoder_tokens, encoder_pos_embeds)

        # Decoder
        decoder_in = torch.zeros(
            (self.chunk_size, batch_size, self.dim_model),
            dtype=encoder_tokens.dtype,
            device=device
        )
        decoder_out = self.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embeds,
            self.decoder_pos_embed.weight.unsqueeze(1)
        )

        # Move to (B, S, C) and apply action head
        decoder_out = decoder_out.transpose(0, 1)
        actions = self.action_head(decoder_out)

        return actions


class TransformerEncoder(nn.Module):
    def __init__(self, dim_model, n_heads, dim_feedforward, n_layers, dropout, pre_norm):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(dim_model, n_heads, dim_feedforward, dropout, pre_norm)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(dim_model) if pre_norm else nn.Identity()

    def forward(self, x, pos_embed):
        for layer in self.layers:
            x = layer(x, pos_embed)
        return self.norm(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim_model, n_heads, dim_feedforward, dropout, pre_norm):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim_model, n_heads, dropout=dropout)
        self.linear1 = nn.Linear(dim_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, dim_model)
        self.norm1 = nn.LayerNorm(dim_model)
        self.norm2 = nn.LayerNorm(dim_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.pre_norm = pre_norm

    def forward(self, x, pos_embed):
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = x + pos_embed
        x = self.self_attn(q, k, value=x)[0]
        x = skip + self.dropout1(x)

        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x

        x = self.linear2(self.dropout(F.relu(self.linear1(x))))
        x = skip + self.dropout2(x)

        if not self.pre_norm:
            x = self.norm2(x)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, dim_model, n_heads, dim_feedforward, n_layers, dropout, pre_norm):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(dim_model, n_heads, dim_feedforward, dropout, pre_norm)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(dim_model)

    def forward(self, x, encoder_out, encoder_pos_embed, decoder_pos_embed):
        for layer in self.layers:
            x = layer(x, encoder_out, encoder_pos_embed, decoder_pos_embed)
        return self.norm(x)


class TransformerDecoderLayer(nn.Module):
    def __init__(self, dim_model, n_heads, dim_feedforward, dropout, pre_norm):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim_model, n_heads, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(dim_model, n_heads, dropout=dropout)
        self.linear1 = nn.Linear(dim_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, dim_model)
        self.norm1 = nn.LayerNorm(dim_model)
        self.norm2 = nn.LayerNorm(dim_model)
        self.norm3 = nn.LayerNorm(dim_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.pre_norm = pre_norm

    def forward(self, x, encoder_out, encoder_pos_embed, decoder_pos_embed):
        # Self attention
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = x + decoder_pos_embed
        x = self.self_attn(q, k, value=x)[0]
        x = skip + self.dropout1(x)

        # Cross attention
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x

        x = self.multihead_attn(
            query=x + decoder_pos_embed,
            key=encoder_out + encoder_pos_embed,
            value=encoder_out
        )[0]
        x = skip + self.dropout2(x)

        # Feed forward
        if self.pre_norm:
            skip = x
            x = self.norm3(x)
        else:
            x = self.norm2(x)
            skip = x

        x = self.linear2(self.dropout(F.relu(self.linear1(x))))
        x = skip + self.dropout3(x)

        if not self.pre_norm:
            x = self.norm3(x)
        return x


class SinusoidalPositionEmbedding2d(nn.Module):
    """2D sinusoidal positional embeddings"""

    def __init__(self, dimension: int):
        super().__init__()
        self.dimension = dimension
        self._two_pi = 2 * math.pi
        self._eps = 1e-6
        self._temperature = 10000

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) feature map
        Returns:
            (1, C, H, W) positional embeddings
        """
        not_mask = torch.ones_like(x[0, :1])
        y_range = not_mask.cumsum(1, dtype=torch.float32)
        x_range = not_mask.cumsum(2, dtype=torch.float32)

        y_range = y_range / (y_range[:, -1:, :] + self._eps) * self._two_pi
        x_range = x_range / (x_range[:, :, -1:] + self._eps) * self._two_pi

        inverse_frequency = self._temperature ** (
            2 * (torch.arange(self.dimension, dtype=torch.float32, device=x.device) // 2) / self.dimension
        )

        x_range = x_range.unsqueeze(-1) / inverse_frequency
        y_range = y_range.unsqueeze(-1) / inverse_frequency

        pos_embed_x = torch.stack((x_range[..., 0::2].sin(), x_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed_y = torch.stack((y_range[..., 0::2].sin(), y_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed = torch.cat((pos_embed_y, pos_embed_x), dim=3).permute(0, 3, 1, 2)

        return pos_embed
