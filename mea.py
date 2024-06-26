import torch
import torch.nn as nn
import torch.nn.functional as F

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple, Type
from monai.losses import DiceCELoss


class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class LayerNorm3d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
        return x


class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        self.norm1 = LayerNorm3d(384)
        self.norm2 = LayerNorm3d(384)
        self.mlp = MLPBlock(384, 4 * 384)
        if self.use_rel_pos:
            assert (
                input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            # initialize relative positional embeddings
            self.rel_pos_d = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[2] - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = (self.norm1(x.permute(0, 4, 1, 2, 3))).permute(0, 2, 3, 4, 1)

        B, D, H, W, _ = x.shape  # [2, 8, 8, 8, 768]

        qkv = (
            self.qkv(x)
            .reshape(B, D * H * W, 3, self.num_heads, -1)
            .permute(2, 0, 3, 1, 4)
        )  # [3, 2, 12, 512, 64]

        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, D * H * W, -1).unbind(
            0
        )  # [24, 512, 64]

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(
                attn,
                q,
                self.rel_pos_d,
                self.rel_pos_h,
                self.rel_pos_w,
                (D, H, W),
                (D, H, W),
            )

        attn = attn.softmax(dim=-1)
        x = (
            (attn @ v)
            .view(B, self.num_heads, D, H, W, -1)
            .permute(0, 2, 3, 4, 1, 5)
            .reshape(B, D, H, W, -1)
        )
        x = self.proj(x)

        x = x + shortcut
        x = (self.norm2(x.permute(0, 4, 1, 2, 3))).permute(0, 2, 3, 4, 1)
        x = self.mlp(x)

        return x


def window_partition3D(
    x: torch.Tensor, window_size: int
) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, D, H, W, C = x.shape

    pad_d = (window_size - D % window_size) % window_size
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size

    if pad_h > 0 or pad_w > 0 or pad_d > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_d))
    Hp, Wp, Dp = H + pad_h, W + pad_w, D + pad_d

    x = x.view(
        B,
        Dp // window_size,
        window_size,
        Hp // window_size,
        window_size,
        Wp // window_size,
        window_size,
        C,
    )
    windows = (
        x.permute(0, 1, 3, 5, 2, 4, 6, 7)
        .contiguous()
        .view(-1, window_size, window_size, window_size, C)
    )
    return windows, (Dp, Hp, Wp)


def window_unpartition3D(
    windows: torch.Tensor,
    window_size: int,
    pad_dhw: Tuple[int, int, int],
    dhw: Tuple[int, int, int],
) -> torch.Tensor:
    """
    Window unpartition into original sequences and removing padding.
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Dp, Hp, Wp = pad_dhw
    D, H, W = dhw
    B = windows.shape[0] // (Dp * Hp * Wp // window_size // window_size // window_size)
    x = windows.view(
        B,
        Dp // window_size,
        Hp // window_size,
        Wp // window_size,
        window_size,
        window_size,
        window_size,
        -1,
    )
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, Hp, Wp, Dp, -1)

    if Hp > H or Wp > W or Dp > D:
        x = x[:, :D, :H, :W, :].contiguous()
    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_d: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int, int],
    k_size: Tuple[int, int, int],
) -> torch.Tensor:
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_d, q_h, q_w = q_size
    k_d, k_h, k_w = k_size

    Rd = get_rel_pos(q_d, k_d, rel_pos_d)
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_d, q_h, q_w, dim)

    rel_d = torch.einsum("bdhwc,dkc->bdhwk", r_q, Rd)
    rel_h = torch.einsum("bdhwc,hkc->bdhwk", r_q, Rh)
    rel_w = torch.einsum("bdhwc,wkc->bdhwk", r_q, Rw)

    attn = (
        attn.view(B, q_d, q_h, q_w, k_d, k_h, k_w)
        + rel_d[:, :, :, :, None, None]
        + rel_h[:, :, :, None, :, None]
        + rel_w[:, :, :, None, None, :]
    ).view(B, q_d * q_h * q_w, k_d * k_h * k_w)

    return attn


class PatchEmbed3D(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16, 16),
        stride: Tuple[int, int] = (16, 16, 16),
        padding: Tuple[int, int] = (0, 0, 0),
        in_chans: int = 1,
        embed_dim: int = 768,
    ) -> None:
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        # B C X Y Z -> B X Y Z C
        x = x.permute(0, 2, 3, 4, 1)
        return x


class new_BiDirectionalAttention(nn.Module):
    def __init__(self, channels, dim):
        super(new_BiDirectionalAttention, self).__init__()
        self.channels = channels
        self.dim = dim
        depth, height, width = dim

        self.prompt_conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.prompt_conv2 = nn.Conv3d(channels, channels, kernel_size=1)

        self.image_conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.image_conv2 = nn.Conv3d(channels, channels, kernel_size=1)

        self.prompt_norm = LayerNorm3d(384)
        self.image_norm = LayerNorm3d(384)

        self.prompt_addnorm1 = LayerNorm3d(384)
        self.prompt_addnorm2 = LayerNorm3d(384)
        self.prompt_addnorm3 = LayerNorm3d(384)
        self.image_addnorm1 = LayerNorm3d(384)
        self.image_addnorm2 = LayerNorm3d(384)
        self.image_addnorm3 = LayerNorm3d(384)

        self.prompt_gelu = nn.GELU()
        self.image_gelu = nn.GELU()

        self.prompt_multiheadattn = Attention(
            dim=384,
            num_heads=6,
            qkv_bias=True,
            use_rel_pos=True,
            rel_pos_zero_init=True,
            input_size=(8, 8, 8),
        )
        self.image_multiheadattn = Attention(
            dim=384,
            num_heads=6,
            qkv_bias=True,
            use_rel_pos=True,
            rel_pos_zero_init=True,
            input_size=(8, 8, 8),
        )

        # MLP layer
        self.mlp1 = MLPBlock(384, 4 * 384)
        self.mlp2 = MLPBlock(384, 4 * 384)

    def forward(self, image_embedding, prompt_embedding):
        batch_size = image_embedding.size(0)

        updated_image_embedding = self.image_conv1(image_embedding)
        updated_image_embedding = self.image_norm(updated_image_embedding)
        updated_image_embedding = self.image_gelu(updated_image_embedding)
        updated_image_embedding = self.image_conv2(updated_image_embedding)
        updated_image_embedding = self.image_addnorm1(
            updated_image_embedding + image_embedding
        )

        updated_prompt_embedding = self.prompt_conv1(prompt_embedding)
        updated_prompt_embedding = self.prompt_norm(updated_prompt_embedding)
        updated_prompt_embedding = self.prompt_gelu(updated_prompt_embedding)
        updated_prompt_embedding = self.prompt_conv2(updated_prompt_embedding)
        updated_prompt_embedding = self.prompt_addnorm1(
            updated_prompt_embedding + prompt_embedding
        )

        qkv_p = image_embedding + updated_prompt_embedding + prompt_embedding
        qkv_i = updated_image_embedding + image_embedding + prompt_embedding

        output_image_embedding = self.image_multiheadattn(
            qkv_i.permute(0, 2, 3, 4, 1)
        ).permute(0, 4, 1, 2, 3)

        output_image_embedding = self.image_addnorm2(
            output_image_embedding + updated_image_embedding
        )

        output_image_embedding = self.image_addnorm3(
            self.mlp1(output_image_embedding.permute(0, 2, 3, 4, 1)).permute(
                0, 4, 1, 2, 3
            )
            + output_image_embedding
        )

        output_prompt_embedding = self.prompt_multiheadattn(
            qkv_p.permute(0, 2, 3, 4, 1)
        ).permute(0, 4, 1, 2, 3)

        output_prompt_embedding = self.prompt_addnorm2(
            output_prompt_embedding + updated_prompt_embedding
        )

        output_prompt_embedding = self.prompt_addnorm3(
            self.mlp2(output_prompt_embedding.permute(0, 2, 3, 4, 1)).permute(
                0, 4, 1, 2, 3
            )
            + output_prompt_embedding
        )

        return (
            output_image_embedding,
            output_prompt_embedding,
        )


class BiDirectionalAttentionNetwork(nn.Module):
    def __init__(self, channels, dim, num_attention_blocks):
        super(BiDirectionalAttentionNetwork, self).__init__()
        self.channels = channels
        self.dim = dim
        depth, height, width = dim

        # N * BiDirectionalAttention
        self.bidirectional_attentions = nn.ModuleList(
            [
                new_BiDirectionalAttention(channels, dim)
                for _ in range(num_attention_blocks)
            ]
        )

        # Muilthead Attention Layer
        self.multi_head_self_attention = Attention(
            dim=384,
            num_heads=6,
            qkv_bias=True,
            use_rel_pos=True,
            rel_pos_zero_init=True,
            input_size=(8, 8, 8),
        )

        # add and Norm
        self.add_norm = LayerNorm3d(384)
        self.add_norm2 = LayerNorm3d(384)
        self.add_norm3 = LayerNorm3d(384)
        self.add_norm4 = LayerNorm3d(384)
        self.mlp = MLPBlock(384, 4 * 384)

    def forward(self, image_embedding, prompt_embedding):

        prompt_embedding = self.add_norm2(prompt_embedding)
        image_embedding = self.add_norm3(image_embedding)
        
        updated_prompt_embedding = prompt_embedding
        updated_image_embedding = image_embedding

        # N * BidirectionalAttentions
        for attention_block in self.bidirectional_attentions:
            updated_image_embedding, updated_prompt_embedding = attention_block(
                updated_image_embedding, updated_prompt_embedding
            )

        qkv = updated_image_embedding + image_embedding + updated_prompt_embedding + prompt_embedding
    
        # # Muilthead Attention Layer
        output = self.multi_head_self_attention(qkv.permute(0, 2, 3, 4, 1)).permute(
            0, 4, 1, 2, 3
        )

        shortcut = output

        output = self.add_norm(output + qkv) 

        output = output + self.mlp(output.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)

        return output


# batch_size, channels, depth, height, width = 1, 384, 8, 8, 8
# image_embedding = torch.rand(1, 384, 8, 8, 8)
# prompt_embedding = torch.rand(batch_size, channels, depth, height, width)

# network = BiDirectionalAttentionNetwork(
#     channels, (depth, height, width), num_attention_blocks=1
# )


# output = network(image_embedding, prompt_embedding)
# print(output.shape)
