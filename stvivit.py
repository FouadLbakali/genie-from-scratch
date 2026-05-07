'''Spatio-Temporal ViViT encoder/decoder in PyTorch.

References:
[1] Dosovitskiy, A. et al. An Image is Worth 16x16 Words: Transformers
    for Image Recognition at Scale. arXiv:2010.11929
[2] Xu, M. et al. Spatial-Temporal Transformer Networks for Traffic Flow
    Forecasting. arXiv:2001.02908
[3] Bruce, J. et al. Genie: Generative Interactive Environments.
    arXiv:2402.15391
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class ReLUSquared(nn.Module):
    def forward(self, x):
        return F.relu(x).pow(2)
    
# patch_size = nb de pixel par patch sur une longueur
# Nb patch = (image_size/patch_size)**2


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, d_model, patch_size, image_size, time_steps):
        super().__init__()
        num_spatial = (image_size // patch_size)**2
        # num_time = time_steps // patch_size
        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=d_model,
            kernel_size=(1, patch_size, patch_size),
            stride=(1, patch_size, patch_size)
            )
        self.pos_embed_spatial = nn.Parameter(torch.randn(1, 1, num_spatial, d_model))
        self.pos_embed_temporal = nn.Parameter(torch.randn(1, time_steps, 1, d_model))
    
    def forward(self, x):
        # (Batch, Time, Channels, Height, Width)
        B, T, C, H, W = x.shape
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.flatten(3, 4)
        x = x.permute(0, 2, 3, 1)
        # (Batch, Time, N_Spatial, Channels)
        x = x + self.pos_embed_spatial + self.pos_embed_temporal
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, is_causal=False):
        super().__init__()
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.is_causal = is_causal

        self.proj_in = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj_out = nn.Linear(d_model, d_model, bias=False)
    
    def forward(self, x):
        # (Batch, Seq_len, Channels)
        B, L, C = x.shape
        q, k, v = self.proj_in(x).chunk(3, dim=-1)
        q = q.reshape(B, L, self.num_heads, self.d_head).transpose(1, 2)
        k = k.reshape(B, L, self.num_heads, self.d_head).transpose(1, 2)
        v = v.reshape(B, L, self.num_heads, self.d_head).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=self.is_causal)
        attn = attn.transpose(-3, -2).reshape(B, L, -1)
        return self.proj_out(attn)

class SpatialTransformer(nn.Module):
    def __init__(self, num_heads, d_model):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
    
    def forward(self, x):
        # (Batch, Time, N, Channels)
        B, T, N, C = x.shape
        x_s = x.reshape(-1, N, C)
        x_s = self.attn(x_s)
        x_s = x_s.reshape(B, T, N, C)
        return x_s

class TemporalTransformer(nn.Module):
    def __init__(self, num_heads, d_model):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads, is_causal=True)
    
    def forward(self, x):
        # (Batch, Time, N, Channels)
        B, T, N, C = x.shape
        x_t = x.transpose(1, 2).reshape(-1, T, C)
        x_t = self.attn(x_t)
        x_t = x_t.reshape(B, N, T, C).transpose(1, 2)
        return x_t

class SpatialTemporalEncoderBlock(nn.Module):
    def __init__(self, num_heads, d_model):
        super().__init__()
        self.norm = nn.RMSNorm(d_model, elementwise_affine=False)
        self.temp_attn = TemporalTransformer(num_heads, d_model)
        self.spatial_attn = SpatialTransformer(num_heads, d_model)
        self.ffw = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            ReLUSquared(),
            nn.Linear(4 * d_model, d_model),
        )
    
    def forward(self, x):
        x_s = self.spatial_attn(self.norm(x)) + x
        x_t = self.temp_attn(self.norm(x_s)) + x_s
        x_out = self.ffw(self.norm(x_t)) + x_t
        return x_out

class STTransformerEncoder(nn.Module):
    def __init__(self, num_heads, d_model, patch_size, num_layers, image_size, in_channels=3, time_steps=16):
        super().__init__()
        self.patch_emb = PatchEmbedding(in_channels, d_model, patch_size, image_size, time_steps)
        self.encoder = nn.Sequential(
            *[SpatialTemporalEncoderBlock(num_heads, d_model) for _ in range(num_layers)]
        )
    
    def forward(self, x):
        x = self.patch_emb(x)
        x = self.encoder(x)
        return x

class SpatialTemporalDecoderBlock(nn.Module):
    def __init__(self, num_heads, d_model):
        super().__init__()
        self.norm = nn.RMSNorm(d_model, elementwise_affine=False)
        self.temp_attn = TemporalTransformer(num_heads, d_model)
        self.spatial_attn = SpatialTransformer(num_heads, d_model)
        self.ffw = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            ReLUSquared(),
            nn.Linear(4 * d_model, d_model),
        )
    
    def forward(self, x):
        x_temp = self.temp_attn(self.norm(x)) + x
        x_spatial = self.spatial_attn(self.norm(x_temp)) + x_temp
        x_out = self.ffw(self.norm(x_spatial)) + x_spatial
        return x_out

class PatchReconstruction(nn.Module):
    def __init__(self, in_channels, d_model, patch_size, image_size):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.proj = nn.Linear(d_model, in_channels*patch_size*patch_size)
    
    def forward(self, x):
        # (Batch, Time, Num_Patches, D_Model)
        B, T, N, C = x.shape
        x = self.proj(x)
        # (Batch, Time, Num_Patches, Pixel_per_patch)
        nb_patches = self.image_size // self.patch_size
        x = x.reshape(B, T, nb_patches, nb_patches, self.patch_size, self.patch_size, self.in_channels)
        x = x.permute(0, 1, 6, 2, 4, 3, 5)
        x = x.reshape(B, T, self.in_channels, self.image_size, self.image_size)
        return x

class STTransformerDecoder(nn.Module):
    def __init__(self, num_heads, d_model, patch_size, num_layers, image_size, in_channels=3, time_steps=16):
        super().__init__()
        self.patch_rec = PatchReconstruction(in_channels, d_model, patch_size, image_size)
        self.decoder = nn.Sequential(
            *[SpatialTemporalDecoderBlock(num_heads, d_model) for _ in range(num_layers)]
        )
    
    def forward(self, x):
        x = self.decoder(x)
        x = self.patch_rec(x)
        return x

class STTransformer_AE(nn.Module):
    def __init__(self, num_heads, d_model, patch_size, image_size, num_layers, in_channels=3, time_steps=16):
        super().__init__()
        self.encoder = STTransformerEncoder(num_heads, d_model, patch_size, image_size, num_layers, in_channels, time_steps)
        self.decoder = STTransformerDecoder(num_heads, d_model, patch_size, image_size, num_layers, in_channels, time_steps)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
