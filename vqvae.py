'''VQ-VAE video tokenizer in PyTorch.

References:
[1] van den Oord, A., Vinyals, O., Kavukcuoglu, K. Neural Discrete
    Representation Learning. arXiv:1711.00937
[2] Bruce, J. et al. Genie: Generative Interactive Environments.
    arXiv:2402.15391
'''
import torch
import torch.nn as nn
from stvivit import STTransformerEncoder, STTransformerDecoder
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25):
        super().__init__()
        self.e = nn.Parameter(torch.rand(num_embeddings, embedding_dim)/num_embeddings)
        self.beta = beta
    
    def _compute_index(self, x):
        B, T, N, C = x.shape
        z_e = x.reshape(B*T*N, C)
        dist = (z_e**2).sum(dim=1, keepdim=True) - 2 * (z_e @ self.e.transpose(0, 1)) + (self.e**2).sum(dim=1)
        index = dist.argmin(dim=1)

        return index, B, T, N, C

    def forward(self, x):
        index, B, T, N, C = self._compute_index(x)
        z_q = self.e[index]
        z_q = z_q.view(B, T, N, C)

        vq_loss = F.mse_loss(x.detach(), z_q) + self.beta * F.mse_loss(x, z_q.detach())
        z_q = x + (z_q - x).detach()
        
        return z_q, vq_loss
    
    def get_index(self, x):
        index, B, T, N, C= self._compute_index(x)

        return index.view(B, T, N)
    
    def get_codebook_entry(self, index):
        z_q = self.e[index]

        return z_q

class VQVAE(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, num_codes, patch_size, latent_dim, image_size, time_steps, in_channels):
        super().__init__()
        self.encoder = STTransformerEncoder(num_heads, d_model, patch_size, num_layers, image_size, in_channels, time_steps)
        self.decoder = STTransformerDecoder(num_heads, d_model, patch_size, num_layers, image_size, in_channels, time_steps)
        self.vq = VectorQuantizer(num_codes, latent_dim)
    
    def forward(self, x):
        x = self.encoder(x)
        x, vq_loss = self.vq(x)
        x = self.decoder(x)
        return x, vq_loss
    
    def get_index(self, x):
        x = self.encoder(x)
        index = self.vq.get_index(x)
        return index
    
    def decode_from_index(self, index):
        x = self.vq.get_codebook_entry(index)
        x = self.decoder(x)
        return x
