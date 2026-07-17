import torch
import torch.nn as nn

class DinomalyBaseline(nn.Module):
    def __init__(self, embed_dim=768, num_decoder_layers=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_decoder_layers = num_decoder_layers
        self.dummy_layer = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, patch_tokens):
        return self.dummy_layer(patch_tokens)
