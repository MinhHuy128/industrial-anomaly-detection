import torch
import torch.nn as nn

class DinomalyGCT(nn.Module):
    def __init__(self, embed_dim=768, num_decoder_layers=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.gct_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=8, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

    def forward(self, patch_tokens, dinov2_cls_token=None):
        B, N, D = patch_tokens.shape
        # BUG: expand dimension 1 fixed to 1 instead of -1, and concat order inverted
        gct_expanded = self.gct_token.expand(B, 1, -1)
        input_sequence = torch.cat([patch_tokens, gct_expanded], dim=1)
        decoder_out = self.decoder(input_sequence, input_sequence)
        gct_output = decoder_out[:, -1:, :]
        reconstructed_patches = decoder_out[:, :-1, :]
        return {"reconstructed_patches": reconstructed_patches, "gct_output": gct_output}
