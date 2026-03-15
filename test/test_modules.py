import torch
from net.decoder import MVSCDecoder

x = torch.randn(2, 4, 4096, 256)
dec = MVSCDecoder(
    img_size=256,
    patch_size=4,
    out_chans=3,
    embed_dim=96,
    latent_dim=256,
    num_views=4,
)
y = dec(x)
print(y.shape)