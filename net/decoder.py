from net.modules import *
import math
import torch
from net.encoder import (
    SwinTransformerBlock,
    SwinTransformerBlock3D,
    AdaptiveModulator,
    ConvResidualBlock2D,
    ViewFusionBlock,
)


class BasicLayer(nn.Module):

    def __init__(self, dim, out_dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 norm_layer=nn.LayerNorm, upsample=None,):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if upsample is not None:
            self.upsample = upsample(input_resolution, dim=dim, out_dim=out_dim, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        for _, blk in enumerate(self.blocks):
            x = blk(x)

        if self.upsample is not None:
            x = self.upsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
            print("blk.flops()", blk.flops())
        if self.upsample is not None:
            flops += self.upsample.flops()
            print("upsample.flops()", self.upsample.flops())
        return flops

    def update_resolution(self, H, W):
        self.input_resolution = (H, W)
        for _, blk in enumerate(self.blocks):
            blk.input_resolution = (H, W)
            blk.update_mask()
        if self.upsample is not None:
            self.upsample.input_resolution = (H, W)


class SwinJSCC_Decoder(nn.Module):
    def __init__(self, model, img_size, embed_dims, depths, num_heads, C,
                 window_size=4, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 bottleneck_dim=16):
        super().__init__()

        self.num_layers = len(depths)
        self.ape = ape
        self.embed_dims = embed_dims
        self.patch_norm = patch_norm
        self.num_features = bottleneck_dim
        self.mlp_ratio = mlp_ratio
        self.H = img_size[0]
        self.W = img_size[1]
        self.patches_resolution = (img_size[0] // 2 ** len(depths), img_size[1] // 2 ** len(depths))
        num_patches = self.H // 4 * self.W // 4
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dims[0]))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dims[i_layer]),
                               out_dim=int(embed_dims[i_layer + 1]) if (i_layer < self.num_layers - 1) else 3,
                               input_resolution=(self.patches_resolution[0] * (2 ** i_layer),
                                                 self.patches_resolution[1] * (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               norm_layer=norm_layer,
                               upsample=PatchReverseMerging)
            self.layers.append(layer)
            print("Decoder ", layer.extra_repr())
        if C != None:
            self.head_list = nn.Linear(C, embed_dims[0])
        self.apply(self._init_weights)
        self.hidden_dim = int(self.embed_dims[0] * 1.5)
        self.layer_num = layer_num = 7
        if model != "SwinJSCC_w/_RA":
            self.bm_list = nn.ModuleList()
            self.sm_list = nn.ModuleList()
            self.sm_list.append(nn.Linear(self.embed_dims[0], self.hidden_dim))
            for i in range(layer_num):
                if i == layer_num - 1:
                    outdim = self.embed_dims[0]
                else:
                    outdim = self.hidden_dim
                self.bm_list.append(AdaptiveModulator(self.hidden_dim))
                self.sm_list.append(nn.Linear(self.hidden_dim, outdim))
            self.sigmoid = nn.Sigmoid()

    def forward(self, x, snr, model):
        if model == 'SwinJSCC_w/o_SAandRA':
            x = self.head_list(x)
            for i_layer, layer in enumerate(self.layers):
                x = layer(x)
            B, L, N = x.shape
            x = x.reshape(B, self.H, self.W, N).permute(0, 3, 1, 2)
            return x

        elif model == 'SwinJSCC_w/_SA':
            B, L, C = x.size()
            device = x.get_device()
            x = self.head_list(x)
            snr_cuda = torch.tensor(snr, dtype=torch.float).to(device)
            snr_batch = snr_cuda.unsqueeze(0).expand(B, -1)
            for i in range(self.layer_num):
                if i == 0:
                    temp = self.sm_list[i](x.detach())
                else:
                    temp = self.sm_list[i](temp)
                bm = self.bm_list[i](snr_batch).unsqueeze(1).expand(-1, L, -1)
                temp = temp * bm
            mod_val = self.sigmoid(self.sm_list[-1](temp))
            x = x * mod_val
            for i_layer, layer in enumerate(self.layers):
                x = layer(x)
            B, L, N = x.shape
            x = x.reshape(B, self.H, self.W, N).permute(0, 3, 1, 2)
            return x

        elif model == 'SwinJSCC_w/_RA':
            for i_layer, layer in enumerate(self.layers):
                x = layer(x)
            B, L, N = x.shape
            x = x.reshape(B, self.H, self.W, N).permute(0, 3, 1, 2)
            return x

        elif model == 'SwinJSCC_w/_SAandRA':
            B, L, C = x.size()
            device = x.get_device()
            snr_cuda = torch.tensor(snr, dtype=torch.float).to(device)
            snr_batch = snr_cuda.unsqueeze(0).expand(B, -1)
            for i in range(self.layer_num):
                if i == 0:
                    temp = self.sm_list[i](x.detach())
                else:
                    temp = self.sm_list[i](temp)
                bm = self.bm_list[i](snr_batch).unsqueeze(1).expand(-1, L, -1)
                temp = temp * bm
            mod_val = self.sigmoid(self.sm_list[-1](temp))
            x = x * mod_val
            for i_layer, layer in enumerate(self.layers):
                x = layer(x)
            B, L, N = x.shape
            x = x.reshape(B, self.H, self.W, N).permute(0, 3, 1, 2)
            return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def flops(self):
        flops = 0
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        return flops

    def update_resolution(self, H, W):
        self.input_resolution = (H, W)
        self.H = H * 2 ** len(self.layers)
        self.W = W * 2 ** len(self.layers)
        for i_layer, layer in enumerate(self.layers):
            layer.update_resolution(H * (2 ** i_layer),
                                    W * (2 ** i_layer))


class SwinTransformerBlock3D_Dec(nn.Module):
    """Decoder-side wrapper for full 3D shifted-window Swin block."""

    def __init__(self, dim, input_resolution, num_heads, window_size=8):
        super().__init__()
        self.block = SwinTransformerBlock3D(
            dim=dim,
            input_resolution=input_resolution,
            num_heads=num_heads,
            window_size=window_size,
        )

    def forward(self, x):
        return self.block(x)

class BasicLayer3D_Dec(nn.Module):
    """
    Stack of decoder-side 3D blocks.
    Input: [B, T, L, C]
    Output: [B, T, L, C]
    """
    def __init__(self, dim, input_resolution, depth, num_heads, window_size=8):
        super().__init__()
        if isinstance(window_size, int):
            window_size_3d = (2, window_size, window_size)
        elif len(window_size) == 2:
            window_size_3d = (2, window_size[0], window_size[1])
        else:
            window_size_3d = tuple(window_size)

        shift_size_3d = (
            window_size_3d[0] // 2,
            window_size_3d[1] // 2,
            window_size_3d[2] // 2,
        )

        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock3D(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size_3d,
                    shift_size=(0, 0, 0) if (i % 2 == 0) else shift_size_3d,
                )
                for i in range(depth)
            ]
        )

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class JSCCUpBlock(nn.Module):
    """Upsampling transposed-conv block followed by residual refinement."""

    def __init__(self, in_ch, out_ch, stride=2, num_res_blocks=3):
        super().__init__()
        if stride == 2:
            self.up = nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.GELU(),
            )
        elif stride == 1:
            self.up = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
                nn.GELU(),
            )
        else:
            raise ValueError(f"Unsupported stride for JSCCUpBlock: {stride}")

        self.res_blocks = nn.Sequential(*[ConvResidualBlock2D(out_ch) for _ in range(num_res_blocks)])

    def forward(self, x):
        x = self.up(x)
        x = self.res_blocks(x)
        return x

class MVSC_JSCC_Decoder(nn.Module):
    """
    Decode channel latent tokens back to semantic tokens.
    Input:  [B, T, L, latent_dim]
    Output: [B, T, L, embed_dim]
    """
    def __init__(self, latent_dim=256, embed_dim=96):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                JSCCUpBlock(latent_dim, embed_dim, stride=2, num_res_blocks=3),
                JSCCUpBlock(embed_dim, embed_dim, stride=2, num_res_blocks=3),
                JSCCUpBlock(embed_dim, embed_dim, stride=2, num_res_blocks=3),
            ]
        )

    def forward(self, x):
        if x.dim() == 4:
            B, T, L, C = x.shape
            h = int(math.sqrt(L))
            if h * h != L:
                raise ValueError(f"MVSC_JSCC_Decoder expects square token map, but got L={L}.")

            x = x.view(B * T, h, h, C).permute(0, 3, 1, 2).contiguous()
            for block in self.blocks:
                x = block(x)

            _, c, h_out, w_out = x.shape
            x = x.permute(0, 2, 3, 1).contiguous().view(B, T, h_out * w_out, c)
            return x

        if x.dim() == 5:
            B, T, V, L, C = x.shape
            h = int(math.sqrt(L))
            if h * h != L:
                raise ValueError(f"MVSC_JSCC_Decoder expects square token map, but got L={L}.")

            x = x.view(B * T * V, h, h, C).permute(0, 3, 1, 2).contiguous()
            for block in self.blocks:
                x = block(x)

            _, c, h_out, w_out = x.shape
            x = x.permute(0, 2, 3, 1).contiguous().view(B, T, V, h_out * w_out, c)
            return x

        raise ValueError(f"MVSC_JSCC_Decoder expects 4D/5D tokens, got shape={tuple(x.shape)}")
    
class MVSC_Commonality_Decoder(nn.Module):
    """
    Recover per-view semantic tokens with spatio-temporal + cross-view fusion.
    Input:  [B, T, V, L, C]
    Output: [B, T, V, L, C]
    """
    def __init__(self, dim, input_resolution, num_views=4, depth=2, num_heads=4):
        super().__init__()
        self.num_views = num_views
        self.temporal_layers = nn.ModuleList(
            [
                BasicLayer3D_Dec(
                    dim=dim,
                    input_resolution=input_resolution,
                    depth=depth,
                    num_heads=num_heads,
                    window_size=8,
                )
                for _ in range(3)
            ]
        )
        self.view_layers = nn.ModuleList(
            [
                ViewFusionBlock(dim=dim, num_heads=num_heads)
                for _ in range(3)
            ]
        )

    def forward(self, x):
        # x: [B, T, V, L, C]
        if x.dim() != 5:
            raise ValueError(f"MVSC_Commonality_Decoder expects [B,T,V,L,C], got shape={tuple(x.shape)}")

        B, T, V, L, C = x.shape
        for temporal_layer, view_layer in zip(self.temporal_layers, self.view_layers):
            # Temporal modeling per view.
            x = x.view(B * V, T, L, C)
            x = temporal_layer(x)
            x = x.view(B, T, V, L, C)

            # Cross-view fusion.
            x = view_layer(x)
        return x


class MVSC_Individual_Decoder(nn.Module):
    """
    Recover RGB frames from per-view tokens.
    Input:  [B, T, V, L, C]
    Output: [B, T, V, 3, H, W]
    """
    def __init__(
        self,
        img_size=256,
        patch_size=4,
        out_chans=3,
        embed_dim=96,
        input_resolution=None,
        num_upsample_stages=None,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.out_chans = out_chans
        self.embed_dim = embed_dim

        self.target_resolution = (img_size // patch_size, img_size // patch_size)
        self.grid_h, self.grid_w = self.target_resolution
        self.num_patches = self.grid_h * self.grid_w

        if input_resolution is None:
            input_resolution = self.target_resolution
        self.input_resolution = input_resolution

        if num_upsample_stages is None:
            inferred_stages = 0
            probe_h, probe_w = self.input_resolution
            while (
                inferred_stages < 2
                and probe_h < self.target_resolution[0]
                and probe_w < self.target_resolution[1]
            ):
                inferred_stages += 1
                probe_h *= 2
                probe_w *= 2
            num_upsample_stages = inferred_stages

        self.num_upsample_stages = max(0, min(2, int(num_upsample_stages)))
        upsample_flags = [
            False,
            self.num_upsample_stages >= 1,
            self.num_upsample_stages >= 2,
        ]

        self.reconstruct_layers = nn.ModuleList()
        cur_resolution = self.input_resolution
        for do_upsample in upsample_flags:
            layer = BasicLayer(
                dim=embed_dim,
                out_dim=embed_dim,
                input_resolution=cur_resolution,
                depth=2,
                num_heads=3,
                window_size=8,
                upsample=PatchReverseMerging if do_upsample else None,
            )
            self.reconstruct_layers.append(layer)
            if do_upsample:
                cur_resolution = (cur_resolution[0] * 2, cur_resolution[1] * 2)

        self.output_resolution = cur_resolution
        if self.output_resolution != self.target_resolution:
            raise ValueError(
                "MVSC_Individual_Decoder output resolution mismatch: "
                f"got {self.output_resolution}, expected {self.target_resolution}."
            )

        self.patch_dim = patch_size * patch_size * out_chans

        self.token_refine = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.head = nn.Linear(embed_dim, self.patch_dim)

    def _tokens_to_image(self, x):
        # x: [N, L, patch_dim], where L = grid_h * grid_w
        N, L, D = x.shape
        gh, gw = self.grid_h, self.grid_w
        p = self.patch_size
        c = self.out_chans
        assert L == gh * gw, f"Unexpected number of patches: {L} vs {gh * gw}"
        assert D == p * p * c, f"Unexpected patch dimension: {D} vs {p * p * c}"

        x = x.view(N, gh, gw, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(N, c, gh * p, gw * p)
        return x

    def forward(self, x):
        # x: [B, T, V, L, C]
        B, T, V, L, C = x.shape
        expected_l = self.input_resolution[0] * self.input_resolution[1]
        if L != expected_l:
            raise ValueError(
                f"MVSC_Individual_Decoder expected token length {expected_l}, got {L}."
            )

        x = x.view(B * T * V, L, C)

        for layer in self.reconstruct_layers:
            x = layer(x)

        x = self.token_refine(x)
        x = self.head(x)
        x = self._tokens_to_image(x)
        x = x.view(B, T, V, self.out_chans, self.img_size, self.img_size)
        return x

class MVSCDecoder(nn.Module):
    """
    Full MVSC decoder.
    Input:  [B, T, L, latent_dim]
    Output: [B, T, V, 3, H, W]
    """
    def __init__(
        self,
        img_size=256,
        patch_size=4,
        out_chans=3,
        embed_dim=96,
        latent_dim=256,
        num_views=4,
        common_depth=2,
        common_heads=4,
    ):
        super().__init__()
        input_resolution = (img_size // patch_size, img_size // patch_size)

        self.jscc = MVSC_JSCC_Decoder(latent_dim=latent_dim, embed_dim=embed_dim)
        self.common = MVSC_Commonality_Decoder(
            dim=embed_dim,
            input_resolution=input_resolution,
            num_views=num_views,
            depth=common_depth,
            num_heads=common_heads,
        )
        self.individual = MVSC_Individual_Decoder(
            img_size=img_size,
            patch_size=patch_size,
            out_chans=out_chans,
            embed_dim=embed_dim,
        )

    def forward(self, x):
        x = self.jscc(x)       # [B, T, L, C]
        x = self.common(x)     # [B, T, V, L, C]
        x = self.individual(x) # [B, T, V, 3, H, W]
        return x