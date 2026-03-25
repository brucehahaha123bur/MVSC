from net.modules import *
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from net.encoder3 import (
    SwinTransformerBlock,
    SwinTransformerBlock3D,
    AdaptiveModulator,
    ConvResidualBlock2D,
    restore_d_to_tv,
)


def _infer_hw_from_l(l_value: int):
    h = int(math.sqrt(l_value))
    w = h
    if h * w != l_value:
        raise ValueError(
            f"Cannot infer non-square token resolution from L={l_value}. "
            f"Please pass hw=(H,W) explicitly."
        )
    return h, w


class BasicLayer(nn.Module):

    def __init__(self, dim, out_dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 norm_layer=nn.LayerNorm, upsample=None,):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 norm_layer=norm_layer)
            for i in range(depth)])

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
        if self.upsample is not None:
            flops += self.upsample.flops()
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
        if C is not None:
            self.head_list = nn.Linear(C, embed_dims[0])
        self.apply(self._init_weights)
        self.hidden_dim = int(self.embed_dims[0] * 1.5)
        self.layer_num = layer_num = 7
        if model != "SwinJSCC_w/_RA":
            self.bm_list = nn.ModuleList()
            self.sm_list = nn.ModuleList()
            self.sm_list.append(nn.Linear(self.embed_dims[0], self.hidden_dim))
            for i in range(layer_num):
                outdim = self.embed_dims[0] if i == layer_num - 1 else self.hidden_dim
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
            device = x.device
            x = self.head_list(x)
            snr_cuda = torch.tensor(snr, dtype=torch.float, device=device)
            snr_batch = snr_cuda.unsqueeze(0).expand(B, -1)
            for i in range(self.layer_num):
                temp = self.sm_list[i](x.detach()) if i == 0 else self.sm_list[i](temp)
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
            device = x.device
            snr_cuda = torch.tensor(snr, dtype=torch.float, device=device)
            snr_batch = snr_cuda.unsqueeze(0).expand(B, -1)
            for i in range(self.layer_num):
                temp = self.sm_list[i](x.detach()) if i == 0 else self.sm_list[i](temp)
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
            layer.update_resolution(H * (2 ** i_layer), W * (2 ** i_layer))


class BasicLayer3D_Dec(nn.Module):
    """
    Stack of decoder-side 3D blocks.
    Input: [B, D, L, C]
    Output: [B, D, L, C]
    """
    def __init__(self, dim, input_resolution, depth, num_heads, window_size=8):
        super().__init__()
        self.input_resolution = input_resolution

        if isinstance(window_size, int):
            window_size_3d = (window_size, window_size, window_size)
        elif len(window_size) == 2:
            window_size_3d = (window_size[0], window_size[0], window_size[1])
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

    def update_resolution(self, H, W):
        self.input_resolution = (H, W)
        for blk in self.blocks:
            blk.input_resolution = (H, W)


class ViewPatchExpand(nn.Module):
    def __init__(self, dim, out_dim=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.out_dim = dim if out_dim is None else out_dim
        self.expand = nn.Linear(dim, self.out_dim * 2, bias=False)
        self.norm = norm_layer(self.out_dim)

    def forward(self, x):
        if x.dim() != 5:
            raise ValueError(f"ViewPatchExpand expects [B,T,V,L,C], got shape={tuple(x.shape)}")

        B, T, V, L, C = x.shape
        x = self.expand(x)
        x = x.view(B, T, V, L, 2, self.out_dim)
        x = x.permute(0, 1, 2, 4, 3, 5).contiguous()
        x = x.view(B, T, V * 2, L, self.out_dim)
        x = self.norm(x)
        return x


class TemporalPatchExpand(nn.Module):
    def __init__(self, dim, out_dim=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.out_dim = dim if out_dim is None else out_dim
        self.expand = nn.Linear(dim, self.out_dim * 2, bias=False)
        self.norm = norm_layer(self.out_dim)

    def forward(self, x):
        if x.dim() != 5:
            raise ValueError(f"TemporalPatchExpand expects [B,T,V,L,C], got shape={tuple(x.shape)}")

        B, T, V, L, C = x.shape
        x = self.expand(x)
        x = x.view(B, T, V, L, 2, self.out_dim)
        x = x.permute(0, 1, 4, 2, 3, 5).contiguous()
        x = x.view(B, T * 2, V, L, self.out_dim)
        x = self.norm(x)
        return x


class JSTemporalExpand(nn.Module):
    def __init__(self, dim, out_dim=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.out_dim = dim if out_dim is None else out_dim
        self.expand = nn.Linear(dim, self.out_dim * 2, bias=False)
        self.norm = norm_layer(self.out_dim)

    def forward(self, x):
        if x.dim() != 5:
            raise ValueError(f"JSTemporalExpand expects [B,T,V,L,C], got shape={tuple(x.shape)}")

        B, T, V, L, C = x.shape
        x = self.expand(x)
        x = x.view(B, T, V, L, 2, self.out_dim)
        x = x.permute(0, 1, 4, 2, 3, 5).contiguous()
        x = x.view(B, T * 2, V, L, self.out_dim)
        x = self.norm(x)
        return x


class JSCCUpBlock(nn.Module):
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
    Decode channel latent tokens back to compressed semantic tokens.

    Input:
      [B, D, L, latent_dim]

    Output:
      [B, Tc, Vc, L_out, 320]
    """
    def __init__(self, latent_dim=128, embed_dim=320, compressed_num_views=2, temporal_upsample_in_jscc=False):
        super().__init__()
        self.compressed_num_views = compressed_num_views
        self.temporal_upsample_in_jscc = temporal_upsample_in_jscc

        # mirror of encoder3 JSCC:
        # 320 -> 256 -> 192 -> 128   with 32->16->8
        # decoder should be:
        # 128 -> 192 -> 256 -> 320   with 8->16->32
        mid2 = 192
        mid1 = 256

        self.blocks = nn.ModuleList(
            [
                JSCCUpBlock(latent_dim, mid2, stride=2, num_res_blocks=3),  # 8 -> 16, 128 -> 192
                JSCCUpBlock(mid2, mid1, stride=2, num_res_blocks=3),        # 16 -> 32, 192 -> 256
                JSCCUpBlock(mid1, embed_dim, stride=1, num_res_blocks=3),   # 32 -> 32, 256 -> 320
            ]
        )

        self.temporal_expand = JSTemporalExpand(dim=embed_dim, out_dim=embed_dim)

    def forward(self, x, hw=None, return_hw=False):
        if x.dim() != 4:
            raise ValueError(f"MVSC_JSCC_Decoder expects [B,D,L,C], got shape={tuple(x.shape)}")

        B, D, L, C = x.shape
        if hw is None:
            hw = _infer_hw_from_l(L)
        H, W = hw
        if H * W != L:
            raise ValueError(f"MVSC_JSCC_Decoder hw mismatch: hw={hw}, but L={L}")

        if D % self.compressed_num_views != 0:
            raise ValueError(
                f"MVSC_JSCC_Decoder depth mismatch: D={D} is not divisible by compressed_num_views={self.compressed_num_views}."
            )

        x = x.view(B * D, H, W, C).permute(0, 3, 1, 2).contiguous()
        for block in self.blocks:
            x = block(x)

        _, c, h_out, w_out = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(B, D, h_out * w_out, c)

        T_comp = D // self.compressed_num_views
        V_comp = self.compressed_num_views
        x = restore_d_to_tv(x, T_comp, V_comp)

        if self.temporal_upsample_in_jscc:
            x = self.temporal_expand(x)

        if return_hw:
            return x, (h_out, w_out)
        return x


class MVSC_Commonality_Decoder(nn.Module):
    """
    Recover per-view semantic tokens using spatio-temporal decoding.
    Input:  [B, T', V', L, C]
    Output: [B, T, V, L, C]
    """
    def __init__(
        self,
        dim,
        input_resolution,
        num_views=4,
        compressed_num_views=None,
        depths=(1, 2),
        num_heads=(10, 8),
        out_dim=192,
    ):
        super().__init__()

        self.num_views = num_views
        self.compressed_num_views = (
            compressed_num_views if compressed_num_views is not None else max(1, num_views // 2)
        )
        if self.compressed_num_views < 1:
            raise ValueError(f"compressed_num_views must be >= 1, got {self.compressed_num_views}")

        if isinstance(depths, int):
            depths = (depths, depths)
        if len(depths) != 2:
            raise ValueError(f"MVSC_Commonality_Decoder expects 2 stage depths, got depths={depths}")

        if isinstance(num_heads, int):
            num_heads = (num_heads, num_heads)
        if len(num_heads) != 2:
            raise ValueError(f"MVSC_Commonality_Decoder expects 2 stage head counts, got num_heads={num_heads}")

        if self.num_views % self.compressed_num_views != 0:
            raise ValueError(
                f"num_views={self.num_views} must be divisible by compressed_num_views={self.compressed_num_views}"
            )

        self.in_dim = dim
        self.stage2_dim = 256
        self.out_dim = out_dim

        self.stage1_swin = BasicLayer3D_Dec(
            dim=self.in_dim,
            input_resolution=input_resolution,
            depth=depths[0],
            num_heads=num_heads[0],
            window_size=4,
        )

        self.view_expand = ViewPatchExpand(dim=self.in_dim, out_dim=self.in_dim)

        self.stage2_expand = TemporalPatchExpand(dim=self.in_dim, out_dim=self.stage2_dim)

        self.stage2_swin = BasicLayer3D_Dec(
            dim=self.stage2_dim,
            input_resolution=input_resolution,
            depth=depths[1],
            num_heads=num_heads[1],
            window_size=4,
        )

        self.out_proj = nn.Linear(self.stage2_dim, self.out_dim)

    def forward(self, x, hw=None, return_hw=False):
        if x.dim() != 5:
            raise ValueError(f"MVSC_Commonality_Decoder expects [B,T,V,L,C], got shape={tuple(x.shape)}")

        if x.shape[2] != self.compressed_num_views:
            raise ValueError(
                f"MVSC_Commonality_Decoder expected compressed view count {self.compressed_num_views}, got V={x.shape[2]}"
            )

        _, _, _, L, _ = x.shape
        if hw is None:
            hw = _infer_hw_from_l(L)
        H, W = hw
        if H * W != L:
            raise ValueError(f"MVSC_Commonality_Decoder hw mismatch: hw={hw}, but L={L}")

        self.stage1_swin.update_resolution(H, W)
        self.stage2_swin.update_resolution(H, W)

        B, T, V, L, C = x.shape
        x = x.contiguous().view(B, T * V, L, C)
        x = self.stage1_swin(x)
        x = restore_d_to_tv(x, T, V)

        x = self.view_expand(x)
        x = self.stage2_expand(x)

        B, T, V, L, C = x.shape
        x = x.contiguous().view(B, T * V, L, C)
        x = self.stage2_swin(x)
        x = restore_d_to_tv(x, T, V)

        x = self.out_proj(x)
        if return_hw:
            return x, hw
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
        patch_size=8,
        out_chans=3,
        embed_dim=192,
        input_resolution=None,
        num_upsample_stages=None,
        depths=(1, 2, 1),
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.out_chans = out_chans
        self.embed_dim = embed_dim

        if isinstance(depths, int):
            depths = (depths, depths, depths)
        if len(depths) != 3:
            raise ValueError(f"MVSC_Individual_Decoder expects 3 stage depths, got depths={depths}")

        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        self.default_img_size = img_size

        self.target_resolution = (img_size[0] // patch_size, img_size[1] // patch_size)

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
        for stage_idx, do_upsample in enumerate(upsample_flags):
            layer = BasicLayer(
                dim=embed_dim,
                out_dim=embed_dim,
                input_resolution=cur_resolution,
                depth=depths[stage_idx],
                num_heads=(3 if stage_idx == 0 else 6 if stage_idx == 1 else 8),
                window_size=8,
                upsample=PatchReverseMerging if do_upsample else None,
            )
            self.reconstruct_layers.append(layer)
            if do_upsample:
                cur_resolution = (cur_resolution[0] * 2, cur_resolution[1] * 2)

        self.output_resolution = cur_resolution
        self.patch_dim = patch_size * patch_size * out_chans

        self.token_refine = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.head = nn.Linear(embed_dim, self.patch_dim)

    def _tokens_to_image(self, x, grid_hw):
        N, L, D = x.shape
        gh, gw = grid_hw
        p = self.patch_size
        c = self.out_chans
        assert L == gh * gw, f"Unexpected number of patches: {L} vs {gh * gw}"
        assert D == p * p * c, f"Unexpected patch dimension: {D} vs {p * p * c}"

        x = x.view(N, gh, gw, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(N, c, gh * p, gw * p)
        return x

    def forward(self, x, hw=None, out_hw=None):
        B, T, V, L, C = x.shape
        if hw is None:
            hw = _infer_hw_from_l(L)
        H_in, W_in = hw
        expected_l = H_in * W_in
        if L != expected_l:
            raise ValueError(
                f"MVSC_Individual_Decoder expected token length {expected_l}, got {L}."
            )

        cur_h, cur_w = H_in, W_in
        for layer in self.reconstruct_layers:
            layer.update_resolution(cur_h, cur_w)
            if layer.upsample is not None:
                cur_h, cur_w = cur_h * 2, cur_w * 2

        x = x.contiguous().view(B * T * V, L, C)
        for layer in self.reconstruct_layers:
            x = layer(x)

        x = self.token_refine(x)
        x = self.head(x)
        x = self._tokens_to_image(x, (cur_h, cur_w))

        if out_hw is not None:
            out_h, out_w = out_hw
            if x.shape[-2:] != (out_h, out_w):
                x = F.interpolate(x, size=(out_h, out_w), mode="bilinear", align_corners=False)

        out_h, out_w = x.shape[-2:]
        x = x.contiguous().view(B, T, V, self.out_chans, out_h, out_w)
        return x


class MVSCDecoder(nn.Module):
    """
    Full MVSC decoder.
    Input:  [B, D, L, latent_dim]
    Output: [B, T, V, 3, H, W]
    """
    def __init__(
        self,
        img_size=256,
        patch_size=8,
        out_chans=3,
        common_dim=320,
        individual_dim=192,
        latent_dim=128,
        num_views=4,
        compressed_num_views=None,
        common_depths=(1, 2),
        common_heads=(10, 8),
        individual_depths=(1, 2, 1),
    ):
        super().__init__()
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        common_input_resolution = (img_size[0] // 8, img_size[1] // 8)
        individual_input_resolution = (img_size[0] // 8, img_size[1] // 8)

        if compressed_num_views is None:
            compressed_num_views = max(1, num_views // 2)

        self.jscc = MVSC_JSCC_Decoder(
            latent_dim=latent_dim,
            embed_dim=common_dim,
            compressed_num_views=compressed_num_views,
            temporal_upsample_in_jscc=False,
        )
        self.common = MVSC_Commonality_Decoder(
            dim=common_dim,
            input_resolution=common_input_resolution,
            num_views=num_views,
            compressed_num_views=compressed_num_views,
            depths=common_depths,
            num_heads=common_heads,
            out_dim=individual_dim,
        )
        self.individual = MVSC_Individual_Decoder(
            img_size=img_size,
            patch_size=patch_size,
            out_chans=out_chans,
            embed_dim=individual_dim,
            input_resolution=individual_input_resolution,
            depths=individual_depths,
        )

    def forward(self, x, hw=None, out_hw=None):
        x, hw = self.jscc(x, hw=hw, return_hw=True)
        x, hw = self.common(x, hw=hw, return_hw=True)
        x = self.individual(x, hw=hw, out_hw=out_hw)
        return x


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    B = 2
    T_comp = 2
    V_comp = 2
    D = T_comp * V_comp
    latent_dim = 128
    common_dim = 320
    individual_dim = 192

    def check_shape(name, x, expected_shape):
        actual_shape = tuple(x.shape)
        expected_shape = tuple(expected_shape)
        ok = actual_shape == expected_shape
        status = "OK" if ok else "FAIL"
        print(f"[{status}] {name}: actual={actual_shape}, expected={expected_shape}")
        if not ok:
            raise AssertionError(f"{name} shape mismatch: actual={actual_shape}, expected={expected_shape}")

    L_jscc_in = 64

    x_jscc = torch.randn(B, D, L_jscc_in, latent_dim).to(device)

    jscc_dec = MVSC_JSCC_Decoder(
        latent_dim=latent_dim,
        embed_dim=common_dim,
        compressed_num_views=V_comp,
        temporal_upsample_in_jscc=False,
    ).to(device)

    with torch.no_grad():
        y_jscc, hw_jscc = jscc_dec(x_jscc, hw=(8, 8), return_hw=True)

    print("[JSCC Decoder]")
    print("input :", x_jscc.shape)
    print("output:", y_jscc.shape, "hw:", hw_jscc)
    check_shape("JSCC Decoder", y_jscc, (B, 2, 2, 1024, common_dim))

    x_common = torch.randn(B, 2, 2, 1024, common_dim).to(device)
    common_dec = MVSC_Commonality_Decoder(
        dim=common_dim,
        input_resolution=(32, 32),
        num_views=4,
        compressed_num_views=2,
        depths=(1, 2),
        num_heads=(10, 8),
        out_dim=individual_dim,
    ).to(device)

    with torch.no_grad():
        y_common, hw_common = common_dec(x_common, hw=(32, 32), return_hw=True)

    print("\n[Commonality Decoder]")
    print("input :", x_common.shape)
    print("output:", y_common.shape, "hw:", hw_common)
    check_shape("Commonality Decoder", y_common, (B, 4, 4, 1024, individual_dim))

    x_ind = torch.randn(B, 4, 4, 1024, individual_dim).to(device)
    ind_dec = MVSC_Individual_Decoder(
        img_size=256,
        patch_size=8,
        out_chans=3,
        embed_dim=individual_dim,
        input_resolution=(32, 32),
        num_upsample_stages=0,
        depths=(1, 2, 1),
    ).to(device)

    with torch.no_grad():
        y_ind = ind_dec(x_ind, hw=(32, 32), out_hw=(256, 256))

    print("\n[Individual Decoder]")
    print("input :", x_ind.shape)
    print("output:", y_ind.shape)
    check_shape("Individual Decoder", y_ind, (B, 4, 4, 3, 256, 256))

    full_dec = MVSCDecoder(
        img_size=256,
        patch_size=8,
        out_chans=3,
        common_dim=common_dim,
        individual_dim=individual_dim,
        latent_dim=latent_dim,
        num_views=4,
        compressed_num_views=2,
        common_depths=(1, 2),
        common_heads=(10, 8),
        individual_depths=(1, 2, 1),
    ).to(device)

    with torch.no_grad():
        y_full = full_dec(x_jscc, hw=(8, 8), out_hw=(256, 256))

    print("\n[Full MVSC Decoder]")
    print("input :", x_jscc.shape)
    print("output:", y_full.shape)
    check_shape("Full MVSC Decoder", y_full, (B, 4, 4, 3, 256, 256))

    print("\nAll decoder self-tests passed.")
