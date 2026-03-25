from net.modules import *
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenPatchEmbedding(nn.Module):
    """
    Wrap PatchEmbed so it can serve as the first Patch Processing stage in the
    Individual Semantic Encoder.

    Input:  [B, C, H, W]
    Output: [B, L, C_embed]
    """

    def __init__(self, img_size=256, patch_size=2, in_chans=3, embed_dim=96):
        super().__init__()
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

    def forward(self, x):
        return self.patch_embed(x)


class AdaptiveModulator(nn.Module):
    """Map channel condition (e.g., SNR) to feature-wise modulation weights."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(self, snr):
        if snr.dim() == 1:
            snr = snr.unsqueeze(-1)
        return self.net(snr)


def _infer_hw_from_l(l_value: int):
    h = int(math.sqrt(l_value))
    w = h
    if h * w != l_value:
        raise ValueError(
            f"Cannot infer non-square token resolution from L={l_value}. "
            f"Please pass hw=(H,W) explicitly."
        )
    return h, w

class SwinTransformerBlock(nn.Module):

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.base_shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
        )

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer
        )

        self._mask_cache = {}

    @staticmethod
    def _make_slices(length, window, shift):
        if shift == 0:
            return (slice(0, length),)
        return (
            slice(0, -window),
            slice(-window, -shift),
            slice(-shift, None),
        )

    def _compute_attn_mask(self, Hp, Wp, shift_size, device):
        if shift_size == 0:
            return None

        key = (Hp, Wp, shift_size, device.type, device.index)
        if key in self._mask_cache:
            return self._mask_cache[key]

        img_mask = torch.zeros((1, Hp, Wp, 1), device=device)

        h_slices = self._make_slices(Hp, self.window_size, shift_size)
        w_slices = self._make_slices(Wp, self.window_size, shift_size)

        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        self._mask_cache[key] = attn_mask
        return attn_mask

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, f"input feature has wrong size: {L} vs {H*W}"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size

        if pad_h > 0 or pad_w > 0:
            x = x.permute(0, 3, 1, 2).contiguous()   # [B,C,H,W]
            x = F.pad(x, (0, pad_w, 0, pad_h))
            x = x.permute(0, 2, 3, 1).contiguous()   # [B,H,W,C]

        Hp, Wp = H + pad_h, W + pad_w

        shift_size = self.shift_size if (Hp > self.window_size and Wp > self.window_size) else 0

        if shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(1, 2))
        else:
            shifted_x = x

        attn_mask = self._compute_attn_mask(Hp, Wp, shift_size, x.device)

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        attn_windows = self.attn(x_windows, add_token=False, mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        if shift_size > 0:
            x = torch.roll(shifted_x, shifts=(shift_size, shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_h > 0 or pad_w > 0:
            x = x[:, :H, :W, :]

        x = x.view(B, H * W, C)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        flops += self.dim * H * W
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        flops += self.dim * H * W
        return flops

    def update_mask(self):
        self._mask_cache = {}


class BasicLayer(nn.Module):
    def __init__(self, dim, out_dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm,
                 downsample=None):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.downsample = None
        self.channel_proj = None

        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, out_dim=out_dim, norm_layer=norm_layer)
            self.block_resolution = (input_resolution[0] // 2, input_resolution[1] // 2)
        else:
            self.block_resolution = input_resolution
            if dim != out_dim:
                self.channel_proj = nn.Linear(dim, out_dim)

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=out_dim,
                                 input_resolution=self.block_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 norm_layer=norm_layer)
            for i in range(depth)])

    def forward(self, x):
        if self.downsample is not None:
            x = self.downsample(x)
        elif self.channel_proj is not None:
            x = self.channel_proj(x)
        for _, blk in enumerate(self.blocks):
            x = blk(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        if self.channel_proj is not None:
            H, W = self.block_resolution
            flops += H * W * self.channel_proj.in_features * self.channel_proj.out_features
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

    def update_resolution(self, H, W):
        self.input_resolution = (H, W)
        if self.downsample is not None:
            self.block_resolution = (H // 2, W // 2)
            self.downsample.input_resolution = (H, W)
        else:
            self.block_resolution = (H, W)
        for _, blk in enumerate(self.blocks):
            blk.input_resolution = self.block_resolution
            blk.update_mask()


class SwinJSCC_Encoder(nn.Module):
    def __init__(self, model, img_size, patch_size, in_chans, embed_dims, depths, num_heads, C,
                 window_size=4, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 norm_layer=nn.LayerNorm, patch_norm=True):
        super().__init__()

        self.model = model
        self.embed_dims = embed_dims
        self.num_layers = len(depths)
        self.patch_size = patch_size
        self.H = img_size[0]
        self.W = img_size[1]
        self.patches_resolution = (self.H // patch_size, self.W // patch_size)

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dims[0],
            norm_layer=norm_layer if patch_norm else None,
        )

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            input_resolution = (
                self.patches_resolution[0] // (2 ** i_layer),
                self.patches_resolution[1] // (2 ** i_layer),
            )
            if i_layer < self.num_layers - 1:
                out_dim = int(embed_dims[i_layer + 1])
                downsample = PatchMerging
            else:
                out_dim = int(embed_dims[i_layer])
                downsample = None

            layer = BasicLayer(
                dim=int(embed_dims[i_layer]),
                out_dim=out_dim,
                input_resolution=input_resolution,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                norm_layer=norm_layer,
                downsample=downsample,
            )
            self.layers.append(layer)

        self.channel_number = C
        if C is not None:
            self.head_list = nn.Linear(embed_dims[-1], C)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, snr, rate, model):
        x = self.patch_embed(x)
        for layer in self.layers:
            x = layer(x)

        if model == 'SwinJSCC_w/o_SAandRA' or model == 'SwinJSCC_w/_SA':
            x = self.head_list(x)
            return x

        if model == 'SwinJSCC_w/_RA' or model == 'SwinJSCC_w/_SAandRA':
            if rate is None:
                raise ValueError('rate must be provided for RA models')
            max_channels = x.size(-1)
            active_channels = max(1, min(int(rate), max_channels))
            mask = torch.zeros_like(x)
            mask[:, :, :active_channels] = 1.0
            x = x * mask
            return x, mask

        raise ValueError(f'Unknown model type: {model}')

    def update_resolution(self, H, W):
        self.H = H
        self.W = W
        self.patches_resolution = (H // self.patch_size, W // self.patch_size)
        for i_layer, layer in enumerate(self.layers):
            input_resolution = (
                self.patches_resolution[0] // (2 ** i_layer),
                self.patches_resolution[1] // (2 ** i_layer),
            )
            layer.update_resolution(input_resolution[0], input_resolution[1])

    def flops(self):
        flops = self.patch_embed.flops()
        for layer in self.layers:
            flops += layer.flops()
        return flops


class SwinTransformerBlock3D(nn.Module):
    """3D Swin block with shifted window attention over a depth axis D and spatial axes [H, W]."""

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=(2, 8, 8),
        shift_size=(0, 0, 0),
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads

        if isinstance(window_size, int):
            window_size = (window_size, window_size, window_size)
        elif len(window_size) == 2:
            window_size = (window_size[0], window_size[0], window_size[1])
        self.window_size = tuple(window_size)

        if isinstance(shift_size, int):
            shift_size = (shift_size, shift_size, shift_size)
        elif len(shift_size) == 2:
            shift_size = (shift_size[0], shift_size[0], shift_size[1])
        self.shift_size = tuple(shift_size)

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3D(
            dim=dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
        )

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU)

        self._mask_cache = {}

    @staticmethod
    def _make_slices(length, window, shift):
        if shift == 0:
            return (slice(0, length),)
        return (
            slice(0, -window),
            slice(-window, -shift),
            slice(-shift, None),
        )

    def _compute_attn_mask(self, Dp, Hp, Wp, shift_size, device):
        if shift_size == (0, 0, 0):
            return None

        key = (Dp, Hp, Wp, shift_size, device.type, device.index)
        if key in self._mask_cache:
            return self._mask_cache[key]

        img_mask = torch.zeros((1, Dp, Hp, Wp, 1), device=device)
        d_slices = self._make_slices(Dp, self.window_size[0], shift_size[0])
        h_slices = self._make_slices(Hp, self.window_size[1], shift_size[1])
        w_slices = self._make_slices(Wp, self.window_size[2], shift_size[2])

        cnt = 0
        for d in d_slices:
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, d, h, w, :] = cnt
                    cnt += 1

        mask_windows = window_partition_3d(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2])
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        self._mask_cache[key] = attn_mask
        return attn_mask

    def forward(self, x):
        H, W = self.input_resolution
        B, T, L, C = x.shape
        assert L == H * W, f"input token size mismatch: {L} vs {H*W}"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, T, H, W, C)

        pad_d = (self.window_size[0] - T % self.window_size[0]) % self.window_size[0]
        pad_h = (self.window_size[1] - H % self.window_size[1]) % self.window_size[1]
        pad_w = (self.window_size[2] - W % self.window_size[2]) % self.window_size[2]

        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            x = x.permute(0, 4, 1, 2, 3).contiguous()
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))
            x = x.permute(0, 2, 3, 4, 1).contiguous()

        Tp, Hp, Wp = T + pad_d, H + pad_h, W + pad_w
        shift_size = (
            self.shift_size[0] if Tp > self.window_size[0] else 0,
            self.shift_size[1] if Hp > self.window_size[1] else 0,
            self.shift_size[2] if Wp > self.window_size[2] else 0,
        )

        if shift_size != (0, 0, 0):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
        else:
            shifted_x = x

        attn_mask = self._compute_attn_mask(Tp, Hp, Wp, shift_size, x.device)

        x_windows = window_partition_3d(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2], C)
        attn_windows = self.attn(x_windows, mask=attn_mask)

        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], self.window_size[2], C)
        shifted_x = window_reverse_3d(attn_windows, self.window_size, Tp, Hp, Wp)

        if shift_size != (0, 0, 0):
            x = torch.roll(shifted_x, shifts=shift_size, dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            x = x[:, :T, :H, :W, :]

        x = x.view(B, T, H * W, C)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class BasicLayer3D(nn.Module):
    """
    Stack of 3D Swin blocks.
    Input: [B, D, L, C], where D is the flattened depth axis formed from
    temporal and view dimensions when needed.
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


class ViewFusionBlock(nn.Module):
    """Cross-view self-attention block on [B, T, V, L, C]."""

    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        B, T, V, L, C = x.shape
        y = x.permute(0, 1, 3, 2, 4).contiguous().view(B * T * L, V, C)

        y_norm = self.norm1(y)
        attn_out, _ = self.attn(y_norm, y_norm, y_norm, need_weights=False)
        y = y + attn_out
        y = y + self.mlp(self.norm2(y))

        y = y.view(B, T, L, V, C).permute(0, 1, 3, 2, 4).contiguous()
        return y


class ConvResidualBlock2D(nn.Module):
    """Residual 2D conv block used by JSCC encoder/decoder stacks."""

    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        return x + self.body(x)


class JSCCDownBlock(nn.Module):
    """Downsampling conv block followed by residual refinement."""

    def __init__(self, in_ch, out_ch, stride=2, num_res_blocks=3):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1),
            nn.GELU(),
        )
        self.res_blocks = nn.Sequential(*[ConvResidualBlock2D(out_ch) for _ in range(num_res_blocks)])

    def forward(self, x):
        x = self.down(x)
        x = self.res_blocks(x)
        return x


class ConvResidualBlock3D(nn.Module):
    """Residual 3D conv block used by JSCC encoder stacks over [T, H, W]."""

    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv3d(channels, channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        return x + self.body(x)


class JSCCDownBlock3D(nn.Module):
    """Downsampling 3D conv block followed by residual refinement over [T, H, W]."""

    def __init__(self, in_ch, out_ch, stride=(2, 2, 2), num_res_blocks=3):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1),
            nn.GELU(),
        )
        self.res_blocks = nn.Sequential(*[ConvResidualBlock3D(out_ch) for _ in range(num_res_blocks)])

    def forward(self, x):
        x = self.down(x)
        x = self.res_blocks(x)
        return x


class MVSC_Individual_Encoder(nn.Module):
    """
    Individual semantic encoder (2D Swin).
    Supports dynamic input resolution as long as H and W are divisible by 8.

    Input:  [B, T, V, 3, H, W]
    Output: [B, T, V, L, C]
    Optional: return spatial token resolution hw=(H/8, W/8)
    """

    def __init__(self, img_size=256, patch_size=2, in_chans=3, embed_dim=96, depths=(1, 2, 1)):
        super().__init__()

        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        if isinstance(depths, int):
            depths = (depths, depths, depths)
        if len(depths) != 3:
            raise ValueError(f"MVSC_Individual_Encoder expects 3 stage depths, got depths={depths}")
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        c1 = 96
        c2 = 144
        c3 = 192

        stage1_resolution = (img_size[0] // patch_size, img_size[1] // patch_size)
        stage2_resolution = (stage1_resolution[0] // 2, stage1_resolution[1] // 2)
        stage3_resolution = (stage2_resolution[0] // 2, stage2_resolution[1] // 2)

        self.patch_processing = nn.ModuleList(
            [
                TokenPatchEmbedding(
                    img_size=img_size,
                    patch_size=patch_size,
                    in_chans=in_chans,
                    embed_dim=c1,
                ),
                PatchMerging(stage1_resolution, dim=c1, out_dim=c2, norm_layer=nn.LayerNorm),
                PatchMerging(stage2_resolution, dim=c2, out_dim=c3, norm_layer=nn.LayerNorm),
            ]
        )

        self.swin_layers = nn.ModuleList(
            [
                BasicLayer(
                    dim=c1,
                    out_dim=c1,
                    input_resolution=stage1_resolution,
                    depth=depths[0],
                    num_heads=3,
                    window_size=8,
                    downsample=None,
                ),
                BasicLayer(
                    dim=c2,
                    out_dim=c2,
                    input_resolution=stage2_resolution,
                    depth=depths[1],
                    num_heads=6,
                    window_size=8,
                    downsample=None,
                ),
                BasicLayer(
                    dim=c3,
                    out_dim=c3,
                    input_resolution=stage3_resolution,
                    depth=depths[2],
                    num_heads=8,
                    window_size=8,
                    downsample=None,
                ),
            ]
        )

        self.output_resolution = stage3_resolution
        self.output_dim = c3

    def _update_dynamic_resolution(self, H, W):
        if H % 8 != 0 or W % 8 != 0:
            raise ValueError(f"MVSC_Individual_Encoder expects H and W divisible by 8, got {(H, W)}")

        stage1_resolution = (H // self.patch_size, W // self.patch_size)
        stage2_resolution = (stage1_resolution[0] // 2, stage1_resolution[1] // 2)
        stage3_resolution = (stage2_resolution[0] // 2, stage2_resolution[1] // 2)

        if hasattr(self.patch_processing[1], "input_resolution"):
            self.patch_processing[1].input_resolution = stage1_resolution
        if hasattr(self.patch_processing[2], "input_resolution"):
            self.patch_processing[2].input_resolution = stage2_resolution

        self.swin_layers[0].update_resolution(*stage1_resolution)
        self.swin_layers[1].update_resolution(*stage2_resolution)
        self.swin_layers[2].update_resolution(*stage3_resolution)

        self.output_resolution = stage3_resolution

    def forward(self, x, return_hw=False):
        B, T, V, C, H, W = x.shape
        self._update_dynamic_resolution(H, W)

        x = x.contiguous().view(B * T * V, C, H, W)

        x = self.patch_processing[0](x)
        x = self.swin_layers[0](x)

        x = self.patch_processing[1](x)
        x = self.swin_layers[1](x)

        x = self.patch_processing[2](x)
        x = self.swin_layers[2](x)

        L = x.shape[1]
        C = x.shape[2]
        expected_l = self.output_resolution[0] * self.output_resolution[1]
        assert L == expected_l, f"Token number mismatch: got L={L}, expected {expected_l}"
        assert C == self.output_dim, f"Channel mismatch: got C={C}, expected {self.output_dim}"

        x = x.contiguous().view(B, T, V, L, C)
        if return_hw:
            return x, self.output_resolution
        return x


class TemporalPatchEmbedding(nn.Module):
    """
    Patch embedding for the spatio-temporal commonality encoder.
    """

    def __init__(self, dim, out_dim=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.out_dim = dim if out_dim is None else out_dim
        self.norm = norm_layer(dim)
        self.proj = nn.Linear(dim, self.out_dim, bias=False)

    def forward(self, x):
        if x.dim() != 5:
            raise ValueError(f"TemporalPatchEmbedding expects [B,T,V,L,C], got shape={tuple(x.shape)}")
        x = self.norm(x)
        x = self.proj(x)
        return x


class TemporalPatchMerging(nn.Module):
    """
    Merge adjacent tokens along the temporal axis while preserving the view
    dimension and the spatial token grid.
    """

    def __init__(self, dim, out_dim=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.out_dim = dim if out_dim is None else out_dim
        self.norm = norm_layer(dim * 2)
        self.reduction = nn.Linear(dim * 2, self.out_dim, bias=False)

    def forward(self, x):
        if x.dim() != 5:
            raise ValueError(f"TemporalPatchMerging expects [B,T,V,L,C], got shape={tuple(x.shape)}")

        B, T, V, L, C = x.shape

        if T % 2 == 1:
            pad = x[:, -1:, :, :, :]
            x = torch.cat([x, pad], dim=1)
            T = T + 1

        x0 = x[:, 0::2, :, :, :]
        x1 = x[:, 1::2, :, :, :]
        x = torch.cat([x0, x1], dim=-1)
        x = self.norm(x)
        x = self.reduction(x)
        return x


def flatten_tv_to_d(x):
    if x.dim() != 5:
        raise ValueError(f"flatten_tv_to_d expects [B,T,V,L,C], got shape={tuple(x.shape)}")
    B, T, V, L, C = x.shape
    return x.contiguous().view(B, T * V, L, C)


def restore_d_to_tv(x, T, V):
    if x.dim() != 4:
        raise ValueError(f"restore_d_to_tv expects [B,D,L,C], got shape={tuple(x.shape)}")
    B, D, L, C = x.shape
    if D != T * V:
        raise ValueError(f"restore_d_to_tv shape mismatch: D={D}, but T*V={T*V}")
    return x.contiguous().view(B, T, V, L, C)


class MVSC_Commonality_Encoder(nn.Module):
    """
    Spatio-temporal commonality encoder.
    Supports dynamic spatial token resolution passed by hw=(H,W).
    """

    def __init__(self, dim, input_resolution, depths=(2, 1), num_heads=(8, 10)):
        super().__init__()

        if isinstance(depths, int):
            depths = (depths, depths)
        if len(depths) != 2:
            raise ValueError(f"MVSC_Commonality_Encoder expects 2 stage depths, got depths={depths}")

        if isinstance(num_heads, int):
            num_heads = (num_heads, num_heads)
        if len(num_heads) != 2:
            raise ValueError(f"MVSC_Commonality_Encoder expects 2 stage head counts, got num_heads={num_heads}")

        self.stage1_dim = 256
        self.stage2_dim = 320

        self.stage1_patch = TemporalPatchEmbedding(dim=dim, out_dim=self.stage1_dim)
        self.stage1_swin = BasicLayer3D(
            dim=self.stage1_dim,
            input_resolution=input_resolution,
            depth=depths[0],
            num_heads=num_heads[0],
            window_size=4,
        )

        self.stage2_patch = TemporalPatchMerging(dim=self.stage1_dim, out_dim=self.stage2_dim)
        self.stage2_swin = BasicLayer3D(
            dim=self.stage2_dim,
            input_resolution=input_resolution,
            depth=depths[1],
            num_heads=num_heads[1],
            window_size=4,
        )
        self.view_merge = ViewPatchMerging(dim=self.stage2_dim, out_dim=self.stage2_dim)
        self.output_dim = self.stage2_dim

    def forward(self, x, hw=None, return_hw=False):
        if x.dim() != 5:
            raise ValueError(f"MVSC_Commonality_Encoder expects [B,T,V,L,C], got shape={tuple(x.shape)}")

        _, _, _, L, _ = x.shape
        if hw is None:
            hw = _infer_hw_from_l(L)
        H, W = hw
        if H * W != L:
            raise ValueError(f"MVSC_Commonality_Encoder hw mismatch: hw={hw}, but L={L}")

        self.stage1_swin.update_resolution(H, W)
        self.stage2_swin.update_resolution(H, W)

        x = self.stage1_patch(x)
        B, T, V, L, C = x.shape
        x = flatten_tv_to_d(x)
        x = self.stage1_swin(x)
        x = restore_d_to_tv(x, T, V)

        x = self.stage2_patch(x)
        B, T, V, L, C = x.shape
        x = flatten_tv_to_d(x)
        x = self.stage2_swin(x)
        x = restore_d_to_tv(x, T, V)

        x = self.view_merge(x)

        if return_hw:
            return x, hw
        return x


class ViewPatchMerging(nn.Module):
    """
    Merge adjacent tokens along the view axis while keeping the temporal
    dimension explicit and preserving the spatial token grid.
    """

    def __init__(self, dim, out_dim=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.out_dim = dim if out_dim is None else out_dim
        self.norm = norm_layer(dim * 2)
        self.reduction = nn.Linear(dim * 2, self.out_dim, bias=False)

    def forward(self, x):
        if x.dim() != 5:
            raise ValueError(f"ViewPatchMerging expects [B,T,V,L,C], got shape={tuple(x.shape)}")

        B, T, V, L, C = x.shape

        if V % 2 == 1:
            pad = x[:, :, -1:, :, :]
            x = torch.cat([x, pad], dim=2)
            V = V + 1

        x0 = x[:, :, 0::2, :, :]
        x1 = x[:, :, 1::2, :, :]
        x = torch.cat([x0, x1], dim=-1)
        x = self.norm(x)
        x = self.reduction(x)
        return x

def _round_channels(value, multiple=8):
    value = int(round(value / multiple) * multiple)
    return max(multiple, value)

class MVSC_JSCC_Encoder(nn.Module):
    """
    JSCC encoder with dynamic spatial token resolution support.

    Input:
      [B, T, V, L, C]  or [B, D, L, C]

    Output:
      [B, D_jscc, L_jscc, C_latent]

    Note:
      - Current compression policy is unchanged: spatial grid is compressed
        twice (H,W) -> (H/2,W/2) -> (H/4,W/4).
      - To support non-square / high-resolution inference, pass hw=(H,W)
        explicitly. If omitted, only square token maps can be inferred.
    """

    def __init__(self, dim, latent_dim=256):
        super().__init__()

        gap = dim - latent_dim
        mid1 = _round_channels(dim - gap / 3)        # 例如 320 -> 256
        mid2 = _round_channels(dim - 2 * gap / 3)    # 例如 320 -> 192

        mid1 = max(latent_dim, min(dim, mid1))
        mid2 = max(latent_dim, min(mid1, mid2))

        self.blocks = nn.ModuleList(
            [
                JSCCDownBlock3D(dim,  mid1,      stride=(1, 2, 2), num_res_blocks=3),  # 32 -> 16
                JSCCDownBlock3D(mid1, mid2,      stride=(1, 2, 2), num_res_blocks=3),  # 16 -> 8
                JSCCDownBlock3D(mid2, latent_dim, stride=(1, 1, 1), num_res_blocks=3), # 8 -> 8
            ]
        )

    def _forward_flattened(self, x, hw):
        B, D, L, C = x.shape
        H, W = hw
        if H * W != L:
            raise ValueError(f"MVSC_JSCC_Encoder hw mismatch: hw={hw}, but L={L}")

        x = x.view(B, D, H, W, C).permute(0, 4, 1, 2, 3).contiguous()
        for block in self.blocks:
            x = block(x)

        _, c, d_out, h_out, w_out = x.shape
        x = x.permute(0, 2, 3, 4, 1).contiguous().view(B, d_out, h_out * w_out, c)
        return x, (h_out, w_out)

    def forward(self, x, hw=None, return_hw=False):
        if x.dim() == 4:
            B, D, L, C = x.shape
            if hw is None:
                hw = _infer_hw_from_l(L)
            x, out_hw = self._forward_flattened(x, hw)
            if return_hw:
                return x, out_hw
            return x

        if x.dim() == 5:
            B, T, V, L, C = x.shape
            if hw is None:
                hw = _infer_hw_from_l(L)

            x = flatten_tv_to_d(x)
            x, out_hw = self._forward_flattened(x, hw)
            if return_hw:
                return x, out_hw
            return x

        raise ValueError(f"MVSC_JSCC_Encoder expects 4D/5D tokens, got shape={tuple(x.shape)}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    B = 2
    T = 4
    V = 4
    embed_dim = 96
    token_dim = 192
    latent_dim = 256

    def check_shape(name, x, expected_shape):
        actual_shape = tuple(x.shape)
        expected_shape = tuple(expected_shape)
        ok = actual_shape == expected_shape
        status = "OK" if ok else "FAIL"
        print(f"[{status}] {name}: actual={actual_shape}, expected={expected_shape}")
        if not ok:
            raise AssertionError(f"{name} shape mismatch: actual={actual_shape}, expected={expected_shape}")

    x_rgb = torch.randn(B, T, V, 3, 256, 256).to(device)
    ind_enc = MVSC_Individual_Encoder(
        img_size=256,
        patch_size=2,
        in_chans=3,
        embed_dim=embed_dim,
        depths=(1, 2, 1),
    ).to(device)

    with torch.no_grad():
        y_ind, hw_ind = ind_enc(x_rgb, return_hw=True)

    print("[Individual Encoder]")
    print("input :", x_rgb.shape)
    print("output:", y_ind.shape, "hw:", hw_ind)
    check_shape("Individual Encoder", y_ind, (B, T, V, 1024, token_dim))

    common_enc = MVSC_Commonality_Encoder(
        dim=token_dim,
        input_resolution=(32, 32),
        depths=(2, 1),
        num_heads=(8, 10),
    ).to(device)

    with torch.no_grad():
        y_common, hw_common = common_enc(y_ind, hw=hw_ind, return_hw=True)

    print("\n[Commonality Encoder]")
    print("input :", y_ind.shape)
    print("output:", y_common.shape, "hw:", hw_common)
    check_shape("Commonality Encoder", y_common, (B, 2, 2, 1024, 320))

    jscc_enc = MVSC_JSCC_Encoder(
        dim=320,
        latent_dim=latent_dim,
    ).to(device)

    with torch.no_grad():
        y_jscc, hw_jscc = jscc_enc(y_common, hw=hw_common, return_hw=True)

    print("\n[JSCC Encoder]")
    print("input :", y_common.shape)
    print("output:", y_jscc.shape, "hw:", hw_jscc)
    check_shape("JSCC Encoder", y_jscc, (B, 4, 64, latent_dim))

    with torch.no_grad():
        y_ind2, hw_ind2 = ind_enc(x_rgb, return_hw=True)
        y_common2, hw_common2 = common_enc(y_ind2, hw=hw_ind2, return_hw=True)
        y_full, hw_full = jscc_enc(y_common2, hw=hw_common2, return_hw=True)

    print("\n[Full MVSC Encoder]")
    print("input :", x_rgb.shape)
    print("output:", y_full.shape, "hw:", hw_full)
    check_shape("Full MVSC Encoder", y_full, (B, 4, 64, latent_dim))

    print("\nAll encoder self-tests passed.")
