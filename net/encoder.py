from net.modules import *
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        B_, N, C = x_windows.shape

        # merge windows
        attn_windows = self.attn(x_windows,
                                 add_token=False,
                                 mask=self.attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops

    def update_mask(self):
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
            self.attn_mask = attn_mask.cuda()
        else:
            pass


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
    """3D Swin block with shifted window attention over [T, H, W]."""

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
        self.input_resolution = input_resolution  # (H, W)
        self.num_heads = num_heads

        if isinstance(window_size, int):
            window_size = (2, window_size, window_size)
        elif len(window_size) == 2:
            window_size = (2, window_size[0], window_size[1])
        self.window_size = tuple(window_size)

        if isinstance(shift_size, int):
            shift_size = (shift_size, shift_size, shift_size)
        elif len(shift_size) == 2:
            shift_size = (0, shift_size[0], shift_size[1])
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
        # x: [B, T, L, C]
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
    Input: [B, T, L, C]
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
        # x: [B, T, V, L, C]
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


class MVSC_Individual_Encoder(nn.Module):
    """
    Individual semantic encoder (2D Swin).
    Processes each (view, frame) independently.
    Input:  [B, T, V, 3, H, W]
    Output: [B, T, V, L, C]
    """

    def __init__(self, img_size=256, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )

        resolution = (img_size // patch_size, img_size // patch_size)
        self.target_resolution = (max(1, img_size // 8), max(1, img_size // 8))

        # Keep a 3-stage hierarchical encoder and reduce spatial tokens toward H/8 x W/8.
        self.num_downsample_stages = 0
        probe_h, probe_w = resolution
        while (
            self.num_downsample_stages < 2
            and probe_h > self.target_resolution[0]
            and probe_w > self.target_resolution[1]
            and probe_h % 2 == 0
            and probe_w % 2 == 0
        ):
            self.num_downsample_stages += 1
            probe_h //= 2
            probe_w //= 2

        downsample_flags = [
            False,
            self.num_downsample_stages >= 1,
            self.num_downsample_stages >= 2,
        ]

        self.layers = nn.ModuleList()
        cur_resolution = resolution
        for do_downsample in downsample_flags:
            layer = BasicLayer(
                dim=embed_dim,
                out_dim=embed_dim,
                input_resolution=cur_resolution,
                depth=2,
                num_heads=3,
                window_size=8,
                downsample=PatchMerging if do_downsample else None,
            )
            self.layers.append(layer)
            if do_downsample:
                cur_resolution = (cur_resolution[0] // 2, cur_resolution[1] // 2)

        self.output_resolution = cur_resolution

    def forward(self, x):
        B, T, V, C, H, W = x.shape
        x = x.view(B * T * V, C, H, W)

        x = self.patch_embed(x)          # [BTV, L, C]
        for layer in self.layers:
            x = layer(x)

        L = x.shape[1]
        C = x.shape[2]

        x = x.view(B, T, V, L, C)
        return x


class MVSC_Commonality_Encoder(nn.Module):
    def __init__(self, dim, input_resolution, depth=2, num_heads=4):
        super().__init__()

        self.temporal_layers = nn.ModuleList(
            [
                BasicLayer3D(
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
        B, T, V, L, C = x.shape
        for temporal_layer, view_layer in zip(self.temporal_layers, self.view_layers):
            # Temporal modeling per view.
            x = x.view(B * V, T, L, C)
            x = temporal_layer(x)
            x = x.view(B, T, V, L, C)

            # Cross-view fusion.
            x = view_layer(x)
        return x


class MVSC_JSCC_Encoder(nn.Module):
    def __init__(self, dim, latent_dim=256):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                JSCCDownBlock(dim, dim, stride=2, num_res_blocks=3),
                JSCCDownBlock(dim, dim, stride=2, num_res_blocks=3),
                JSCCDownBlock(dim, latent_dim, stride=2, num_res_blocks=3),
            ]
        )

    def forward(self, x):
        if x.dim() == 4:
            B, T, L, C = x.shape
            h = int(math.sqrt(L))
            if h * h != L:
                raise ValueError(f"MVSC_JSCC_Encoder expects square token map, but got L={L}.")

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
                raise ValueError(f"MVSC_JSCC_Encoder expects square token map, but got L={L}.")

            x = x.view(B * T * V, h, h, C).permute(0, 3, 1, 2).contiguous()
            for block in self.blocks:
                x = block(x)

            _, c, h_out, w_out = x.shape
            x = x.permute(0, 2, 3, 1).contiguous().view(B, T, V, h_out * w_out, c)
            return x

        raise ValueError(f"MVSC_JSCC_Encoder expects 4D/5D tokens, got shape={tuple(x.shape)}")