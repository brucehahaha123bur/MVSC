import torch
import torch.nn as nn
from random import choice

from net.encoder import (
    SwinJSCC_Encoder,
    MVSC_Individual_Encoder,
    MVSC_Commonality_Encoder,
    MVSC_JSCC_Encoder,
)

from net.decoder import (
    SwinJSCC_Decoder,
    MVSC_JSCC_Decoder,
    MVSC_Commonality_Decoder,
    MVSC_Individual_Decoder,
)

from net.channel import Channel
from loss.distortion import Distortion


def create_encoder(**kwargs):
    return SwinJSCC_Encoder(**kwargs)


def create_decoder(**kwargs):
    return SwinJSCC_Decoder(**kwargs)



class SwinJSCC(nn.Module):
    def __init__(self, args, config):
        super(SwinJSCC, self).__init__()
        self.config = config
        encoder_kwargs = config.encoder_kwargs
        decoder_kwargs = config.decoder_kwargs
        self.encoder = create_encoder(**encoder_kwargs)
        self.decoder = create_decoder(**decoder_kwargs)
        if config.logger is not None:
            config.logger.info("Network config: ")
            config.logger.info("Encoder: ")
            config.logger.info(encoder_kwargs)
            config.logger.info("Decoder: ")
            config.logger.info(decoder_kwargs)
        self.distortion_loss = Distortion(args)
        self.channel = Channel(args, config)
        self.pass_channel = config.pass_channel
        self.squared_difference = torch.nn.MSELoss(reduction='none')
        self.H = self.W = 0
        self.multiple_snr = args.multiple_snr.split(",")
        for i in range(len(self.multiple_snr)):
            self.multiple_snr[i] = int(self.multiple_snr[i])
        self.cbr_bits_per_component = float(getattr(args, "cbr_bits_per_component", 3.0))
        self.channel_number = args.C.split(",")
        for i in range(len(self.channel_number)):
            self.channel_number[i] = int(self.channel_number[i])
        self.downsample = config.downsample
        self.model = args.model

    def distortion_loss_wrapper(self, x_gen, x_real):
        distortion_loss = self.distortion_loss.forward(x_gen, x_real, normalization=self.config.norm)
        return distortion_loss

    def feature_pass_channel(self, feature, chan_param, avg_pwr=False):
        noisy_feature = self.channel.forward(feature, chan_param, avg_pwr)
        return noisy_feature

    def forward(self, input_image, given_SNR=None, given_rate=None):
        B, _, H, W = input_image.shape

        if H != self.H or W != self.W:
            self.encoder.update_resolution(H, W)
            self.decoder.update_resolution(H // (2 ** self.downsample), W // (2 ** self.downsample))
            self.H = H
            self.W = W

        if given_SNR is None:
            SNR = choice(self.multiple_snr)
            chan_param = SNR
        else:
            chan_param = given_SNR

        if given_rate is None:
            channel_number = choice(self.channel_number)
        else:
            channel_number = given_rate

        if self.model == 'SwinJSCC_w/o_SAandRA' or self.model == 'SwinJSCC_w/_SA':
            feature = self.encoder(input_image, chan_param, channel_number, self.model)
            transmitted_bits = float(feature.numel()) * self.cbr_bits_per_component
            CBR = transmitted_bits / float(input_image.numel())
            if self.pass_channel:
                noisy_feature = self.feature_pass_channel(feature, chan_param)
            else:
                noisy_feature = feature

        elif self.model == 'SwinJSCC_w/_RA' or self.model == 'SwinJSCC_w/_SAandRA':
            feature, mask = self.encoder(input_image, chan_param, channel_number, self.model)
            # IQ-component + bit-depth convention:
            # CBR = (bits per transmitted feature component) * (transmitted components / source values)
            # Here transmitted components per source value = channel_number / (3 * 2^(2*downsample)).
            CBR = self.cbr_bits_per_component * channel_number / (3 * 2 ** (self.downsample * 2))
            avg_pwr = torch.sum(feature ** 2) / mask.sum()
            if self.pass_channel:
                noisy_feature = self.feature_pass_channel(feature, chan_param, avg_pwr)
            else:
                noisy_feature = feature
            noisy_feature = noisy_feature * mask

        recon_image = self.decoder(noisy_feature, chan_param, self.model)
        mse = self.squared_difference(input_image * 255., recon_image.clamp(0., 1.) * 255.)
        loss_G = self.distortion_loss.forward(input_image, recon_image.clamp(0., 1.))
        return recon_image, CBR, chan_param, mse.mean(), loss_G.mean()



class MVSCNet(nn.Module):
    """
    Full MVSC pipeline:
        x_g [B, T, V, 3, H, W]
          -> Individual Encoder
          -> Commonality Encoder
          -> JSCC Encoder
          -> Channel
          -> JSCC Decoder
          -> Commonality Decoder
          -> Individual Decoder
          -> x_hat [B, T, V, 3, H, W]
    """

    def __init__(self, args, config):
        super().__init__()

        self.args = args
        self.config = config
        self.cbr_weight = float(getattr(args, "cbr_weight", 0.0))
        self.cbr_bits_per_component = float(getattr(args, "cbr_bits_per_component", 3.0))
        self.snr_candidates = []
        raw_snr = getattr(args, "multiple_snr", None)
        if raw_snr is not None:
            for token in str(raw_snr).split(","):
                token = token.strip()
                if token:
                    self.snr_candidates.append(float(token))

        # -------------------------------------------------
        # config defaults
        # -------------------------------------------------
        if hasattr(config, "image_dims"):
            # e.g. (3, 256, 256)
            _, img_h, img_w = config.image_dims
            assert img_h == img_w, "Current MVSCNet assumes square training crop."
            img_size = img_h
        else:
            img_size = getattr(config, "img_size", 256)

        patch_size = getattr(config, "mvsc_patch_size", 4)
        embed_dim = getattr(config, "mvsc_embed_dim", 96)
        latent_dim = getattr(config, "mvsc_latent_dim", 256)
        num_views = getattr(config, "mvsc_num_views", 4)
        common_depth = getattr(config, "mvsc_common_depth", 2)
        common_heads = getattr(config, "mvsc_common_heads", 4)

        # -------------------------------------------------
        # encoders
        # -------------------------------------------------
        self.individual_encoder = MVSC_Individual_Encoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=3,
            embed_dim=embed_dim,
        )

        token_resolution = self.individual_encoder.output_resolution

        self.commonality_encoder = MVSC_Commonality_Encoder(
            dim=embed_dim,
            input_resolution=token_resolution,
            depth=common_depth,
            num_heads=common_heads,
        )

        self.jscc_encoder = MVSC_JSCC_Encoder(
            dim=embed_dim,
            latent_dim=latent_dim,
        )

        # -------------------------------------------------
        # channel
        # -------------------------------------------------
        self.channel = Channel(args, config)

        # -------------------------------------------------
        # decoders
        # -------------------------------------------------
        self.jscc_decoder = MVSC_JSCC_Decoder(
            latent_dim=latent_dim,
            embed_dim=embed_dim,
        )

        self.commonality_decoder = MVSC_Commonality_Decoder(
            dim=embed_dim,
            input_resolution=token_resolution,
            num_views=num_views,
            depth=common_depth,
            num_heads=common_heads,
        )

        self.individual_decoder = MVSC_Individual_Decoder(
            img_size=img_size,
            patch_size=patch_size,
            out_chans=3,
            embed_dim=embed_dim,
            input_resolution=token_resolution,
            num_upsample_stages=self.individual_encoder.num_downsample_stages,
        )

        # -------------------------------------------------
        # distortion
        # -------------------------------------------------
        self.distortion_loss = Distortion(args)

    def encode(self, x_g):
        """
        Args:
            x_g: [B, T, V, 3, H, W]

        Returns:
            L_g: [B, T, V, L, C]
            S_g: [B, T, V, L, C]
            y:   [B, T, V, Lc, latent_dim]
        """
        L_g = self.individual_encoder(x_g)
        S_g = self.commonality_encoder(L_g)
        y = self.jscc_encoder(S_g)
        return L_g, S_g, y

    def decode(self, y_hat):
        """
        Args:
            y_hat: [B, T, V, Lc, latent_dim]

        Returns:
            S_hat: [B, T, V, L, C]
            L_hat: [B, T, V, L, C]
            x_hat: [B, T, V, 3, H, W]
        """
        S_hat = self.jscc_decoder(y_hat)
        L_hat = self.commonality_decoder(S_hat)
        x_hat = self.individual_decoder(L_hat)
        return S_hat, L_hat, x_hat

    def _compute_distortion(self, x_g, x_hat):
        """
        Distortion module in your project is image-oriented.
        We flatten [B, T, V, C, H, W] -> [B*T*V, C, H, W]
        before feeding it into Distortion for compatibility.
        """
        if x_g.dim() == 6:
            B, T, V, C, H, W = x_g.shape
            x_g_ = x_g.reshape(B * T * V, C, H, W)
            x_hat_ = x_hat.reshape(B * T * V, C, H, W)
        else:
            x_g_ = x_g
            x_hat_ = x_hat
        return self.distortion_loss.forward(x_g_, x_hat_)

    def _compute_cbr(self, x_g, y):
        """
        CBR uses IQ-component bit accounting:
            cbr = (num_transmitted_components * bits_per_component) / num_source_values

        - num_transmitted_components counts all transmitted real components (I and Q both included).
        - bits_per_component defaults to 3 to match the paper's 3-bit quantization convention.
        """
        source_values = float(x_g.numel())
        transmitted_components = float(y.numel())
        transmitted_bits = transmitted_components * self.cbr_bits_per_component
        cbr_value = transmitted_bits / max(source_values, 1.0)
        return y.new_tensor(cbr_value)

    def forward(self, x_g, given_SNR=None):
        """
        Args:
            x_g: [B, T, V, 3, H, W]
            given_SNR: optional scalar / tensor, passed to channel

        Returns:
            x_hat: [B, T, V, 3, H, W]
            used_snr: given_SNR or selected fallback
            loss: scalar total loss (distortion + cbr term)
            aux: dict with detached scalar terms for logging
        """
        # encode
        L_g, S_g, y = self.encode(x_g)

        # choose snr (fallback if None)
        if given_SNR is None:
            if self.snr_candidates:
                used_snr = choice(self.snr_candidates)
            else:
                used_snr = 10.0
        else:
            if torch.is_tensor(given_SNR):
                used_snr = float(given_SNR.item())
            else:
                used_snr = float(given_SNR)

        # channel (use explicit forward to match original API)
        y_hat = self.channel.forward(y, used_snr)

        # decode
        S_hat, L_hat, x_hat = self.decode(y_hat)

        # clamp to image range
        x_hat = x_hat.clamp(0.0, 1.0)

        # loss (flatten for image-oriented Distortion and pass normalization)
        if x_g.dim() == 6:
            B, T, V, C, H, W = x_g.shape
            x_g_ = x_g.reshape(B * T * V, C, H, W)
            x_hat_ = x_hat.reshape(B * T * V, C, H, W)
        else:
            x_g_ = x_g
            x_hat_ = x_hat

        distortion = self.distortion_loss.forward(x_g_, x_hat_, normalization=self.config.norm)
        cbr = self._compute_cbr(x_g, y)
        loss = distortion + self.cbr_weight * cbr

        aux = {
            "distortion": distortion.detach(),
            "cbr": cbr.detach(),
            "cbr_weight": self.cbr_weight,
        }
        return x_hat, used_snr, loss, aux