"""
Author: David Pukanec (FIT BUT)

This code was produced by David Pukanec (FIT VUT) and was inspired by:
  - DiffComplete: https://github.com/dvlab-research/DiffComplete
  - ControlNet:   https://github.com/lllyasviel/ControlNet
"""

from torch.nn import Conv3d
from model.network.net_util import *


def build_activation(net_cfg):
    """
    Construct the activation module used throughout the network.

    Defaults to SiLU if no activation is explicitly specified.
    """
    act_name = getattr(net_cfg, "unet_activation", "SiLU")

    if act_name == "SiLU":
        return nn.SiLU()
    if act_name == "ReLU":
        return nn.ReLU()

    raise NotImplementedError(f"Unsupported activation: {act_name}")

class ContextCondNet(nn.Module):
    """
     Conditioning encoder used to extract multi-scale control features
     from the partial / hint input.

     The network produces a list of feature tensors aligned with the
     diffusion U-Net blocks. These features are later injected into the
     main denoising model.

     Supports:
         - timestep conditioning
         - optional class conditioning
         - optional classifier-free setup
         - optional correction mode
         - selectable attention type
     """
    def __init__(self, config):

        super().__init__()
        self.resolution = config.exp.res
        self.class_free = config.diffusion.class_free
        self.config = config.net
        self.in_channels = self.config.in_channels
        self.activation = build_activation(self.config)
        self.dropout = self.config.dropout if hasattr(self.config, 'dropout') else 0
        self.channel_mult = list(self.config.channel_mult)
        self.model_channels = self.config.model_channels
        self.resolution = config.exp.res
        self.mask_thresh = 2

        if isinstance(config.net.channel_mult, str):
            self.channel_mult = list(map(int, config.net.channel_mult.split(',')))
        else:
            self.channel_mult = list(config.net.channel_mult)

        if self.config.attention_resolutions:
            self.attention_resolutions = list(self.config.attention_resolutions)
        else:
            self.attention_resolutions = []
        self.conv_resample = True
        self.res_dims = 3
        self.num_res_blocks = self.config.num_res_blocks
        self.attention_heads = self.config.attention_heads
        self.out_channels = (2 if hasattr(config.diffusion, 'diffusion_learn_sigma')
                                  and config.diffusion.diffusion_learn_sigma else 1)  # 1
        self.hint_channels = self.config.hint_channels

        time_embed_dim = self.model_channels * 4
        label_embed_dim = self.model_channels * 4

        # To turn off some of the resnet properties
        self.use_checkpoint = False
        self.use_scale_shift_norm = False

        if hasattr(self.config, 'correction') and self.config.correction:
            self.partial_channels = self.model_channels
            self.model_channels = self.model_channels * 2


        if hasattr(config, 'class_cond'):
            self.num_classes = config.class_cond.num_classes
        else:
            self.num_classes = None
        if config.diffusion.class_free:
            # For classiefier free diffusion
            self.uncond_features =  nn.Parameter(torch.randn(1, 1, 64, 64, 64))


        # Time embedding layer
        self.time_embed = nn.Sequential(
            nn.Linear(self.model_channels, time_embed_dim),
            self.activation,
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # If FDI conditioned create embedding layer
        if self.num_classes is not None:
            if self.class_free:
                self.label_emb = nn.Embedding(self.num_classes + 1, label_embed_dim)
            else:
                self.label_emb = nn.Embedding(self.num_classes, label_embed_dim)


        if hasattr(self.config, 'correction') and self.config.correction:
            self.input_hint_block = TimestepEmbedSequential(
                nn.Conv3d(self.in_channels, self.partial_channels, 3, padding=1),
            )
            self.input_blocks = nn.ModuleList([TimestepEmbedSequential(
                nn.Conv3d(self.in_channels, self.partial_channels, 3, padding=1)
            )])
        else:
            self.input_hint_block = TimestepEmbedSequential(
                Conv3d( self.hint_channels, 16, 3, padding=1),
                nn.SiLU(),
                zero_module(Conv3d(16, self.in_channels, 3, padding=1))
            )
            self.input_blocks = nn.ModuleList(
                [
                    TimestepEmbedSequential(
                        nn.Conv3d(self.in_channels, self.model_channels, 3, padding=1)
                    )
                ]
            )

        self.zero_convs = nn.ModuleList([self.make_zero_conv(self.model_channels)])

        input_block_chans = [self.model_channels]
        ch = self.model_channels
        ds_index = 1
        for level, mult in enumerate(self.channel_mult):
            for _ in range(self.num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        self.dropout,
                        out_channels=mult * self.model_channels,
                        use_checkpoint=self.use_checkpoint,
                        use_scale_shift_norm=self.use_scale_shift_norm,
                        activation=self.activation
                    )
                ]
                ch = mult * self.model_channels
                if ds_index in self.attention_resolutions:
                    layers.append(
                       self.select_attention(ch, self.resolution / ds_index)
                    )

                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                input_block_chans.append(ch)
            if level != len(self.channel_mult) - 1:
                self.input_blocks.append(
                    TimestepEmbedSequential(Downsample(ch, self.conv_resample))
                )
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                ds_index *= 2

        middle_layers = [ResBlock(
            ch,
            time_embed_dim,
            self.dropout,
            use_checkpoint=self.use_checkpoint,
            use_scale_shift_norm=self.use_scale_shift_norm,
            activation=self.activation
        )]
        if not hasattr(self.config, 'middle_attention') or self.config.middle_attention:
            middle_layers.append(self.select_attention(ch, self.resolution / ds_index))
        middle_layers.append(ResBlock(
            ch,
            time_embed_dim,
            self.dropout,
            use_checkpoint=self.use_checkpoint,
            use_scale_shift_norm=self.use_scale_shift_norm,
            activation=self.activation
        ), )
        self.middle_block = TimestepEmbedSequential(*middle_layers)
        self.middle_block_out = self.make_zero_conv(ch)

    def select_attention(self, ch, feat_dim=None):
        if self.config.attention_type == 'normal':
            return AttentionBlock(
                ch, use_checkpoint=False, num_heads=self.attention_heads,
            )
        elif self.config.attention_type == 'efficient':
            return EfficientSelfAttentionBlock(ch, feat_dim, self.attention_heads)
        else:
            raise NotImplementedError

    @property
    def inner_dtype(self):
        """
        Get the dtype used by the torso of the model.
        """
        return torch.float32  # FIXED
        # return next(self.input_blocks.parameters()).dtype

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(Conv3d(channels, channels, 1, padding=0)))

    def forward(self, x, hint, timesteps, y=None):
        """
        Run the conditioning encoder.

        Parameters
        ----------
        x : torch.Tensor
            Main diffusion input tensor of shape [B, C, D, H, W]
        hint : torch.Tensor
            Conditioning tensor, usually incomplete geometry / partial SDF
        timesteps : torch.Tensor
            Diffusion timestep indices of shape [B]
        y : torch.Tensor or None
            Class labels when class conditioning is enabled

        Returns
        -------
        list[torch.Tensor]
            Multi-scale control features aligned with the backbone U-Net.
        """
        assert (y is not None) == (self.num_classes is not None), (
            "must specify y iff the model is class-conditional"
        )

        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))  # (16. 256)

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)


        guided_hint = self.input_hint_block(hint, emb)

        outs = []
        h = x.type(self.inner_dtype)
        if hasattr(self.config, 'correction') and self.config.correction:
            for module, zero_conv in zip(self.input_blocks, self.zero_convs):
                # First input projection
                if guided_hint is not None:
                    h = module(h, emb)
                    h = torch.cat([h, guided_hint], dim=1)
                    guided_hint = None
                else:
                    h = module(h, emb)
                outs.append(zero_conv(h, emb))

        else:

            for module, zero_conv in zip(self.input_blocks, self.zero_convs):
                if guided_hint is not None:
                    h = module(h, emb)
                    h += guided_hint
                    guided_hint = None
                else:
                    h = module(h, emb)

                outs.append(zero_conv(h, emb))

        h = self.middle_block(h, emb)
        outs.append(self.middle_block_out(h, emb))


        return outs



    def get_uncond_features(self, hint, batch_size):
        """
        Return learned unconditional hint features expanded to batch size.

        The `hint` argument is kept for interface compatibility.
        """
        return self.uncond_features.expand(batch_size, -1, -1, -1, -1)