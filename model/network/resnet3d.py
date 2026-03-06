
"""
Author: David Pukanec (FIT BUT)

This code was produced by David Pukanec (FIT VUT) and was inspired by:
  - DiffComplete: https://github.com/dvlab-research/DiffComplete
  - ControlNet:   https://github.com/lllyasviel/ControlNet
"""


import hydra
from model.network.net_util import *


def get_activation(config):
    if hasattr(config, 'unet_activation') and config.unet_activation:
        if config.unet_activation == 'SiLU':
            return nn.SiLU()
        elif config.unet_activation == 'ReLU':
            return nn.ReLU()
        else:
            raise NotImplementedError
    else:
        return nn.SiLU()

class DiffUNet(nn.Module):
    """
    3D diffusion U-Net with optional:
      - timestep conditioning
      - class conditioning
      - external context injection into decoder skip features
      - antagonist feature injection
    """
    def __init__(self, config):

        super().__init__()
        self.config = config.net
        self.class_free = config.diffusion.class_free
        self.resolution = config.exp.res

        if hasattr(config, 'antag') and  hasattr(config, 'type'):
            self.antag_cond_type = config.antag.type

        else:
            self.antag_cond_type = None

        self.activation = get_activation(self.config)
        self.dropout = self.config.dropout if hasattr(self.config, 'dropout') else 0

        if isinstance(config.net.channel_mult, str):
            self.channel_mult = list(map(int, config.net.channel_mult.split(',')))
        else:
            self.channel_mult = list(config.net.channel_mult)

        self.model_channels = self.config.model_channels
        self.in_channels = self.config.in_channels
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

        time_embed_dim = self.model_channels * 4
        label_embed_dim = self.model_channels * 4

        # To turn off some of the resnet properties
        self.use_checkpoint = False
        self.use_scale_shift_norm = False

        if hasattr(config, 'class_cond'):
            self.num_classes = config.class_cond.num_classes
        else:
            self.num_classes = None


        if hasattr(self.config, 'correction') and self.config.correction:
            self.model_channels = self.model_channels * 2


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

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    nn.Conv3d(self.in_channels, self.model_channels, 3, padding=1, stride=1)
                )
            ]
        )
        # Encoder definition
        input_block_chans = [self.model_channels]
        # Current channel size and downsample index
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
                        use_checkpoint= self.use_checkpoint,
                        use_scale_shift_norm=self.use_scale_shift_norm,
                        activation = self.activation
                    )
                ]
                ch = mult * self.model_channels
                if ds_index in self.attention_resolutions:
                    layers.append(
                        self.select_attention(ch, tensor_res=self.resolution / ds_index)
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(self.channel_mult) - 1:
                self.input_blocks.append(
                    TimestepEmbedSequential(Downsample(ch, self.conv_resample))
                )
                input_block_chans.append(ch)
                ds_index *= 2
        # Middle block definition
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                self.dropout,
                use_checkpoint=self.use_checkpoint,
                use_scale_shift_norm=self.use_scale_shift_norm,
                activation = self.activation
            ),
            self.select_attention(ch, tensor_res=self.resolution / ds_index),
            ResBlock(
                ch,
                time_embed_dim,
                self.dropout,
                use_checkpoint=self.use_checkpoint,
                use_scale_shift_norm=self.use_scale_shift_norm,
                activation=self.activation
            ),
        )

        # Decoder definition
        self.out_cross = nn.ModuleList([])
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(self.channel_mult))[::-1]:
            for i in range(self.num_res_blocks + 1):
                layers = [
                    ResBlock(
                        ch + input_block_chans.pop(),
                        time_embed_dim,
                        self.dropout,
                        out_channels=self.model_channels * mult,
                        use_checkpoint=self.use_checkpoint,
                        use_scale_shift_norm=self.use_scale_shift_norm,
                        activation = self.activation
                    )
                ]
                ch = self.model_channels * mult
                if ds_index in self.attention_resolutions:
                    layers.append(
                       self.select_attention(ch, tensor_res=self.resolution / ds_index)
                    )
                if level and i == self.num_res_blocks:
                    layers.append(Upsample(ch, self.conv_resample,))
                    ds_index //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            nn.GroupNorm(32, ch),
            self.activation,
            zero_module(nn.Conv3d(self.model_channels, self.out_channels, 3, padding=1)),
        )


    def select_attention(self, ch, tensor_res = None):
        if self.config.attention_type == 'normal':
            return AttentionBlock(
                ch, use_checkpoint=False, num_heads=self.attention_heads,
            )
        elif self.config.attention_type == 'efficient':
            return EfficientSelfAttentionBlock(ch, tensor_res, self.attention_heads)
        else:
            raise NotImplementedError

    @property
    def inner_dtype(self):
        """
        Get the dtype used by the torso of the model.
        """
        return torch.float32 # FIXED
        # return next(self.input_blocks.parameters()).dtype

    def forward(self, x, timesteps, context=None, y=None, antag_feats=None, **kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :param low_cond: an [N x C x ...]  Tensor of condition.
        :return: an [N x C x ...] Tensor of outputs.
        """

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        # If FDI conditioned
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)
            
            
        h = x.type(self.inner_dtype)
        # Encoder pass
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)


        h = self.middle_block(h, emb)

        # Decoder pass
        if context is not None:
            h = h + context.pop()
        if  self.antag_cond_type == 'encoder' and antag_feats is not None:
            h = h + antag_feats.pop()


        for i, module in enumerate(self.output_blocks):
            if hs[-1].size(-1) < h.size(-1):
                h = h[..., :-1]
            if hs[-1].size(-2) < h.size(-2):
                h = h[..., :-1, :]
            if hs[-1].size(-3) < h.size(-3):
                h = h[..., :-1, :, :]

            # Controlling the model on partial condition or antagonist
            dec_hs = hs.pop()
            if context is not None:
                dec_hs = dec_hs + context.pop()
            if self.antag_cond_type == 'encoder' and antag_feats is not None:
                dec_hs = dec_hs + antag_feats.pop()

            h = torch.cat([h, dec_hs], dim=1)
            h = module(h, emb)

        h = h.type(x.dtype)
        return self.out(h)

    def get_feature_vectors(self, x, timesteps, y=None):
        """
        Apply the model and return all of the intermediate tensors.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: a dict with the following keys:
                 - 'down': a list of hidden state tensors from downsampling.
                 - 'middle': the tensor of the output of the lowest-resolution
                             block in the model.
                 - 'up': a list of hidden state tensors from upsampling.
        """
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)
        result = dict(down=[], up=[])
        h = x.type(self.inner_dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
            result["down"].append(h.type(x.dtype))
        h = self.middle_block(h, emb)
        result["middle"] = h.type(x.dtype)
        for module in self.output_blocks:
            cat_in = torch.cat([h, hs.pop()], dim=1)
            h = module(cat_in, emb)
            result["up"].append(h.type(x.dtype))
        return result
