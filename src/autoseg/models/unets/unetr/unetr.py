import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def UNetResConcat(a, b):
    """Center crop a so that it can be concatenated with b
    a: [b, c, z, y, x]
    b: [b, c, z, y, x]
    """
    a_shape = a.shape
    b_shape = b.shape

    # Calculate offsets for center cropping
    z_diff = a_shape[2] - b_shape[2]
    y_diff = a_shape[3] - b_shape[3]
    x_diff = a_shape[4] - b_shape[4]

    z_start = z_diff // 2
    y_start = y_diff // 2
    x_start = x_diff // 2

    # Perform center cropping
    a_cropped = a[
        :,
        :,
        z_start : z_start + b_shape[2],
        y_start : y_start + b_shape[3],
        x_start : x_start + b_shape[4],
    ]

    # Concatenate along the channel dimension
    return torch.cat([a_cropped, b], dim=1)


class SingleDeconv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, upsample_factors=2):
        super().__init__()
        self.block = nn.ConvTranspose3d(
            in_planes,
            out_planes,
            kernel_size=upsample_factors,
            stride=upsample_factors,
            padding=0,
            output_padding=0,
        )

    def forward(self, x):
        return self.block(x)


class SingleConv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, pad=True):
        super().__init__()
        self.block = nn.Conv3d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=1,
            padding=((kernel_size - 1) // 2) if pad else 0,
        )

    def forward(self, x):
        return self.block(x)


class Conv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, pad=True):
        super().__init__()
        self.block = nn.Sequential(
            SingleConv3DBlock(in_planes, out_planes, kernel_size, pad=pad),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.block(x)


class Deconv3DBlock(nn.Module):
    def __init__(
        self, in_planes, out_planes, kernel_size=3, pad=True, upsample_factors=2
    ):
        super().__init__()
        self.block = nn.Sequential(
            SingleDeconv3DBlock(
                in_planes, out_planes, upsample_factors=upsample_factors
            ),
            SingleConv3DBlock(out_planes, out_planes, kernel_size, pad=pad),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.block(x)


class SelfAttention(nn.Module):
    def __init__(self, num_heads, embed_dim, dropout):
        super().__init__()
        self.num_attention_heads = num_heads
        self.attention_head_size = int(embed_dim / num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(embed_dim, self.all_head_size)
        self.key = nn.Linear(embed_dim, self.all_head_size)
        self.value = nn.Linear(embed_dim, self.all_head_size)

        self.out = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        self.softmax = nn.Softmax(dim=-1)

        self.vis = False

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, return_raw_scores=False):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        if not return_raw_scores:
            return attention_output, weights
        else:
            return attention_output, weights, attention_scores


class Mlp(nn.Module):
    def __init__(self, in_features, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, in_features)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1()
        x = self.act(x)
        x = self.drop(x)
        return x


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model=786, d_ff=2048, dropout=0.1):
        super().__init__()
        # Torch linears have a `b` by default.
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, input_dim, embed_dim, cube_size, patch_size, dropout):
        super().__init__()
        self.n_patches = int(
            (cube_size[0] * cube_size[1] * cube_size[2])
            / (patch_size[0] * patch_size[1] * patch_size[2])
        )
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_embeddings = nn.Conv3d(
            in_channels=input_dim,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, self.n_patches, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, cube_size, patch_size):
        super().__init__()
        self.attention_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp_dim = int(
            (cube_size[0] * cube_size[1] * cube_size[2])
            / (patch_size[0] * patch_size[1] * patch_size[2])
        )
        self.mlp = PositionwiseFeedForward(embed_dim, 2048)
        self.attn = SelfAttention(num_heads, embed_dim, dropout)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h
        h = x

        x = self.mlp_norm(x)
        x = self.mlp(x)

        x = x + h
        return x, weights


class Transformer(nn.Module):
    def __init__(
        self,
        input_dim,
        embed_dim,
        cube_size,
        patch_size,
        num_heads,
        num_layers,
        dropout,
        extract_layers,
    ):
        super().__init__()
        print(input_dim, embed_dim, cube_size, patch_size)
        self.embeddings = Embeddings(
            input_dim, embed_dim, cube_size, patch_size, dropout
        )
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.extract_layers = extract_layers
        for _ in range(num_layers):
            layer = TransformerBlock(
                embed_dim, num_heads, dropout, cube_size, patch_size
            )
            self.layer.append(copy.deepcopy(layer))

    def forward(self, x):
        extract_layers = []
        hidden_states = self.embeddings(x)

        for depth, layer_block in enumerate(self.layer):
            hidden_states, _ = layer_block(hidden_states)
            if depth + 1 in self.extract_layers:
                extract_layers.append(hidden_states)

        return extract_layers


class UNETR(nn.Module):
    def __init__(
        self,
        img_shape=(128, 128, 128),
        input_dim=4,
        output_dim=3,
        embed_dim=768,
        patch_size=[16, 16, 16],
        num_heads=12,
        dropout=0.1,
        pad_decoder=True,
        upsample_factors=[[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.img_shape = img_shape
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.pad_decoder = pad_decoder
        self.num_layers = 12
        # self.num_layers = 6
        self.ext_layers = [3, 6, 9, 12]
        # self.ext_layers = [3, 6]
        if not isinstance(patch_size, list) and not isinstance(patch_size, tuple):
            patch_size = [patch_size for _ in img_shape]

        for i in range(3):
            mul = 1
            for f in upsample_factors:
                mul *= f[i]
            assert (
                mul == patch_size[i]
            ), f"Upsample factors must multiply to patch size. Error in dim {i}, upsample factors: {[f[i] for f in upsample_factors]}, patch size: {patch_size[i]}"

        self.patch_dim = [int(x / p_size) for x, p_size in zip(img_shape, patch_size)]

        # Transformer Encoder
        self.transformer = Transformer(
            input_dim,
            embed_dim,
            img_shape,
            patch_size,
            num_heads,
            self.num_layers,
            dropout,
            self.ext_layers,
        )

        # U-Net Decoder
        self.decoder0 = nn.Sequential(
            Conv3DBlock(input_dim, 32, 3, pad=pad_decoder),
            Conv3DBlock(32, 64, 3, pad=pad_decoder),
        )

        self.decoder3 = nn.Sequential(
            Deconv3DBlock(
                embed_dim, 512, pad=pad_decoder, upsample_factors=upsample_factors[0]
            ),
            Deconv3DBlock(
                512, 256, pad=pad_decoder, upsample_factors=upsample_factors[1]
            ),
            Deconv3DBlock(
                256, 128, pad=pad_decoder, upsample_factors=upsample_factors[2]
            ),
        )

        self.decoder6 = nn.Sequential(
            Deconv3DBlock(
                embed_dim, 512, pad=pad_decoder, upsample_factors=upsample_factors[0]
            ),
            Deconv3DBlock(
                512, 256, pad=pad_decoder, upsample_factors=upsample_factors[1]
            ),
        )

        self.decoder9 = Deconv3DBlock(
            embed_dim, 512, pad=pad_decoder, upsample_factors=upsample_factors[0]
        )

        self.decoder12_upsampler = SingleDeconv3DBlock(
            embed_dim, 512, upsample_factors=upsample_factors[0]
        )

        self.decoder9_upsampler = nn.Sequential(
            Conv3DBlock(1024, 512, pad=pad_decoder),
            Conv3DBlock(512, 512, pad=pad_decoder),
            Conv3DBlock(512, 512, pad=pad_decoder),
            SingleDeconv3DBlock(512, 256, upsample_factors=upsample_factors[1]),
        )

        self.decoder6_upsampler = nn.Sequential(
            Conv3DBlock(512, 256, pad=pad_decoder),
            Conv3DBlock(256, 256, pad=pad_decoder),
            SingleDeconv3DBlock(256, 128, upsample_factors=upsample_factors[2]),
        )

        self.decoder3_upsampler = nn.Sequential(
            Conv3DBlock(256, 128, pad=pad_decoder),
            Conv3DBlock(128, 128, pad=pad_decoder),
            SingleDeconv3DBlock(128, 64, upsample_factors=upsample_factors[3]),
        )

        self.decoder0_header = nn.Sequential(
            Conv3DBlock(128, 64, pad=pad_decoder),
            Conv3DBlock(64, 64, pad=pad_decoder),
            SingleConv3DBlock(64, output_dim, 1),
        )

    def forward(self, x):
        z = self.transformer(x)
        z0, z3, z6, z9, z12 = x, *z
        # z0, z3, z6 = x, *z
        z3 = z3.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z6 = z6.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z9 = z9.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z12 = z12.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)

        z12 = self.decoder12_upsampler(z12)

        z9 = self.decoder9(z9)
        if self.pad_decoder:
            inp_ = torch.cat([z9, z12], dim=1)
        else:
            # we need to crop z12
            inp_ = UNetResConcat(z12, z9)
        z9 = self.decoder9_upsampler(inp_)

        z6 = self.decoder6(z6)
        if self.pad_decoder:
            inp_ = torch.cat([z6, z9], dim=1)
        else:
            # reverse order of cropping for all other skip connections than the first
            inp_ = UNetResConcat(z6, z9)
        z6 = self.decoder6_upsampler(inp_)

        z3 = self.decoder3(z3)
        if self.pad_decoder:
            inp_ = torch.cat([z3, z6], dim=1)
        else:
            # we need to crop z12
            inp_ = UNetResConcat(z3, z6)
        z3 = self.decoder3_upsampler(inp_)

        z0 = self.decoder0(z0)
        if self.pad_decoder:
            inp_ = torch.cat([z0, z3], dim=1)
        else:
            # we need to crop z12
            inp_ = UNetResConcat(z0, z3)
        output = self.decoder0_header(inp_)
        return output
