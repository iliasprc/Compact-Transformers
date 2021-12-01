import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.nn import Module, ModuleList, Linear, Dropout, LayerNorm, Identity, Parameter, init

from src.utils.lds import batch_image_to_Om
from src.utils.riemmanian_utils import log_dist
from .stochastic_depth import DropPath


################  GRASSMANN LIASSS ######################################33
class NonTrainableObsMatrixModule(nn.Module):
    def __init__(self, image_size=224, patch_size=16, channels=3, m=13, lds_size=3):
        super().__init__()
        # image_height, image_width = pair(image_size)
        # patch_height, patch_width = pair(patch_size)

        assert image_size % patch_size == 0, 'Image dimensions must be ' \
                                             'divisible by the patch size.'

        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        self.to_patch_embedding = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)

        self.project = True

        self.m = m
        self.lds_size = lds_size
        self.num_input_channels = 3
        self.num_patches = num_patches
        self.patc_dim = patch_dim

    def forward(self, img):
        with torch.no_grad():
            x = self.to_patch_embedding(img)

            x = batch_image_to_Om(x, lds_size=self.lds_size, m=self.m, num_channels=self.num_input_channels)
        return x


class ObsMatrixTokenizer(nn.Module):
    def __init__(self, image_size=224, patch_size=16, channels=3, m=4, lds_size=4, return_gradients=True):
        super().__init__()
        # image_height, image_width = pair(image_size)
        # patch_height, patch_width = pair(patch_size)
        #
        # assert image_size % patch_size == 0, f' { image_size} {patch_size} Image dimensions must be ' \
        #                                      'divisible by the patch size.'

        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        self.project = True

        self.m = m
        self.lds_size = lds_size
        self.num_input_channels = channels
        self.return_gradients = return_gradients

    def forward(self, x):
        if self.return_gradients:
            x = batch_image_to_Om(x, lds_size=self.lds_size, m=self.m)
        else:
            with torch.no_grad():
                x = batch_image_to_Om(x, lds_size=self.lds_size, m=self.m)
        return x


class ManifoldAttention(nn.Module):
    def __init__(self, dim, num_heads=8, attention_dropout=0.1, projection_dropout=0.1, sequence_length=-1):
        super().__init__()

        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5

        self.qkv = Linear(dim, dim * 3, bias=False)

        self.attn_drop = Dropout(attention_dropout)
        self.proj = Linear(dim, dim)
        self.proj_drop = Dropout(projection_dropout)
        self.sequence_len = sequence_length
        self.attn_matrix = nn.Parameter(torch.randn(sequence_length, sequence_length))
        self.conv_attn = nn.Sequential(

            nn.Conv2d(in_channels=2 * self.num_heads, out_channels=num_heads, kernel_size=(1, 1), bias=False)
        )

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        old_shape = q.shape

        # qgr = Bgrassmanian_point(q)
        # kgr = Bgrassmanian_point(k)
        # # rieem_attn = log_dist(q, k, use_covariance=True, use_log=False)
        #
        # qgr = rearrange(qgr, 'b h t d -> b h d t').unsqueeze(-1)
        # kgr = rearrange(kgr, 'b h t d -> b h d t').unsqueeze(-2)
        # dots = torch.matmul(qgr, kgr)  # * self.scale
        #
        # attn_grassmman = torch.linalg.norm(dots, dim=2) ** 2.
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_riemmanian = log_dist(q, k, use_covariance=True, use_log=False)

        attn_ = self.conv_attn(self.attn_drop(torch.cat((attn, attn_riemmanian), dim=1))).softmax(dim=-1)

        out = torch.matmul(attn_, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.proj_drop(self.proj(out))


class ManifoldEncoderLayer(Module):
    """
    Inspired by torch.nn.TransformerEncoderLayer and timm.
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 attention_dropout=0.1, drop_path_rate=0.1, sequence_length=-1):
        super(ManifoldEncoderLayer, self).__init__()
        self.pre_norm = LayerNorm(d_model)
        self.self_attn = ManifoldAttention(dim=d_model, num_heads=nhead,
                                           attention_dropout=attention_dropout, projection_dropout=dropout,
                                           sequence_length=sequence_length)

        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout1 = Dropout(dropout)
        self.norm1 = LayerNorm(d_model)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.dropout2 = Dropout(dropout)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else Identity()

        self.activation = F.gelu

    def forward(self, src: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        src = src + self.drop_path(self.self_attn(self.pre_norm(src)))
        src = self.norm1(src)
        src2 = self.linear2(self.dropout1(self.activation(self.linear1(src))))
        src = src + self.drop_path(self.dropout2(src2))
        return src


class ManifoldformerClassifier(Module):
    def __init__(self,
                 seq_pool=True,
                 embedding_dim=768,
                 num_layers=12,
                 num_heads=12,
                 mlp_ratio=4.0,
                 num_classes=1000,
                 dropout=0.1,
                 attention_dropout=0.1,
                 stochastic_depth=0.1,
                 positional_embedding='learnable',
                 use_grassman=True,
                 sequence_length=None):
        super().__init__()
        positional_embedding = positional_embedding if \
            positional_embedding in ['sine', 'learnable', 'none'] else 'sine'
        dim_feedforward = int(embedding_dim * mlp_ratio)
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.seq_pool = seq_pool

        assert sequence_length is not None or positional_embedding == 'none', \
            f"Positional embedding is set to {positional_embedding} and" \
            f" the sequence length was not specified."

        if not seq_pool:
            sequence_length += 1
            self.class_emb = Parameter(torch.zeros(1, 1, self.embedding_dim),
                                       requires_grad=True)
        else:
            self.attention_pool = Linear(self.embedding_dim, 1)

        if positional_embedding != 'none':
            if positional_embedding == 'learnable':
                self.positional_emb = Parameter(torch.zeros(1, sequence_length, embedding_dim),
                                                requires_grad=True)
                init.trunc_normal_(self.positional_emb, std=0.2)
            else:
                self.positional_emb = Parameter(self.sinusoidal_embedding(sequence_length, embedding_dim),
                                                requires_grad=False)
        else:
            self.positional_emb = None

        self.dropout = Dropout(p=dropout)
        dpr = [x.item() for x in torch.linspace(0, stochastic_depth, num_layers)]
        self.blocks = ModuleList([
            ManifoldEncoderLayer(d_model=embedding_dim, nhead=num_heads,
                                 dim_feedforward=dim_feedforward, dropout=dropout,
                                 attention_dropout=attention_dropout, drop_path_rate=dpr[i],
                                 sequence_length=sequence_length)
            for i in range(num_layers)])
        self.norm = LayerNorm(embedding_dim)

        self.fc = Linear(embedding_dim, num_classes)
        self.apply(self.init_weight)

    def forward(self, x):
        if self.positional_emb is None and x.size(1) < self.sequence_length:
            x = F.pad(x, (0, 0, 0, self.n_channels - x.size(1)), mode='constant', value=0)

        if not self.seq_pool:
            cls_token = self.class_emb.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        if self.positional_emb is not None:
            x += self.positional_emb

        x = self.dropout(x)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        if self.seq_pool:
            x = torch.matmul(F.softmax(self.attention_pool(x), dim=1).transpose(-1, -2), x).squeeze(-2)

        else:
            x = x[:, 0]

        x = self.fc(x)
        return x

    @staticmethod
    def init_weight(m):
        if isinstance(m, Linear):
            init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, Linear) and m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, LayerNorm):
            init.constant_(m.bias, 0)
            init.constant_(m.weight, 1.0)

    @staticmethod
    def sinusoidal_embedding(n_channels, dim):
        pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                                for p in range(n_channels)])
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        return pe.unsqueeze(0)
