import sys

sys.path.append('.')

from collections import OrderedDict

import torch
import torch.nn as nn
from ViT3d_cls import ViT


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


import math


class IntermediateSequential(nn.Sequential):
    def __init__(self, *args, return_intermediate=False):
        super().__init__(*args)
        self.return_intermediate = return_intermediate

    def forward(self, *input):
        if not self.return_intermediate:
            return super().forward(*input)

        intermediate_outputs = {}
        output = input
        for name, module in self.named_children():
            output = intermediate_outputs[name] = module(*output)

        return output, intermediate_outputs

import warnings
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)




class QueryNet(nn.Module):

    def __init__(self, embed_dim=768, mlp_ratio=4, num_heads=8):
        super(QueryNet, self).__init__()

        self.vis_proj = Mlp(in_features=embed_dim,
                            hidden_features=embed_dim * mlp_ratio,
                            out_features=embed_dim)
        self.lin_proj = Mlp(in_features=embed_dim,
                            hidden_features=embed_dim * mlp_ratio,
                            out_features=embed_dim)
        self.value_proj = Mlp(in_features=embed_dim,
                              hidden_features=embed_dim * mlp_ratio,
                              out_features=embed_dim)

        self.norm = nn.LayerNorm(embed_dim)
        self.num_heads = num_heads

        head_dim = embed_dim // num_heads
        self.scale = head_dim ** -0.5
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, lang_f, cur_v):
        _v = cur_v
        # _m_l = torch.mean(_loc_mem, dim=1)
        # _m_g = torch.mean(_glo_mem, dim=1)

        B, L, C = lang_f.shape
        _v_p = self.vis_proj(_v).view(B, self.num_heads, C // self.num_heads, 1)  # * B,H,D,1
        _l_p = self.lin_proj(lang_f)  # * B,L,C
        _l_v = self.value_proj(lang_f)  # * B,L,C

        k = _l_p.reshape(B, L, self.num_heads, C // self.num_heads).transpose(1, 2)  # * B,H,L,D

        _attn = torch.matmul(k, _v_p).squeeze(-1) * self.scale  # * B,H,L
        _attn = torch.softmax(_attn, dim=-1)  # * B,H,L

        query = torch.bmm(_attn, _l_v)

        query = self.norm(query)  # * B,H,C

        return query, _attn

def _n2p(w, t=True):
    if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
        w = w.flatten()
    if t:
        if w.ndim == 4:
            w = w.transpose([3, 2, 0, 1])
        elif w.ndim == 3:
            w = w.transpose([2, 0, 1])
        elif w.ndim == 2:
            w = w.transpose([1, 0])
    return torch.from_numpy(w)

import numpy as np
@torch.no_grad()
def _load_block_weights(model, checkpoint_path, prefix='', start_i=0):
    w = np.load(checkpoint_path)
    if not prefix and 'opt/target/embedding/kernel' in w:
        prefix = 'opt/target/'

    print('Load Vit with First [{}] Layers from [{}]'.format(len(model.blocks), start_i))
    for _i, block in enumerate(model.blocks.children()):
        i = start_i + _i

        block_prefix = f'{prefix}Transformer/encoderblock_{i}/'
        mha_prefix = block_prefix + 'MultiHeadDotProductAttention_1/'
        block.norm1.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale']))
        block.norm1.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias']))
        block.attn.qkv.weight.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/kernel'], t=False).flatten(1).T for n in ('query', 'key', 'value')]))
        block.attn.qkv.bias.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/bias'], t=False).reshape(-1) for n in ('query', 'key', 'value')]))
        block.attn.proj.weight.copy_(_n2p(w[f'{mha_prefix}out/kernel']).flatten(1))
        block.attn.proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias']))
        for r in range(2):
            getattr(block.mlp, f'fc{r + 1}').weight.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/kernel']))
            getattr(block.mlp, f'fc{r + 1}').bias.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/bias']))
        block.norm2.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/scale']))
        block.norm2.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/bias']))
class EncodeNet(nn.Module):
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, init_depth=6,
                 cascade_lang_fuse=False, post_depth=1):
        super(EncodeNet, self).__init__()

        self.modal_embed = nn.Parameter(torch.zeros(1, 2, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.cascade_lang_fuse = cascade_lang_fuse

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.pos_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(post_depth)])

        self.norm = norm_layer(embed_dim)

        trunc_normal_(self.modal_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix='', start_i=0):
        _load_block_weights(self, checkpoint_path, prefix, start_i)

    def forward(self, vis, lang):
        N_v = vis.shape[1]
        B, N_l, C = lang.shape
        # B,1+N_v,C
        vis_pos = vis + self.modal_embed[:, :1]
        lang_pos = lang + self.modal_embed[:, 1:]

        if vis.shape[0] != B:
            T = int(vis.shape[0] / B)
            lang_pos = lang_pos.view(B, 1, N_l, C).repeat(1, T, 1, 1).contiguous().view(-1, N_l, C)

        modal_f = torch.cat([vis_pos, lang_pos], dim=1)  # * B,N_v+N_l,C

        for blk in self.blocks:
            modal_f = blk(modal_f)
        modal_f = self.norm(modal_f)

        modal_f = modal_f[:, :N_v]
        if self.cascade_lang_fuse:
            _lang = modal_f[:, N_v:]
        else:
            _lang = lang

        for pos_blk in self.pos_blocks:
            modal_f = pos_blk(modal_f)

        modal_f = self.norm(modal_f)

        return (modal_f, _lang)


class ProjectDotSurv(nn.Module):
    def __init__(self, dim, num_heads=8, bias=True, scale=None, cat_enc_mode=False, norm=False):
        super().__init__()
        # print(num_heads)
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=bias)
        self.k = nn.Linear(dim, dim, bias=bias)

        dec_dim = num_heads * num_heads
        if cat_enc_mode == 'concat':
            dec_dim += dim
        elif cat_enc_mode == 'residule':
            self.res_proj = nn.Linear(dim, dec_dim, bias=bias)

        # self.mask_head = self._pup_predict_block(in_c=dec_dim)
        self.cat_enc_mode = cat_enc_mode
        self.norm_out = nn.LayerNorm(768 + 64, eps=1e-6)
        self.surv_head = nn.Linear(768 + 64, 1)
        self.norm = norm

    def forward(self, params):
        x, y = params
        B, N, C = x.shape
        _, M, _ = y.shape

        q = self.q(x).reshape(B, N, 1, self.num_heads, C // self.num_heads)
        q = q.permute(2, 0, 3, 1, 4)
        k = self.k(y).reshape(B, M, 1, self.num_heads, C // self.num_heads)
        k = k.permute(2, 0, 3, 1, 4)
        q, k = q[0], k[0]  # * B,H,N,D

        if self.norm:  # * in cosine distance
            q = torch.nn.functional.normalize(q, p=2, dim=-1)
            k = torch.nn.functional.normalize(k, p=2, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # * B,H,N,M

        if self.num_heads != 1:
            P1 = 256 // 16
            P2 = 64 // 16
            # print(self.num_heads)
            attn = attn.transpose(-1, -2).reshape(B, -1, N)  # * B,H*M,N
            if self.cat_enc_mode == 'concat':
                # * B,N,D -> B,D,N -> B,D+H*M,N
                _x = x.transpose(-1, -2)
                attn = torch.cat([_x, attn], dim=1)
            elif self.cat_enc_mode == 'residule':
                _x = self.res_proj(x).transpose(-1, -2)
                attn += _x

            attn = attn.mean(-1)
            attn = self.norm_out(attn)
            out = self.surv_head(attn)

        return out


class VLFTrans(nn.Module):
    def __init__(
            self,
            img_dim=320,
            patch_dim=16.,
            in_chans=3,
            embedding_dim=768,
            K=3,
            global_mem_size=1.5,
            local_mem_size=2,
            fuse_depth=2,
            post_depth=2,
            vision_depth=6,
            cat_enc_mode='concat'
    ):
        super(VLFTrans, self).__init__()

        self.img_dim = img_dim
        self.embedding_dim = embedding_dim
        self.patch_dim = patch_dim
        self.patch_num = int(img_dim / self.patch_dim) ** 2

        self.in_chans = in_chans
        self.K = K

        self.local_mem_size = local_mem_size * self.patch_num
        self.global_mem_size = global_mem_size * self.patch_num

        self._local_mem = None
        self._global_mem = None
        self.norm_ref = nn.LayerNorm(768, eps=1e-6)
        self.surv_ref_head = nn.Linear(768, 1)

        self.vision_encoder = ViT(
            image_size=512,  # image size
            frames=512,  # max number of frames
            image_patch_size=32,  # image patch size
            frame_patch_size=4,  # frame patch size
            dim=768,
            depth=12,
            heads=8,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1
        )

        self.ref_encoder = ViT(
            image_size=512,  # image size
            frames=512,  # max number of frames
            image_patch_size=32,  # image patch size
            frame_patch_size=4,  # frame patch size
            dim=768,
            depth=12,
            heads=8,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1
        )
        self.trans_fusion = IntermediateSequential(OrderedDict(
            [(str(i), EncodeNet(depth=fuse_depth, post_depth=post_depth, cascade_lang_fuse=False))
             for i in range(K)]), return_intermediate=True)

        self.query_generator = QueryNet(embed_dim=self.embedding_dim, num_heads=8)

        self.main_surv_net = ProjectDotSurv(dim=self.embedding_dim, num_heads=8, cat_enc_mode=cat_enc_mode, norm=False)

    def encode(self, vis, lang):
        # B,T,C,H,W = vis.shape
        # vis = vis.view(-1,C,H,W)

        vis_f, pos1 = self.vision_encoder(vis)

        ref_f, pos2 = self.ref_encoder(lang)

        return vis_f, ref_f

    def forward(self, ref, vis):
        B, C, H, W, D = vis.shape

        # B,T,C,H,W
        vis_f, ref_f = self.encode(vis, ref)

        # fusion
        (fused_v, _post_ref), _interm_fused_v = self.trans_fusion(vis_f, ref_f)

        cur_v = fused_v[:, 1]

        ref_out = ref_f[:, 1:].mean(dim=1)
        ref_out = self.norm_ref(ref_out)
        ref_out = self.surv_ref_head(ref_out)

        # gerneate query
        query, _query_attn = self.query_generator(ref_f[:, 1:], cur_v, )  # * B,C
        out = self.main_surv_net((vis_f[:, 1:], query))
        return ref_out, out

    def _get_padding(self, padding_type, kernel_size):
        assert padding_type in ['SAME', 'VALID']
        if padding_type == 'SAME':
            _list = [(k - 1) // 2 for k in kernel_size]
            return tuple(_list)
        return tuple(0 for _ in kernel_size)

    def _reshape_output(self, x, embedding_dim):
        x = x.view(
            x.size(0),
            int(self.img_dim / self.patch_dim),
            int(self.img_dim / self.patch_dim),
            embedding_dim,
        )
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

    def load_state_dict(self, new_state):
        state = self.state_dict()
        for layer in state:
            if layer in new_state:
                if state[layer].size() == new_state[layer].size():
                    state[layer] = new_state[layer]
            elif 'module.' + layer in new_state:
                new_layer = 'module.' + layer
                if state[layer].size() == new_state[new_layer].size():
                    state[layer] = new_state[new_layer]
            else:
                print(layer)

        super().load_state_dict(state)

class ProjectDotSurv_cli(nn.Module):
    def __init__(self, dim, cli_dim=25, num_heads=8, bias=True, scale=None, cat_enc_mode=False, norm=False):
        super().__init__()
        # print(num_heads)
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=bias)
        self.k = nn.Linear(dim, dim, bias=bias)

        dec_dim = num_heads * num_heads
        if cat_enc_mode == 'concat':
            dec_dim += dim
        elif cat_enc_mode == 'residule':
            self.res_proj = nn.Linear(dim, dec_dim, bias=bias)

        # self.mask_head = self._pup_predict_block(in_c=dec_dim)
        self.cat_enc_mode = cat_enc_mode
        self.norm_out = nn.LayerNorm(768 + 64, eps=1e-6)
        self.surv_head1 = nn.Sequential(
            nn.Linear(768 + 64, 4),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(),
        )

        self.surv_head2 = nn.Sequential(
            nn.Linear(cli_dim, 2),
            nn.BatchNorm1d(2),
            nn.LeakyReLU(),
        )
        self.surv_head3 = nn.Linear(4 + 2, 1)

        self.norm = norm

    def forward(self, params):
        x, y, cli = params
        B, N, C = x.shape
        _, M, _ = y.shape

        q = self.q(x).reshape(B, N, 1, self.num_heads, C // self.num_heads)
        q = q.permute(2, 0, 3, 1, 4)
        k = self.k(y).reshape(B, M, 1, self.num_heads, C // self.num_heads)
        k = k.permute(2, 0, 3, 1, 4)
        q, k = q[0], k[0]  # * B,H,N,D

        if self.norm:  # * in cosine distance
            q = torch.nn.functional.normalize(q, p=2, dim=-1)
            k = torch.nn.functional.normalize(k, p=2, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # * B,H,N,M

        if self.num_heads != 1:
            P1 = 256 // 16
            P2 = 64 // 16
            # print(self.num_heads)
            attn = attn.transpose(-1, -2).reshape(B, -1, N)  # * B,H*M,N
            if self.cat_enc_mode == 'concat':
                # * B,N,D -> B,D,N -> B,D+H*M,N
                _x = x.transpose(-1, -2)
                attn = torch.cat([_x, attn], dim=1)
            elif self.cat_enc_mode == 'residule':
                _x = self.res_proj(x).transpose(-1, -2)
                attn += _x

            attn = attn.mean(-1)
            attn = self.norm_out(attn)
            mid1 = self.surv_head1(attn)
            mid2 = self.surv_head2(cli)
            out = self.surv_head3(torch.cat([mid1, mid2], dim=-1))

        return out

class ProjectDotSurv_cli2(nn.Module):
    def __init__(self, dim, cli_dim=25, num_heads=8, bias=True, scale=None, cat_enc_mode=False, norm=False):
        super().__init__()
        # print(num_heads)
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=bias)
        self.k = nn.Linear(dim, dim, bias=bias)

        dec_dim = num_heads * num_heads
        if cat_enc_mode == 'concat':
            dec_dim += dim
        elif cat_enc_mode == 'residule':
            self.res_proj = nn.Linear(dim, dec_dim, bias=bias)

        # self.maskK_head = self._pup_predict_block(in_c=dec_dim)
        self.cat_enc_mode = cat_enc_mode
        self.norm_out = nn.LayerNorm(768 + 64, eps=1e-6)
        self.surv_head1 = nn.Sequential(
            nn.Linear(768 + 64, 4),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(),
        )

        self.surv_head2 = nn.Sequential(
            nn.Linear(cli_dim, 2),
            nn.BatchNorm1d(2),
            nn.LeakyReLU(),
        )


        self.surv_head3 = nn.Linear(4 + 2, 1)


        self.surv_head4 = nn.Linear(2, 1)

        self.norm = norm

    def forward(self, params):
        x, y, cli = params
        B, N, C = x.shape
        _, M, _ = y.shape

        q = self.q(x).reshape(B, N, 1, self.num_heads, C // self.num_heads)
        q = q.permute(2, 0, 3, 1, 4)
        k = self.k(y).reshape(B, M, 1, self.num_heads, C // self.num_heads)
        k = k.permute(2, 0, 3, 1, 4)
        q, k = q[0], k[0]  # * B,H,N,D

        if self.norm:  # * in cosine distance
            q = torch.nn.functional.normalize(q, p=2, dim=-1)
            k = torch.nn.functional.normalize(k, p=2, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # * B,H,N,M

        if self.num_heads != 1:
            P1 = 256 // 16
            P2 = 64 // 16
            # print(self.num_heads)
            attn = attn.transpose(-1, -2).reshape(B, -1, N)  # * B,H*M,N
            if self.cat_enc_mode == 'concat':
                # * B,N,D -> B,D,N -> B,D+H*M,N
                _x = x.transpose(-1, -2)
                attn = torch.cat([_x, attn], dim=1)
            elif self.cat_enc_mode == 'residule':
                _x = self.res_proj(x).transpose(-1, -2)
                attn += _x

            attn = attn.mean(-1)
            attn = self.norm_out(attn)
            mid1 = self.surv_head1(attn)
            mid2 = self.surv_head2(cli)
            out = self.surv_head3(torch.cat([mid1, mid2], dim=-1))
            cli_out = self.surv_head4(mid2)
            # cli_out = mid2

        return cli_out, out

class VLFTrans_cli(nn.Module):
    def __init__(
            self,
            img_dim=320,
            patch_dim=16.,
            in_chans=3,
            embedding_dim=768,
            K=3,
            global_mem_size=1.5,
            local_mem_size=2,
            fuse_depth=2,
            post_depth=2,
            vision_depth=6,
            cat_enc_mode='concat'
    ):
        super(VLFTrans_cli, self).__init__()

        self.img_dim = img_dim
        self.embedding_dim = embedding_dim
        self.patch_dim = patch_dim
        self.patch_num = int(img_dim / self.patch_dim) ** 2

        self.in_chans = in_chans
        self.K = K

        self.local_mem_size = local_mem_size * self.patch_num
        self.global_mem_size = global_mem_size * self.patch_num

        self._local_mem = None
        self._global_mem = None
        self.norm_ref = nn.LayerNorm(768, eps=1e-6)
        self.surv_ref_head = nn.Linear(768, 1)

        self.vision_encoder = ViT(
            image_size=512,  # image size
            frames=512,  # max number of frames
            image_patch_size=32,  # image patch size
            frame_patch_size=4,  # frame patch size
            dim=768,
            depth=12,
            heads=8,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1
        )

        self.ref_encoder = ViT(
            image_size=512,  # image size
            frames=512,  # max number of frames
            image_patch_size=32,  # image patch size
            frame_patch_size=4,  # frame patch size
            dim=768,
            depth=12,
            heads=8,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1
        )
        self.trans_fusion = IntermediateSequential(OrderedDict(
            [(str(i), EncodeNet(depth=fuse_depth, post_depth=post_depth, cascade_lang_fuse=False))
             for i in range(K)]), return_intermediate=True)

        self.query_generator = QueryNet(embed_dim=self.embedding_dim, num_heads=8)

        self.main_surv_net = ProjectDotSurv_cli(dim=self.embedding_dim, num_heads=8, cat_enc_mode=cat_enc_mode, norm=False)

    def encode(self, vis, lang):
        # B,T,C,H,W = vis.shape
        # vis = vis.view(-1,C,H,W)

        vis_f, pos1 = self.vision_encoder(vis)

        ref_f, pos2 = self.ref_encoder(lang)

        return vis_f, ref_f

    def forward(self, ref, vis, cli):
        B, C, H, W, D = vis.shape

        # B,T,C,H,W
        vis_f, ref_f = self.encode(vis, ref)

        # fusion
        (fused_v, _post_ref), _interm_fused_v = self.trans_fusion(vis_f, ref_f)

        cur_v = fused_v[:, 1]

        ref_out = ref_f[:, 1:].mean(dim=1)
        ref_out = self.norm_ref(ref_out)
        ref_out = self.surv_ref_head(ref_out)

        # gerneate query
        query, _query_attn = self.query_generator(ref_f[:, 1:], cur_v, )  # * B,C
        out = self.main_surv_net((vis_f[:, 1:], query, cli))
        return ref_out, out

    def _get_padding(self, padding_type, kernel_size):
        assert padding_type in ['SAME', 'VALID']
        if padding_type == 'SAME':
            _list = [(k - 1) // 2 for k in kernel_size]
            return tuple(_list)
        return tuple(0 for _ in kernel_size)

    def _reshape_output(self, x, embedding_dim):
        x = x.view(
            x.size(0),
            int(self.img_dim / self.patch_dim),
            int(self.img_dim / self.patch_dim),
            embedding_dim,
        )
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

    def load_state_dict(self, new_state):
        state = self.state_dict()
        for layer in state:
            if layer in new_state:
                if state[layer].size() == new_state[layer].size():
                    state[layer] = new_state[layer]
            elif 'module.' + layer in new_state:
                new_layer = 'module.' + layer
                if state[layer].size() == new_state[new_layer].size():
                    state[layer] = new_state[new_layer]
            else:
                print(layer)

        super().load_state_dict(state)


class VLFTrans_cli2(nn.Module):
    def __init__(
            self,
            img_dim=320,
            patch_dim=16.,
            in_chans=3,
            embedding_dim=768,
            K=3,
            global_mem_size=1.5,
            local_mem_size=2,
            fuse_depth=2,
            post_depth=2,
            vision_depth=6,
            cli_dim=25,
            cat_enc_mode='concat'
    ):
        super(VLFTrans_cli2, self).__init__()

        self.img_dim = img_dim
        self.embedding_dim = embedding_dim
        self.patch_dim = patch_dim
        self.patch_num = int(img_dim / self.patch_dim) ** 2

        self.in_chans = in_chans
        self.K = K

        self.local_mem_size = local_mem_size * self.patch_num
        self.global_mem_size = global_mem_size * self.patch_num

        self._local_mem = None
        self._global_mem = None
        self.norm_ref = nn.LayerNorm(768, eps=1e-6)
        self.surv_ref_head = nn.Linear(768, 1)

        self.vision_encoder = ViT(
            image_size=512,  # image size
            frames=512,  # max number of frames
            image_patch_size=32,  # image patch size
            frame_patch_size=4,  # frame patch size
            dim=768,
            depth=12,
            heads=8,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1
        )

        self.ref_encoder = ViT(
            image_size=512,  # image size
            frames=512,  # max number of frames
            image_patch_size=32,  # image patch size
            frame_patch_size=4,  # frame patch size
            dim=768,
            depth=12,
            heads=8,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1
        )
        self.trans_fusion = IntermediateSequential(OrderedDict(
            [(str(i), EncodeNet(depth=fuse_depth, post_depth=post_depth, cascade_lang_fuse=False))
             for i in range(K)]), return_intermediate=True)

        self.query_generator = QueryNet(embed_dim=self.embedding_dim, num_heads=8)

        self.main_surv_net = ProjectDotSurv_cli2(dim=self.embedding_dim, cli_dim=cli_dim, num_heads=8,
                                                 cat_enc_mode=cat_enc_mode, norm=False)

    def encode(self, vis, lang):
        # B,T,C,H,W = vis.shape
        # vis = vis.view(-1,C,H,W)

        vis_f, pos1 = self.vision_encoder(vis)

        ref_f, pos2 = self.ref_encoder(lang)

        return vis_f, ref_f

    def forward(self, ref, vis, cli):
        B, C, H, W, D = vis.shape

        # B,T,C,H,W
        vis_f, ref_f = self.encode(vis, ref)

        # fusion
        (fused_v, _post_ref), _interm_fused_v = self.trans_fusion(vis_f, ref_f)

        cur_v = fused_v[:, 1]

        ref_out = ref_f[:, 1:].mean(dim=1)
        ref_out = self.norm_ref(ref_out)
        ref_out = self.surv_ref_head(ref_out)

        # gerneate query
        query, _query_attn = self.query_generator(ref_f[:, 1:], cur_v, )  # * B,C
        # print(query.shape)
        cli_out, out = self.main_surv_net((vis_f[:, 1:], query, cli))
        return ref_out, cli_out, out

    def _get_padding(self, padding_type, kernel_size):
        assert padding_type in ['SAME', 'VALID']
        if padding_type == 'SAME':
            _list = [(k - 1) // 2 for k in kernel_size]
            return tuple(_list)
        return tuple(0 for _ in kernel_size)

    def _reshape_output(self, x, embedding_dim):
        x = x.view(
            x.size(0),
            int(self.img_dim / self.patch_dim),
            int(self.img_dim / self.patch_dim),
            embedding_dim,
        )
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

    def load_state_dict(self, new_state):
        state = self.state_dict()
        for layer in state:
            if layer in new_state:
                if state[layer].size() == new_state[layer].size():
                    state[layer] = new_state[layer]
            elif 'module.' + layer in new_state:
                new_layer = 'module.' + layer
                if state[layer].size() == new_state[new_layer].size():
                    state[layer] = new_state[new_layer]
            else:
                print(layer)

        super().load_state_dict(state)


class VLFTrans_cli2_one_step_fusion(nn.Module):
    def __init__(
            self,
            img_dim=320,
            patch_dim=16.,
            in_chans=3,
            embedding_dim=768,
            K=3,
            global_mem_size=1.5,
            local_mem_size=2,
            fuse_depth=2,
            post_depth=2,
            vision_depth=6,
            cli_dim=25,
            cat_enc_mode='concat'
    ):
        super(VLFTrans_cli2, self).__init__()

        self.img_dim = img_dim
        self.embedding_dim = embedding_dim
        self.patch_dim = patch_dim
        self.patch_num = int(img_dim / self.patch_dim) ** 2

        self.in_chans = in_chans
        self.K = K

        self.local_mem_size = local_mem_size * self.patch_num
        self.global_mem_size = global_mem_size * self.patch_num

        self._local_mem = None
        self._global_mem = None
        self.norm_ref = nn.LayerNorm(768, eps=1e-6)
        self.surv_ref_head = nn.Linear(768, 1)

        self.vision_encoder = ViT(
            image_size=512,  # image size
            frames=512,  # max number of frames
            image_patch_size=32,  # image patch size
            frame_patch_size=4,  # frame patch size
            dim=768,
            depth=12,
            heads=8,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1
        )

        self.ref_encoder = ViT(
            image_size=512,  # image size
            frames=512,  # max number of frames
            image_patch_size=32,  # image patch size
            frame_patch_size=4,  # frame patch size
            dim=768,
            depth=12,
            heads=8,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1
        )
        self.trans_fusion = IntermediateSequential(OrderedDict(
            [(str(i), EncodeNet(depth=fuse_depth, post_depth=0, cascade_lang_fuse=False))
             for i in range(K)]), return_intermediate=True)

        self.query_generator = QueryNet(embed_dim=self.embedding_dim, num_heads=8)

        self.main_surv_net = ProjectDotSurv_cli2(dim=self.embedding_dim, cli_dim=cli_dim, num_heads=8,
                                                 cat_enc_mode=cat_enc_mode, norm=False)

    def encode(self, vis, lang):
        # B,T,C,H,W = vis.shape
        # vis = vis.view(-1,C,H,W)

        vis_f, pos1 = self.vision_encoder(vis)

        ref_f, pos2 = self.ref_encoder(lang)

        return vis_f, ref_f

    def forward(self, ref, vis, cli):
        B, C, H, W, D = vis.shape

        # B,T,C,H,W
        vis_f, ref_f = self.encode(vis, ref)

        # fusion
        (fused_v, _post_ref), _interm_fused_v = self.trans_fusion(vis_f, ref_f)

        cur_v = fused_v[:, 1]

        ref_out = ref_f[:, 1:].mean(dim=1)
        ref_out = self.norm_ref(ref_out)
        ref_out = self.surv_ref_head(ref_out)

        # gerneate query
        query, _query_attn = self.query_generator(ref_f[:, 1:], cur_v, )  # * B,C
        # print(query.shape)
        cli_out, out = self.main_surv_net((vis_f[:, 1:], query, cli))
        return ref_out, cli_out, out

    def _get_padding(self, padding_type, kernel_size):
        assert padding_type in ['SAME', 'VALID']
        if padding_type == 'SAME':
            _list = [(k - 1) // 2 for k in kernel_size]
            return tuple(_list)
        return tuple(0 for _ in kernel_size)

    def _reshape_output(self, x, embedding_dim):
        x = x.view(
            x.size(0),
            int(self.img_dim / self.patch_dim),
            int(self.img_dim / self.patch_dim),
            embedding_dim,
        )
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

    def load_state_dict(self, new_state):
        state = self.state_dict()
        for layer in state:
            if layer in new_state:
                if state[layer].size() == new_state[layer].size():
                    state[layer] = new_state[layer]
            elif 'module.' + layer in new_state:
                new_layer = 'module.' + layer
                if state[layer].size() == new_state[new_layer].size():
                    state[layer] = new_state[new_layer]
            else:
                print(layer)

        super().load_state_dict(state)
class VLFTrans_cli2_abpf(nn.Module):
    def __init__(
            self,
            img_dim=320,
            patch_dim=16.,
            in_chans=3,
            embedding_dim=768,
            K=3,
            global_mem_size=1.5,
            local_mem_size=2,
            fuse_depth=2,
            post_depth=2,
            vision_depth=6,
            cli_dim=25,
            cat_enc_mode='concat'
    ):
        super(VLFTrans_cli2_abpf, self).__init__()

        self.img_dim = img_dim
        self.embedding_dim = embedding_dim
        self.patch_dim = patch_dim
        self.patch_num = int(img_dim / self.patch_dim) ** 2

        self.in_chans = in_chans
        self.K = K

        self.local_mem_size = local_mem_size * self.patch_num
        self.global_mem_size = global_mem_size * self.patch_num

        self._local_mem = None
        self._global_mem = None
        self.norm_ref = nn.LayerNorm(768, eps=1e-6)
        self.surv_ref_head = nn.Linear(768, 1)

        self.vision_encoder = ViT(
            image_size=512,  # image size
            frames=512,  # max number of frames
            image_patch_size=32,  # image patch size
            frame_patch_size=4,  # frame patch size
            dim=768,
            depth=12,
            heads=8,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1
        )

        self.ref_encoder = ViT(
            image_size=512,  # image size
            frames=512,  # max number of frames
            image_patch_size=32,  # image patch size
            frame_patch_size=4,  # frame patch size
            dim=768,
            depth=12,
            heads=8,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1
        )




        self.query_generator = QueryNet(embed_dim=self.embedding_dim, num_heads=8)

        self.main_surv_net = ProjectDotSurv_cli2(dim=self.embedding_dim, cli_dim=cli_dim, num_heads=8,
                                                 cat_enc_mode=cat_enc_mode, norm=False)

    def encode(self, vis, lang):
        # B,T,C,H,W = vis.shape
        # vis = vis.view(-1,C,H,W)

        vis_f, pos1 = self.vision_encoder(vis)

        ref_f, pos2 = self.ref_encoder(lang)

        return vis_f, ref_f

    def forward(self, ref, vis, cli):
        B, C, H, W, D = vis.shape

        # B,T,C,H,W
        vis_f, ref_f = self.encode(vis, ref)
        cur_v = torch.cat([vis_f[:, 1:], ref_f[:, 1:]], dim=1).mean(dim=1)

        ref_out = ref_f[:, 1:].mean(dim=1)
        ref_out = self.norm_ref(ref_out)
        ref_out = self.surv_ref_head(ref_out)

        # gerneate query
        query, _query_attn = self.query_generator(ref_f[:, 1:], cur_v, )  # * B,C
        # print(query.shape)
        cli_out, out = self.main_surv_net((vis_f[:, 1:], query, cli))
        return ref_out, cli_out, out

    def _get_padding(self, padding_type, kernel_size):
        assert padding_type in ['SAME', 'VALID']
        if padding_type == 'SAME':
            _list = [(k - 1) // 2 for k in kernel_size]
            return tuple(_list)
        return tuple(0 for _ in kernel_size)

    def _reshape_output(self, x, embedding_dim):
        x = x.view(
            x.size(0),
            int(self.img_dim / self.patch_dim),
            int(self.img_dim / self.patch_dim),
            embedding_dim,
        )
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

    def load_state_dict(self, new_state):
        state = self.state_dict()
        for layer in state:
            if layer in new_state:
                if state[layer].size() == new_state[layer].size():
                    state[layer] = new_state[layer]
            elif 'module.' + layer in new_state:
                new_layer = 'module.' + layer
                if state[layer].size() == new_state[new_layer].size():
                    state[layer] = new_state[new_layer]
            else:
                print(layer)

        super().load_state_dict(state)

class VLFTrans_cli2_abca(nn.Module):
    def __init__(
            self,
            img_dim=320,
            patch_dim=16.,
            in_chans=3,
            embedding_dim=768,
            K=3,
            global_mem_size=1.5,
            local_mem_size=2,
            fuse_depth=2,
            post_depth=2,
            vision_depth=6,
            cli_dim=25,
            cat_enc_mode='concat'
    ):
        super(VLFTrans_cli2_abca, self).__init__()

        self.img_dim = img_dim
        self.embedding_dim = embedding_dim
        self.patch_dim = patch_dim
        self.patch_num = int(img_dim / self.patch_dim) ** 2

        self.in_chans = in_chans
        self.K = K

        self.local_mem_size = local_mem_size * self.patch_num
        self.global_mem_size = global_mem_size * self.patch_num

        self._local_mem = None
        self._global_mem = None
        self.norm_ref = nn.LayerNorm(768, eps=1e-6)
        self.surv_ref_head = nn.Linear(768, 1)

        self.norm_vis = nn.LayerNorm(768, eps=1e-6)
        # self.surv_vis_head = nn.Linear(768, 64)

        self.vision_encoder = ViT(
            image_size=512,  # image size
            frames=512,  # max number of frames
            image_patch_size=32,  # image patch size
            frame_patch_size=4,  # frame patch size
            dim=768,
            depth=12,
            heads=8,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1
        )

        self.ref_encoder = ViT(
            image_size=512,  # image size
            frames=512,  # max number of frames
            image_patch_size=32,  # image patch size
            frame_patch_size=4,  # frame patch size
            dim=768,
            depth=12,
            heads=8,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1
        )
        self.trans_fusion = IntermediateSequential(OrderedDict(
            [(str(i), EncodeNet(depth=fuse_depth, post_depth=post_depth, cascade_lang_fuse=False))
             for i in range(K)]), return_intermediate=True)


        self.surv_head1 = nn.Sequential(
            nn.Linear(768, 4),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(),
        )

        self.surv_head2 = nn.Sequential(
            nn.Linear(cli_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
        )

        self.surv_head3 = nn.Linear(64, 1)
        self.surv_head4 = nn.Linear(768 + 768 + 64, 1)


    def encode(self, vis, lang):
        # B,T,C,H,W = vis.shape
        # vis = vis.view(-1,C,H,W)

        vis_f, pos1 = self.vision_encoder(vis)

        ref_f, pos2 = self.ref_encoder(lang)

        return vis_f, ref_f

    def forward(self, ref, vis, cli):
        B, C, H, W, D = vis.shape

        # B,T,C,H,W
        vis_f, ref_f = self.encode(vis, ref)

        # fusion
        (fused_v, _post_ref), _interm_fused_v = self.trans_fusion(vis_f, ref_f)

        cur_v = fused_v[:, 1]

        ref_out = ref_f[:, 1:].mean(dim=1)
        ref_out = self.norm_ref(ref_out)
        ref_out = self.surv_ref_head(ref_out)

        mid1 = vis_f[:, 1:].mean(dim=1)
        mid1 = self.norm_vis(mid1)

        mid2 = self.surv_head2(cli)
        cli_out = self.surv_head3(mid2)
        out = self.surv_head4(torch.cat([mid1, cur_v, mid2], dim=-1))

        return ref_out, cli_out, out

    def _get_padding(self, padding_type, kernel_size):
        assert padding_type in ['SAME', 'VALID']
        if padding_type == 'SAME':
            _list = [(k - 1) // 2 for k in kernel_size]
            return tuple(_list)
        return tuple(0 for _ in kernel_size)

    def _reshape_output(self, x, embedding_dim):
        x = x.view(
            x.size(0),
            int(self.img_dim / self.patch_dim),
            int(self.img_dim / self.patch_dim),
            embedding_dim,
        )
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

    def load_state_dict(self, new_state):
        state = self.state_dict()
        for layer in state:
            if layer in new_state:
                if state[layer].size() == new_state[layer].size():
                    state[layer] = new_state[layer]
            elif 'module.' + layer in new_state:
                new_layer = 'module.' + layer
                if state[layer].size() == new_state[new_layer].size():
                    state[layer] = new_state[new_layer]
            else:
                print(layer)

        super().load_state_dict(state)

class VLFTrans_cli2x(nn.Module):
    def __init__(
            self,
            img_dim=320,
            patch_dim=16.,
            in_chans=3,
            embedding_dim=768,
            K=3,
            global_mem_size=1.5,
            local_mem_size=2,
            fuse_depth=2,
            post_depth=2,
            vision_depth=6,
            cli_dim=20,
            cat_enc_mode='concat'
    ):
        super(VLFTrans_cli2x, self).__init__()

        self.img_dim = img_dim
        self.embedding_dim = embedding_dim
        self.patch_dim = patch_dim
        self.patch_num = int(img_dim / self.patch_dim) ** 2

        self.in_chans = in_chans
        self.K = K

        self.local_mem_size = local_mem_size * self.patch_num
        self.global_mem_size = global_mem_size * self.patch_num

        self._local_mem = None
        self._global_mem = None
        self.norm_ref = nn.LayerNorm(768, eps=1e-6)
        self.surv_ref_head = nn.Linear(768, 1)

        self.vision_encoder = ViT(
            image_size=512,  # image size
            frames=512,  # max number of frames
            image_patch_size=32,  # image patch size
            frame_patch_size=4,  # frame patch size
            dim=768,
            depth=12,
            heads=8,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1
        )

        self.ref_encoder = ViT(
            image_size=512,  # image size
            frames=512,  # max number of frames
            image_patch_size=32,  # image patch size
            frame_patch_size=4,  # frame patch size
            dim=768,
            depth=12,
            heads=8,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1
        )
        self.trans_fusion = IntermediateSequential(OrderedDict(
            [(str(i), EncodeNet(depth=fuse_depth, post_depth=post_depth, cascade_lang_fuse=False))
             for i in range(K)]), return_intermediate=True)

        self.query_generator = QueryNet(embed_dim=self.embedding_dim, num_heads=8)

        self.main_surv_net = ProjectDotSurv_cli2(dim=self.embedding_dim, cli_dim=cli_dim, num_heads=8,
                                                 cat_enc_mode=cat_enc_mode, norm=False)

    def encode(self, vis, lang):
        # B,T,C,H,W = vis.shape
        # vis = vis.view(-1,C,H,W)

        vis_f, pos1 = self.vision_encoder(vis)

        ref_f, pos2 = self.ref_encoder(lang)

        return vis_f, ref_f

    def forward(self, ref, vis, cli):
        B, C, H, W, D = vis.shape

        # B,T,C,H,W
        vis_f, ref_f = self.encode(vis, ref)

        # fusion
        (fused_v, _post_ref), _interm_fused_v = self.trans_fusion(vis_f, ref_f)

        cur_v = fused_v[:, 1]

        ref_out = ref_f[:, 1:].mean(dim=1)
        ref_out = self.norm_ref(ref_out)
        ref_out = self.surv_ref_head(ref_out)

        # gerneate query
        query, _query_attn = self.query_generator(ref_f[:, 1:], cur_v, )  # * B,C
        # print(query.shape)
        cli_out, out = self.main_surv_net((vis_f[:, 1:], query, cli))
        return ref_out, cli_out, out

    def _get_padding(self, padding_type, kernel_size):
        assert padding_type in ['SAME', 'VALID']
        if padding_type == 'SAME':
            _list = [(k - 1) // 2 for k in kernel_size]
            return tuple(_list)
        return tuple(0 for _ in kernel_size)

    def _reshape_output(self, x, embedding_dim):
        x = x.view(
            x.size(0),
            int(self.img_dim / self.patch_dim),
            int(self.img_dim / self.patch_dim),
            embedding_dim,
        )
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

    def load_state_dict(self, new_state):
        state = self.state_dict()
        for layer in state:
            if layer in new_state:
                if state[layer].size() == new_state[layer].size():
                    state[layer] = new_state[layer]
            elif 'module.' + layer in new_state:
                new_layer = 'module.' + layer
                if state[layer].size() == new_state[new_layer].size():
                    state[layer] = new_state[new_layer]
            else:
                print(layer)

        super().load_state_dict(state)


class VLFTrans_cli2y(nn.Module):
    def __init__(
            self,
            img_dim=320,
            patch_dim=16.,
            in_chans=3,
            embedding_dim=768,
            K=3,
            global_mem_size=1.5,
            local_mem_size=2,
            fuse_depth=2,
            post_depth=2,
            vision_depth=6,
            cli_dim=7,
            cat_enc_mode='concat'
    ):
        super(VLFTrans_cli2y, self).__init__()

        self.img_dim = img_dim
        self.embedding_dim = embedding_dim
        self.patch_dim = patch_dim
        self.patch_num = int(img_dim / self.patch_dim) ** 2

        self.in_chans = in_chans
        self.K = K

        self.local_mem_size = local_mem_size * self.patch_num
        self.global_mem_size = global_mem_size * self.patch_num

        self._local_mem = None
        self._global_mem = None
        self.norm_ref = nn.LayerNorm(768, eps=1e-6)
        self.surv_ref_head = nn.Linear(768, 1)

        self.vision_encoder = ViT(
            image_size=512,  # image size
            frames=512,  # max number of frames
            image_patch_size=32,  # image patch size
            frame_patch_size=4,  # frame patch size
            dim=768,
            depth=12,
            heads=8,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1
        )

        self.ref_encoder = ViT(
            image_size=512,  # image size
            frames=512,  # max number of frames
            image_patch_size=32,  # image patch size
            frame_patch_size=4,  # frame patch size
            dim=768,
            depth=12,
            heads=8,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1
        )
        self.trans_fusion = IntermediateSequential(OrderedDict(
            [(str(i), EncodeNet(depth=fuse_depth, post_depth=post_depth, cascade_lang_fuse=False))
             for i in range(K)]), return_intermediate=True)

        self.query_generator = QueryNet(embed_dim=self.embedding_dim, num_heads=8)

        self.main_surv_net = ProjectDotSurv_cli2(dim=self.embedding_dim, cli_dim=cli_dim, num_heads=8,
                                                 cat_enc_mode=cat_enc_mode, norm=False)

    def encode(self, vis, lang):
        # B,T,C,H,W = vis.shape
        # vis = vis.view(-1,C,H,W)

        vis_f, pos1 = self.vision_encoder(vis)

        ref_f, pos2 = self.ref_encoder(lang)

        return vis_f, ref_f

    def forward(self, ref, vis, cli):
        B, C, H, W, D = vis.shape

        # B,T,C,H,W
        vis_f, ref_f = self.encode(vis, ref)

        # fusion
        (fused_v, _post_ref), _interm_fused_v = self.trans_fusion(vis_f, ref_f)

        cur_v = fused_v[:, 1]

        ref_out = ref_f[:, 1:].mean(dim=1)
        ref_out = self.norm_ref(ref_out)
        ref_out = self.surv_ref_head(ref_out)

        # gerneate query
        query, _query_attn = self.query_generator(ref_f[:, 1:], cur_v, )  # * B,C
        # print(query.shape)
        cli_out, out = self.main_surv_net((vis_f[:, 1:], query, cli))
        return ref_out, cli_out, out

    def _get_padding(self, padding_type, kernel_size):
        assert padding_type in ['SAME', 'VALID']
        if padding_type == 'SAME':
            _list = [(k - 1) // 2 for k in kernel_size]
            return tuple(_list)
        return tuple(0 for _ in kernel_size)

    def _reshape_output(self, x, embedding_dim):
        x = x.view(
            x.size(0),
            int(self.img_dim / self.patch_dim),
            int(self.img_dim / self.patch_dim),
            embedding_dim,
        )
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

    def load_state_dict(self, new_state):
        state = self.state_dict()
        for layer in state:
            if layer in new_state:
                if state[layer].size() == new_state[layer].size():
                    state[layer] = new_state[layer]
            elif 'module.' + layer in new_state:
                new_layer = 'module.' + layer
                if state[layer].size() == new_state[new_layer].size():
                    state[layer] = new_state[new_layer]
            else:
                print(layer)

        super().load_state_dict(state)

if __name__ == '__main__':
    input_image1 = torch.randn([2, 3, 256, 256, 64])
    input_image2 = torch.randn([2, 3, 256, 256, 64])
    cli_in = torch.randn([2, 25])
    # dict_image = {}
    # dict_image['vis'] = input_image1
    # dict_image['lang'] = input_image2
    model = VLFTrans_cli2_abpf(img_dim=256,
                     patch_dim=16.,
                     in_chans=1,
                     embedding_dim=768)
    out1, out2 = model(input_image1, input_image2, cli_in)
    s = 0
