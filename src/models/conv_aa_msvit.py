import math
from functools import partial
import logging
import torch
from torch import nn
from timm.models.layers import DropPath, trunc_normal_, to_2tuple
from .layers import (
    Long2DSCSelfAttention,
    FastAttention,
    PerformerSelfAttention,
    LinformerSelfAttention,
    SRSelfAttention
)
# from .longformer2d_cuda import Longformer2DSelfAttention
from .resnet_reference import BasicBlock, Bottleneck
from typing import Type, Any, Callable, Union, List, Optional

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
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


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0.,
                 rpe=False, wx=14, wy=14, nglo=1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Inspired by swin transformer:
        # https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py#L88-L103
        # define parameter tables for local and global relative position bias
        self.rpe = rpe
        if rpe:
            self.wx = wx
            self.wy = wy
            self.nglo = nglo
            self.local_relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * wx - 1) * (2 * wy - 1),
                            num_heads))  # (2*wx-1, 2*wy-1, nH)
            trunc_normal_(self.local_relative_position_bias_table, std=.02)
            if nglo >= 1:
                self.g2l_relative_position_bias = nn.Parameter(
                    torch.zeros(2, num_heads, nglo))  # (2, nH, nglo)
                self.g2g_relative_position_bias = nn.Parameter(
                    torch.zeros(num_heads, nglo, nglo))  # (nH, nglo, nglo)
                trunc_normal_(self.g2l_relative_position_bias, std=.02)
                trunc_normal_(self.g2g_relative_position_bias, std=.02)

            # get pair-wise relative position index
            coords_h = torch.arange(wx)
            coords_w = torch.arange(wy)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, wx, wy
            coords_flatten = torch.flatten(coords, 1)  # 2, Wx*Wy
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wx*Wy, Wx*Wy
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wx*Wy, Wx*Wy, 2
            relative_coords[:, :, 0] += wx - 1  # shift to start from 0
            relative_coords[:, :, 1] += wy - 1
            relative_coords[:, :, 0] *= 2 * wy - 1
            relative_position_index = relative_coords.sum(-1)  # Wx*Wy, Wx*Wy
            self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x, nx=None, ny=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if self.rpe:
            assert N == self.nglo + self.wx*self.wy, "For relative position, N != self.nglo + self.wx*self.wy!"
            local_relative_position_bias = self.local_relative_position_bias_table[
                self.relative_position_index.view(-1)].view(
                self.wx*self.wy, self.wx*self.wy, -1)  # Wh*Ww, Wh*Ww,nH
            relative_position_bias = local_relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            if self.nglo > 0:
                # relative position embedding of global tokens
                global_relative_position_bias = torch.cat([
                    self.g2g_relative_position_bias,
                    self.g2l_relative_position_bias[0].unsqueeze(-1).expand(-1, -1, self.wx*self.wy)
                ], dim=-1)  # nH, nglo, N
                # relative position embedding of local tokens
                local_relative_position_bias = torch.cat([
                    self.g2l_relative_position_bias[1].unsqueeze(1).expand(-1, self.wx*self.wy, -1),
                    relative_position_bias,
                ], dim=-1)  # nH, Wh*Ww, N
                relative_position_bias = torch.cat([
                    global_relative_position_bias,
                    local_relative_position_bias,
                ], dim=1)  # nH, N, N
            attn = attn + relative_position_bias.unsqueeze(0)

        attn = (attn - torch.max(attn, dim=-1, keepdim=True)[0]).softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    @staticmethod
    def compute_macs(module, input, output):
        # T: num_token
        # S: num_token
        input = input[0]
        _, T, C = input.shape
        S = T
        macs = 0
        n_params = 0

        # Scaled-dot-product macs
        # [B x T x C] x [B x C x S] --> [B x T x S]
        # multiplication-addition is counted as 1 because operations can be fused
        num_macs_kq = T * S * C
        # [B x T x S] x [B x S x C] --> [B x T x C]
        num_macs_v = T * C * S

        macs += num_macs_kq + num_macs_v
        # print('macs att', macs / 1e8)

        # self attention: T should be equal to S
        assert T == S
        qkv_params = sum([p.numel() for p in module.qkv.parameters()])
        n_params += qkv_params
        # multiply by Seq length
        macs += qkv_params * T
        # print('macs qkv', qkv_params * T / 1e8)

        proj_params = sum([p.numel() for p in module.proj.parameters()])
        n_params += proj_params
        macs += (proj_params * T)
        # print('macs proj', proj_params * T / 1e8)

        module.__flops__ += macs
        # return n_params, macs


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, patch_size, nx, ny, in_chans=3, embed_dim=768, nglo=1,
                 norm_layer=nn.LayerNorm, norm_embed=True, drop_rate=0.0,
                 ape=True):
        # maximal global/x-direction/y-direction tokens: nglo, nx, ny
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=patch_size)

        self.norm_embed = norm_layer(embed_dim) if norm_embed else None

        self.nx = nx
        self.ny = ny
        self.Nglo = nglo
        if nglo >= 1:
            self.cls_token = nn.Parameter(torch.zeros(1, nglo, embed_dim))
            trunc_normal_(self.cls_token, std=.02)
        else:
            self.cls_token = None
        self.ape = ape
        if ape:
            self.cls_pos_embed = nn.Parameter(torch.zeros(1, nglo, embed_dim))
            self.x_pos_embed = nn.Parameter(torch.zeros(1, nx, embed_dim // 2))
            self.y_pos_embed = nn.Parameter(torch.zeros(1, ny, embed_dim // 2))
            trunc_normal_(self.cls_pos_embed, std=.02)
            trunc_normal_(self.x_pos_embed, std=.02)
            trunc_normal_(self.y_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

    def forward(self, xtuple):
        x, nx, ny = xtuple
        B = x.shape[0]

        x = self.proj(x)
        nx, ny = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)
        assert nx == self.nx and ny == self.ny, "Fix input size!"

        if self.norm_embed:
            x = self.norm_embed(x)

        # concat cls_token
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(
                B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x), dim=1)

        if self.ape:
            # add position embedding
            pos_embed_2d = torch.cat([
                self.x_pos_embed.unsqueeze(2).expand(-1, -1, ny, -1),
                self.y_pos_embed.unsqueeze(1).expand(-1, nx, -1, -1),
            ], dim=-1).flatten(start_dim=1, end_dim=2)
            x = x + torch.cat([self.cls_pos_embed, pos_embed_2d], dim=1).expand(
                B, -1, -1)

        x = self.pos_drop(x)

        return x, nx, ny


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# for Performer, start
def get_module_device(module):
    return next(module.parameters()).device


def find_modules(nn_module, type):
    return [module for module in nn_module.modules() if isinstance(module, type)]

# for Performer, end


class AttnBlock(nn.Module):
    """ Meta Attn Block
    """

    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None, drop=0.,
                 attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm,
                 attn_type='full', w=7, d=1, sharew=False, nglo=1,
                 only_glo=False,
                 seq_len=None, num_feats=256, share_kv=False, sw_exact=0,
                 rratio=2, rpe=False, wx=14, wy=14, mode=0):
        super().__init__()
        self.norm = norm_layer(dim)
        if attn_type == 'full':
            self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                  qk_scale=qk_scale, attn_drop=attn_drop,
                                  proj_drop=drop,
                                  rpe=rpe, wx=wx, wy=wy, nglo=nglo)
        # elif attn_type == 'longformer_cuda':
        #     self.attn = Longformer2DSelfAttention(
        #         dim, num_heads=num_heads, qkv_bias=qkv_bias,
        #         qk_scale=qk_scale, attn_drop=attn_drop,
        #         proj_drop=drop, w=w, d=d, sharew=sharew,
        #         nglo=nglo, only_glo=only_glo)
        elif attn_type == 'longformerhand':
            self.attn = Long2DSCSelfAttention(
                dim, exact=sw_exact, num_heads=num_heads, qkv_bias=qkv_bias,
                qk_scale=qk_scale, attn_drop=attn_drop,
                proj_drop=drop, w=w, d=d, sharew=sharew,
                nglo=nglo, only_glo=only_glo, autograd=False,
                rpe=rpe, mode=mode
            )
        elif attn_type == 'longformerauto':
            self.attn = Long2DSCSelfAttention(
                dim, exact=sw_exact, num_heads=num_heads, qkv_bias=qkv_bias,
                qk_scale=qk_scale, attn_drop=attn_drop,
                proj_drop=drop, w=w, d=d, sharew=sharew,
                nglo=nglo, only_glo=only_glo, autograd=True,
                rpe=rpe, mode=mode
            )
        elif attn_type == 'linformer':
            assert seq_len is not None, "seq_len must be provided for Linformer!"
            self.attn = LinformerSelfAttention(
                dim, seq_len, num_feats=num_feats,
                num_heads=num_heads, qkv_bias=qkv_bias,
                qk_scale=qk_scale, attn_drop=attn_drop,
                proj_drop=drop, share_kv=share_kv,
            )
        elif attn_type == 'srformer':
            self.attn = SRSelfAttention(
                dim, rratio=rratio,
                num_heads=num_heads, qkv_bias=qkv_bias,
                qk_scale=qk_scale, attn_drop=attn_drop,
                proj_drop=drop
            )
        elif attn_type == 'performer':
            self.attn = PerformerSelfAttention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias,
                qk_scale=qk_scale, attn_drop=attn_drop,
                proj_drop=drop, nb_features=num_feats,
            )
        else:
            raise ValueError(
                "Not supported attention type {}".format(attn_type))
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, xtuple):
        x, nx, ny = xtuple
        x = x + self.drop_path(self.attn(self.norm(x), nx, ny))
        return x, nx, ny


class MlpBlock(nn.Module):
    """ Meta MLP Block
    """

    def __init__(self, dim, out_dim=None, mlp_ratio=4., drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       out_features=out_dim, act_layer=act_layer, drop=drop)
        self.shortcut = nn.Identity()
        if out_dim is not None and out_dim != dim:
            self.shortcut = nn.Sequential(nn.Linear(dim, out_dim),
                                          nn.Dropout(drop))

    def forward(self, xtuple):
        x, nx, ny = xtuple
        x = self.shortcut(x) + self.drop_path(self.mlp(self.norm(x)))
        return x, nx, ny


class MsViTAA(nn.Module):
    """ Multiscale Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, arch, img_size=512, in_chans=3,
                 num_classes=1000,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 norm_embed=False, w=7, d=1, sharew=False, only_glo=False,
                 share_kv=False,
                 attn_type='longformerhand', sw_exact=0, mode=0, **args):
        super().__init__()
        self.num_classes = num_classes
        if 'ln_eps' in args:
            ln_eps = args['ln_eps']
            self.norm_layer = partial(nn.LayerNorm, eps=ln_eps)
            logging.info("Customized LayerNorm EPS: {}".format(ln_eps))
        else:
            self.norm_layer = norm_layer
        self.drop_path_rate = drop_path_rate
        self.attn_type = attn_type

        # for performer, start
        if attn_type == "performer":
            self.auto_check_redraw = True  # TODO: make this an choice
            self.feature_redraw_interval = 1
            self.register_buffer('calls_since_last_redraw', torch.tensor(0))
        # for performer, end

        self.attn_args = dict({
            'attn_type': attn_type,
            'qkv_bias': qkv_bias,
            'qk_scale': qk_scale,
            'drop': drop_rate,
            'attn_drop': attn_drop_rate,
            'w': w,
            'd': d,
            'sharew': sharew,
            'only_glo': only_glo,
            'share_kv': share_kv,
            'sw_exact': sw_exact,
            'norm_layer': norm_layer,
            'mode': mode,
        })
        self.patch_embed_args = dict({
            'norm_layer': norm_layer,
            'norm_embed': norm_embed,
            'drop_rate': drop_rate,
        })
        self.mlp_args = dict({
            'mlp_ratio': 4.0,
            'norm_layer': norm_layer,
            'act_layer': nn.GELU,
            'drop': drop_rate,
        })

        self.Nx = img_size
        self.Ny = img_size

        def parse_arch(arch):
            layer_cfgs = []
            for layer in arch.split('_'):
                layer_cfg = {'l': 1, 'h': 3, 'd': 192, 'n': 1, 's': 1, 'g': 1,
                             'p': 2, 'f': 7, 'a': 1}  # defaults
                for attr in layer.split(','):
                    layer_cfg[attr[0]] = int(attr[1:])
                layer_cfgs.append(layer_cfg)
            return layer_cfgs

        self.layer_cfgs = parse_arch(arch)
        self.num_layers = len(self.layer_cfgs)
        self.depth = sum([cfg['n'] for cfg in self.layer_cfgs])
        self.out_planes = self.layer_cfgs[-1]['d']
        self.Nglos = [cfg['g'] for cfg in self.layer_cfgs]
        self.avg_pool = args['avg_pool'] if 'avg_pool' in args else False

        dprs = torch.linspace(0, drop_path_rate, self.depth).split(
            [cfg['n'] for cfg in self.layer_cfgs]
        )  # stochastic depth decay rule


        #Resnet Layers
        self._norm_layer = nn.BatchNorm2d

        self.inplanes = 64
        self.dilation = 1
        self.groups = 1
        self.base_width = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        block = BasicBlock
        layers = [2, 2, 2, 2]

        self.res_layer1 = self._make_res_layer(block, 64, layers[0])
        self.res_layer2 = self._make_res_layer(block, 128, layers[1], stride=2, dilate=False)
        self.res_layer3 = self._make_res_layer(block, 256, layers[2], stride=2, dilate=False)
        self.res_layer4 = self._make_res_layer(block, 512, layers[3], stride=2, dilate=False)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Atten layers to replace CONV
        self.aa_layer1 = self._make_layer(64, self.layer_cfgs[0],
                                       dprs=dprs[0], layerid=1)
        # Atten Layers
        self.attn_layer1 = self._make_layer(in_chans, self.layer_cfgs[0],
                                       dprs=dprs[0], layerid=1)
        self.attn_layer2 = self._make_layer(self.layer_cfgs[0]['d'],
                                       self.layer_cfgs[1], dprs=dprs[1],
                                       layerid=2)
        self.attn_layer3 = self._make_layer(self.layer_cfgs[1]['d'],
                                       self.layer_cfgs[2], dprs=dprs[2],
                                       layerid=3)
        if self.num_layers == 3:
            self.attn_layer4 = None
        elif self.num_layers == 4:
            self.attn_layer4 = self._make_layer(self.layer_cfgs[2]['d'],
                                           self.layer_cfgs[3], dprs=dprs[3],
                                           layerid=4)
        else:
            raise ValueError("Numer of layers {} not implemented yet!".format(self.num_layers))
        self.norm = norm_layer(self.out_planes)

        # Classifier head
        self.head = nn.Linear(self.out_planes,
                              num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)
    
    def _make_res_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _make_layer(self, in_dim, layer_cfg, dprs, layerid=0):
        layer_id, num_heads, dim, num_block, is_sparse_attn, nglo, patch_size, num_feats, ape \
            = layer_cfg['l'], layer_cfg['h'], layer_cfg['d'], layer_cfg['n'], layer_cfg['s'], layer_cfg['g'], layer_cfg['p'], layer_cfg['f'], layer_cfg['a']

        assert layerid == layer_id, "Error in _make_layer: layerid {} does not equal to layer_id {}".format(layerid, layer_id)
        self.Nx = nx = self.Nx // patch_size
        self.Ny = ny = self.Ny // patch_size
        seq_len = nx * ny + nglo

        self.attn_args['nglo'] = nglo
        self.patch_embed_args['nglo'] = nglo
        self.attn_args['num_feats'] = num_feats  # shared for linformer and performer
        self.attn_args['rratio'] = num_feats  # srformer reuses this parameter
        self.attn_args['w'] = num_feats  # longformer reuses this parameter
        if is_sparse_attn == 0:
            self.attn_args['attn_type'] = 'full'

        # patch embedding
        layers = [
            PatchEmbed(patch_size, nx, ny, in_chans=in_dim, embed_dim=dim, ape=ape,
                       **self.patch_embed_args)
        ]
        for dpr in dprs:
            layers.append(AttnBlock(
                dim, num_heads, drop_path=dpr, seq_len=seq_len, rpe=not ape,
                wx=nx, wy=ny,
                **self.attn_args
            ))
            layers.append(MlpBlock(dim, drop_path=dpr, **self.mlp_args))
        return nn.Sequential(*layers)

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
        no_decay = {'pos_embed', 'cls_token',
                    'norm.weight', 'norm.bias',
                    'norm_embed', 'head.bias',
                    'relative_position'}
        return no_decay

    def get_classifier(self):
        return self.head

    def forward_features(self, x):
        B = x.shape[0]
        x, nx, ny = self.attn_layer1((x, None, None))
        x = x[:, self.Nglos[0]:].transpose(-2, -1).reshape(B, -1, nx, ny)

        x, nx, ny = self.attn_layer2((x, nx, ny))
        x = x[:, self.Nglos[1]:].transpose(-2, -1).reshape(B, -1, nx, ny)

        x, nx, ny = self.attn_layer3((x, nx, ny))
        if self.attn_layer4 is not None:
            x = x[:, self.Nglos[2]:].transpose(-2, -1).reshape(B, -1, nx, ny)
            x, nx, ny = self.attn_layer4((x, nx, ny))

        x = self.norm(x)

        if self.Nglos[-1] > 0 and (not self.avg_pool):
            return x[:, 0]
        else:
            return torch.mean(x, dim=1)

    def _forward_resnet(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.res_layer1(x)
        x = self.res_layer2(x)
        x = self.res_layer3(x)
        x = self.res_layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def _forward_aa_resnet(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.res_layer1(x)

        B = x.shape[0]
        x, nx, ny = self.aa_layer1((x, None, None))
        x = x[:, self.Nglos[0]:].transpose(-2, -1).reshape(B, -1, nx, ny)

        x = self.norm(x)

        #x = self.res_layer2(x)
        x = self.res_layer3(x)
        x = self.res_layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


    def check_redraw_projections(self):
        if not self.training:
            return

        if self.calls_since_last_redraw >= self.feature_redraw_interval:
            device = get_module_device(self)
            fast_attentions = find_modules(self, FastAttention)
            for fast_attention in fast_attentions:
                fast_attention.redraw_projection_matrix(device)

            self.calls_since_last_redraw.zero_()
            return

        self.calls_since_last_redraw += 1

    def reset_vil_mode(self, mode):
        longformer_attentions = find_modules(self, Long2DSCSelfAttention)
        for longformer_attention in longformer_attentions:
            mode_old = longformer_attention.mode
            if mode_old != mode:
                longformer_attention.mode = mode
                logging.info(
                    "Change vil attention mode from {} to {} in " "layer {}"
                        .format(mode_old, mode, longformer_attention))
        return

    def forward(self, x):

        #raw attn foward path
        # if self.attn_type == "performer" and self.auto_check_redraw:
        #     self.check_redraw_projections()
        # x = self.forward_features(x)
        # x = self.head(x)

        #raw resnet forward path
        #x = self._forward_resnet(x)
        #AA forward
        x = self._forward_aa_resnet(x)
        return x
