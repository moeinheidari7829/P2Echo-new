import torch
import torch.nn as nn
import torch.nn.functional as F

# from models.encoders.vmamba import CVSSDecoderBlock
from ..encoders.vmamba import SS2D, Mlp
import torch.utils.checkpoint as checkpoint
import math
from ..encoders.vmamba import cross_selective_scan
from einops import rearrange, repeat
from einops import rearrange
from typing import Callable, Any
from functools import partial
from timm.layers import DropPath

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

from .MambaDecoder import PatchExpand, UpsampleExpand, FinalUpsample_X4
from ..modules.dseb import DSEBlock


class LocalBranch(nn.Module):
    def __init__(self, dim, hidden_dim, depth_wise=False, act=nn.SiLU, out_dim=None):
        super().__init__()
        out_dim = dim if out_dim is None else out_dim

        self.gate = nn.Linear(dim, out_dim, bias=True)
        self.act = act()

        self.linear_d2h = nn.Linear(dim, hidden_dim, bias=True)
        self.conv = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            padding=1,
            groups=hidden_dim if depth_wise else 1,
            bias=True,
        )
        print(f"LocalBranch: depth_wise={depth_wise}, hidden_dim={hidden_dim}, out_dim={out_dim}, groupnorm={hidden_dim//4}")
        self.bn = nn.GroupNorm(hidden_dim//4, hidden_dim) # Medical/seg workloads often use B∈{1,2}. BN statistics get noisy. Consider GroupNorm(32) or LayerNorm in LocalBranch instead of BN (or SyncBN if you truly have multi-GPU and large global batch). Also two different norms in close succession can fight each other. If you switch BN→GN/LN, the block becomes more consistent.
        self.edge = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            padding=1,
            groups=hidden_dim if depth_wise else 1,
            bias=True,
        )

        self.linear_h2d = nn.Linear(hidden_dim, out_dim, bias=True)
        self.ln = nn.LayerNorm(out_dim)

        nn.init.zeros_(self.edge.weight)
        nn.init.zeros_(self.edge.bias)

    def forward(self, x):
        # gate
        # g = self.act(self.gate(x))
        g = self.gate(x)

        # value
        x = self.linear_d2h(x)
        x = x.permute(0, 3, 1, 2).contiguous()

        x = self.bn(self.conv(x))
        x_ = x
        x = self.act(x)
        x = self.edge(x) + x_

        # x_ = self.conv(x)
        # x = self.act(x_)
        # x = self.bn(x)
        # x = self.edge(x) + x_

        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.linear_h2d(x)
        v = self.ln(x)

        return torch.sigmoid(g) * v


class SS2D_pure(nn.Module):
    def __init__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        # ssm_ratio=2,
        dt_rank="auto",
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        # ======================
        softmax_version=False,
        # ======================
        **kwargs,
    ):

        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.softmax_version = softmax_version
        self.out_norm = nn.LayerNorm(d_model)  # necessary for vmamba
        self.d_model = d_model
        self.d_state = (
            math.ceil(self.d_model / 6) if d_state == "auto" else d_state
        )  # 20240109
        self.d_inner = self.d_model
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        # x proj; dt proj ============================
        self.K = 4
        self.x_proj = [
            nn.Linear(
                self.d_inner,
                (self.dt_rank + self.d_state * 2),
                bias=False,
                **factory_kwargs,
            )
            for _ in range(self.K)
        ]
        self.x_proj_weight = nn.Parameter(
            torch.stack([t.weight for t in self.x_proj], dim=0)
        )  # (K, N, inner)
        del self.x_proj

        self.dt_projs = [
            self.dt_init(
                self.dt_rank,
                self.d_inner,
                dt_scale,
                dt_init,
                dt_min,
                dt_max,
                dt_init_floor,
                **factory_kwargs,
            )
            for _ in range(self.K)
        ]
        self.dt_projs_weight = nn.Parameter(
            torch.stack([t.weight for t in self.dt_projs], dim=0)
        )  # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(
            torch.stack([t.bias for t in self.dt_projs], dim=0)
        )  # (K, inner)
        del self.dt_projs

        # A, D =======================================
        self.K2 = self.K
        self.A_logs = self.A_log_init(
            self.d_state, self.d_inner, copies=self.K2, merge=True
        )  # (K * D, N)
        self.Ds = self.D_init(self.d_inner, copies=self.K2, merge=True)  # (K * D)

    @staticmethod
    def dt_init(
        dt_rank,
        d_inner,
        dt_scale=1.0,
        dt_init="random",
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        **factory_kwargs,
    ):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor, nrows=-1):
        return cross_selective_scan(
            x,
            self.x_proj_weight,
            None,
            self.dt_projs_weight,
            self.dt_projs_bias,
            self.A_logs,
            self.Ds,
            getattr(self, "out_norm", None),
            self.softmax_version,
            nrows=nrows,
        )

    def forward(self, x: torch.Tensor, **kwargs):
        # shape of x must be (b, h, w, d)
        x = x.permute(0, 3, 1, 2).contiguous()  # (b, d, h, w)
        # print(x.shape)
        # exit()
        y = self.forward_core(x)  # (b, h, w, d)
        return y


class GlobalBranch(nn.Module):
    def __init__(self, dim, hidden_dim, act=nn.SiLU, 
                 d_state=16, 
                 dt_rank="auto",
                 softmax_version=False,
                 **kwargs):
        super().__init__()
        self.gate = nn.Linear(dim, dim, bias=True)
        self.act = act()

        self.linear_d2h = nn.Linear(dim, hidden_dim, bias=True)
        # self.dwconv = nn.Conv2d(
        #     hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim, bias=True)
        self.ss2d = SS2D_pure(d_model=hidden_dim, d_state=d_state, dt_rank=dt_rank, softmax_version=softmax_version, **kwargs)
        self.linear_h2d = nn.Linear(hidden_dim, dim, bias=True)
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        # gate
        # g = self.act(self.gate(x))
        g = self.gate(x)

        # value
        x = self.linear_d2h(x)
        # x = x.permute(0, 3, 1, 2).contiguous()
        # x = self.act(self.dwconv(x))
        # x = x.permute(0, 2, 3, 1).contiguous()
        x = self.ss2d(self.act(x))
        x = self.linear_h2d(x)
        v = self.ln(x)

        return torch.sigmoid(g) * v


class DualGateDecoderBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        drop_path: float = 0.0,
        norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        dt_rank: Any = "auto",
        ssm_ratio: float = 2,
        shared_ssm: bool = False,
        softmax_version: bool = False,
        use_checkpoint: bool = False,
        mlp_ratio: float = 2.0,
        act_layer=nn.GELU,
        drop: float = 0.0,
        fuse_mode: str = "cat",  # "learnable", "sum", "cat", or "mlp"
        cnv_ratio: float = 1,
        **kwargs,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.fuse_mode = fuse_mode
        C = hidden_dim

        self.init_norm = norm_layer(C)
        self.local_path = LocalBranch(
            dim=C, hidden_dim=int(C * cnv_ratio), depth_wise=True,
            out_dim=None if not "cat" in fuse_mode.lower() else int(C * cnv_ratio)
        )
        self.global_path = GlobalBranch(dim=C, hidden_dim=int(C * ssm_ratio), 
                                        d_state=d_state, 
                                        dt_rank=dt_rank,
                                        softmax_version=softmax_version,
                                        **kwargs)
        # self.local_path = nn.Identity()
        # self.global_path = nn.Identity()  # ablation

        # local-global fuser
        if self.fuse_mode == "learnable":
            raise ValueError("DO NOT USE learnable FUSER!")
            '''“learnable” fuse is per-channel static self.fuse_params is broadcast [1,1,1,C], i.e., global, data-independent weights. That’s fine, but you might get better adaptability with data-dependent mixing (your mlp fuse does this). If “learnable” underperforms, prefer mlp fuse.'''
            self.fuse_params = nn.Parameter(torch.zeros(1, 1, 1, C))  # [B,H,W,C]
        elif self.fuse_mode == "mlp":
            # self.fuse = nn.Linear(2*C, C, bias=True)
            self.ln_fuse = nn.LayerNorm(C)
            # self.fuse = nn.Sequential(
            #     nn.Linear(2 * C, C, bias=False),
            #     nn.GELU(),
            #     nn.Linear(C, C, bias=True),
            # )
            self.fuse = nn.Conv2d(2 * C, C, kernel_size=1, bias=True)
        elif self.fuse_mode == "cat":
            self.fuse = nn.Sequential(
                nn.Linear(C + int(C * cnv_ratio), C, bias=False),
                nn.GELU(),
                nn.LayerNorm(C),
                nn.Linear(C, C, bias=True),
            )
        elif not self.fuse_mode in ["sum", "add"]:
            raise NotImplementedError(f"Unknown fuse mode: {self.fuse_mode}")

        # self.linear = nn.Linear(C, C)
        self.scale = nn.Parameter(torch.ones(C)*1e-3)
        self.drop_path_global = DropPath(drop_path)
        self.drop_path_local = DropPath(drop_path)

        # output FFN
        self.mlp_branch = mlp_ratio > 0
        if self.mlp_branch:
            self.norm2 = norm_layer(C)
            mlp_hidden_dim = int(C * mlp_ratio)
            self.mlp = Mlp(
                in_features=C,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop=drop,
                channels_first=False,
            )
            self.scale_mlp = nn.Parameter(torch.ones(C))

        # print(
        #     f"DualGateDecoderBlock params: hidden_dim={hidden_dim}, drop_path={drop_path}, fuse_mode={fuse_mode}, ssm_ratio={ssm_ratio}, cnv_ratio={cnv_ratio}, local_variant={local_variant}, mlp_ratio={mlp_ratio}"
        # )

    def mixer(self, x_g, x_l):  # x: [B,H,W,C]
        # if self.fuse_mode == "mlp":
        #     gap_g = x_g.mean(dim=(1, 2))
        #     gap_l = x_l.mean(dim=(1, 2))
        #     alpha = torch.sigmoid(self.fuse(torch.cat([self.ln_fuse(gap_g), self.ln_fuse(gap_l)], dim=-1)))  # [B,C]
        #     alpha = alpha.unsqueeze(1).unsqueeze(1)  # [B,1,1,C]
        #     y = alpha * x_g + (1 - alpha) * x_l
        if self.fuse_mode == "mlp":
            _g = self.ln_fuse(x_g).permute(0, 3, 1, 2).contiguous()
            _l = self.ln_fuse(x_l).permute(0, 3, 1, 2).contiguous()
            alpha = torch.sigmoid(self.fuse(torch.cat([_g, _l], dim=1)))
            alpha = alpha.permute(0, 2, 3, 1).contiguous()
            y = alpha * x_g + (1 - alpha) * x_l
        elif self.fuse_mode == "learnable":
            y = self.fuse_params * x_g + (1 - self.fuse_params) * x_l
        elif self.fuse_mode == "cat":
            y = self.fuse(torch.cat([x_g, x_l], dim=-1))
        else:
            y = 0.5 * x_g + 0.5 * x_l
        return y

    def _forward(self, x):  # x: [B,H,W,C]
        # exit()
        x_norm = self.init_norm(x)
        x_g = self.drop_path_global(self.global_path(x_norm))
        x_l = self.drop_path_local(self.local_path(x_norm))
        z = self.mixer(x_g, x_l)

        y = x + self.scale * z
        if self.mlp_branch:
            y = y + self.scale_mlp*self.mlp(self.norm2(y))
        return y

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)


class Mamba_up(nn.Module):
    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        dt_rank="auto",
        d_state=4,
        ssm_ratio=2.0,
        attn_drop_rate=0.0,
        drop_rate=0.0,
        mlp_ratio=4.0,
        drop_path=0.1,
        norm_layer=nn.LayerNorm,
        upsample=None,
        shared_ssm=False,
        softmax_version=False,
        use_checkpoint=False,
        **kwargs,
    ):

        super().__init__()
        self.input_resolution = input_resolution
        self.depth = depth
        if isinstance(drop_path, float):
            drop_path = [drop_path] * depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        cnv_ratios = [1/16, 1/8, 1/2, 1]
        self.blocks = nn.ModuleList(
            [
                DualGateDecoderBlock(
                    hidden_dim=dim,
                    drop_path=drop_path[i],
                    norm_layer=norm_layer,
                    d_state=d_state,
                    dt_rank=dt_rank,
                    ssm_ratio=ssm_ratio,
                    shared_ssm=shared_ssm,
                    softmax_version=softmax_version,
                    use_checkpoint=use_checkpoint,
                    mlp_ratio=mlp_ratio,
                    act_layer=nn.GELU,
                    drop=drop_rate,
                    local_kernel=3,
                    local_dilation=1,
                    use_deformable_local=False,
                    balance_temperature=1.0,
                    min_balance=0.05,  # prevent collapse (0 or 1)

                    fuse_mode="mlp",
                    cnv_ratio=cnv_ratios[i],
                )
                for i in range(depth)
            ]
        )

        # patch merging layer
        if upsample is not None:
            # self.upsample = PatchExpand(input_resolution, dim=dim, dim_scale=2, norm_layer=norm_layer)
            self.upsample = UpsampleExpand(
                input_resolution, dim=dim, patch_size=2, norm_layer=norm_layer
            )
        else:
            self.upsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


class DGDecoder(nn.Module):
    def __init__(
        self,
        img_size=[480, 640],
        in_channels=[96, 192, 384, 768],  # [64, 128, 320, 512],
        num_classes=40,
        dropout_ratio=0.1,
        embed_dim=96,
        align_corners=False,
        patch_size=4,
        depths=[4, 2, 1, 1],
        mlp_ratio=4.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        use_checkpoint=False,
        deep_supervision=False,
        **kwargs,
    ):
        super().__init__()

        self.num_classes = num_classes
        # actually only three depths are used. The last feature is simply upexpanded
        self.num_layers = len(depths)
        self.mlp_ratio = mlp_ratio
        self.patch_size = patch_size
        self.patches_resolution = [img_size[0] // patch_size, img_size[1] // patch_size]
        self.deep_supervision = deep_supervision
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        # Build input projection layers for encoder->decoder channel mismatches
        # Decoder expects channels: [embed_dim, embed_dim*2, embed_dim*4, embed_dim*8]
        # e.g., with embed_dim=64: [64, 128, 256, 512]
        # PVT-v2-B2 outputs: [64, 128, 320, 512] -> only stage 2 (320 vs 256) needs projection
        self.input_projs = nn.ModuleList()
        for i in range(self.num_layers):
            encoder_ch = in_channels[i]  # encoder output channel for this stage
            decoder_ch = int(embed_dim * (2 ** i))  # decoder expected channel
            if encoder_ch != decoder_ch:
                self.input_projs.append(nn.Conv2d(encoder_ch, decoder_ch, kernel_size=1, bias=False))
            else:
                self.input_projs.append(nn.Identity())

        # DSEB blocks for skip connections (replaces additive skip)
        self.dseb_blocks = nn.ModuleList()
        for i_layer in range(1, self.num_layers):  # Skip first layer (no skip connection)
            dim = int(embed_dim * 2 ** (self.num_layers - 1 - i_layer))
            resolution = self.patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer))
            num_heads = max(2, dim // 32)  # Adaptive heads based on dim
            self.dseb_blocks.append(
                DSEBlock(
                    dim=dim,
                    scale_factors=[0.8, 0.4],
                    num_heads=num_heads,
                    input_size=resolution,
                    mode='add',
                    use_command='dat-fea',
                    depth=i_layer,
                )
            )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers_up = nn.ModuleList()
        for i_layer in range(self.num_layers):
            if i_layer == 0:
                # B, 768, 15, 20 -> B, 384, 30, 40
                layer_up = PatchExpand(
                    input_resolution=(
                        self.patches_resolution[0]
                        // (2 ** (self.num_layers - 1 - i_layer)),
                        self.patches_resolution[1]
                        // (2 ** (self.num_layers - 1 - i_layer)),
                    ),
                    dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                    dim_scale=2,
                    norm_layer=norm_layer,
                )
            else:
                # B, 30, 40, 384 -> B, 60, 80, 192
                # B, 60, 80, 192 -> B, 120, 160, 96
                # B, 120, 160, 96 -> B, 120, 160, 96
                layer_up = Mamba_up(
                    dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                    input_resolution=(
                        self.patches_resolution[0]
                        // (2 ** (self.num_layers - 1 - i_layer)),
                        self.patches_resolution[1]
                        // (2 ** (self.num_layers - 1 - i_layer)),
                    ),
                    depth=depths[(self.num_layers - 1 - i_layer)],
                    mlp_ratio=self.mlp_ratio,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[
                        sum(depths[: (self.num_layers - 1 - i_layer)]) : sum(
                            depths[: (self.num_layers - 1 - i_layer) + 1]
                        )
                    ],
                    norm_layer=norm_layer,
                    upsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
                    use_checkpoint=use_checkpoint,
                )
            self.layers_up.append(layer_up)

        self.norm_up = norm_layer(embed_dim)
        if self.deep_supervision:
            self.norm_ds = nn.ModuleList(
                [
                    norm_layer(embed_dim * 2 ** (self.num_layers - 2 - i_layer))
                    for i_layer in range(self.num_layers - 1)
                ]
            )
            self.output_ds = nn.ModuleList(
                [
                    nn.Conv2d(
                        in_channels=embed_dim * 2 ** (self.num_layers - 2 - i_layer),
                        out_channels=self.num_classes,
                        kernel_size=1,
                        bias=False,
                    )
                    for i_layer in range(self.num_layers - 1)
                ]
            )

        # print("---final upsample expand_first---")
        # self.up = FinalPatchExpand_X4(input_resolution=(img_size[0] // patch_size, img_size[1] // patch_size),
        #                                 patch_size=4, dim=embed_dim)
        self.up = FinalUpsample_X4(
            input_resolution=(img_size[0] // patch_size, img_size[1] // patch_size),
            patch_size=4,
            dim=embed_dim,
        )
        self.output = nn.Conv2d(
            in_channels=embed_dim,
            out_channels=self.num_classes,
            kernel_size=1,
            bias=False,
        )

    def forward_up_features(self, inputs):  # B, C, H, W
        if not self.deep_supervision:
            for inx, layer_up in enumerate(self.layers_up):
                if inx == 0:
                    # Apply input projection if needed (handles encoder->decoder channel mismatch)
                    skip_feat = self.input_projs[3 - inx](inputs[3 - inx])  # B, C', H, W
                    x = skip_feat.permute(0, 2, 3, 1).contiguous()  # B, H, W, C'
                    y = layer_up(x)
                else:
                    # Apply input projection if needed
                    skip_feat = self.input_projs[3 - inx](inputs[3 - inx])  # B, C', H, W
                    # interpolate y to input size (only pst900 dataset needs)
                    B, C, H, W = skip_feat.shape
                    y = (
                        F.interpolate(
                            y.permute(0, 3, 1, 2).contiguous(),
                            size=(H, W),
                            mode="bilinear",
                            align_corners=False,
                        )
                        .permute(0, 2, 3, 1)
                        .contiguous()
                    )

                    # DSEB skip connection (replaces additive skip)
                    y_chw = y.permute(0, 3, 1, 2).contiguous()  # B, C, H, W
                    x_chw = self.dseb_blocks[inx - 1](skip=skip_feat, dec=y_chw)  # DSEB
                    x = x_chw.permute(0, 2, 3, 1).contiguous()  # B, H, W, C
                    y = layer_up(x)

            x = self.norm_up(y)

            return x
        else:
            # if deep supervision
            x_upsample = []
            for inx, layer_up in enumerate(self.layers_up):
                if inx == 0:
                    # Apply input projection if needed (handles encoder->decoder channel mismatch)
                    skip_feat = self.input_projs[3 - inx](inputs[3 - inx])  # B, C', H, W
                    x = skip_feat.permute(0, 2, 3, 1).contiguous()  # B, H, W, C'
                    y = layer_up(x)
                    x_upsample.append(self.norm_ds[inx](y))
                else:
                    # Apply input projection if needed
                    skip_feat = self.input_projs[3 - inx](inputs[3 - inx])  # B, C', H, W
                    # Interpolate y to match skip_feat size if needed
                    B, C, H, W = skip_feat.shape
                    y_interp = F.interpolate(
                        y.permute(0, 3, 1, 2).contiguous(),
                        size=(H, W),
                        mode="bilinear",
                        align_corners=False,
                    )
                    # DSEB skip connection (replaces additive skip)
                    x_chw = self.dseb_blocks[inx - 1](skip=skip_feat, dec=y_interp)  # DSEB
                    x = x_chw.permute(0, 2, 3, 1).contiguous()  # B, H, W, C
                    y = layer_up(x)
                    if inx != self.num_layers - 1:
                        x_upsample.append((self.norm_ds[inx](y)))

            x = self.norm_up(y)

            return x, x_upsample

    def forward(self, inputs):
        if not self.deep_supervision:
            x = self.forward_up_features(inputs)  # B, H, W, C
            x_last = self.up_x4(x, self.patch_size)
            return x_last
        else:
            x, x_upsample = self.forward_up_features(inputs)
            x_last = self.up_x4(x, self.patch_size)
            x_output_0 = self.output_ds[0](
                F.interpolate(
                    x_upsample[0].permute(0, 3, 1, 2).contiguous(),
                    scale_factor=16,
                    mode="bilinear",
                    align_corners=False,
                )
            )
            x_output_1 = self.output_ds[1](
                F.interpolate(
                    x_upsample[1].permute(0, 3, 1, 2).contiguous(),
                    scale_factor=8,
                    mode="bilinear",
                    align_corners=False,
                )
            )
            x_output_2 = self.output_ds[2](
                F.interpolate(
                    x_upsample[2].permute(0, 3, 1, 2).contiguous(),
                    scale_factor=4,
                    mode="bilinear",
                    align_corners=False,
                )
            )
            return x_last, x_output_0, x_output_1, x_output_2

    def up_x4(self, x, pz):
        B, H, W, C = x.shape

        x = self.up(x)
        x = x.view(B, pz * H, pz * W, -1)
        x = x.permute(0, 3, 1, 2).contiguous()  # B, C, 4H, 4W
        x = self.output(x)

        return x
