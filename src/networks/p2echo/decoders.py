import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from .modules.cfam import CFAModule
from .modules.dseb import DSEBlock
from .modules.blocks import UpConv, UpTConv, UpRb, EUCB


def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer


class Decoder(nn.Module):
    def __init__(self, 
                 channels=[512,320,128,64], 
                 input_size=[14,28,56,112] , 
                 scale_factors=[0.8,0.4], 
                 skip_mode='add', # 'add' or 'cat',
                 num_heads=[2,2,2],
                 up_block='eucb',
                 num_classes=6,
                 writer=None):

        super(Decoder,self).__init__()

        assert up_block in ["uprb", "eucb", "upcn", "uptc"], f"Invalid up_block: {up_block}"

        up_ks = 3 # kernel size for eucb
        self.input_size = input_size
        self.writer = writer
        self.num_classes = num_classes

        if up_block == "uprb":
            up_block = partial(UpRb, kernel_size=up_ks, scale_factor=2)
        elif up_block == "eucb":
            up_block = partial(EUCB, kernel_size=up_ks, stride=up_ks//2, activation='leakyrelu')
        elif up_block == "upcn":
            up_block = partial(UpConv, kernel_size=up_ks, stride=1, activation='leakyrelu')
        elif up_block == "uptc":
            up_block = partial(UpTConv, kernel_size=up_ks, stride=2, activation='leakyrelu')
        else:
            raise ValueError(f"Invalid up_block: {up_block}")

        mca_rates_list = [[2,3,5], [1,2,4], [1,2,3], [1,2,2]] # spatial-size: [56x56, 28x28, 14x14, 7x7]
        # mca_rates_list = [[4,6,8], [2,3,4], [2,3,4], [1,2,3]]
        decoder = partial(CFAModule, ffn_ratio=4, drop_rate=0, 
                               drop_path_rate=0, 
                               act_type='GELU',
                               norm_type="BN",
                               init_value=1e-6,
                               attn_channel_split=[1,3,4],
                               attn_act_type="SiLU",
                            )
        #skip_block = partial(DSEBlock, scale_factors=scale_factors, mode=skip_mode, writer=writer)
        
        self.dec4 = decoder(embed_dims=channels[0], mca_rates=mca_rates_list[3])
        self.up3 = up_block(in_channels=channels[0], out_channels=channels[1])
        #self.skip_enhancer3 = skip_block(dim=channels[1], num_heads=num_heads[0], input_size=input_size[0], depth=4, label="S14")
        
        self.dec3 = decoder(embed_dims=channels[1], mca_rates=mca_rates_list[2])
        self.up2 = up_block(in_channels=channels[1], out_channels=channels[2])
        #self.skip_enhancer2 = skip_block(dim=channels[2], num_heads=num_heads[1], input_size=input_size[1], depth=3, label="S28")
        
        self.dec2 = decoder(embed_dims=channels[2], mca_rates=mca_rates_list[1])
        self.up1 = up_block(in_channels=channels[2], out_channels=channels[3])
        #self.skip_enhancer1 = skip_block(dim=channels[3], num_heads=num_heads[2], input_size=input_size[2], depth=2, label="S56")
        
        self.dec1 = decoder(embed_dims=channels[3], mca_rates=mca_rates_list[0])

        # Text injection fusers for stages 1-3 (channels[1], channels[2], channels[3]).
        self.inject_convs = nn.ModuleList()
        for ch in channels[1:]:
            self.inject_convs.append(nn.Sequential(
                nn.Conv2d(2 * ch, ch, kernel_size=1, bias=False),
                nn.GroupNorm(max(1, ch // 16), ch),
                nn.GELU(),
            ))

        self.output = nn.Conv2d(
            in_channels=channels[3],
            out_channels=num_classes,
            kernel_size=1,
            bias=False,
        )

    def _inject_text(self, y, mask_embed, stage_idx):
        """
        Args:
            y: [B, C, H, W] decoder feature map.
            mask_embed: [B, N, C] projected text embeddings for this stage.
            stage_idx: 1-based stage index in {1, 2, 3}.
        Returns:
            [B, C, H, W] text-enriched feature map.
        """
        # Per-prompt spatial affinity.
        attn = torch.einsum("b c h w, b n c -> b n h w", y, mask_embed)
        # Aggregate text embeddings at each location.
        text_feat = torch.einsum("b n h w, b n c -> b c h w", attn.sigmoid(), mask_embed)
        # Fuse visual and text features.
        return self.inject_convs[stage_idx - 1](torch.cat([y, text_feat], dim=1))
        
    def forward(self, features, mask_embeds):
        # features are encoder outputs in shallow->deep order: [f1, f2, f3, f4]
        f1, f2, f3, x = features
        skips_3, skips_2, skips_1 = f3, f2, f1
        
        d4 = self.dec4(x)
        
        d3 = self.up3(d4)
        #skips_3 = self.skip_enhancer3(skips[0], d3)
        d3 = self.dec3(d3 + skips_3)
        d3 = self._inject_text(d3, mask_embeds[0], stage_idx=1)
        
        d2 = self.up2(d3)
        #skips_2 = self.skip_enhancer2(skips[1], d2)
        d2 = self.dec2(d2 + skips_2)
        d2 = self._inject_text(d2, mask_embeds[1], stage_idx=2)
        
        d1 = self.up1(d2)
        #skips_1 = self.skip_enhancer1(skips[2], d1)
        d1 = self.dec1(d1 + skips_1)
        d1 = self._inject_text(d1, mask_embeds[2], stage_idx=3)

        logits = self.output(d1)
        logits = F.interpolate(logits, scale_factor=4, mode="bilinear", align_corners=False)
        return logits
