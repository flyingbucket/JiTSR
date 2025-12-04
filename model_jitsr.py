import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from util.model_util import VisionRotaryEmbeddingFast, get_2d_sincos_pos_embed, RMSNorm
from model_jit import (
    BottleneckPatchEmbed,
    TimestepEmbedder,
    Attention,
    SwiGLUFFN,
    FinalLayer,
    JiTBlock,
    JiT,
)


class JiTSR(JiT):
    def __init__(
        self,
        input_size=256,
        patch_size=16,
        in_channels=3,
        lr_input_size=64,
        lr_patch_size=4,
        hidden_size=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        attn_drop=0.0,
        proj_drop=0.0,
        bottleneck_dim=128,
        in_context_start=8,
    ):
        nn.Module.__init__(self)

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.input_size = input_size

        # ----- Important: SR in-context tokens = LR patches -----
        self.lr_embedder = BottleneckPatchEmbed(
            lr_input_size, lr_patch_size, in_channels, bottleneck_dim, hidden_size
        )
        self.in_context_len = self.lr_embedder.num_patches
        self.in_context_start = in_context_start

        # LR positional embedding
        self.lr_posemb = nn.Parameter(
            torch.zeros(1, self.in_context_len, hidden_size), requires_grad=True
        )
        torch.nn.init.normal_(self.lr_posemb, 0.02)

        # Time embedding
        self.t_embedder = TimestepEmbedder(hidden_size)

        # HR patch embedder
        self.x_embedder = BottleneckPatchEmbed(
            input_size, patch_size, in_channels, bottleneck_dim, hidden_size
        )

        # Fixed HR pos embed
        num_hr_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_hr_patches, hidden_size), requires_grad=False
        )

        # ----- RoPE -----
        half_head_dim = hidden_size // num_heads // 2
        hw_seq_len = input_size // patch_size

        # No in-context before insertion
        self.feat_rope = VisionRotaryEmbeddingFast(
            dim=half_head_dim, pt_seq_len=hw_seq_len, num_cls_token=0
        )
        # After LR tokens are inserted
        self.feat_rope_incontext = VisionRotaryEmbeddingFast(
            dim=half_head_dim, pt_seq_len=hw_seq_len, num_cls_token=self.in_context_len
        )

        # Transformer backbone
        self.blocks = nn.ModuleList(
            [
                JiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_drop=attn_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                    proj_drop=proj_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                )
                for i in range(depth)
            ]
        )

        # Output layer
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        # ----- 1. Initialize transformer linear layers -----
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # ----- 2. Initialize HR position embedding -----
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.x_embedder.num_patches**0.5)
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # ----- 3. Initialize HR patch embedder -----
        w1 = self.x_embedder.proj1.weight.data
        nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))

        w2 = self.x_embedder.proj2.weight.data
        nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj2.bias, 0)

        # ----- 4. Initialize LR patch embedder -----
        w1_lr = self.lr_embedder.proj1.weight.data
        nn.init.xavier_uniform_(w1_lr.view([w1_lr.shape[0], -1]))

        w2_lr = self.lr_embedder.proj2.weight.data
        nn.init.xavier_uniform_(w2_lr.view([w2_lr.shape[0], -1]))
        nn.init.constant_(self.lr_embedder.proj2.bias, 0)

        # ----- 5. Initialize timestep embedding MLP -----
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # ----- 6. Zero-out AdaLN modulation layers in transformer blocks -----
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # ----- 7. Zero-out FinalLayer modulation and output projection -----
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, hr, t, lr):
        """
        hr: (B, C, H, W)  -- noisy high-res
        lr: (B, C, H_lr, W_lr) -- low-res condition
        t:  (B,)
        """
        # ---- Condition embedding ----
        t_emb = self.t_embedder(t)  # (B, H)

        # LR tokens
        lr_emb = self.lr_embedder(lr)  # (B, T_lr, H)
        lr_emb = lr_emb + self.lr_posemb  # learnable LR position embedding

        # global LR embedding for AdaLN conditioning
        lr_global_emb = lr_emb.mean(dim=1)
        c = t_emb + lr_global_emb  # AdaLN conditioning vector

        # ---- HR tokens ----
        x = self.x_embedder(hr)  # (B, T_hr, H)
        x = x + self.pos_embed  # fixed 2D sin/cos

        # ---- Transformer blocks ----
        inserted = False
        for i, block in enumerate(self.blocks):
            # insert LR tokens only once
            if (not inserted) and (i == self.in_context_start):
                x = torch.cat([lr_emb, x], dim=1)
                inserted = True

            # choose RoPE (offset after LR tokens insertion)
            rope = self.feat_rope_incontext if inserted else self.feat_rope

            x = block(x, c, rope)

        # ---- remove LR tokens ----
        if inserted:
            x = x[:, self.in_context_len :]  # remove LR tokens

        # ---- Final stage ----
        x = self.final_layer(x, c)
        out = self.unpatchify(x, self.patch_size)

        return out
