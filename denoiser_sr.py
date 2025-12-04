import torch
import torch.nn as nn
from model_jitsr import JiTSR


class DenoiserSR(nn.Module):
    """
    Denoiser for Super-Resolution Diffusion.
    Uses LR image as condition through JiTSR.
    Trains with EDM v-prediction.
    """

    def __init__(self, args):
        super().__init__()

        # ----- Build JiTSR backbone -----
        self.net = JiTSR(
            input_size=args.hr_size,
            patch_size=args.hr_patch,
            in_channels=3,
            lr_input_size=args.lr_size,
            lr_patch_size=args.lr_patch,
            hidden_size=args.hidden_size,
            depth=args.depth,
            num_heads=args.num_heads,
            mlp_ratio=args.mlp_ratio,
            attn_drop=args.attn_dropout,
            proj_drop=args.proj_dropout,
            bottleneck_dim=args.bottleneck_dim,
            in_context_start=args.in_context_start,
        )

        # ----- Noise parameters -----
        self.noise_scale = args.noise_scale
        self.t_eps = args.t_eps
        self.P_mean = args.P_mean
        self.P_std = args.P_std

        # ----- EMA parameters -----
        self.ema_decay1 = args.ema_decay1
        self.ema_decay2 = args.ema_decay2
        self.ema_params1 = None
        self.ema_params2 = None

        # ----- Sampling hyperparameters -----
        self.steps = args.num_sampling_steps
        self.method = args.sampling_method
        self.cfg_scale = args.cfg
        self.cfg_interval = (args.interval_min, args.interval_max)

        # store target HR size
        self.hr_size = args.hr_size

    # ======================================================
    #           Utility: sample continuous t in EDM style
    # ======================================================
    def sample_t(self, n: int, device=None):
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)  # (n, )

    # ======================================================
    #              TRAINING: v-prediction loss
    # ======================================================
    def forward(self, hr, lr):
        """
        hr : (B, C, H, W) high-resolution ground truth
        lr : (B, C, h, w) low-resolution condition
        """
        B = hr.shape[0]
        device = hr.device

        # sample continuous t
        t = self.sample_t(B, device=device)
        t_broadcast = t.view(-1, *([1] * (hr.ndim - 1)))

        # sample noise
        e = torch.randn_like(hr) * self.noise_scale

        # generate noisy HR input
        z = t_broadcast * hr + (1 - t_broadcast) * e

        # target v
        v = (hr - z) / (1 - t_broadcast).clamp_min(self.t_eps)

        # predicted HR from JiTSR
        hr_pred = self.net(z, t, lr)
        v_pred = (hr_pred - z) / (1 - t_broadcast).clamp_min(self.t_eps)

        loss = ((v - v_pred) ** 2).mean()
        return loss

    # ======================================================
    #       CFG: conditional / unconditional v-pred
    # ======================================================
    @torch.no_grad()
    def _forward_sample(self, z, t, lr):
        """
        CFG for SR:
        cond     = use LR as condition
        uncond   = zero-LR, equivalent to "ignore LR"
        """
        # conditional
        x_cond = self.net(z, t.flatten(), lr)
        v_cond = (x_cond - z) / (1 - t).clamp_min(self.t_eps)

        # unconditional (zero LR)
        zero_lr = torch.zeros_like(lr)
        x_uncond = self.net(z, t.flatten(), zero_lr)
        v_uncond = (x_uncond - z) / (1 - t).clamp_min(self.t_eps)

        # cfg interval
        low, high = self.cfg_interval
        interval_mask = (t < high) & (t > low)
        cfg_scale_interval = torch.where(interval_mask, self.cfg_scale, 1.0)

        return v_uncond + cfg_scale_interval * (v_cond - v_uncond)

    # ======================================================
    #            ODE sampler (Euler and Heun)
    # ======================================================
    @torch.no_grad()
    def _euler_step(self, z, t, t_next, lr):
        v_pred = self._forward_sample(z, t, lr)
        return z + (t_next - t) * v_pred

    @torch.no_grad()
    def _heun_step(self, z, t, t_next, lr):
        v_pred_t = self._forward_sample(z, t, lr)

        # Euler step
        z_euler = z + (t_next - t) * v_pred_t
        v_pred_next = self._forward_sample(z_euler, t_next, lr)

        # Heun step
        v_pred = 0.5 * (v_pred_t + v_pred_next)
        return z + (t_next - t) * v_pred

    # ======================================================
    #                   SAMPLING (SR)
    # ======================================================
    @torch.no_grad()
    def generate(self, lr):
        """
        SR sampling:
        lr : (B, C, h, w)
        returns HR image (denoised)
        """
        device = lr.device
        B = lr.size(0)

        # initial noise
        z = self.noise_scale * torch.randn(
            B, 3, self.hr_size, self.hr_size, device=device
        )

        # ODE timesteps
        timesteps = (
            torch.linspace(0.0, 1.0, self.steps + 1, device=device)
            .view(-1, *([1] * z.ndim))
            .expand(-1, B, -1, -1, -1)
        )

        step_fn = self._euler_step if self.method == "euler" else self._heun_step

        # ODE evolution
        for i in range(self.steps - 1):
            z = step_fn(z, timesteps[i], timesteps[i + 1], lr)

        # final step
        z = self._euler_step(z, timesteps[-2], timesteps[-1], lr)

        return z

    # ======================================================
    #                      EMA update
    # ======================================================
    @torch.no_grad()
    def update_ema(self):
        source_params = list(self.parameters())

        for targ, src in zip(self.ema_params1, source_params):
            targ.data.mul_(self.ema_decay1).add_(src.data, alpha=1 - self.ema_decay1)

        for targ, src in zip(self.ema_params2, source_params):
            targ.data.mul_(self.ema_decay2).add_(src.data, alpha=1 - self.ema_decay2)
