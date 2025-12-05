import yaml
import os
import sys
import math
from math import log10

from PIL import Image
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

import torch.nn.functional as F
import torchvision.transforms as transforms

import util.misc as misc
import util.lr_sched as lr_sched
import copy

DEFAULT_CFG = dict(
    # ---------- architecture ----------
    hr_size=256,
    lr_size=64,
    hr_patch=16,
    lr_patch=4,
    hidden_size=1024,
    depth=24,
    num_heads=16,
    mlp_ratio=4.0,
    bottleneck_dim=128,
    in_context_start=8,
    attn_dropout=0.0,
    proj_dropout=0.0,
    # ---------- training ----------
    epochs=200,
    warmup_epochs=5,
    batch_size=128,
    blr=5e-5,  # base lr
    lr=None,  # 实际 lr，默认用 blr * batch_size / 256 算
    min_lr=0.0,
    lr_schedule="constant",
    weight_decay=0.0,
    seed=0,
    start_epoch=0,
    num_workers=12,
    pin_mem=True,
    # ---------- EDM ----------
    noise_scale=1.0,
    P_mean=-0.8,
    P_std=0.8,
    t_eps=0.05,
    ema_decay1=0.9999,
    ema_decay2=0.9996,
    # ---------- sampling ----------
    sampling_method="heun",
    num_sampling_steps=50,
    cfg=1.0,
    interval_min=0.0,
    interval_max=1.0,
    gen_bsz=64,
    online_eval=False,
    evaluate_gen=False,
    # 如果你的代码里有 args.eval_freq，就加上：
    eval_freq=10,
    # ---------- dataset ----------
    data_path="./data/imagenet",
    # ---------- logging ----------
    output_dir="./experiments/jitsr",
    save_last_freq=5,
    log_freq=100,
    # ---------- distributed ----------
    device="cuda",
    world_size=1,
    local_rank=-1,
    dist_on_itp=False,
    dist_url="env://",
)

FLOAT_FIELDS = [
    "blr",
    "lr",
    "min_lr",
    "weight_decay",
    "noise_scale",
    "P_mean",
    "P_std",
    "t_eps",
    "ema_decay1",
    "ema_decay2",
    "cfg",
    "interval_min",
    "interval_max",
]

INT_FIELDS = [
    "epochs",
    "batch_size",
    "warmup_epochs",
    "num_workers",
    "save_last_freq",
    "log_freq",
    "num_sampling_steps",
    "gen_bsz",
    "world_size",
    "local_rank",
    "depth",
    "num_heads",
    "hr_size",
    "lr_size",
    "hr_patch",
    "lr_patch",
    "bottleneck_dim",
    "in_context_start",
]

BOOL_FIELDS = ["pin_mem", "online_eval", "evaluate_gen", "dist_on_itp"]


def load_config(args):
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # flatten YAML groups
    yaml_flat = {}
    for k, v in cfg.items():
        if isinstance(v, dict):
            yaml_flat.update(v)
        else:
            yaml_flat[k] = v

    # load default values
    for k, v in DEFAULT_CFG.items():
        setattr(args, k, v)

    # override defaults with YAML
    for k, v in yaml_flat.items():
        setattr(args, k, v)

    # apply casting
    for f in FLOAT_FIELDS:
        if hasattr(args, f):
            try:
                v = getattr(args, f)
                setattr(args, f, float(v) if v is not None else None)
            except:
                raise ValueError(f"Invalid float value for '{f}': {v}")

    for f in INT_FIELDS:
        if hasattr(args, f):
            try:
                setattr(args, f, int(getattr(args, f)))
            except:
                raise ValueError(f"Invalid int value for '{f}': {v}")

    for f in BOOL_FIELDS:
        if hasattr(args, f):
            v = getattr(args, f)
            if isinstance(v, str):
                setattr(args, f, v.lower() in ["1", "true", "yes"])

    return args


# ========================================================================
#                         HR / LR Data Pipeline
# ========================================================================
def build_sr_transform(args):
    hr_size = args.hr_size
    lr_size = args.lr_size

    def make_pair(pil_image):
        # 1. convert to tensor first (safe, fast)
        img = transforms.functional.to_tensor(pil_image)  # [C,H,W]

        # 2. center crop using PyTorch (safe) instead of PIL
        _, H, W = img.shape
        crop = min(H, W)
        y1 = (H - crop) // 2
        x1 = (W - crop) // 2
        img = img[:, y1 : y1 + crop, x1 : x1 + crop]

        # 3. resize to HR using torch (safe)
        img = img.unsqueeze(0)
        hr = F.interpolate(
            img, size=(hr_size, hr_size), mode="bicubic", align_corners=False
        ).squeeze(0)

        # 4. generate LR using torch (safe)
        lr = F.interpolate(
            hr.unsqueeze(0),
            size=(lr_size, lr_size),
            mode="bicubic",
            align_corners=False,
        ).squeeze(0)

        return hr, lr

    return transforms.Lambda(make_pair)


def train_one_epoch(
    model,
    model_without_ddp,
    data_loader,
    optimizer,
    device,
    epoch,
    log_writer=None,
    args=None,
):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))

    header = f"Epoch: [{epoch}]"
    print_freq = 20

    optimizer.zero_grad()

    if log_writer is not None:
        print(f"log_dir: {log_writer.log_dir}")

    for data_iter_step, (hr, lr) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        # ------------------------------
        # adjust LR per iteration
        # ------------------------------
        lr_sched.adjust_learning_rate(
            optimizer,
            data_iter_step / len(data_loader) + epoch,
            args,
        )

        hr = hr.to(device, non_blocking=True).float()  # (B,3,H,W)
        lr = lr.to(device, non_blocking=True).float()  # (B,3,h,w)

        # ------------------------------
        # forward loss
        # ------------------------------
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(hr, lr)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        # ------------------------------
        # backward
        # ------------------------------
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()

        # ------------------------------
        # EMA update
        # ------------------------------
        model_without_ddp.update_ema()

        # ------------------------------
        # logging
        # ------------------------------
        metric_logger.update(loss=loss_value)
        cur_lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=cur_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)

        # tensorboard
        if log_writer is not None:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            if data_iter_step % args.log_freq == 0:
                log_writer.add_scalar("train_loss", loss_value_reduce, epoch_1000x)
                log_writer.add_scalar("lr", cur_lr, epoch_1000x)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class SRFlatFolderDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.paths = [
            os.path.join(root, f)
            for f in os.listdir(root)
            if f.lower().endswith(("jpg", "jpeg", "png"))
        ]
        self.paths.sort()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img)  # transform returns (HR, LR)


@torch.no_grad()
def evaluate(model_without_ddp, args, epoch, batch_size=32, log_writer=None):
    """
    Super-Resolution evaluation with PSNR, SSIM, TensorBoard images.
    No external libs (no skimage, no torchvision.metrics).
    """

    # --------------------------------------
    # Define PSNR
    # --------------------------------------
    def calc_psnr(sr, hr):
        # sr, hr in [0,1], shape (B,3,H,W)
        mse = F.mse_loss(sr, hr, reduction="mean")
        if mse.item() == 0:
            return 100
        return 10 * log10(1.0 / mse.item())

    # --------------------------------------
    # Define SSIM (PyTorch implementation)
    # --------------------------------------
    def calc_ssim(sr, hr):
        """Compute SSIM for (B,3,H,W), channel-wise averaged."""
        C1 = 0.01**2
        C2 = 0.03**2

        # Gaussian-like 11x11 window (but using a fixed average kernel is okay for SR)
        kernel = torch.ones((3, 1, 11, 11), device=sr.device) / (11 * 11)

        mu_x = F.conv2d(sr, kernel, padding=5, groups=3)
        mu_y = F.conv2d(hr, kernel, padding=5, groups=3)

        mu_x2 = mu_x * mu_x
        mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y

        sigma_x2 = F.conv2d(sr * sr, kernel, padding=5, groups=3) - mu_x2
        sigma_y2 = F.conv2d(hr * hr, kernel, padding=5, groups=3) - mu_y2
        sigma_xy = F.conv2d(sr * hr, kernel, padding=5, groups=3) - mu_xy

        ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / (
            (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
        )

        return ssim_map.mean().item()

    # --------------------------------------
    # Setup
    # --------------------------------------
    model_without_ddp.eval()
    device = torch.device(args.device)
    world_size = misc.get_world_size()
    rank = misc.get_rank()

    # --------------------------------------
    # Build validation dataset
    # --------------------------------------
    val_folder = os.path.join(args.data_path, "val")
    dataset_val = SRFlatFolderDataset(val_folder, transform=build_sr_transform(args))

    sampler_val = torch.utils.data.DistributedSampler(
        dataset_val, num_replicas=world_size, rank=rank, shuffle=False
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=batch_size,
        sampler=sampler_val,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=lambda batch: (
            torch.stack([b[0] for b in batch]),  # HR
            torch.stack([b[1] for b in batch]),  # LR
        ),
    )

    # --------------------------------------
    # Output folder
    # --------------------------------------
    save_folder = os.path.join(args.output_dir, f"sr_epoch{epoch}")
    if rank == 0:
        os.makedirs(save_folder, exist_ok=True)

    # --------------------------------------
    # Switch to EMA1
    # --------------------------------------
    model_state = copy.deepcopy(model_without_ddp.state_dict())
    ema_state = copy.deepcopy(model_without_ddp.state_dict())
    for i, (name, _) in enumerate(model_without_ddp.named_parameters()):
        ema_state[name] = model_without_ddp.ema_params1[i]
    print("[Eval] Switch to EMA1")
    model_without_ddp.load_state_dict(ema_state)

    # --------------------------------------
    # Evaluate
    # --------------------------------------
    global_psnr = []
    global_ssim = []

    # Only visualize first 3 batches
    log_visual_limit = 3

    for it, (hr, lr) in enumerate(data_loader_val):
        hr = hr.to(device)
        lr = lr.to(device)

        # SR generation
        sr = model_without_ddp.generate(lr)
        sr = torch.clamp(sr, 0, 1)

        # --- Compute metrics ---
        psnr_val = calc_psnr(sr, hr)
        ssim_val = calc_ssim(sr, hr)

        global_psnr.append(psnr_val)
        global_ssim.append(ssim_val)

        # --- TensorBoard visualization ---
        if log_writer is not None and rank == 0 and it < log_visual_limit:
            # LR → upsample to HR shape for visual comparison
            lr_up = F.interpolate(
                lr, size=sr.shape[-2:], mode="bicubic", align_corners=False
            )

            # Combine LR_up | SR | HR into a single wide image
            grid = torch.cat([lr_up.cpu(), sr.cpu(), hr.cpu()], dim=3)

            log_writer.add_images(f"eval/samples_batch{it}", grid, epoch)

        # --- Save SR images ---
        sr_np = (sr.cpu().numpy() * 255).astype(np.uint8)
        for b in range(sr_np.shape[0]):
            idx = it * batch_size + b
            img = sr_np[b].transpose(1, 2, 0)
            cv2.imwrite(os.path.join(save_folder, f"{str(idx).zfill(5)}.png"), img)

    torch.distributed.barrier()

    # --------------------------------------
    # Log metrics
    # --------------------------------------
    if log_writer is not None and rank == 0:
        log_writer.add_scalar("eval/psnr", sum(global_psnr) / len(global_psnr), epoch)
        log_writer.add_scalar("eval/ssim", sum(global_ssim) / len(global_ssim), epoch)

    # --------------------------------------
    # Restore original params
    # --------------------------------------
    print("[Eval] Restore original weights")
    model_without_ddp.load_state_dict(model_state)

    print(f"[Eval] SR results saved to: {save_folder}")

    return save_folder
