import argparse
import datetime
import numpy as np
import os
import time
from pathlib import Path

import yaml
from types import SimpleNamespace

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F

from util.crop import center_crop_arr
import util.misc as misc

import copy
from engine_jit import train_one_epoch, evaluate

from denoiser_sr import DenoiserSR


def get_args_parser():
    parser = argparse.ArgumentParser("JiTSR SR Training", add_help=True)

    parser.add_argument(
        "--config", type=str, required=True, help="Path to config yaml file"
    )

    parser.add_argument("--resume", default="", type=str)

    return parser


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

    # load default values
    for k, v in DEFAULT_CFG.items():
        setattr(args, k, v)

    # override defaults with YAML
    for k, v in yaml_flat.items():
        setattr(args, k, v)

    # -----------------------------
    # 强制类型修正（关键）
    # -----------------------------
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
    """Return transform that makes HR + LR pairs."""
    hr_size = args.hr_size
    lr_size = args.lr_size

    def make_pair(img):
        img = center_crop_arr(img, hr_size)
        img = transforms.functional.to_tensor(img)
        hr = img

        lr = F.interpolate(
            hr.unsqueeze(0),
            size=(lr_size, lr_size),
            mode="bicubic",
            align_corners=False,
        ).squeeze(0)

        return hr, lr  # each is (C,H,W)

    return transforms.Lambda(make_pair)


# ========================================================================
#                              MAIN
# ========================================================================
def main(args):
    args = get_args_parser().parse_args()
    args = load_config(args)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    misc.init_distributed_mode(args)
    device = torch.device(args.device)

    # seed
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    # logging
    if global_rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        log_writer = SummaryWriter(args.output_dir)
    else:
        log_writer = None

    # ------------------------------------------------------------
    #                   Dataset: HR/LR pairs
    # ------------------------------------------------------------
    dataset_train = datasets.ImageFolder(
        os.path.join(args.data_path, "train"), transform=build_sr_transform(args)
    )

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        collate_fn=lambda batch: (
            torch.stack([b[0] for b in batch]),  # HR batch
            torch.stack([b[1] for b in batch]),  # LR batch
        ),
    )

    # ------------------------------------------------------------
    #                     Create SR Denoiser
    # ------------------------------------------------------------
    model = DenoiserSR(args)
    model.to(device)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    model_without_ddp = model.module

    # learning rate
    eff_batch_size = args.batch_size * num_tasks
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    param_groups = misc.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))

    # ------------------------------------------------------------
    #                    Resume from checkpoint
    # ------------------------------------------------------------
    checkpoint_path = (
        os.path.join(args.resume, "checkpoint-last.pth") if args.resume else None
    )
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])

        ema1 = checkpoint["model_ema1"]
        ema2 = checkpoint["model_ema2"]
        model_without_ddp.ema_params1 = [
            ema1[name].cuda() for name, _ in model_without_ddp.named_parameters()
        ]
        model_without_ddp.ema_params2 = [
            ema2[name].cuda() for name, _ in model_without_ddp.named_parameters()
        ]

        optimizer.load_state_dict(checkpoint["optimizer"])
        args.start_epoch = checkpoint["epoch"] + 1
        print("Loaded checkpoint from", args.resume)

    else:
        model_without_ddp.ema_params1 = copy.deepcopy(
            list(model_without_ddp.parameters())
        )
        model_without_ddp.ema_params2 = copy.deepcopy(
            list(model_without_ddp.parameters())
        )
        print("Training from scratch")

    # ------------------------------------------------------------
    #                 Evaluate-only mode (sampling)
    # ------------------------------------------------------------
    if args.evaluate_gen:
        with torch.no_grad():
            evaluate(
                model_without_ddp,
                args,
                0,
                batch_size=args.gen_bsz,
                log_writer=log_writer,
            )
        return

    # ------------------------------------------------------------
    #                         Training Loop
    # ------------------------------------------------------------
    print("Start training...")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(
            model,
            model_without_ddp,
            data_loader_train,
            optimizer,
            device,
            epoch,
            log_writer=log_writer,
            args=args,
        )

        # save "last"
        if epoch % args.save_last_freq == 0 or epoch + 1 == args.epochs:
            misc.save_model(
                args=args,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                epoch=epoch,
                epoch_name="last",
            )

        if epoch % 100 == 0 and epoch > 0:
            misc.save_model(
                args=args,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                epoch=epoch,
            )

        # online evaluation
        if args.online_eval and (
            epoch % args.eval_freq == 0 or epoch + 1 == args.epochs
        ):
            with torch.no_grad():
                evaluate(
                    model_without_ddp,
                    args,
                    epoch,
                    batch_size=args.gen_bsz,
                    log_writer=log_writer,
                )

        if log_writer is not None:
            log_writer.flush()

    total_time = time.time() - start_time
    print(
        "Training finished. Total time:",
        str(datetime.timedelta(seconds=int(total_time))),
    )


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
