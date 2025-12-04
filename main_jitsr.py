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

from denoiser_sr import DenoiserSR
from engine_jitsr import load_config, train_one_epoch, evaluate


def get_args_parser():
    parser = argparse.ArgumentParser("JiTSR SR Training", add_help=True)

    parser.add_argument(
        "--config", type=str, required=True, help="Path to config yaml file"
    )

    parser.add_argument("--resume", default="", type=str)

    return parser


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
