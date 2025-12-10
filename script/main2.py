#!/usr/bin/env python3
"""
main_retrain.py
Training + resume-aware checkpointing for UNet3D diffusion model.
Saves/loads full checkpoint: model, optimizer, lr_scheduler, noise_scheduler, epoch, global_step.
"""

import glob
import logging
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
import os
import argparse

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

import diffusers
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.utils import is_tensorboard_available
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration

from monai.visualize import matshow3d
from monai.data import CacheDataset
from monai.transforms import (
    LoadImage,
    EnsureChannelFirst,
    Lambda,
    Compose,
    ScaleIntensityRange,
    EnsureType,
    Resize,
)

from UNet3D_2D import UNet3DModel

# silence annoying warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"


@dataclass
class TrainingConfig:
    data_dir: str
    image_size: int
    scan_depth: int
    batch_size: int
    num_epochs: int
    learning_rate: float
    lr_warmup_steps: int
    save_image_epochs: int
    save_model_epochs: int
    output_dir: str
    seed: int
    checkpoint_path: str
    logging_dir: str
    first_epoch: int


# -------------------------
# Utilities: checkpoint save/load
# -------------------------
def _move_state_to_device(state, device):
    """
    Move all tensors in a nested state dict to `device`.
    Works for optimizer.state (which is a dict of dicts), lr_scheduler.state_dict etc.
    """
    if isinstance(state, dict):
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
            else:
                _move_state_to_device(v, device)
    elif isinstance(state, (list, tuple)):
        for i, v in enumerate(state):
            if isinstance(v, torch.Tensor):
                state[i] = v.to(device)
            else:
                _move_state_to_device(v, device)


def save_checkpoint(path, model, optimizer, lr_scheduler, noise_scheduler,
                    epoch, global_step, accelerator=None):

    ckpt = {
        "model": accelerator.get_state_dict(model) if accelerator else model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        # zamiast state_dict() używamy config

        # # Zapisujemy PEŁNY scheduler
        # "noise_scheduler": noise_scheduler.to_dict(),
        #"noise_scheduler": noise_scheduler.config, 

                # scheduler config + parametry faktyczne
        "noise_scheduler_config": noise_scheduler.config,
        "noise_scheduler_betas": noise_scheduler.betas.cpu(),

        "epoch": epoch,
        "global_step": global_step,
    }

    torch.save(ckpt, path)


def load_checkpoint(path, model, optimizer, lr_scheduler, noise_scheduler, *, device, accelerator=None):

    ckpt = torch.load(path, map_location=device)


    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    lr_scheduler.load_state_dict(ckpt["lr_scheduler"])

    # Odtwarzamy scheduler z configu
    #noise_scheduler = DDPMScheduler.from_config(ckpt["noise_scheduler"])
    noise_scheduler = DDPMScheduler.from_config(ckpt["noise_scheduler_config"])
    # Podmieniamy bety
    noise_scheduler.betas = ckpt["noise_scheduler_betas"].to(device)
    noise_scheduler.alphas = 1.0 - noise_scheduler.betas
    noise_scheduler.alphas_cumprod = noise_scheduler.alphas.cumprod(0)

    # Odtwarzamy scheduler 1:1
    #noise_scheduler = DDPMScheduler.from_dict(ckpt["noise_scheduler"])

    epoch = ckpt.get("epoch", 0)
    global_step = ckpt.get("global_step", 0)

    return model, optimizer, lr_scheduler, noise_scheduler, epoch, global_step


# -------------------------
# Evaluate
# -------------------------
@torch.no_grad()
def evaluate(model, config, epoch, noise_scheduler, device, retrain=""):
    generator = torch.Generator(device=device)
    generator.manual_seed(config.seed)
    image_shape = (config.batch_size, 1, config.scan_depth, config.image_size, config.image_size)
    image = torch.randn(image_shape, generator=generator, device=device).to(device)

    for t in tqdm(noise_scheduler.timesteps):
        model_output = model(image, t).sample
        image = noise_scheduler.step(model_output, t, image, generator=generator).prev_sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 1, 3, 4, 2).reshape(config.batch_size, config.image_size, config.image_size, config.scan_depth).numpy()
    fig = plt.figure(figsize=(15, 15))
    _ = matshow3d(volume=image, fig=fig, every_n=1, frame_dim=-1, cmap="gray")
    test_dir = os.path.join(config.output_dir, 'samples')
    os.makedirs(test_dir, exist_ok=True)
    fig.savefig(f"{test_dir}/{retrain}{epoch:04d}.png")
    plt.close(fig)


@torch.no_grad()
def generate(n, model, config, noise_scheduler, device):
    gen_dir = os.path.join(config.output_dir, "generated_examples")
    os.makedirs(gen_dir, exist_ok=True)
    for i in range(n):
        print(f"Generating: {i + 1}/{n} scan")
        eval_device = device
        generator = torch.Generator(device=eval_device)
        image_shape = (config.batch_size, 1, config.scan_depth, config.image_size, config.image_size)
        image = torch.randn(image_shape, generator=generator, device=eval_device).to(eval_device)
        for t in tqdm(noise_scheduler.timesteps):
            model_output = model.to(eval_device)(image, t).sample
            image = noise_scheduler.step(model_output, t, image, generator=generator).prev_sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 1, 3, 4, 2).reshape(config.batch_size, config.image_size, config.image_size, config.scan_depth).numpy()
        fig = plt.figure(figsize=(15, 15))
        _ = matshow3d(volume=image, fig=fig, every_n=1, frame_dim=-1, cmap="gray")
        fig.savefig(f"{gen_dir}/{i+1}.png")
        plt.close(fig)


# -------------------------
# Argument parsing
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Training script configuration (resume-ready)")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to data directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--scan_depth", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=4000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_warmup_steps", type=int, default=1000)
    parser.add_argument("--save_image_epochs", type=int, default=100)
    parser.add_argument("--save_model_epochs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint_path", type=str, default="", help="Path to checkpoint to resume from")
    parser.add_argument("--first_epoch", type=int, default=0, help="If not resuming, start epoch")
    parser.add_argument("--logging_dir", type=str, default=None, help="Where to save logs (default: output_dir/logs)")
    return parser.parse_args()


# -------------------------
# Main
# -------------------------
def main():
    args = parse_args()
    cfg = TrainingConfig(
        data_dir=args.data_dir,
        image_size=args.image_size,
        scan_depth=args.scan_depth,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        lr_warmup_steps=args.lr_warmup_steps,
        save_image_epochs=args.save_image_epochs,
        save_model_epochs=args.save_model_epochs,
        output_dir=args.output_dir,
        seed=args.seed,
        checkpoint_path=args.checkpoint_path,
        logging_dir=args.logging_dir if args.logging_dir is not None else os.path.join(args.output_dir, "logs"),
        first_epoch=args.first_epoch,
    )

    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, "models"), exist_ok=True)

    logger = get_logger(__name__, log_level="INFO")
    accelerator_project_config = ProjectConfiguration(project_dir=cfg.output_dir, logging_dir=cfg.logging_dir)

    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))
    accelerator = Accelerator(log_with="tensorboard", project_config=accelerator_project_config, kwargs_handlers=[kwargs])

    if not is_tensorboard_available():
        raise ImportError("tensorboard not found")

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        diffusers.utils.logging.set_verbosity_info()
    else:
        diffusers.utils.logging.set_verbosity_error()

    # -------------------------
    # Build model (uninitialized weights)
    # -------------------------
    model = UNet3DModel(
        sample_size=cfg.image_size,
        sample_depth=cfg.scan_depth,
        in_channels=1,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(32, 64, 64, 128, 256, 512, 512),
        down_block_types=("DownBlock3D", "DownBlock2D", "DownBlock3D", "DownBlock2D", "DownBlock3D", "AttnDownBlock3D", "DownBlock3D"),
        up_block_types=("UpBlock3D", "AttnUpBlock3D", "UpBlock2D", "UpBlock3D", "UpBlock2D", "UpBlock3D", "UpBlock3D"),
        norm_num_groups=32,
        dropout=0.0,
    )

    # noise scheduler (we create it now so we can save/load its state)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1500)

    # optimizer & lr scheduler (based on model.parameters)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    total_training_steps = None  # will set after dataloader size known
    # create a dummy lr_scheduler; will re-create after dataloader set to have correct num_training_steps
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=cfg.lr_warmup_steps, num_training_steps=1000)

    # -------------------------
    # Data transforms and dataset
    # -------------------------
    win_wid = 400
    win_lev = 60
    transforms = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        Resize((cfg.image_size, cfg.image_size, cfg.scan_depth)),
        ScaleIntensityRange(a_min=win_lev - (win_wid / 2), a_max=win_lev + (win_wid / 2), b_min=0.0, b_max=1.0, clip=True),
        EnsureType()
    ])
    images = sorted(glob.glob(os.path.join(cfg.data_dir, '*.nii.gz')))
    dataset = CacheDataset(images, transforms)
    val_size = len(dataset) // 5
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [len(dataset) - val_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, num_workers=10, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.batch_size, num_workers=10, shuffle=True)

    # Now that we know dataloader size, recreate lr_scheduler with correct num_training_steps
    num_training_steps = len(train_loader) * cfg.num_epochs
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=cfg.lr_warmup_steps, num_training_steps=num_training_steps)

  
    # -------------------------
    # Optionally resume from checkpoint
    # -------------------------
    start_epoch = cfg.first_epoch
    global_step = 0
    if cfg.checkpoint_path:
        # Attempt to load checkpoint into prepared objects
        model, optimizer, lr_scheduler, noise_scheduler, ckpt_epoch, ckpt_step = load_checkpoint(cfg.checkpoint_path, model, 
                                                optimizer, lr_scheduler, noise_scheduler, 
                                                device=accelerator.device, accelerator=accelerator)
        start_epoch = ckpt_epoch + 1  # resume from next epoch
        global_step = ckpt_step


    # -------------------------
    # Prepare with accelerator (wrap / move to devices)
    # -------------------------
    model, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, lr_scheduler
    )
    

    # init trackers
    if accelerator.is_main_process:
        run = Path(__file__).stem
        accelerator.init_trackers(run)

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num Epochs = {cfg.num_epochs}")

    # -------------------------
    # Training loop
    # -------------------------
    for epoch in range(start_epoch, start_epoch + cfg.num_epochs):
        model.train()
        progress_bar = tqdm(total=len(train_loader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        train_loss = 0.0
        for step, batch in enumerate(train_loader):
            # prepare batch: MONAI CacheDataset returns MetaTensor with shape (C,H,W,D) probably, we want (B,C,D,H,W) later
            # batch is a dict or tensor depending on dataset; assuming data tensor returned directly
            clean_images = batch  # MONAI CacheDataset yields the tensor already shaped like (B, C, H, W, D)?? -> user used permute previously
            # ensure correct layout: we used in training earlier .permute(0,1,4,2,3)
            # If your dataset yields (B,1,H,W,D), we convert to (B,1,D,H,W):
            clean_images = clean_images.permute(0, 1, 4, 2, 3).to(device=accelerator.device)

            # forward / diffusion training step
            noise = torch.randn_like(clean_images)
            bs = clean_images.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device).long()
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            model_output = model(noisy_images, timesteps).sample
            loss = F.mse_loss(model_output.float(), noise.float())
            train_loss += loss.detach().item()

            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # housekeeping
            torch.cuda.empty_cache()
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

        # validation
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for step, batch in enumerate(val_loader):
                clean_images = batch.permute(0, 1, 4, 2, 3).to(device=accelerator.device)
                noise = torch.randn_like(clean_images)
                bs = clean_images.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device).long()
                noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                val_loss += F.mse_loss(noise_pred, noise).detach().item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        accelerator.log({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)

        progress_bar.close()

        # Save images and checkpoint on main process only
        if accelerator.is_main_process:
            retrain_tag = "" if cfg.checkpoint_path == "" else "retrain_"
            # evaluation images
            if epoch == (start_epoch + cfg.num_epochs - 1) or (epoch + 1) % cfg.save_image_epochs == 0:
                evaluate(accelerator.unwrap_model(model), cfg, epoch, noise_scheduler, accelerator.device, retrain=retrain_tag)

            # checkpoint
            if epoch == (start_epoch + cfg.num_epochs - 1) or (epoch + 1) % cfg.save_model_epochs == 0:
                ckpt_path = os.path.join(cfg.output_dir, "models", f"{retrain_tag}ckpt_epoch_{epoch}.pt")
                save_checkpoint(ckpt_path, model, optimizer, lr_scheduler, noise_scheduler, epoch, global_step, accelerator=accelerator)

        accelerator.wait_for_everyone()

    # final generate
    if accelerator.is_main_process:
        generate(10, accelerator.unwrap_model(model), cfg, noise_scheduler, accelerator.device)

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
