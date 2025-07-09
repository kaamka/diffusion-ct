import os
import glob
import logging
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

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
from monai.data import Dataset
from monai.transforms import (
    LoadImage,
    EnsureChannelFirst,
    ScaleIntensityRange,
    EnsureType,
    Compose,
    RandSpatialCrop,
    Lambda
)

from UNet3D_2D import UNet3DModel
import argparse

# silence warning
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
    load_model_from_file: bool
    checkpoint_path: str
    logging_dir = str

def get_patch_coords(full_size, patch_size, overlap=0):
    d, h, w = full_size
    pd, ph, pw = patch_size
    coords = []
    for z in range(0, d - pd + 1, pd - overlap):
        for y in range(0, h - ph + 1, ph - overlap):
            for x in range(0, w - pw + 1, pw - overlap):
                coords.append((z, y, x))
    return coords

def stitch_patches(patches, coords, full_size, patch_size):
    stitched = torch.zeros((1, *full_size))
    count_map = torch.zeros_like(stitched)
    for patch, (z, y, x) in zip(patches, coords):
        stitched[:, z:z+patch_size[0], y:y+patch_size[1], x:x+patch_size[2]] += patch
        count_map[:, z:z+patch_size[0], y:y+patch_size[1], x:x+patch_size[2]] += 1
    return stitched / count_map

def get_patch_transforms(config):
    win_wid = 400
    win_lev = 60
    return Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        ScaleIntensityRange(a_min=win_lev - (win_wid / 2), a_max=win_lev + (win_wid / 2), b_min=0.0, b_max=1.0, clip=True),
        RandSpatialCrop(roi_size=(config.scan_depth, 256, 256), random_size=False),
        EnsureType()
    ])

def load_patch_datasets(config):
    images_pattern = os.path.join(config.data_dir, '*.nii.gz')
    images = sorted(glob.glob(images_pattern))
    transforms = get_patch_transforms(config)
    dataset = Dataset(data=images, transform=transforms)
    val_size = len(dataset) // 5
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [len(dataset) - val_size, val_size])
    return train_dataset, val_dataset

def prepare_batch(batch_data, device=None, non_blocking=False):
    t = Compose([
        Lambda(lambda t: (t * 2) - 1)
    ])
    return t(batch_data.permute(0, 1, 4, 2, 3).to(device=device, non_blocking=non_blocking))

@torch.no_grad()
def generate(n, model, config, noise_scheduler, device):
    gen_dir = os.path.join(config.output_dir, "generated_examples")
    os.makedirs(gen_dir, exist_ok=True)

    patch_size = (config.scan_depth, 256, 256)
    full_size = (config.scan_depth, config.image_size, config.image_size)
    coords = get_patch_coords(full_size, patch_size, overlap=64)

    for i in range(n):
        print(f"Generating: {i + 1}/{n} scan")
        generator = torch.Generator(device=device).manual_seed(config.seed + i)
        generated_patches = []

        for (z, y, x) in tqdm(coords, desc="Patches"):
            noise = torch.randn((1, 1, *patch_size), generator=generator, device=device)
            for t in noise_scheduler.timesteps:
                model_output = model(noise, t).sample
                noise = noise_scheduler.step(model_output, t, noise, generator=generator).prev_sample
            patch = (noise / 2 + 0.5).clamp(0, 1).cpu().squeeze(0)
            generated_patches.append(patch)

        volume = stitch_patches(generated_patches, coords, full_size, patch_size)
        volume_np = volume.squeeze(0).permute(1, 2, 0).numpy()

        fig = plt.figure(figsize=(15, 15))
        _ = matshow3d(volume=volume_np, fig=fig, every_n=1, frame_dim=-1, cmap="gray")
        fig.savefig(f"{gen_dir}/{i+1}.png")
        plt.close(fig)

@torch.no_grad()
def evaluate(model, config, epoch, noise_scheduler, device):
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
    fig.savefig(f"{test_dir}/{epoch:04d}.png")
    plt.close(fig)

def parse_args():
    parser = argparse.ArgumentParser(description="Patch-based 3D diffusion training")

    parser.add_argument("--data_dir", type=str, required=True, help="Path to NIfTI data directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs and models")
    parser.add_argument("--image_size", type=int, default=512, help="XY size of the full volume")
    parser.add_argument("--scan_depth", type=int, default=32, help="Z depth of the volume")
    parser.add_argument("--batch_size", type=int, default=1, help="Training batch size")
    parser.add_argument("--num_epochs", type=int, default=4000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_warmup_steps", type=int, default=1000)
    parser.add_argument("--save_image_epochs", type=int, default=100)
    parser.add_argument("--save_model_epochs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load_model_from_file", action="store_true", help="Load model from checkpoint")
    parser.add_argument("--checkpoint_path", type=str, default="", help="Path to model checkpoint")
    parser.add_argument("--logging_dir", type=str, default=None, help="Optional tensorboard logging dir")

    return parser.parse_args()

def main():
    args = parse_args()