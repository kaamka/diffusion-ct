import argparse
import os
import torch
import nibabel as nib
import numpy as np
from tqdm import tqdm
from diffusers import DDPMScheduler
from UNet3D_2D import UNet3DModel
import matplotlib.pyplot as plt
from monai.visualize import matshow3d
from data_utils import window_scale_intensity_range, inverse_scale_intensity_range



def load_model(model_path, image_size, scan_depth, device):
    model = UNet3DModel(
        sample_size=image_size,
        sample_depth=scan_depth,
        in_channels=1,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(32, 64, 64, 128, 256, 512, 512),
        down_block_types=(
            "DownBlock3D",
            "DownBlock2D",
            "DownBlock3D",
            "DownBlock2D",
            "DownBlock3D",
            "AttnDownBlock3D",
            "DownBlock3D",
        ),
        up_block_types=(
            "UpBlock3D",
            "AttnUpBlock3D",
            "UpBlock2D",
            "UpBlock3D",
            "UpBlock2D",
            "UpBlock3D",
            "UpBlock3D",
        ),
        norm_num_groups=32,
        dropout=0.0,
    )

    # Wczytanie checkpointu
    state_dict = torch.load(model_path, map_location=device)

    # Naprawa prefixu 'module.' jeśli checkpoint był trenowany w DataParallel
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v  # usuń 'module.'
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def generate_sample(model, scheduler, image_size, scan_depth, device, seed=None):
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)

    image_shape = (1, 1, scan_depth, image_size, image_size)
    image = torch.randn(image_shape, generator=generator, device=device)

    for t in tqdm(scheduler.timesteps, desc="Denoising"):
        model_output = model(image, t).sample
        image = scheduler.step(model_output, t, image, generator=generator).prev_sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().numpy()[0, 0]
    image = np.transpose(image, (1, 2, 0)) # (H, W, D)
    return image  # shape (H, W, D)


# def generate_sample(model, scheduler, image_size, scan_depth, device, seed=None):
#     generator = torch.Generator(device=device)
#     if seed is not None:
#         generator.manual_seed(seed)

#     image_shape = (1, 1, scan_depth, image_size, image_size)
#     image = torch.randn(image_shape, generator=generator, device=device)

#     for t in tqdm(scheduler.timesteps, desc="Denoising"):
#         with torch.no_grad():
#             model_output = model(image, t)
#         image = scheduler.step(model_output, t, image, generator=generator).prev_sample

#     image = (image / 2 + 0.5).clamp(0, 1)
#     image = image.cpu().numpy()[0, 0]
#     return image  # shape (D, H, W)



def save_png(volume, out_path):
    fig = plt.figure(figsize=(12, 12))
    _ = matshow3d(volume=volume, fig=fig, every_n=1, frame_dim=0, cmap="gray")
    fig.savefig(out_path)
    plt.close(fig)


def save_nifti(volume, out_path):
    affine = np.eye(4)
    img = nib.Nifti1Image(volume.astype(np.float32), affine)
    nib.save(img, out_path)


def main():
    parser = argparse.ArgumentParser(description="Generate samples from trained diffusion model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save generated scans")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--scan_depth", type=int, default=32)
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading model...")
    model = load_model(args.model_path, args.image_size, args.scan_depth, device)
    # print(model.keys())

    scheduler = DDPMScheduler(num_train_timesteps=1500)

    print(f"Generating {args.num_samples} samples...")
    for i in range(args.num_samples):
        print(f"Sample {i+1}/{args.num_samples}")
        volume = generate_sample(model, scheduler, args.image_size, args.scan_depth, device, seed=args.seed)

        png_path = os.path.join(args.output_dir, f"sample_2_{i+1}.png")
        nii_path = os.path.join(args.output_dir, f"sample_2_{i+1}.nii.gz")

        save_png(volume, png_path)
        save_nifti(volume, nii_path)

    print("Done.")


if __name__ == "__main__":
    main()
