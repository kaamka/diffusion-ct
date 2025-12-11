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


# -------------------------------------------------------
# Load model + scheduler
# -------------------------------------------------------
def load_from_checkpoint(ckpt_path, image_size, scan_depth, device,
                         beta_schedule=None,
                         prediction_type=None,
                         timestep_spacing=None,
                         num_inference_steps=None,
                         rescale_betas_zero_snr=None):

    ckpt = torch.load(ckpt_path, map_location=device)

    # Odtwarzamy model tak jak w treningu
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

    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)
    model.eval()

    # Odtwarzamy scheduler z checkpointu
    scheduler_cfg = ckpt["noise_scheduler_config"]

    # Nadpisujemy parametry jeśli podano
    if beta_schedule is not None:
        scheduler_cfg["beta_schedule"] = beta_schedule

    if prediction_type is not None:
        scheduler_cfg["prediction_type"] = prediction_type

    if timestep_spacing is not None:
        scheduler_cfg["timestep_spacing"] = timestep_spacing

    if rescale_betas_zero_snr is not None:
        scheduler_cfg["rescale_betas_zero_snr"] = rescale_betas_zero_snr

    noise_scheduler = DDPMScheduler.from_config(scheduler_cfg)

    # Zmiana liczby kroków
    if num_inference_steps is not None:
        noise_scheduler.set_timesteps(num_inference_steps)
    else:
        noise_scheduler.set_timesteps(scheduler_cfg.get("num_train_timesteps", 1000))

    return model, noise_scheduler


# -------------------------------------------------------
# Generate single volume
# -------------------------------------------------------
@torch.no_grad()
def generate_sample(model, scheduler, H, D, device, seed=None):

    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)

    x = torch.randn((1, 1, D, H, H), generator=generator, device=device)

    for t in tqdm(scheduler.timesteps, desc="Denoising"):
        noise_pred = model(x, t).sample
        step = scheduler.step(noise_pred, t, x)
        x = step.prev_sample

    x = (x / 2 + 0.5).clamp(0, 1)
    return x.cpu().numpy()[0, 0]


# -------------------------------------------------------
# Save utilities
# -------------------------------------------------------
def save_png(volume, out_path):
    fig = plt.figure(figsize=(12, 12))
    _ = matshow3d(volume=volume, fig=fig, every_n=1, frame_dim=0, cmap="gray")
    fig.savefig(out_path)
    plt.close(fig)


def save_nifti(volume, out_path, spacing=(1.0, 1.0, 2.5)):
    vol = np.asarray(volume).astype(np.float32)
    vol = np.transpose(vol, (1, 2, 0))

    sx, sy, sz = spacing
    affine = np.array([
        [sx, 0, 0, 0],
        [0, sy, 0, 0],
        [0, 0, sz, 0],
        [0, 0, 0, 1],
    ], dtype=np.float32)

    header = nib.Nifti1Header()
    header.set_data_dtype(np.float32)
    header["pixdim"][1:4] = [sx, sy, sz]
    header["sform_code"] = 1
    header["qform_code"] = 1

    img = nib.Nifti1Image(vol, affine=affine, header=header)
    img.set_sform(affine, code=1)
    img.set_qform(affine, code=1)

    nib.save(img, out_path)


# -------------------------------------------------------
# Main
# -------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate samples from DDPM checkpoint")

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--image_size", type=int, default=400)
    parser.add_argument("--scan_depth", type=int, default=32)
    parser.add_argument("--seed", type=int, default=None)

    # Nowe DDPM-legalne parametry
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--beta_schedule", type=str, default=None)
    parser.add_argument("--prediction_type", type=str, default=None)
    parser.add_argument("--timestep_spacing", type=str, default=None)
    parser.add_argument("--rescale_betas_zero_snr", action="store_true")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading checkpoint...")
    model, scheduler = load_from_checkpoint(
        args.model_path,
        args.image_size,
        args.scan_depth,
        device,
        beta_schedule=args.beta_schedule,
        prediction_type=args.prediction_type,
        timestep_spacing=args.timestep_spacing,
        num_inference_steps=args.num_inference_steps,
        rescale_betas_zero_snr=args.rescale_betas_zero_snr,
    )

    print(f"Generating {args.num_samples} samples...")
    for i in range(args.num_samples):
        print(f"\nSample {i+1}/{args.num_samples}")
        vol = generate_sample(model, scheduler, args.image_size, args.scan_depth, device, seed=args.seed)

        png_path = os.path.join(args.output_dir, f"sample_{i+1}.png")
        nii_path = os.path.join(args.output_dir, f"sample_{i+1}.nii.gz")

        save_png(vol, png_path)
        save_nifti(vol, nii_path)

    print("Done.")


if __name__ == "__main__":
    main()
