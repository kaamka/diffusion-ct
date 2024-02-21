# setup TrainingConfig, setup model, python ./main.py
# TODO saving how data is split between test and train

from diffusers.optimization import get_cosine_schedule_with_warmup
import torch.nn.functional as F
from dataclasses import dataclass
from tqdm import tqdm
import glob
import copy
from torchvision import transforms
import torch
import os
from monai.visualize import matshow3d
import matplotlib.pyplot as plt

from monai.data import CacheDataset
from monai.utils import first

from diffusers import DDPMScheduler
from UNet3D_2D import UNet3DModel

import time

def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        func(*args, **kwargs)
        end_time = time.time()
        exec_time = end_time - start_time
        print(f'{func.__name__} took {exec_time:.2f} seconds')
    return wrapper



from monai.transforms import (
    LoadImage,
    AddChannel,
    ToTensor,
    Lambda,
    Compose,
    RandFlip,
    RandRotate,
    RandZoom,
    ScaleIntensity,
    ScaleIntensityRange,
    EnsureType,
    Transform,
    Resize,
)

print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(0))
device = torch.device("cuda:0")


@dataclass
class TrainingConfig:
    data_dir = "/data1/dose-3d-generative/data_med/PREPARED/FOR_AUG/ct_images_prostate_only_26fixed"
    image_size = 256
    scan_depth = 26
    batch_size = 1
    num_epochs = 2000
    learning_rate = 1e-4
    lr_warmup_steps = 1000
    save_image_epochs = 200
    save_model_epochs = 200
    output_dir = "ct_256"
    seed = 0
    load_model_from_file = False


config = TrainingConfig()

win_wid = 400
win_lev = 60

transforms = Compose(
    [
        LoadImage(image_only=True),
        AddChannel(),
        Resize((config.image_size, config.image_size, config.scan_depth)),
        # ScaleIntensity(),
        ScaleIntensityRange(a_min=win_lev-(win_wid/2), a_max=win_lev+(win_wid/2), b_min=0.0, b_max=1.0, clip=True),
        # RandFlip(spatial_axis=1, prob=0.5),
        # RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
        EnsureType()
    ]
)

images_pattern = os.path.join(config.data_dir, "*.nii.gz")
images = sorted(glob.glob(images_pattern))
dataset = CacheDataset(images, transforms)

val_size = len(dataset) // 5
train_set, val_set = torch.utils.data.random_split(
    dataset, [len(dataset) - val_size, val_size])

train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=config.batch_size, num_workers=10, shuffle=True, pin_memory=torch.cuda.is_available())
val_loader = torch.utils.data.DataLoader(
    val_set, batch_size=config.batch_size, num_workers=10, shuffle=True, pin_memory=torch.cuda.is_available())

print(len(images))

# create directories and save few images from dataset as png
if not config.load_model_from_file:
    os.makedirs(config.output_dir, exist_ok=True)
    examples_dir = config.output_dir + '/examples_from_dataset'
    os.makedirs(examples_dir, exist_ok=True)
    for i in range(min(10, len(val_loader))):
        im = val_set[i]
        im = first(val_loader)
        fig = plt.figure(figsize=(15, 15))
        _ = matshow3d(
            volume=im,
            fig=fig,
            every_n=1,
            frame_dim=-1,
            cmap="gray",
        )
        fig.savefig(examples_dir + f"/{i}.png")
        plt.close(fig)

# model config
model = UNet3DModel(
    sample_size=config.image_size,
    sample_depth=config.scan_depth,
    in_channels=1,  # data are in grayscale, so always 1
    out_channels=1,  # ^
    layers_per_block=2,  # number of resnet blocks in each down_block/up_block
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

with open(config.output_dir + "/config.txt", 'w') as fp:
    fp.write(
        f"{model.block_out_channels}\n{model.down_block_types}\n{model.up_block_types}\n")

if config.load_model_from_file:
    model.load_state_dict(torch.load(config.output_dir + '/model'))


def prepare_batch(batch_data, device=None, non_blocking=False):
    t = Compose(
        [
            Lambda(lambda t: (t * 2) - 1),
        ]
    )
    return t(batch_data.permute(0, 1, 4, 2, 3).to(device=device, non_blocking=non_blocking))


sample_image = prepare_batch(val_set[0][None, :], device)

print("Input shape:", sample_image.shape)

print("Output shape:", model.to(device)(sample_image, timestep=0).sample.shape)


noise_scheduler = DDPMScheduler(num_train_timesteps=1000)


# generate data (with set seed) and save to file
@torch.no_grad()
def evaluate(config, epoch):
    eval_device = device
    model.to(eval_device)
    generator = torch.Generator(device=eval_device)
    generator.manual_seed(config.seed)
    image_shape = (config.batch_size, 1, config.scan_depth,
                   config.image_size, config.image_size)
    image = torch.randn(image_shape, generator=generator,
                        device=eval_device).to(eval_device)
    
    # t: num_train_timesteps ... 0 (1000, 999, ..., 0)
    for t in tqdm(noise_scheduler.timesteps):
        # 1. predict noise model_output
        model_output = model.to(eval_device)(image, t).sample
        # 2. compute previous image: x_t -> x_t-1 until x_0 - a generated clean image
        image = noise_scheduler.step(
            model_output, t, image, generator=generator).prev_sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 1, 3, 4, 2).reshape(config.batch_size,
                                                       config.image_size, config.image_size, config.scan_depth).numpy()
    fig = plt.figure(figsize=(15, 15))
    _ = matshow3d(
        volume=image,
        fig=fig,
        every_n=1,
        frame_dim=-1,
        cmap="gray",
    )

    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    fig.savefig(f"{test_dir}/{epoch:04d}.png")
    plt.close(fig)


optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_loader) * config.num_epochs),
)

@timing_decorator
def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    os.makedirs(config.output_dir, exist_ok=True)
    global_step = 0
    epoch_begin = 0

    count = 0
    if config.load_model_from_file:
        with open(config.output_dir + "/loss.txt", 'r') as fp:
            for count, line in enumerate(fp):
                pass
    epoch_begin = count + 1
    loss_file = open(config.output_dir + "/loss.txt",
                     "a" if config.load_model_from_file else "w")

    for epoch in range(config.num_epochs):
        epoch += epoch_begin
        train_loss = 0
        val_loss = 0
        progress_bar = tqdm(total=len(train_dataloader))
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in enumerate(train_dataloader):
            clean_images = prepare_batch(batch, device)

            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device
            ).long().to(clean_images.device) # generates a tensor of shape (1,) with random int from range [0, 1000) ?

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(
                clean_images, noise, timesteps)

            noise_pred = model(noisy_images, timesteps,
                               return_dict=False)[0].to(device)
            loss = F.mse_loss(noise_pred, noise.to(device))

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            train_loss += loss.detach().item()
            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[
                0], "step": global_step}
            progress_bar.set_postfix(**logs)
            global_step += 1

        train_loss /= len(train_dataloader)
        # calculate validation loss
        with torch.no_grad():
            for step, batch in enumerate(val_loader):
                clean_images = prepare_batch(batch, device)
                noise = torch.randn(clean_images.shape).to(clean_images.device)
                bs = clean_images.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device
                ).long().to(clean_images.device)
                noisy_images = noise_scheduler.add_noise(
                    clean_images, noise, timesteps)
                noise_pred = model(noisy_images, timesteps,
                                   return_dict=False)[0].to(device)
                val_loss += F.mse_loss(noise_pred,
                                       noise.to(device)).detach().item()
            val_loss /= len(val_loader)

        logs = {"train_loss": train_loss, "val_loss": val_loss, "lr":
                lr_scheduler.get_last_lr()[0], "step": global_step}
        progress_bar.set_postfix(**logs)

        loss_file.write(f"{train_loss} {val_loss}\n")
        loss_file.flush()

        if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
            evaluate(config, epoch + 1)

        if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
            torch.save(model.state_dict(), config.output_dir + '/model')


train_loop(config, model.to(device), noise_scheduler,
           optimizer, train_loader, lr_scheduler)


# generate final n examples
@torch.no_grad()
def generate(n, model):
    gen_dir = os.path.join(config.output_dir, "generated_examples")
    os.makedirs(gen_dir, exist_ok=True)
    for i in range(n):
        print(f"Generating: {i + 1}/{n} scan")
        eval_device = device
        generator = torch.Generator(device=eval_device)
        print(f"seed: {generator.seed()} {generator.initial_seed()}")
        image_shape = (config.batch_size, 1, config.scan_depth,
                       config.image_size, config.image_size)
        image = torch.randn(image_shape, generator=generator,
                            device=eval_device).to(eval_device)
        for t in tqdm(noise_scheduler.timesteps):
            model_output = model.to(eval_device)(image, t).sample
            image = noise_scheduler.step(
                model_output, t, image, generator=generator).prev_sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 1, 3, 4, 2).reshape(config.batch_size,
                                                           config.image_size, config.image_size, config.scan_depth).numpy()
        fig = plt.figure(figsize=(15, 15))
        _ = matshow3d(
            volume=image,
            fig=fig,
            every_n=1,
            frame_dim=-1,
            cmap="gray",
        )

        fig.savefig(f"{gen_dir}/{i+1}.png")
        plt.close(fig)


generate(10, model)
