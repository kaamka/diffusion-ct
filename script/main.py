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
import argparse

# silence warning
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

@dataclass
class TrainingConfig:
    data_dir: str
    image_size: int
    scan_depth: int
    batch_size: int
    num_epochs: int
    learning_rate: float
    lr_warmup_steps : int
    save_image_epochs : int
    save_model_epochs : int
    output_dir: str
    seed: int
    load_model_from_file: bool
    checkpoint_path: str
    logging_dir = str


# @dataclass
# class TrainingConfig:
#     data_dir: str
#     # data_dir = "/sekhemet/scratch/kamkal/Augm/data_v2_prostate_32slices_34_plus_100/"
#     # image_size = 256
#     image_size: int
#     scan_depth = 32
#     batch_size = 1
#     num_epochs = 4000
#     learning_rate = 1e-4
#     lr_warmup_steps = 1000
#     save_image_epochs = 100
#     save_model_epochs = 500
#     output_dir = "ct_256"
#     seed = 0
#     load_model_from_file = False
#     checkpoint_path = ""
#     logging_dir = f"{output_dir}/logs"



# evaluate model and save results
@torch.no_grad()
def evaluate(model, config, epoch, noise_scheduler, device):
    generator = torch.Generator(device=device)
    generator.manual_seed(config.seed)
    image_shape = (config.batch_size, 1, config.scan_depth,
                config.image_size, config.image_size)
    image = torch.randn(image_shape, generator=generator, device=device).to(device)
    # t: num_train_timesteps ... 0 (1000, 999, ..., 0)
    for t in tqdm(noise_scheduler.timesteps):
        # 1. predict noise model_output
        model_output = model(image, t).sample
        # 2. compute previous image: x_t -> x_t-1 until x_0 - a generated clean image
        image = noise_scheduler.step(model_output, t, image, generator=generator).prev_sample
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
    test_dir = os.path.join(config.output_dir, 'samples')
    os.makedirs(test_dir, exist_ok=True)
    fig.savefig(f"{test_dir}/{epoch:04d}.png")
    plt.close(fig)

# generate final n examples
@torch.no_grad()
def generate(n, model, config, noise_scheduler, device):
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

def parse_args():
    parser = argparse.ArgumentParser(description="Training script configuration")

    parser.add_argument("--data_dir", type=str, required=True, help="Path to data directory")    
    parser.add_argument("--output_dir", type=str, help="Directory to save output files")

    parser.add_argument("--image_size", type=int, default=256, help="Size of the input image")
    parser.add_argument("--scan_depth", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=4000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_warmup_steps", type=int, default=1000)
    parser.add_argument("--save_image_epochs", type=int, default=100)
    parser.add_argument("--save_model_epochs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--load_model_from_file", action="store_true", help="Load model from checkpoint")
    parser.add_argument("--checkpoint_path", type=str, default="", help="Path to model checkpoint")
    parser.add_argument("--logging_dir", type=str, default=None, help="Where to save logs (default: output_dir/logs)")
    return parser.parse_args()
    


def main():
    args = parse_args()
    print(args.data_dir)

    # Set logging_dir if not provided
    #logging_dir = args.logging_dir or f"{args.output_dir}/logs"

    config = TrainingConfig(
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
        load_model_from_file=args.load_model_from_file,
        checkpoint_path=args.checkpoint_path,
        #logging_dir=args.logging_dir
        #logging_dir=f"{args.output_dir}/logs" if args.logging_dir is None else args.logging_dir
    )
    os.makedirs(config.output_dir + f'/models', exist_ok=True)
    logging_dir=f"{args.output_dir}/logs" if args.logging_dir is None else args.logging_dir
    logger = get_logger(__name__, log_level="INFO")
    #config = TrainingConfig()
    accelerator_project_config = ProjectConfiguration(
        project_dir=config.output_dir,
        logging_dir=logging_dir
    )

    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))  # a big number for high resolution or big dataset
    accelerator = Accelerator(
        log_with="tensorboard",
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs]
    )

    if not is_tensorboard_available():
        raise ImportError("tensorboard not found")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        diffusers.utils.logging.set_verbosity_info()
    else:
        diffusers.utils.logging.set_verbosity_error()

    # initialize model
    model = UNet3DModel(
        sample_size=config.image_size,
        sample_depth=config.scan_depth,
        in_channels=1,  # data are in grayscale, so always 1
        out_channels=1,  # data are in grayscale, so always 1
        layers_per_block=2,  # number of resnet blocks in each down_block/up_block
        block_out_channels=(32, 64, 64, 128, 256, 512, 512),
        #block_out_channels=(32, 64, 128, 256),
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
        fp.write(f"{model.block_out_channels}\n{model.down_block_types}\n{model.up_block_types}\n")

    if config.load_model_from_file:
        model.load_state_dict(torch.load(config.output_dir + '/model'))

    # initialize noise scheduler
    noise_scheduler = DDPMScheduler(num_train_timesteps=1500)

    # initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    # prepare dataset
    images_pattern = os.path.join(config.data_dir, '*.nii.gz')
    images = sorted(glob.glob(images_pattern))


    win_wid = 400
    win_lev = 60

    transforms = Compose(
        [
            LoadImage(image_only=True),
            EnsureChannelFirst(),
            Resize((config.image_size, config.image_size, config.scan_depth)),
            ScaleIntensityRange(a_min=win_lev-(win_wid/2), a_max=win_lev+(win_wid/2), b_min=0.0, b_max=1.0, clip=True),
            EnsureType()
        ]
    )

    dataset = CacheDataset(images, transforms)
    val_size = len(dataset) // 5
    print(val_size)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [len(dataset) - val_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, num_workers=10, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.batch_size, num_workers=10, shuffle=True)

    def prepare_batch(batch_data, device=None, non_blocking=False):
        t = Compose(
            [
                Lambda(lambda t: (t * 2) - 1),
            ]
        )
        return t(batch_data.permute(0, 1, 4, 2, 3).to(device=device, non_blocking=non_blocking))
    
    logger.info(f"Dataset size: {len(dataset)}")

    # initialize learning rate scheduler
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_loader) * config.num_epochs),
    )

    # prepare everyting with accelerator
    model, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, lr_scheduler
    )

    # initialize trackers on main process
    if accelerator.is_main_process:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run)

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num Epochs = {config.num_epochs}")

    global_step = 0
    first_epoch = 0

    # train the model
    for epoch in range(first_epoch, config.num_epochs):
        model.train()
        
        progress_bar = tqdm(total=len(train_loader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        train_loss = 0
        for step, batch in enumerate(train_loader):
            clean_images = prepare_batch(batch)

            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device
            ).long() # generates a tensor of shape (1,) with random int from range [0, 1000)
    
            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            
            # predict the noise residual
            model_output = model(noisy_images, timesteps).sample
            loss = F.mse_loss(model_output.float(), noise.float())
            train_loss += loss.detach().item()

            # os.system("nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,utilization.memory --format=csv")
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
                
            # check if the accelerator has performed an optimization step
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

        # calculate validation loss
        val_loss = 0
        with torch.no_grad():
            for step, batch in enumerate(val_loader):
                clean_images = prepare_batch(batch)
                noise = torch.randn(clean_images.shape).to(clean_images.device)
                bs = clean_images.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device).long()
                noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                val_loss += F.mse_loss(noise_pred, noise).detach().item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        logs = {"train_loss": train_loss, "val_loss": val_loss, "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
        progress_bar.set_postfix(**logs)
        accelerator.log({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)

        progress_bar.close()

        if accelerator.is_main_process:
            if epoch == config.num_epochs - 1 or (epoch + 1) % config.save_image_epochs == 0:
                logger.info(f"Model evaluation: ")
                evaluate(model, config, epoch, noise_scheduler, accelerator.device)

            if epoch == config.num_epochs - 1 or (epoch + 1) % config.save_model_epochs == 0:
                torch.save(model.state_dict(), config.output_dir + f'/models/model_{epoch}')

        accelerator.wait_for_everyone()

    if accelerator.is_main_process:       
        generate(10, model, config, noise_scheduler, accelerator.device)

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
