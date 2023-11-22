from dataclasses import dataclass
from torchvision import transforms
from diffusers import UNet2DModel
import torch
from PIL import Image
from diffusers import DDPMScheduler
import sys
from accelerate import notebook_launcher
import glob
from torch.utils.data import DataLoader
from diffusers.optimization import get_cosine_schedule_with_warmup
import matplotlib.pyplot as plt
from trainer import train_loop
from butterfly_dataset import ButterflyDSDataset

gettrace = getattr(sys, 'gettrace', None)
if gettrace is None:
    print('No sys.gettrace')
    num_workers = 0
elif gettrace(): # If debugger is attached
    num_workers = 0
else:
    num_workers = 4

@dataclass
class TrainingConfig:
    image_size = 128  # the generated image resolution
    train_batch_size = 16
    eval_batch_size = 4  # how many images to sample during evaluation
    num_epochs = 50
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 1
    save_model_epochs = 30
    mixed_precision = "no"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "results"  # the model name locally and on the HF Hub

    push_to_hub = False  # whether to upload the saved model to the HF Hub
    hub_private_repo = False
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0


config = TrainingConfig()

transform = transforms.Normalize(3 * [0.5], 3 * [0.5])

train_data = ButterflyDSDataset('train-00000-of-00001.parquet', transform=transform, im_size=config.image_size, donwsample_scale=8)
train_loader = DataLoader(train_data, batch_size=config.train_batch_size, shuffle=True, num_workers=num_workers)

test_data = ButterflyDSDataset('train-00000-of-00001.parquet', transform=transform, im_size=config.image_size, donwsample_scale=8)
test_loader = DataLoader(test_data, batch_size=config.eval_batch_size, shuffle=False, num_workers=num_workers)

model = UNet2DModel(
    sample_size=config.image_size,  # the target image resolution
    in_channels=6,  # the number of input channels, 3 for RGB images
    out_channels=3,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    norm_num_groups=32, # the number of groups for GroupNorm
    block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
    down_block_types=(
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",  # a regular ResNet upsampling block
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
)

noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
# noise_scheduler.num_inference_steps = 5

optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_loader) * config.num_epochs),
)

args = (config, model, noise_scheduler, optimizer, train_loader, test_loader, lr_scheduler)

notebook_launcher(train_loop, args, num_processes=1)