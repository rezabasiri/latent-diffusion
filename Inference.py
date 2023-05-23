import timeit
starttime = timeit.default_timer()

from dataclasses import dataclass
import torch
import math
import os
import shutil
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from tqdm.auto import tqdm
from pathlib import Path
from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline, AutoencoderKL, VQModel
from datasets import load_dataset
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator, notebook_launcher
from huggingface_hub import HfFolder, Repository, whoami
#####################################################################
#identifier
scriptversion = os.path.basename(__file__)
realpath = os.path.realpath(__file__)
run_version = "foot"
name_tag = "inference"
# tf.config.list_physical_devices('GPU')
#####################################################################
## Calgary
folder = '/home/rbasiri/Dataset/saved_models/Diffusion/latent/StableDiffusionModel_{}_{}'.format(run_version, name_tag)
pathPreTrained="/home/rbasiri/Dataset/saved_models/Diffusion/latent/StableDiffusionModel_256_foot"
########################
print('Model Saved in:', folder)
# os.umask(0)
os.makedirs(folder, exist_ok = True)
os.chdir(folder)
shutil.copy(realpath, "./")
#####################################################################
@dataclass
class TrainingConfig:
    image_size1 = 32  #the generated image resolution
    image_size2 = 32  #the generated image resolution
    sample_batch_size = 1 #to monitor the progress
    layers_per_block=2
    num_epochs =2
    num_train_timesteps=1000 #be careful dont go above 2000. 1000 is good!
    gradient_accumulation_steps =1
    learning_rate = 1e-3
    mixed_precision = "no"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "./"  # the model name locally
    ID = "test"
    saved_model = "saved_model"

    seed = 0
config = TrainingConfig()
#####################################################################
if os.path.isdir(os.path.join(pathPreTrained, config.saved_model,"scheduler")):
    print("Loading from a different Saved Model (Transfer)")
    noise_scheduler = DDPMScheduler.from_pretrained(os.path.join(pathPreTrained, config.saved_model,"scheduler"))
    model = UNet2DModel.from_pretrained(os.path.join(pathPreTrained, config.saved_model,"unet"))
    
elif os.path.isdir(os.path.join(config.output_dir, config.saved_model,"scheduler")):
    print("Loading from Same Saved Model (cont.)")
    noise_scheduler = DDPMScheduler.from_pretrained(os.path.join(config.output_dir, config.saved_model,"scheduler"))
    model = UNet2DModel.from_pretrained(os.path.join(config.output_dir, config.saved_model,"unet"))
else:
    model = UNet2DModel(
    sample_size=(config.image_size1, config.image_size2),  # the target image resolution
    # sample_size=config.image_size,  # the target image resolution
    in_channels=3,  # the number of input channels, 3 for RGB images
    out_channels=3,  # the number of output channels
    layers_per_block=config.layers_per_block,  # how many ResNet layers to use per UNet block
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
    noise_scheduler = DDPMScheduler(config.num_train_timesteps) 
    print("Running model from scratch")
#####################################################################
def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid

def evaluate(config, epoch, pipeline):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    images = pipeline(
        batch_size=config.sample_batch_size,
        generator=torch.manual_seed(config.seed),
    ).images

    # Make a grid out of the images
    image_grid = make_grid(images, rows=4, cols=4)

    # Save the images
    test_dir = os.path.join(config.output_dir, config.ID)
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")
#####################################################################
accelerator = Accelerator(
    mixed_precision=config.mixed_precision,
    gradient_accumulation_steps=config.gradient_accumulation_steps,
)
if accelerator.is_main_process:
    def train_loop(config, model, noise_scheduler):
        pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)
        for epoch in range(config.num_epochs):
            evaluate(config, epoch, pipeline)

def main():
    train_loop(config, model, noise_scheduler)
    
if __name__ == "__main__":
    main()
    
print("Time Duration", f'{timeit.default_timer()-starttime:.5f}')