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
# from tqdm.auto import tqdm
from tqdm.autonotebook import tqdm
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
run_version = "woundonly"
name_tag = "Model512"
samplefile_tag="Run1"
# tf.config.list_physical_devices('GPU')
pathPreTrained="None"
#####################################################################
## Calgary
pathimg = '/home/rbasiri/Dataset/GAN/train_woundonly'
folder = '/home/rbasiri/Dataset/saved_models/Diffusion/latent/StableDiffusionModel_{}_{}'.format(run_version, name_tag)
# pathPreTrained="/home/rbasiri/Dataset/saved_models/Diffusion/latent/StableDiffusionModel_256_foot"
########################
## Mehdy
# pathimg = '/home/graduate1/segmentation/Dataset/GAN/train_orig'
# folder = '/home/graduate1/segmentation/saved_models/StableDiffusionModel_{}_{}'.format(run_version, name_tag)
########################
print('Model Saved in:', folder)
# os.umask(0)
os.makedirs(folder, exist_ok = True)
os.chdir(folder)
shutil.copy(realpath, "./")

from functools import partialmethod
# tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
#####################################################################
@dataclass
class TrainingConfig:
    image_size1 = 512  #the generated image resolution
    image_size2 = 512  #the generated image resolution
    train_batch_size = 4
    eval_batch_size = 4 #how many images to sample during evaluation
    sample_batch_size = 16 #to monitor the progress
    layers_per_block=4
    num_epochs =900
    num_train_timesteps=1000 #be careful dont go above 2000. 1000 is good!
    num_inference_steps=1000
    gradient_accumulation_steps =1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 20
    save_model_epochs = 20
    mixed_precision = "no"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "./"  # the model name locally and on the HF Hub
    cache_dir = "cache"
    saved_model = "saved_model"

    push_to_hub = False  # whether to upload the saved model to the HF Hub
    hub_private_repo = True
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0
config = TrainingConfig()

test_dir = os.path.join(config.output_dir, config.cache_dir)
os.makedirs(test_dir, exist_ok=True)
test_dir = os.path.join(config.output_dir, config.saved_model)
os.makedirs(test_dir, exist_ok=True)

dataset = load_dataset(pathimg, cache_dir=config.cache_dir, split="train")

# import matplotlib.pyplot as plt
# fig, axs = plt.subplots(1, 4, figsize=(16, 4))
# for i, image in enumerate(dataset[:4]["image"]):
#     axs[i].imshow(image)
#     axs[i].set_axis_off()
# fig.show()

preprocess = transforms.Compose(
    [
        transforms.Resize((config.image_size1, config.image_size2)),
        # transforms.Resize((config.image_size, config.image_size)),
        transforms.RandomHorizontalFlip(p=0.4),
        #transforms.RandomGrayscale(p=0.1)
        #transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter(brightness=(1, 3), contrast=(1, 3), saturation=(1, 2), hue=0),]), p=0.2)
        # transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter(),]), p=0.3)
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

def transform(examples):
    images = [preprocess(image.convert("RGB")) for image in examples["image"]]
    return {"images": images}

dataset.set_transform(transform)
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)

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
# print(model)

sample_image = dataset[0]["images"].unsqueeze(0)
print("Input shape:", sample_image.shape)
print("Output shape:", model(sample_image, timestep=0).sample.shape)

# noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
noise = torch.randn(sample_image.shape)
timesteps = torch.LongTensor([50])
noisy_image = noise_scheduler.add_noise(sample_image, noise, timesteps)

noise_pred = model(noisy_image, timesteps).sample
loss = F.mse_loss(noise_pred, noise)

optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)

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
        num_inference_steps=config.num_inference_steps,
    ).images

    # Make a grid out of the images
    image_grid = make_grid(images, rows=4, cols=4)

    # Save the images
    test_dir = os.path.join(config.output_dir, 'samples_{}_{}'.format(run_version, samplefile_tag))
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")

def get_full_repo_name(model_id: str, organization: str = None, token: str = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"
# model = DDPMPipeline.from_pretrained(os.path.join(config.output_dir, config.saved_model)).to("cuda")

def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    if accelerator.is_main_process:
        if config.push_to_hub:
            repo_name = get_full_repo_name(Path(config.output_dir).name)
            repo = Repository(config.output_dir, clone_from=repo_name)
        elif config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("train_example")

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0
    # progress_bar = tqdm(total=config.num_epochs, disable=not accelerator.is_local_main_process)

    # Now you train the model
    for epoch in range(config.num_epochs):
        # progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        # progress_bar = tqdm(total=config.num_epochs, disable=not accelerator.is_local_main_process)
        # progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch["images"]
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.num_train_timesteps, (bs,), device=clean_images.device
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                if step % config.gradient_accumulation_steps == 0:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

            # if global_step % 50 == 0 or len(train_dataloader) == global_step: 
            # if len(train_dataloader) == global_step: 
                logs2 = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
                accelerator.log(logs2, step=global_step) 
            #     progress_bar.update(global_step)
            #     progress_bar.set_postfix(**logs)
            global_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(config, epoch, pipeline)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                if config.push_to_hub:
                    repo.push_to_hub(commit_message=f"Epoch {epoch}", blocking=True)
                else:
                    pipeline.save_pretrained(os.path.join(config.output_dir, config.saved_model))

        logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
        print("Epoch:", epoch)
        for label, value in logs.items():
            print(f'{label} {value:.4e}', end=' ')
        # progress_bar.update(epoch)
        # progress_bar.set_postfix(**logs) 

# def main():
#     args = (config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)
#     notebook_launcher(train_loop, args, num_processes=2)
# if __name__ == "__main__":
#     main()

def main():
    train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)
if __name__ == "__main__":
    main()
    
print("Time Duration", f'{timeit.default_timer()-starttime:.5f}')

# import torch.multiprocessing as mp
# torch.multiprocessing.spawn(train_loop, args=(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler,), nprocs=2, join=True, daemon=False, start_method='spawn')
# cxx = mp.get_context("spawn")
# if __name__ == '__main__':
#     # mp.set_start_method('spawn')
#     args = (config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)
#     notebook_launcher(train_loop, args, num_processes=2)
# if __name__ == '__main__':
#     num_processes = 2
#     cxx = mp.get_context("spawn")
#     # model.share_memory()
#     # NOTE: this is required for the ``fork`` method to work

#     processes = []
#     for rank in range(num_processes):
#         p = cxx.Process(target=train_loop, args=(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler,))
#         p.start()
#         processes.append(p)
#     for p in processes:
#         p.join()
