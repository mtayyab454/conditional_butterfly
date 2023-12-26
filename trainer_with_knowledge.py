import torch
import numpy as np
import os
import torch.nn.functional as F
from PIL import Image
import wandb

import matplotlib.pyplot as plt
from accelerate import Accelerator
from tqdm.auto import tqdm
from diffusers import DDPMPipeline
from diffusers.utils import make_image_grid, numpy_to_pil

def Upsample(y, scale):
    y = y.repeat_interleave(scale, dim=2)
    y = y.repeat_interleave(scale, dim=3)

    return y

A = torch.nn.AvgPool2d(kernel_size=8, stride=8)
A = A.cuda()
Ap = lambda z: Upsample(z, 8)

def get_pred_original(noise_scheduler, timesteps, noise_pred, noisy_images):

    pred_original_samples = []
    for i, t in enumerate(timesteps):
        alpha_prod_t = noise_scheduler.alphas_cumprod[t]
        beta_prod_t = 1 - alpha_prod_t
        pd = (noisy_images[i] - beta_prod_t ** (0.5) * noise_pred[i]) / alpha_prod_t ** (0.5)
        pred_original_samples.append(pd)

    pred_original_samples = torch.stack(pred_original_samples, dim=0)

    return pred_original_samples
def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, test_loader, lr_scheduler):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="wandb",
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    if accelerator.is_main_process:
        accelerator.init_trackers("Conditional_Butterfly")

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, test_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, test_loader, lr_scheduler
    )

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters: %.2fM" % (num_params / 1e6))

    global_step = 0

    # evaluate(accelerator, global_step, model, noise_scheduler, test_loader)

    loss_diffusion = torch.tensor(0.0).to('cuda')
    loss_guidance = torch.tensor(0.0).to('cuda')

    # Now you train the model
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process, position=0, leave=True)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_x = batch[0]
            clean_y = batch[1]
            # Sample noise to add to the images
            noise = torch.randn(clean_x.shape).to(clean_x.device)
            bs = clean_x.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_x.device
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_x, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                # model_input = torch.cat([noisy_images, clean_y], dim=1)
                # noise_pred = model(model_input, timesteps, return_dict=False)[0]
                # loss = F.mse_loss(noise_pred, noise)

                if torch.rand(1) < 0.5:
                    model_input = torch.cat([noisy_images, clean_y], dim=1)
                    noise_pred = model(model_input, timesteps, return_dict=False)[0]
                    loss_diffusion = F.mse_loss(noise_pred, noise)
                    loss = loss_diffusion
                else:
                    model_input = torch.cat([noisy_images, clean_y], dim=1)
                    noise_pred = model(model_input, timesteps, return_dict=False)[0]
                    pred_original_sample = get_pred_original(noise_scheduler, timesteps, noise_pred, noisy_images)
                    y_ = A(pred_original_sample)
                    y_ = Ap(y_)
                    loss_guidance = F.mse_loss(y_, clean_y)
                    loss = loss_guidance

                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step, "diffusion_loss": loss_diffusion.item(), "guidance_loss": loss_guidance.item()}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

            # evaluate(accelerator, global_step, model, noise_scheduler, test_loader)

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(accelerator, global_step, model, noise_scheduler, test_loader)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                pipeline.save_pretrained(wandb.run.dir)

def gen_images(model, scheduler, y):
    sample_size = model.config.sample_size
    noise = torch.randn((y.shape[0], 3, sample_size, sample_size)).to("cuda")
    input = noise

    for t in scheduler.timesteps:
        with torch.no_grad():
            model_input = torch.cat([input, y], dim=1)
            noisy_residual = model(model_input, t).sample
        previous_noisy_sample = scheduler.step(noisy_residual, t, input).prev_sample
        input = previous_noisy_sample

    # image = (input / 2 + 0.5).clamp(0, 1)
    # image = image.cpu().permute(0, 2, 3, 1).numpy()
    # image = numpy_to_pil(image)
    # image.show()

    return input

def log_images(accelerator, global_step, gt_images, pd_images, caption=None):

    gt_images = gt_images.chunk(4, dim=0)
    pd_images = pd_images.chunk(4, dim=0)

    fig, axs = plt.subplots(2, 4)

    for i in range(4):
        axs[0, i].imshow(gt_images[i].squeeze(0))
        axs[0, i].set_title('Traget')

        axs[1, i].imshow(pd_images[i].squeeze(0))
        axs[1, i].set_title('Prediction')

    # Remove axis for better visualization
    for ax in axs.flat:
        ax.axis('off')

    accelerator.log({caption+'': wandb.Image(plt)}, step=global_step)
    plt.close()

def evaluate(accelerator, global_step, model, scheduler, test_loader):
    model.eval()

    # loop for first 4 images
    for i in range(1):
        batch = next(iter(test_loader))
        x, y = batch
        x = x.to("cuda")
        y = y.to("cuda")
        x_hat = gen_images(model, scheduler, y)

    image = (x_hat / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1)

    gt_image = (x / 2 + 0.5).clamp(0, 1)
    gt_image = gt_image.detach().cpu().permute(0, 2, 3, 1)

    log_images(accelerator, global_step, gt_image, image, caption='Image')