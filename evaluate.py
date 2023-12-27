from dataclasses import dataclass
from torchvision import transforms
import torch
from diffusers import DDPMScheduler, DDPMPipeline
import sys
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from butterfly_dataset import ButterflyDSDataset
import argparse
import torch.nn.functional as F
import os

def Upsample(y, scale):
    y = y.repeat_interleave(scale, dim=2)
    y = y.repeat_interleave(scale, dim=3)

    return y

A = torch.nn.AvgPool2d(kernel_size=8, stride=8)
Ap = lambda z: Upsample(z, 8)

def ddnm(x0t, y):
    # Eq 19
    x0t = x0t + Ap(y - A(x0t))

    return x0t

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

def gen_images_ddnm(model, scheduler, y, y_):
    sample_size = model.config.sample_size
    noise = torch.randn((y.shape[0], 3, sample_size, sample_size)).to("cuda")
    input = noise

    for t in scheduler.timesteps:
        with torch.no_grad():
            model_input = torch.cat([input, y], dim=1)
            noisy_residual = model(model_input, t).sample
        previous_noisy_sample = scheduler.step(noisy_residual, t, input).prev_sample
        previous_noisy_sample = ddnm(previous_noisy_sample, y_)
        input = previous_noisy_sample

    # image = (input / 2 + 0.5).clamp(0, 1)
    # image = image.cpu().permute(0, 2, 3, 1).numpy()
    # image = numpy_to_pil(image)
    # image.show()

    return input

# def evaluate_ddnm(model, scheduler, test_loader):
#     model.eval()
#
#     loader_iter = iter(test_loader)
#     # loop for first 4 images
#     for i in range(len(test_loader)):
#
#         print(i, len(test_loader))
#         batch = next(loader_iter)
#         x, y = batch
#         x = x.to("cuda")
#         y = y.to("cuda")
#         y_ = A(x).to("cuda")
#         x_hat = gen_images_ddnm(model, scheduler, y, y_)
#
#         image = (x_hat / 2 + 0.5).clamp(0, 1)
#         image = image.detach().cpu().permute(0, 2, 3, 1)
#
#         gt_image = (x / 2 + 0.5).clamp(0, 1)
#         gt_image = gt_image.detach().cpu().permute(0, 2, 3, 1)
#
#         log_images(gt_image, image, fname='results/ddnm/batch'+str(i)+'.png')

def log_images(gt_images, pd_images, fname=None):

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
    plt.savefig(fname)

# def evaluate(model, scheduler, test_loader):
#     model.eval()
#
#     # loop for first 4 images
#     loader_iter = iter(test_loader)
#     for i in range(len(test_loader)):
#         print(i, len(test_loader))
#         batch = next(loader_iter)
#         x, y = batch
#         x = x.to("cuda")
#         y = y.to("cuda")
#         x_hat = gen_images(model, scheduler, y)
#
#         image = (x_hat / 2 + 0.5).clamp(0, 1)
#         image = image.detach().cpu().permute(0, 2, 3, 1)
#
#         gt_image = (x / 2 + 0.5).clamp(0, 1)
#         gt_image = gt_image.detach().cpu().permute(0, 2, 3, 1)
#
#         log_images(gt_image, image, fname='results/default/batch'+str(i)+'.png')

def evaluate(model, scheduler, test_loader, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    # loop for first 4 images
    guidance_loss = []
    loader_iter = iter(test_loader)
    for i in range(len(test_loader)):
        print(i, len(test_loader))
        batch = next(loader_iter)
        x, y = batch
        x = x.to("cuda")
        y = y.to("cuda")
        x_hat = gen_images(model, scheduler, y)

        temp = (x_hat / 2 + 0.5).clamp(0, 1) * 255.0
        # temp = x
        # temp = (temp / 2 + 0.5).clamp(0, 1) * 255.0
        y_ = A(temp)
        y_ = Ap(y_)
        y_ = test_loader.dataset.transform(y_ / 255.0)
        ll = F.mse_loss(y_, y)
        guidance_loss.append(ll.item())
        print('Guidance loss: ', sum(guidance_loss)/len(guidance_loss))

        image = (x_hat / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1)

        gt_image = (x / 2 + 0.5).clamp(0, 1)
        gt_image = gt_image.detach().cpu().permute(0, 2, 3, 1)

        log_images(gt_image, image, fname=os.path.join(output_dir, 'batch'+str(i)+'.png'))

def evaluate_ddnm(model, scheduler, test_loader, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    # loop for first 4 images
    guidance_loss = []
    loader_iter = iter(test_loader)
    for i in range(len(test_loader)):
        print(i, len(test_loader))
        batch = next(loader_iter)
        x, y = batch
        x = x.to("cuda")
        y = y.to("cuda")
        y_ = A(x).to("cuda")
        x_hat = gen_images_ddnm(model, scheduler, y, y_)

        temp = (x_hat / 2 + 0.5).clamp(0, 1) * 255.0
        # temp = x
        # temp = (temp / 2 + 0.5).clamp(0, 1) * 255.0
        y_ = A(temp)
        y_ = Ap(y_)
        y_ = test_loader.dataset.transform(y_ / 255.0)
        ll = F.mse_loss(y_, y)
        guidance_loss.append(ll.item())
        print('Guidance loss: ', sum(guidance_loss)/len(guidance_loss))

        image = (x_hat / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1)

        gt_image = (x / 2 + 0.5).clamp(0, 1)
        gt_image = gt_image.detach().cpu().permute(0, 2, 3, 1)

        log_images(gt_image, image, fname=os.path.join(output_dir, 'batch'+str(i)+'.png'))

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

def main(args):
    config = TrainingConfig()

    transform = transforms.Normalize(3 * [0.5], 3 * [0.5])

    # train_data = ButterflyDSDataset('train-00000-of-00001.parquet', transform=transform, im_size=config.image_size,
    #                                 donwsample_scale=8)
    # train_loader = DataLoader(train_data, batch_size=config.train_batch_size, shuffle=True, num_workers=num_workers)

    test_data = ButterflyDSDataset('train-00000-of-00001.parquet', transform=transform, im_size=config.image_size,
                                   donwsample_scale=8)
    test_loader = DataLoader(test_data, batch_size=config.eval_batch_size, shuffle=False, num_workers=num_workers)

    pipeline = DDPMPipeline.from_pretrained(args.ckpt)
    pipeline.to('cuda')

    model = pipeline.unet
    noise_scheduler = pipeline.scheduler

    if args.method == 'default':
        evaluate(model, noise_scheduler, test_loader, args.output_dir)
    elif args.method == 'ddnm':
        evaluate_ddnm(model, noise_scheduler, test_loader, args.output_dir)

# parse arguments
parser = argparse.ArgumentParser(description='Conditional Image Super-Resolution')
parser.add_argument('--method', type=str, default='default', choices=['default', 'ddnm'], help='Method to evaluate [default, ddnm')
parser.add_argument('--ckpt', type=str, default='wandb/run-20231119_154332-m1djegwy/files', help='Path to the pretrained model (trained without knowledge)')
# parser.add_argument('--ckpt', type=str, default='wandb/run-20231226_152903-uqxjycl8/files', help='Path to the pretrained model (trained with knowledge)')
parser.add_argument('--output_dir', type=str, default='results/test', help='Path to save the results')

args = parser.parse_args()

if __name__ == "__main__":
    main(args)

