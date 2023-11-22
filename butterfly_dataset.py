import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import io
from torchvision import transforms
from measurement_system import DownsampleModel
import matplotlib.pyplot as plt

class ButterflyDataset(Dataset):
    def __init__(self, parquet_file, transform=None):
        self.parquet_file = parquet_file
        self.dataframe = pd.read_parquet(parquet_file, columns=['image'])
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image_row = self.dataframe.iloc[idx]
        image = Image.open(io.BytesIO(image_row['image']))

        if self.transform:
            image = self.transform(image)

        return image

class ButterflyDSDataset(ButterflyDataset):
    def __init__(self, parquet_file, transform, im_size, donwsample_scale=8):
        super().__init__(parquet_file, transform=None)
        self.apply_A = DownsampleModel(donwsample_scale).apply_A
        self.transform = transform
        self.im_size = im_size
        self.donwsample_scale = donwsample_scale

    def __getitem__(self, idx):
        image_row = self.dataframe.iloc[idx]
        x = Image.open(io.BytesIO(image_row['image']['bytes']))
        x = transforms.Resize((self.im_size, self.im_size))(x)
        x = torch.Tensor(np.array(x))
        x = x.permute(2, 0, 1)

        y = self.apply_A(x)

        # add noise to x and plot x and save without axis
        # x = x + torch.randn_like(x)*100
        # x = x.permute(1, 2, 0)
        # x = x.detach().cpu().numpy()
        # plt.imshow(x/255.0)
        # plt.axis('off')
        # plt.savefig('x.png', bbox_inches='tight', pad_inches=0)

        # plot y and save without axis
        # y = y.permute(1, 2, 0)
        # y = y.detach().cpu().numpy()
        # plt.imshow(y/255.0)
        # plt.axis('off')
        # plt.savefig('y.png', bbox_inches='tight', pad_inches=0)


        y = y.repeat_interleave(self.donwsample_scale, dim=1)
        y = y.repeat_interleave(self.donwsample_scale, dim=2)

        x = x/255
        y = y/(255)

        x = self.transform(x)
        y = self.transform(y)

        return x, y

# main function for testing
if __name__ == '__main__':

    transform = transforms.Compose([
        transforms.Normalize(3*[0.5], 3*[0.5]),
    ])

    dataset = ButterflyDSDataset('train-00000-of-00001.parquet', transform=transform, im_size=128)
    x1, y1 = dataset[0]  # Returns a list of four images
