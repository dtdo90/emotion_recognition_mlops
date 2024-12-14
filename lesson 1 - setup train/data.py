import torch
import numpy as np
import csv
from PIL import Image
import pytorch_lightning as pl

from torchvision import transforms
from torch.utils.data import DataLoader



class FERDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=128, cut_size=44, device="mps"):
        super().__init__()
        self.batch_size = batch_size
        self.cut_size = cut_size
        self.device = device

        # Transformations
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(cut_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

        self.transform_test = transforms.Compose([
            transforms.TenCrop(cut_size),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))
        ])

    def prepare_data(self):
        """ Load dataset, put into numpy arrays and reshape to 48*48"""
        file = "fer2013.csv"

        self.train_x, self.train_y = [], []
        self.public_test_x, self.public_test_y = [], []
        self.private_test_x, self.private_test_y = [], []

        with open(file, 'r') as csvin:
            data = csv.reader(csvin)
            for row in data:
                if row[-1] == "Training":
                    temp = [int(pixel) for pixel in row[-2].split()]
                    self.train_x.append(temp)
                    self.train_y.append(int(row[0]))

                if row[-1] == "PublicTest":
                    temp = [int(pixel) for pixel in row[-2].split()]
                    self.public_test_x.append(temp)
                    self.public_test_y.append(int(row[0]))

                if row[-1] == "PrivateTest":
                    temp = [int(pixel) for pixel in row[-2].split()]
                    self.private_test_x.append(temp)
                    self.private_test_y.append(int(row[0]))

        # convert data into numpy arrays
        self.train_x, self.train_y = np.asarray(self.train_x), np.asarray(self.train_y)
        self.public_test_x, self.public_test_y = np.asarray(self.public_test_x), np.asarray(self.public_test_y)
        self.private_test_x, self.private_test_y = np.asarray(self.private_test_x), np.asarray(self.private_test_y)

        # reshape into 48*48
        self.train_x = self.train_x.reshape(-1, 48, 48)
        self.public_test_x = self.public_test_x.reshape(-1, 48, 48)
        self.private_test_x = self.private_test_x.reshape(-1, 48, 48)


    def setup(self, stage=None):
        # Initialize datasets based on the stage (train/val/test)
        if stage == 'fit' or stage is None:
            self.train_data = self._get_dataset('train')
            self.val_data = self._get_dataset('public')

        if stage == 'test' or stage is None:
            self.test_data = self._get_dataset('private')

    def _get_dataset(self, split):
        # load data
        self.prepare_data() 
        # Handle different splits
        if split == 'train':
            data_x, data_y = self.train_x, self.train_y
        elif split == 'public':
            data_x, data_y = self.public_test_x, self.public_test_y
        elif split == 'private':
            data_x, data_y = self.private_test_x, self.private_test_y

        return [(torch.tensor(x, dtype=torch.float32).unsqueeze(0), y)
                for x, y in zip(data_x, data_y)]

    def _transform_dataset(self, data, transform):
        # Apply transforms to the dataset (used in dataloader)
        for idx, (img, label) in enumerate(data):
            img = Image.fromarray(img.squeeze(0).numpy().astype(np.uint8))
            data[idx] = (transform(img).to(self.device), torch.tensor(label, device=self.device))
        return data

    def train_dataloader(self):
        transformed_train_data = self._transform_dataset(self.train_data, self.transform_train)
        return DataLoader(transformed_train_data, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        transformed_val_data = self._transform_dataset(self.val_data, self.transform_test)
        return DataLoader(transformed_val_data, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        transformed_test_data = self._transform_dataset(self.test_data, self.transform_test)
        return DataLoader(transformed_test_data, batch_size=self.batch_size, shuffle=False)


if __name__ == "__main__":
    datamodule = FERDataModule()
    datamodule.setup('fit')

    train_loader = datamodule.train_dataloader()
    batch, label = next(iter(train_loader))
    print(f"Train batch: {batch.shape} | Label: {label.shape}")
    print(f"Device: {batch[0].device}")
