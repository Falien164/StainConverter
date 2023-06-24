import torch
import torchvision.transforms as trans
from typing import Tuple

from torch_library.stain_dataset import StainDataset
from torch.utils.data import DataLoader as torch_data_loader

import matplotlib.pyplot as plt
import os


class DataLoader:
    def __init__(self, train_he_images_path, train_pas_images_path, test_he_images_path, test_pas_images_path,
                 img_height=256, img_width=256, buffer_size=400, batch_size=1, seed=1234):
        self.train_he_images_path = train_he_images_path
        self.train_pas_images_path = train_pas_images_path
        self.test_he_images_path = test_he_images_path
        self.test_pas_images_path = test_pas_images_path
        self.img_height = img_height
        self.img_width = img_width
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.seed = seed

        torch.manual_seed(self.seed)

        self.transforms = [trans.ToTensor(),
                           trans.Resize([self.img_width, self.img_height]),
                           trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]

    def load_dataset(self) -> Tuple:
        g = torch.Generator()
        g.manual_seed(self.seed)

        train_dataset = StainDataset(self.train_he_images_path, self.train_pas_images_path, transforms=self.transforms)
        train_data = torch_data_loader(train_dataset, batch_size=self.batch_size, shuffle=True, generator=g)

        test_dataset = StainDataset(self.test_he_images_path, self.test_pas_images_path, transforms=self.transforms)
        test_data = torch_data_loader(test_dataset, batch_size=self.batch_size, shuffle=False, generator=g)

        return train_data, test_data

    @staticmethod
    def display_sample_pair(dataset, path: str) -> None:
        images = next(iter(dataset))
        plt.figure()
        for idx, (stain, img) in enumerate(images.items()):
            plt.subplot(1, 2, idx + 1)
            plt.title(f"Domain {stain}")
            plt.imshow(img[0].permute(1, 2, 0) * 0.5 + 0.5)
            plt.axis('off')
        path = os.path.join(path, 'sample.png')
        plt.savefig(path)
