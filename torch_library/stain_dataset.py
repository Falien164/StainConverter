from torch.utils.data import Dataset
from PIL import Image

import torchvision.transforms as t
import glob


class StainDataset(Dataset):
    def __init__(self, he_images_path, pas_images_path, transforms=None):
        self.transform = t.Compose(transforms)
        self.files_HE = sorted(glob.glob(he_images_path + '*.png'))
        self.files_PAS = sorted(glob.glob(pas_images_path + '*.png'))

    def __getitem__(self, index):
        item_HE = self.transform(Image.open(self.files_HE[index % len(self.files_HE)]))
        item_PAS = self.transform(Image.open(self.files_PAS[index % len(self.files_PAS)]))
        return {'HE': item_HE, 'PAS': item_PAS}

    def __len__(self):
        return max(len(self.files_HE), len(self.files_PAS))
