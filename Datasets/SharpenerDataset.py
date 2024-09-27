# --------------------------------------------------------------------------------------------------
# ------------------------------------ Sharpener Dataset Loader ------------------------------------
# --------------------------------------------------------------------------------------------------
from torch.utils.data import Dataset
from torchvision import transforms

from PIL import ImageFile
from PIL import Image
ImageFile.LOAD_TRUNCATED_IMAGES = True

import contextlib
import pathlib
import random
import torch
import io
import os
import shutil


# --------------------------------------------------------------------------------------------------
# ----------------------------------- CLASS :: Sharpener Dataset -----------------------------------
# --------------------------------------------------------------------------------------------------
class SharpenerDataset(Dataset):
    def __init__(self):

        self.images = []
        self.images.extend(pathlib.Path('/Volumes/T7/Database/Marvel Snap').rglob('*.png'))
        self.images.extend(pathlib.Path('/Volumes/T7/Database/MTG').rglob('*.png'))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:

        source = Image.open(str(self.images[index])).convert('RGB')
        source = transforms.RandomCrop((384, 512))(source)
        target = source.copy()

        buffer = io.BytesIO()
        target.save(buffer, format='JPEG', quality=95)
        buffer.seek(0)
        target = Image.open(buffer)

        contrast = random.uniform(0.0, 0.5)
        brightness = random.uniform(0.0, 0.5)
        target = transforms.ColorJitter(brightness=brightness, contrast=contrast)(target)

        r = random.uniform(0.25, 0.75)
        w = int(target.size[0] * r)
        h = int(target.size[1] * r)

        target = transforms.Resize((h,     w), interpolation=Image.NEAREST)(target)
        target = transforms.Resize((384, 512), interpolation=Image.BICUBIC)(target)

        return transforms.ToTensor()(source), transforms.ToTensor()(target)