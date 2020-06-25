import os
from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image

from .transforms import UNetDataAugmentations, UNetBaseTransform

class CustomDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, transform=None):
        # super().__init__()
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.ids = [splitext(file)[0] for file in listdir(imgs_dir) if not file.startswith(".")]
        self.transform = transform

        logging.info(f"Creating dataset with {len(self.ids)} items")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        id = self.ids[index]
        image_file = glob(os.path.join(self.imgs_dir, id + ".*"))
        mask_file = glob(os.path.join(self.masks_dir, id + ".*"))

        assert len(mask_file) == 1, \
            f"Either no mask or multiple masks found for the ID {id}: {mask_file}."
        assert len(image_file) == 1, \
            f"Either no image or multiple images found for the ID {id}: {image_file}."

        image = Image.open(image_file[0])
        mask = Image.open(mask_file[0])

        assert image.size == mask.size, \
            f"Image and mask {id} should be the same size, but are {image.size} and {mask.size}."

        if self.transform is not None:
            (image, mask) = self.transform(image, mask)

        return {"image": image, "mask": mask}

class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        assert 0 < scale <= 1, "Scale must be between 0 and 1"

        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.transform = UNetBaseTransform(scale)

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith(".")]
        logging.info(f"Creating dataset with {len(self.ids)} examples")

    def __len__(self):
        return len(self.ids)

    def preprocess(self, image, mask):
        return self.transform(image, mask)

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + ".*")
        img_file = glob(self.imgs_dir + idx + ".*")

        assert len(mask_file) == 1, \
            f"Either no mask or multiple masks found for the ID {idx}: {mask_file}"
        assert len(img_file) == 1, \
            f"Either no image or multiple images found for the ID {idx}: {img_file}"

        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])

        assert img.size == mask.size, \
            f"Image and mask {idx} should be the same size, but are {img.size} and {mask.size}"

        img, mask = self.preprocess(img, mask)

        return {"image": img, "mask": mask}
