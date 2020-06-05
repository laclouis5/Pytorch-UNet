from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image

from .transforms import UNetDataAugmentations, UNetBaseTransform

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
