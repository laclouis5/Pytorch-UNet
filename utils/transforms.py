import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import numpy as np
import PIL
import random
import numbers

# MARK: - Main classes

class UNetDataAugmentations:
    def __init__(self, rescale_size=None):
        self.transform = Compose([
            Resize(rescale_size) if rescale_size is not None else Identity(),
            RandomHorizontalFlip(),
            RandomColorJitter(),
            ToNumpyArray(),
            Normalize(),
            ToTensor()
        ])

    def __call__(self, image, mask):
        return self.transform(image, mask)

class UNetBaseTransform:
    def __init__(self, rescale_size=None):
        self.transform = Compose([
            Resize(rescale_size) if rescale_size is not None else Identity(),
            ToNumpyArray(),
            Normalize(),
            ToTensor()
        ])

    def __call__(self, image, mask):
        return self.transform(image, mask)

class TestTimeImageTransform:
    """
    Args:
        - size (int,  int): (height, width) of the output image.
    Returns:
        - image (FloatTensor): tensor representing the image.
    """
    def __init__(self, size=None):
        self.size = size

    def __call__(self, image):
        img = image  # PIL image

        if self.size is not None:
            img = F.resize(img, self.size)  # PIL image

        img = np.array(img)  # NP array (H, W, C)

        if img.max() > 1:
            img = img / 255  # NP array (H, W, C)

        img = img.transpose((2, 0, 1))  # NP array (C, H, W)

        return torch.from_numpy(img).type(torch.FloatTensor)  # FloatTensor (C, H, W)

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask):
        for t in self.transforms:
            (image, mask) = t(image, mask)
        return image, mask

class Identity:
    def __call__(self, image, mask):
        return (image, mask)

class Optional:
    def __init__(self, transform, condition):
        self.transform = transform
        self.condition = condition

    def __call__(self, image, mask):
        if self.condition:
            return self.transform(image, mask)
        else:
            return (image, mask)

class Lambda:
    def __init__(self, lambda_image, lambda_mask):
        self.l1 = lambda_image
        self.l2 = lambda_mask

    def __call__(self, image, mask):
        return l1(image), l2(mask)

class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask):
        if  random.random() < self.p:
            return F.hflip(image), F.hflip(mask)
        else:
            return image, mask

class Resize:
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.width = int(size)
            self.height = int(size)
        else:
            self.height = int(size[0])
            self.width = int(size[1])

        self.img_transform = transforms.Resize((self.height, self.width))
        self.mask_transform = transforms.Resize((self.height, self.width),
            interpolation=PIL.Image.NEAREST)

    def __call__(self, image, mask):
        return (
            self.img_transform(image),
            self.mask_transform(mask))

class Rescale:
    def __init__(self, scale):
        assert 0 < scale <= 1, "'scale' must be in 0<..1"

        self.scale = scale

    def __call__(self, image, mask):
        (img_w, img_h) = image.size
        (w, h) = (int(img_w * self.scale), int(img_h  * self.scale))

        image_out = F.resize(image, (h, w))
        mask_out = F.resize(mask, (h, w), interpolation=PIL.Image.NEAREST)

        return image_out, mask_out

class RandomColorJitter:
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05):
        self.transform = transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue)

    def __call__(self, image, mask):
        return self.transform(image), mask

class ToTensor:
    def __call__(self, image, mask):
        return (
            torch.from_numpy(image).type(torch.FloatTensor), torch.from_numpy(mask).type(torch.FloatTensor)
        )

class Crop:
    def __init__(self, top, left, height, width):
        self.top = top
        self.left = left
        self.height = height
        self.width = width

    def call(self, image, mask):
        return crop(image, mask, self.top, self.left, self.height, self.width)

class RandomCrop:
    """
    ```size`` is a (width, height) tuple.
    """
    def __init__(self, size, pos_ratio=None):
        if isinstance(size, numbers.Number):
            self.height = int(size)
            self.width = int(size)
        else:
            self.height = int(size[1])
            self.width = int(size[0])

        self.ratio = pos_ratio

    def __call__(self, image, mask):
        img_w, img_h = image.size

        if self.ratio is not None:
            x_min = int(self.width / 2)
            x_max = int(img_w - self.width / 2)
            y_min = int(self.height / 2)
            y_max = int(img_h - self.height / 2)

            np_mask = np.array(mask)[y_min:y_max, x_min:x_max]

            if random.random() < self.ratio:
                indices = np.argwhere(np_mask)
            else:
                indices = np.argwhere(~np_mask)

            if len(indices) == 0:
                (x, y) = (
                    int(img_w / 2 - self.width / 2),
                    int(img_h / 2 - self.height / 2))
            else:
                (y, x) = random.choice(indices)

        else:
            x_max = img_w - self.width
            y_max = img_h - self.height

            x = random.randint(0, x_max)
            y = random.randint(0, y_max)

        return crop(image, mask, y, x, self.height, self.width)

class ToNumpyArray:
    def __call__(self, image, mask):
        return np.array(image), np.array(mask)

class Normalize:
    #Can add normalisation from imagenet (mean, std, ...)
    def __call__(self, image, mask):
        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, 2)

        if image.max() > 1:
            image = image / 255
        if mask.max() > 1:
            mask = mask / 255

        image = image.transpose((2, 0, 1))
        mask = mask.transpose((2, 0, 1))

        return image, mask

# MARK: - Function helpers

def crop(image, mask, top, left, height, width):
    return (
        F.crop(image, top, left, height, width),
        F.crop(mask, top, left, height, width))

# MARK: - Tests

def main():
    import matplotlib.pyplot as plt

    image = PIL.Image.open("../data/val/im_02018.jpg")
    mask = PIL.Image.open("../data/val/im_02018.png")

    transform = Compose([
        RandomHorizontalFlip(),
        # RandomCrop(size=(612, 412), pos_ratio=0.5),
        Rescale(0.01) if False else Identity(),
        RandomColorJitter(),
        ToNumpyArray(),
        Normalize(),
        ToTensor()
    ])

    figure = plt.figure()
    grid = plt.GridSpec(nrows=3, ncols=6, hspace=0.1, wspace=0.1)

    image_axis = figure.add_subplot(grid[0, :3], xticklabels=[], yticklabels=[])
    mask_axis = figure.add_subplot(grid[0, 3:], xticklabels=[], yticklabels=[])

    image_axis.imshow(image)
    mask_axis.imshow(mask, cmap="gray")

    for i in range(6):
        for j in range(1, 3, 2):
            (sub_img, sub_mask) = transform(image, mask)

            subplot_image = figure.add_subplot(grid[j, i], xticklabels=[], yticklabels=[])
            subplot_mask = figure.add_subplot(grid[j + 1, i], xticklabels=[], yticklabels=[])

            subplot_image.imshow(transforms.ToPILImage()(sub_img))
            subplot_mask.imshow(transforms.ToPILImage()(sub_mask.squeeze()), cmap="gray")

    plt.show()

if __name__ == "__main__":
    main()
