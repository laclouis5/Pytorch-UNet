import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from unet import UNet
from utils.data_vis import plot_img_and_mask
from utils.dataset import BasicDataset

from utils.transforms import UNetDataAugmentations, UNetBaseTransform
import segmentation_models_pytorch as smp
from utils.transforms import TestTimeImageTransform

def predict_img(
    net,
    full_img,
    device,
    image_size=None,
    out_threshold=0.5
):
    net.eval()
    transform = TestTimeImageTransform(image_size)

    img = transform(full_img)  # FloatTensor (C, H, W)
    img = img.unsqueeze(0)  # FloatTensor (B, C, H, W)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)

        probs = probs.squeeze(0)

        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((full_img.size[1], full_img.size[0])),
            transforms.ToTensor()
        ])

        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()

    return full_mask > out_threshold

def get_args():
    parser = argparse.ArgumentParser(
        description='Predict masks from input images',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--weights', '-s',
        default='MODEL.pth',
        metavar='FILE',
        help="Specify the file in which the weights are stored.")
    parser.add_argument('--input', '-i',
        metavar='INPUT',
        nargs='+',
        help='filenames of input images',
        required=True)
    parser.add_argument("--model", "-m",
        type=str,
        choices=["unet", "fpn"],
        default="unet",
        help="The network model.")
    parser.add_argument("--backbone", "-r",
        type=str,
        choices=["resnet18", "resnet34"],
        default="resnet34",
        help="Backbone to use. Should be the same as the one the model was trained on.")
    parser.add_argument('--output', '-o',
        metavar='INPUT',
        nargs='+',
        help='Filenames of ouput images')
    parser.add_argument('--viz', '-v',
        action='store_true',
        help="Visualize the images as they are processed",
        default=False)
    parser.add_argument('--no-save', '-n',
        action='store_true',
        help="Do not save the output masks",
        default=False)
    parser.add_argument('--mask-threshold', '-t',
        type=float,
        help="Minimum probability value to consider a mask pixel white",
        default=0.5)
    parser.add_argument("--height",
        type=int,
        help="Resize height of input image.",
        default=None)
    parser.add_argument("--width",
        type=int,
        help="Resize width of input image.",
        default=None)

    return parser.parse_args()

def get_output_filenames(args):
    in_files = args.input
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        logging.error("Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files

def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))

if __name__ == "__main__":
    args = get_args()
    in_files = args.input
    out_files = get_output_filenames(args)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    logging.info(f"Images size (H, W): {(args.height, args.width)}")

    image_size = (args.height, args.width) if (args.height is not None and args.width is not None) else None

    if args.model == "unet":
        net = smp.Unet(args.backbone)
    elif args.model == "fpn":
        net = smp.FPN(args.backbone)
    else:
        raise SystemExit()

    setattr(net, "n_classes", 1)
    setattr(net, "n_channels", 3)
    setattr(net, "bilinear", None)

    logging.info(
        f"Network:\n"
        f"  Model: {args.model}, backbone: {args.backbone}\n"
        f"  {net.n_channels} input channels\n"
        f"  {net.n_classes} output channels (classes)"
    )

    logging.info("Loading weights {}".format(args.weights))

    net.to(device=device)
    net.load_state_dict(torch.load(args.weights, map_location=device))

    logging.info("Model loaded !")

    for i, fn in enumerate(in_files):
        logging.info("Predicting image {}...".format(fn))

        img = Image.open(fn)

        mask = predict_img(
            net=net,
            full_img=img,
            image_size=image_size,
            out_threshold=args.mask_threshold,
            device=device)

        if not args.no_save:
            out_fn = out_files[i]
            result = mask_to_image(mask)
            result.save(out_files[i])

            logging.info("Mask saved to {}".format(out_files[i]))

        if args.viz:
            logging.info("Visualizing results for image {}, close to continue...".format(fn))
            plot_img_and_mask(img, mask)
