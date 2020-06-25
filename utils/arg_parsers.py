import argparse

def get_train_args():
    parser = argparse.ArgumentParser(
        description="Train segmentation model on images and target masks.")
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
    parser.add_argument("-e", "--epochs",
        metavar="E",
        type=int,
        default=5,
        help="Number of epochs")
    parser.add_argument("-b", "--batch-size",
        metavar="B",
        type=int,
        nargs="?",
        default=1,
        help="Batch size",
        dest="batchsize")
    parser.add_argument("-l", "--learning-rate",
        metavar="LR",
        type=float,
        nargs="?",
        default=0.01,
        help="Learning rate",
        dest="lr")
    parser.add_argument("-f", "--load",
        dest="load",
        type=str,
        default=False,
        help="Load model from a .pth file")
    parser.add_argument("--width",
        dest="width",
        type=int,
        help="Resize width for input image.")
    parser.add_argument("--height",
        dest="height",
        type=int,
        help="Resize height for input image.")

    return parser.parse_args()

def get_eval_args():
    parser = argparse.ArgumentParser(
        description="Evaluate segmentation network with Dice Coeff.")
    parser.add_argument("--image_dir", "-i",
        type=str,
        default="data/val/images/",
        help="Image directory to perform validation.")
    parser.add_argument("--mask_dir", "-t",
        type=str,
        default="data/val/masks/",
        help="Mask directory to perform validation.")
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
    parser.add_argument("--weights", "-s",
        type=str,
        help="Path to model weights (.pth).")
    parser.add_argument("--height",
        type=int,
        help="Resize height of input images.")
    parser.add_argument("--width",
        type=int,
        help="Resize width of inut images.")

    return parser.parse_args()

def get_predict_args():
    parser = argparse.ArgumentParser(
        description='Predict masks from input images')
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
