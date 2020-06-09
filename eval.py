import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dice_loss import dice_coeff
from utils.dataset import CustomDataset

def eval_net(net, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch['image'], batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                mask_pred = net(imgs)

            if net.n_classes > 1:
                tot += F.cross_entropy(mask_pred, true_masks).item()
            else:
                pred = torch.sigmoid(mask_pred)
                pred = (pred > 0.5).float()
                tot += dice_coeff(pred, true_masks).item()
            pbar.update()

    net.train()
    return tot / n_val

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate segmentation network with Dice Coeff.")
    parser.add_argument("--image_dir", "-i", type=str,
        help="Image directory to perform validation.")
    parser.add_argument("--mask_dir", "-t", type=str,
        help="Mask directory to perform validation.")
    parser.add_argument("--model", "-m", type=str,
        help="Path to model weights (.pth).")
    parser.add_argument("--height", type=int,
        help="Resize height of input images.")
    parser.add_argument("--width", type=int,
        help="Resize width of inut images.")

if __name__ == "__main__":
    args = parse_args()
    # TODO
