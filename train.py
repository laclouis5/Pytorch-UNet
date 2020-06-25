import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from eval import eval_net
from unet import UNet

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset, CustomDataset
from utils.arg_parsers import get_train_args
from torch.utils.data import DataLoader, random_split

from utils.transforms import UNetDataAugmentations, UNetBaseTransform
import segmentation_models_pytorch as smp

dir_checkpoint = "checkpoints/"

def train_net(net,
    device,
    epochs=5,
    batch_size=1,
    lr=0.001,
    save_cp=False,
    img_size=None
):
    # dataset = BasicDataset(dir_img, dir_mask, img_scale)
    # n_val = int(len(dataset) * val_percent)
    # n_train = len(dataset) - n_val
    # (train, val) = random_split(dataset, [n_train, n_val])
    # train.transform = UNetDataAugmentations(img_scale)
    train = CustomDataset("data/train/images/", "data/train/masks/",
        transform=UNetDataAugmentations(img_size))
    val = CustomDataset("data/val/images/", "data/val/masks/",
        transform=UNetBaseTransform(img_size))

    train_loader = DataLoader(train,
        batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val,
        batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

    writer = SummaryWriter(comment=f"LR_{lr}_BS_{batch_size}_SIZE_{img_size}")
    global_step = 0

    logging.info(
        "Starting training:\n"
        f"\tEpochs:             {epochs}\n"
        f"\tBatch size:         {batch_size}\n"
        f"\tLearning rate:      {lr}\n"
        f"\tTraining size:      {len(train)}\n"
        f"\tValidation size:    {len(val)}\n"
        f"\tCheckpoints:        {save_cp}\n"
        f"\tDevice:             {device.type}\n"
        f"\tImages size (H, W): {img_size}")

    criterion = nn.CrossEntropyLoss() if net.n_classes > 1 else nn.BCEWithLogitsLoss()
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
    #     "min" if net.n_classes > 1 else "max", patience=2)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=int(epochs/5))

    best_model_dice_coeff = 0.0

    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=len(train), desc=f"Epoch {epoch + 1}/{epochs}", unit="img") as pbar:
            for batch in train_loader:
                imgs = batch["image"]
                true_masks = batch["mask"]

                assert imgs.shape[1] == net.n_channels, \
                    f"Network has been defined with {net.n_channels} input channels, " \
                    f"but loaded images have {imgs.shape[1]} channels. Please check that " \
                    "the images are loaded correctly."

                imgs = imgs.to(device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device, dtype=mask_type)

                masks_pred = net(imgs)
                loss = criterion(masks_pred, true_masks)
                epoch_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                writer.add_scalar("Loss/train", loss.item(), global_step)
                pbar.set_postfix(**{"loss (batch)": loss.item()})
                pbar.update(imgs.shape[0])

                global_step += 1
                if global_step % (len(train) // (5 * batch_size)) == 0:
                    val_score = eval_net(net, val_loader, device)
                    # scheduler.step(val_score)  # PlateauScheduler

                    if val_score > best_model_dice_coeff:
                        best_model_dice_coeff = val_score
                        torch.save(net.state_dict(),
                            dir_checkpoint + f"CP_best.pth")

                    writer.add_scalar("learning_rate",
                        optimizer.param_groups[0]["lr"],
                        global_step)

                    if net.n_classes > 1:
                        logging.info("Validation cross entropy: {}".format(val_score))
                        writer.add_scalar("Loss/test", val_score, global_step)
                    else:
                        logging.info("Validation Dice Coeff: {}".format(val_score))
                        writer.add_scalar("Dice/test", val_score, global_step)

                    writer.add_images("images", imgs, global_step)

                    if net.n_classes == 1:
                        writer.add_images("masks/true",
                            true_masks,
                            global_step)
                        writer.add_images("masks/pred",
                            torch.sigmoid(masks_pred) > 0.5,
                            global_step)

        # LRStep Scheduler
        scheduler.step()

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info("Created checkpoint directory")
            except OSError:
                logging.error("Can't create save directory")
                pass

            torch.save(net.state_dict(),
                dir_checkpoint + f"CP_epoch{epoch + 1}.pth")
            logging.info(f"Checkpoint {epoch + 1} saved !")

    writer.close()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = get_train_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device {device}")

    if args.model == "unet":
        net = smp.Unet(args.backbone)
    elif args.model == "fpn":
        net = smp.FPN(args.backbone)
    else:
        logging.info("Not a valid model")
        raise SystemExit()

    setattr(net, "n_classes", 1)
    setattr(net, "n_channels", 3)
    setattr(net, "bilinear", None)

    logging.info(
        f"Network:\n"
        f"Model: {args.model}, backbone: {args.backbone}"
        f"\t{net.n_channels} input channels\n"
        f"\t{net.n_classes} output channels (classes)\n")

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f"Model loaded from {args.load}")

    net.to(device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        train_net(net,
            epochs=args.epochs,
            batch_size=args.batchsize,
            lr=args.lr,
            device=device,
            img_size=(args.height, args.width))
    except KeyboardInterrupt:
        torch.save(net.state_dict(), "INTERRUPTED.pth")
        logging.info("Saved interrupt")

        try: sys.exit(0)
        except SystemExit: os._exit(0)
