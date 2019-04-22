import argparse
import gzip
import os
import pickle
import shutil
import sys

import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

from models import ConvVAE, MlpVAE, bce_loss, bce_loss_v2, mse_loss

def preprocess_rgb_frame(frame):
    frame = frame[:, :, :3]                  # RGBA -> RGB
    frame = frame.astype(np.float32) / 255.0 # [0, 255] -> [0, 1]
    return frame

def preprocess_seg_frame_road_only(frame):
    frame = frame[:, :, :1]                 # RGBA -> R
    frame = frame == 7                      # Create a binary mask for the road class
    frame = frame.astype(np.float32)        # To float
    return frame

def preprocess_seg_frame(frame):
    frame = frame[:, :, :1]                 # RGBA -> R
    frame = frame.astype(np.float32) / 12.0 # [0, 12=num_classes] -> [0, 1]
    return frame

def load_images(dir_path, preprocess_fn):
    images = []
    for filename in os.listdir(dir_path):
        _, ext = os.path.splitext(filename)
        if ext == ".png":
            filepath = os.path.join(dir_path, filename)
            frame = preprocess_fn(np.asarray(Image.open(filepath)))
            images.append(frame)
    return np.stack(images, axis=0)

def train_val_split(images, val_portion=0.1):
    val_split = int(images.shape[0] * val_portion)
    train_images = images[val_split:]
    val_images = images[:val_split]
    return train_images, val_images

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trains a VAE with RGB images as source and RGB or segmentation images as target")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="data")
    parser.add_argument("--use_segmentations", type=bool, default=False)
    parser.add_argument("--loss_type", type=str, default="bce")
    parser.add_argument("--model_type", type=str, default="cnn")
    parser.add_argument("--beta", type=int, default=1)
    parser.add_argument("--z_dim", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_decay", type=float, default=0.98)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--kl_tolerance", type=float, default=0.0)
    args = parser.parse_args()

    # Load images from dataset/rgb and dataset/segmentation folders
    # (for every rgb/xxx.png we expect a corresponding segmentation/xxx.png)
    rgb_images = load_images(os.path.join(args.dataset, "rgb"), preprocess_rgb_frame)
    if args.use_segmentations:
        seg_images = load_images(os.path.join(args.dataset, "segmentation"), preprocess_seg_frame)
    
    # Split into train and val sets
    np.random.seed(0)
    train_source_images, val_source_images = train_val_split(rgb_images, val_portion=0.1)
    if args.use_segmentations:
        train_target_images, val_target_images = train_val_split(seg_images, val_portion=0.1)
    else:
        train_target_images, val_target_images = train_source_images, val_source_images

    # Get source and target image sizes
    # (may be different e.g. RGB and grayscale)
    source_shape = train_source_images.shape[1:]
    target_shape = train_target_images.shape[1:] if args.use_segmentations else source_shape

    print("train_source_images.shape", train_source_images.shape)
    print("val_source_images.shape", val_source_images.shape)
    print("train_target_images.shape", train_target_images.shape)
    print("val_target_images.shape", val_target_images.shape)

    print("")
    print("Training parameters:")
    for k, v, in vars(args).items(): print(f"  {k}: {v}")
    print("")

    # Set model name from params
    if args.model_name is None:
        args.model_name = "{}_{}_{}_zdim{}_beta{}_kl_tolerance{}_{}".format(
            "seg" if args.use_segmentations else "rgb",
            args.loss_type, args.model_type, args.z_dim, args.beta, args.kl_tolerance,
            os.path.splitext(os.path.basename(args.dataset))[0])

    if args.loss_type == "bce": loss_fn = bce_loss
    elif args.loss_type == "bce_v2": loss_fn = bce_loss_v2
    elif args.loss_type == "mse": loss_fn = mse_loss
    else: raise Exception("No loss function \"{}\"".format(args.loss_type))

    if args.model_type == "cnn": VAEClass = ConvVAE
    elif args.model_type == "mlp": VAEClass = MlpVAE    
    else: raise Exception("No model type \"{}\"".format(args.model_type))

    # Create VAE model
    vae = VAEClass(source_shape=source_shape,
                   target_shape=target_shape,
                   z_dim=args.z_dim,
                   beta=args.beta,
                   learning_rate=args.learning_rate,
                   lr_decay=args.lr_decay,
                   kl_tolerance=args.kl_tolerance,
                   loss_fn=loss_fn,
                   model_name=args.model_name)
    vae.init_session()

    # Training loop
    min_val_loss = float("inf")
    counter = 0
    print("Training")
    while True:
        epoch = vae.get_step_idx()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}")
        
        # Calculate evaluation metrics
        val_loss, _ = vae.evaluate(val_source_images, val_target_images, args.batch_size)
        
        # Early stopping
        if val_loss < min_val_loss:
            counter = 0
            min_val_loss = val_loss
            vae.save() # Save if better
        else:
            counter += 1
            if counter >= 10:
                print("No improvement in last 10 epochs, stopping")
                break

        # Train one epoch
        vae.train_one_epoch(train_source_images, train_target_images, args.batch_size)
