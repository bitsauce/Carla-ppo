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

def preprocess_frame(frame):
    frame = frame[:, :, :3]                  # RGBA -> RGB
    frame = frame.astype(np.float32) / 255.0 # [0, 255] -> [0, 1]
    return frame

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trains a VAE on input data")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="data")
    parser.add_argument("--loss_type", type=str, default="bce")
    parser.add_argument("--model_type", type=str, default="cnn")
    parser.add_argument("--beta", type=int, default=1)
    parser.add_argument("--z_dim", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--kl_tolerance", type=float, default=0.0)
    args = parser.parse_args()

    images = []
    for filename in os.listdir(args.dataset):
        _, ext = os.path.splitext(filename)
        if ext == ".png":
            filepath = os.path.join(args.dataset, filename)
            frame = preprocess_frame(np.asarray(Image.open(filepath)))
            images.append(frame)
            print(frame.shape)
            eixt()
    images = np.stack(images, axis=0)
    
    np.random.seed(0)
    np.random.shuffle(images)

    val_split = int(images.shape[0] * 0.1)
    train_images = images[val_split:]
    val_images = images[:val_split]

    input_shape = images.shape[1:]
    print("train_images.shape", train_images.shape)
    print("val_images.shape", val_images.shape)

    print("")
    print("Training parameters:")
    for k, v, in vars(args).items(): print(f"  {k}: {v}")
    print("")

    if args.model_name is None:
        args.model_name = "{}_{}_zdim{}_beta{}_kl_tolerance{}_{}".format(
            args.loss_type, args.model_type, args.z_dim, args.beta, args.kl_tolerance,
            os.path.splitext(os.path.basename(args.dataset))[0])

    if args.loss_type == "bce": loss_fn = bce_loss
    elif args.loss_type == "bce_v2": loss_fn = bce_loss_v2
    elif args.loss_type == "mse": loss_fn = mse_loss
    else: raise Exception("No loss function \"{}\"".format(args.loss_type))

    if args.model_type == "cnn": VAEClass = ConvVAE
    elif args.model_type == "mlp": VAEClass = MlpVAE    
    else: raise Exception("No model type \"{}\"".format(args.model_type))

    vae = VAEClass(input_shape=input_shape,
                   z_dim=args.z_dim,
                   beta=args.beta,
                   learning_rate=args.learning_rate,
                   kl_tolerance=args.kl_tolerance,
                   loss_fn=loss_fn,
                   model_name=args.model_name)
    vae.init_session()

    min_val_loss = float("inf")
    counter = 0
    print("Training")
    while True:
        epoch = vae.get_step_idx()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}")
        
        # Save model
        vae.save()

        # Calculate evaluation metrics
        val_loss, _ = vae.evaluate(val_images, args.batch_size)
        
        # Train one epoch
        vae.train_one_epoch(train_images, args.batch_size)
        
        # Early stopping
        if val_loss < min_val_loss:
            counter = 0
            min_val_loss = val_loss
        else:
            counter += 1
            if counter >= 10:
                print("No improvement in last 10 epochs, stopping")
                break

