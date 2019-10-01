from tkinter import *
from tkinter.ttk import *

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import argparse
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import gzip
import pickle
from train_vae import preprocess_rgb_frame

from models import MlpVAE, ConvVAE

import tensorflow as tf

parser = argparse.ArgumentParser(description="Visualizes the features learned by the VAE")
parser.add_argument("--model_dir", type=str, required=True)
parser.add_argument("--model_type", type=str, default="cnn")
parser.add_argument("--z_dim", type=int, default=64)
args = parser.parse_args()

seg_as_target = "seg_" in args.model_dir
source_shape = (80, 160, 3)
target_shape = (80, 160, 1 if seg_as_target else 3)

beta = int(re.findall("beta(\d+)", args.model_dir)[0])
loss_type = "MSE" if "mse_" in args.model_dir else "BCE"

title = "{}; {}; $z_{{dim}}={}$; $\\beta={}$".format(loss_type, args.model_type.upper(), args.z_dim, beta)

if args.model_type == "cnn": VAEClass = ConvVAE
elif args.model_type == "mlp": VAEClass = MlpVAE    
else: raise Exception("No model type \"{}\"".format(args.model_type))

vae = VAEClass(source_shape, target_shape, z_dim=args.z_dim, model_dir=args.model_dir, training=False)
vae.init_session(init_logging=False)
if not vae.load_latest_checkpoint():
    print("Failed to load latest checkpoint for model \"{}\"".format(args.model_dir))

#image = preprocess_rgb_frame(np.asarray(Image.open("data_old/rgb/1535.png")))
image = preprocess_rgb_frame(np.asarray(Image.open("data/rgb/0.png")))

z_range = 10

import carla
def image_to_city_scapes(image):
    classes = {
        0: [0, 0, 0], # None
        1: [70, 70, 70], # Buildings
        2: [190, 153, 153], # Fences
        3: [72, 0, 90], # Other
        4: [220, 20, 60], # Pedestrians
        5: [153, 153, 153], # Poles
        6: [157, 234, 50], # RoadLines
        7: [128, 64, 128], # Roads
        8: [244, 35, 232], # Sidewalks
        9: [107, 142, 35], # Vegetation
        10: [0, 0, 255], # Vehicles
        11: [102, 102, 156], # Walls
        12: [220, 220, 0] # TrafficSigns
    }
    new_image = np.ones((image.shape[0], image.shape[1], 4))
    segimg = np.round((image[:, :, 0] * 12)).astype(np.uint8)
    img = np.zeros((image.shape[0], image.shape[1], 3))
    for j in range(image.shape[0]):
        for i in range(image.shape[1]):
            img[j, i] = classes[segimg[j, i]]
    return img / 255

fig, ax = plt.subplots(min(16, args.z_dim), int(np.ceil(args.z_dim / 16)), sharex=True, figsize=(12, 12))

if len(ax.shape) == 1:
    ax = np.expand_dims(ax, axis=-1)

seeded_z = vae.sess.run(vae.sample, feed_dict={vae.source_states: [image]})[0]
for k in range(int(np.ceil(args.z_dim / 16))):
    for i in range(16):
        z_index = i + k * 16
        if z_index >= args.z_dim:
            break
        w = source_shape[1]
        h = source_shape[0]
        compound_image = np.zeros((h, w * 5, 3))
        for j, zi in enumerate(np.linspace(-z_range, z_range, 5)):
            z = seeded_z.copy()
            z[z_index] += zi
            generated_image = vae.generate_from_latent([z])[0].reshape(h, w, target_shape[-1])
            if seg_as_target:
                generated_image = image_to_city_scapes(generated_image)
            compound_image[:, j*w:(j+1)*w, :] = generated_image
        ax[i, k].imshow(compound_image, vmin=0.0, vmax=1.0)
        ax[i, k].set_xticks(np.linspace(w/2, w*5-w/2, 5))
        ax[i, k].set_xticklabels(np.linspace(-z_range, z_range, 5))
        ax[i, k].set_yticks([h/2])
        ax[i, k].set_yticklabels([z_index])
fig.text(0.04, 0.5, "z index", va="center", rotation="vertical")
fig.suptitle(title)

plt.savefig(title, dpi=700)
plt.show()
