from tkinter import *
from tkinter.ttk import *
from tkinter.filedialog import askopenfilename

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import argparse
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from models import MlpVAE, ConvVAE
from train_vae import preprocess_rgb_frame

parser = argparse.ArgumentParser(description="Visualizes the features learned by the VAE")
parser.add_argument("--model_dir", type=str, required=True)
parser.add_argument("--model_type", type=str, default="cnn")
parser.add_argument("--z_dim", type=int, default=64)
parser.add_argument("--source_shape", type=str, default="160x80x3")
parser.add_argument("--target_shape", type=str, default="160x80x3",
                    help="Remember to set this one to 160x80x1 if model was trained on segmentation maps")
args = parser.parse_args()

source_shape = np.array([int(x) for x in args.source_shape.split("x")])[[1, 0, 2]]
target_shape = np.array([int(x) for x in args.target_shape.split("x")])[[1, 0, 2]]

if args.model_type == "cnn": VAEClass = ConvVAE
elif args.model_type == "mlp": VAEClass = MlpVAE    
else: raise Exception("No model type \"{}\"".format(args.model_type))

vae = VAEClass(source_shape=source_shape,
               target_shape=target_shape,
               z_dim=args.z_dim,
               model_dir=args.model_dir,
               training=False)
vae.init_session(init_logging=False)
if not vae.load_latest_checkpoint():
    print("Failed to load latest checkpoint for model \"{}\"".format(args.model_dir))

class UI():
    def __init__(self, z_dim, generate_fn, slider_range=3, image_scale=4):
        # Setup tkinter window
        self.window = Tk()
        self.window.title("VAE Inspector")
        self.window.style = Style()
        self.window.style.theme_use("clam") # ('clam', 'alt', 'default', 'classic')

        # Setup image
        self.image = Label(self.window)
        self.image.pack(side=LEFT, padx=50, pady=20)

        self.image_scale = image_scale
        self.update_image(np.ones(target_shape) * 127)

        self.generate_fn = generate_fn

        self.browse = Button(self.window, text="Set z by image", command=self.set_z_by_image)
        self.browse.pack(side=BOTTOM, padx=50, pady=20)

        # Setup sliders for latent vector z
        slider_frames = []
        self.z_vars = [DoubleVar() for _ in range(z_dim)]
        self.update_label_fns = []
        for i in range(z_dim):
            # On slider change event
            def create_slider_event(i, z_i, label):
                def event(_=None, generate=True):
                    label.configure(text="z[{}]={}{:.2f}".format(i, "" if z_i.get() < 0 else " ", z_i.get()))
                    if generate: self.generate_fn(np.array([z_i.get() for z_i in self.z_vars]))
                return event

            if i % 20 == 0:
                sliders_frame = Frame(self.window)
                slider_frames.append(sliders_frame)

            # Create widgets
            inner_frame = Frame(sliders_frame) # Frame for side-by-side label and slider layout
            label = Label(inner_frame, font="TkFixedFont")

            # Create event function
            on_value_changed = create_slider_event(i, self.z_vars[i], label)
            on_value_changed(generate=False) # Call once to set label text
            self.update_label_fns.append(on_value_changed)

            # Create slider
            slider = Scale(inner_frame, value=0.0, variable=self.z_vars[i], orient=HORIZONTAL, length=200,
                        from_=-slider_range, to=slider_range, command=on_value_changed)

            # Pack
            slider.pack(side=RIGHT, pady=10)
            label.pack(side=LEFT, padx=10)
            inner_frame.pack(side=TOP)
        for f in reversed(slider_frames): f.pack(side=RIGHT, padx=20, pady=20)

    def set_z_by_image(self):
        filepath = askopenfilename()
        if filepath is not None:
            frame = preprocess_rgb_frame(np.asarray(Image.open(filepath)))
            z = vae.sess.run(vae.sample, feed_dict={vae.source_states: [frame]})[0]
            for i in range(len(self.z_vars)):
                self.z_vars[i].set(z[i])
                self.update_label_fns[i](generate=False)
            self.generate_fn(np.array([z_i.get() for z_i in self.z_vars]))

    def update_image(self, image_array):
        image_array = image_array.astype(np.uint8).squeeze()
        image_size = target_shape[:2] * self.image_scale
        pil_image = Image.fromarray(image_array, mode="L" if target_shape[-1] == 1 else None)
        pil_image = pil_image.resize((image_size[1], image_size[0]), resample=Image.NEAREST)
        self.tkimage = ImageTk.PhotoImage(image=pil_image)
        self.image.configure(image=self.tkimage)

    def mainloop(self):
        self.generate_fn(np.array([z_i.get() for z_i in self.z_vars]))
        self.window.mainloop()

def generate(z):
    generated_image = vae.generate_from_latent([z])[0] * 255
    ui.update_image(generated_image.reshape(target_shape))

ui = UI(vae.sample.shape[1], generate, slider_range=10)
ui.mainloop()
