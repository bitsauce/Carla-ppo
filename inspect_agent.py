from tkinter import *
from tkinter.ttk import *
from tkinter.filedialog import askopenfilename

import os
import gym
import numpy as np
import argparse
from PIL import Image, ImageTk

from ppo import PPO
from vae.models import MlpVAE, ConvVAE
from vae_common import preprocess_frame, load_vae

parser = argparse.ArgumentParser(description="Visualizes the policy learned by the agent")

# VAE parameters
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--vae_model", type=str, default="vae/models/seg_bce_cnn_zdim64_beta1_kl_tolerance0.0_data/")
parser.add_argument("--vae_model_type", type=str, default=None)
parser.add_argument("--vae_z_dim", type=int, default=None)

args = parser.parse_args()

# Load VAE
vae = load_vae(args.vae_model, args.vae_z_dim, args.vae_model_type)

# State encoding fn
measurements_to_include = set(["steer", "throttle", "speed"])
#encode_state_fn = create_encode_state_fn(vae, measurements_to_include)

# Load PPO agent
input_shape = np.array([vae.z_dim + len(measurements_to_include)])
action_space = gym.spaces.Box(np.array([-1, 0]), np.array([1, 1]), dtype=np.float32) # steer, throttle
model = PPO(input_shape, action_space, model_dir=os.path.join("models", args.model_name))
model.init_session(init_logging=False)
if not model.load_latest_checkpoint():
    raise Exception("Failed to load PPO agent")

class UI():
    def __init__(self, z_dim, generate_fn, slider_range=3, image_scale=4):
        # Setup tkinter window
        self.window = Tk()
        self.window.title("Agent Inspector")
        self.window.style = Style()
        self.window.style.theme_use("clam") # ('clam', 'alt', 'default', 'classic')

        # Setup image
        top_frame = Frame(self.window)
        top_frame.pack(side=TOP, padx=50, pady=20)

        top_frame0 = Frame(top_frame)
        top_frame0.pack(side=LEFT)
        top_frame1 = Frame(top_frame)
        top_frame1.pack(side=LEFT)
        top_frame2 = Frame(top_frame)
        top_frame2.pack(side=LEFT)
        
        self.image = Label(top_frame1)
        self.image.pack(side=LEFT)

        # Create generate fn
        def call_generate_fn(event=None):
            z             = [z_i.get() for z_i in self.z_vars]
            measurements  = [m.get() for m in self.measurement_vars]
            encoded_state = np.append(z, measurements)
            generate_fn(z, encoded_state)
        self.generate_fn = call_generate_fn

        # Create sliders for input measurements
        label = Label(top_frame0, text="Input Measurements")
        label.pack(side=TOP)
        num_measurements = 3
        measurement_labels = ["Steer", "Throttle", "Speed"]
        measurements_min = [-1, 0, 0]
        measurements_max = [ 1, 1, 50]
        self.measurement_vars = [DoubleVar() for _ in range(num_measurements)]
        for i in range(num_measurements):
            slider_frame = Frame(top_frame0)
            action_slider_label = Label(slider_frame, text="{:20s}".format(measurement_labels[i]))
            action_slider_label.pack(side=LEFT, padx=10)
            action_slider = Scale(slider_frame, value=0.0, variable=self.measurement_vars[i],
                                  orient=HORIZONTAL, length=200,
                                  from_=measurements_min[i], to=measurements_max[i],
                                  command=self.generate_fn)
            action_slider.pack(side=RIGHT)
            slider_frame.pack(side=TOP)

        # Create sliders to show output action
        label = Label(top_frame2, text="Output Action")
        label.pack(side=TOP)
        num_actions = action_space.shape[0]
        action_labels = ["Steer", "Throttle"]
        self.action_vars = [DoubleVar() for _ in range(num_actions)]
        for i in range(num_actions):
            slider_frame = Frame(top_frame2)
            action_slider_label = Label(slider_frame, text="{:20s}".format(action_labels[i]))
            action_slider_label.pack(side=LEFT, padx=10)
            action_slider = Scale(slider_frame, value=0.0, variable=self.action_vars[i],
                                  orient=HORIZONTAL, length=200,
                                  from_=action_space.low[i], to=action_space.high[i])
            action_slider.pack(side=RIGHT)
            slider_frame.pack(side=TOP)

        self.image_scale = image_scale
        self.update_image(np.ones(vae.target_shape) * 127)

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
                    label.configure(text="z[{:2d}]={:6.2f}".format(i, z_i.get()))
                    if generate: self.generate_fn()
                return event

            if i % 16 == 0:
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

        for f in reversed(slider_frames):
            f.pack(side=RIGHT, padx=0, pady=20)

    def set_z_by_image(self):
        filepath = askopenfilename()
        if filepath is not None:
            frame = preprocess_frame(np.asarray(Image.open(filepath)))
            z = vae.sess.run(vae.sample, feed_dict={vae.input_states: [frame]})[0]
            for i in range(len(self.z_vars)):
                self.z_vars[i].set(z[i])
                self.update_label_fns[i](generate=False)
            self.generate_fn()

    def update_image(self, image_array):
        image_array = image_array.astype(np.uint8)
        image_size = vae.target_shape[:2] * self.image_scale
        if image_array.shape[-1] == 1: image_array = image_array.squeeze(-1)
        pil_image = Image.fromarray(image_array)
        pil_image = pil_image.resize((image_size[1], image_size[0]), resample=Image.NEAREST)
        self.tkimage = ImageTk.PhotoImage(image=pil_image)
        self.image.configure(image=self.tkimage)

    def mainloop(self):
        self.generate_fn()
        self.window.mainloop()

def generate(z, encoded_state):
    generated_image = vae.generate_from_latent([z])[0]
    ui.update_image(generated_image.reshape(vae.target_shape) * 255)

    # Update output action
    action, _ = model.predict(encoded_state, greedy=True)
    for i in range(len(action)):
        ui.action_vars[i].set(action[i])

ui = UI(vae.sample.shape[1], generate, slider_range=10)
ui.mainloop()
