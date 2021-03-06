# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Minimal script for generating an image using pre-trained StyleGAN generator."""

import sys
try:
    import PIL.Image
except ModuleNotFoundError:
    print("you forgot to run:\nsource activate stylegan")
    sys.exit(1)
    

# Turn off deprecation warning spam
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import os
import pickle
import numpy as np
import dnnlib
import dnnlib.tflib as tflib
import config

# The pickle formats load _G, _D, Gs.
# _G = Snapshot of the generator. Mainly useful for resuming a previous training run.
# _D = Snapshot of the discriminator. Mainly useful for resuming a previous training run.
# Gs = Long-term average of the generator.
#      Yields higher-quality results than the snapshot.

def load_nvidia_face_model():
    # karras2019stylegan-ffhq-1024x1024.pkl
    url = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ'

    with dnnlib.util.open_url(url, cache_dir=config.cache_dir) as f:
        _G, _D, Gs = pickle.load(f)

    # Print network details.
    # Gs.print_layers()
    return Gs


def load_gwern_model():
    fname = "/home/lacker/models/2019-03-08-stylegan-animefaces-network-02051-021980.pkl"
    _G, _D, Gs = pickle.load(open(fname, "rb"))
    return Gs


def generate_latents(seed):
    # Pick latent vector.
    rnd = np.random.RandomState(seed)
    latents = rnd.randn(1, 512)    
    return latents


# This is for the resumable version. But let's make an unresumable version first
class State(object):
    def __init__(self, latents=None, index=None, seed=None):
        if latents is None:
            assert index is None
            assert seed is not None
            self.latents = generate_latents(seed)
            self.index = 0
            return

        assert index is not None
        assert seed is None
        self.latents = latents
        self.index = index


# How far is this from being a picture of AJ
AJ = PIL.Image.open(os.path.join(config.result_dir, "aj.png"))
AJ_RAVELED = np.ravel(np.array(AJ))
def aj_distance(image):
    small = image.resize((64, 64), PIL.Image.ANTIALIAS)
    raveled = np.ravel(np.array(small))
    diff = np.subtract(AJ_RAVELED, raveled)
    return np.linalg.norm(diff, 1) / diff.size
        
        
def generate_image(Gs, latents):
    # Generate image.
    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    images = Gs.run(latents, None,
                    truncation_psi=0.7,
                    randomize_noise=False,
                    output_transform=fmt)
    image = PIL.Image.fromarray(images[0], "RGB")
    return image


def save_image(image, outfile):
    os.makedirs(config.result_dir, exist_ok=True)
    png_filename = os.path.join(config.result_dir, outfile + '.png')
    image.save(png_filename)
    print("generated", png_filename, "with aj_distance", aj_distance(image))


# latents are a standard normal distribution
# we scale the fuzz by scale
# amount will be something uniformly between -amount and +amount
def fuzz(latents, scale):
    noise = np.random.randn(*latents.shape) * scale
    return np.add(latents, noise)
    
def main():
    print("initializing...")
    
    # Initialize TensorFlow.
    tflib.init_tf()

    Gs = load_gwern_model()

    latents = generate_latents(3)
    image = generate_image(Gs, latents)
    distance = aj_distance(image)
    counter = 0
    save_image(image, "example0")
    fails = 0

    while True:
        new_latents = fuzz(latents, 0.05)
        new_image = generate_image(Gs, new_latents)
        new_distance = aj_distance(new_image)

        if new_distance >= distance:
            fails += 1
            continue
        print(f"success after {fails} fails")
        
        fails = 0        
        counter += 1
        save_image(new_image, f'example{counter}')
        image = new_image
        distance = new_distance
        latents = new_latents

        
if __name__ == "__main__":
    main()
