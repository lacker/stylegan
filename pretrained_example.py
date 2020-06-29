# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Minimal script for generating an image using pre-trained StyleGAN generator."""

# Turn off deprecation warning spam
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import os
import pickle
import numpy as np
import PIL.Image
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


def generate(Gs, seed, outfile):
    # Pick latent vector.
    rnd = np.random.RandomState(seed)
    latents = rnd.randn(1, Gs.input_shape[1])

    # Generate image.
    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    images = Gs.run(latents, None,
                    truncation_psi=0.7,
                    randomize_noise=False,
                    output_transform=fmt)

    # Save image.
    os.makedirs(config.result_dir, exist_ok=True)
    png_filename = os.path.join(config.result_dir, outfile + '.png')
    PIL.Image.fromarray(images[0], 'RGB').save(png_filename)
    print("generated", png_filename)
    
    
def main():
    # Initialize TensorFlow.
    tflib.init_tf()

    Gs = load_gwern_model()

    for seed in range(10):
        generate(Gs, seed, f'example{seed}')

    
if __name__ == "__main__":
    main()
