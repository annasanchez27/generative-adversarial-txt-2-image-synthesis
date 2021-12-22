import tensorflow as tf
from matplotlib import pyplot as plt


def display_images(out, label=None):
    """Inspired by: https://github.com/Atcold/pytorch-Deep-Learning/blob/master/11-VAE.ipynb"""
    out_pic = tf.reshape(out, [-1, 64, 64, 3])
    plt.figure(figsize=(18, 6))
    for i in range(4):
        plt.subplot(1, 4, i + 1)
        plt.imshow(out_pic[i + 4])
        plt.axis("off")
        plt.title(label[i + 4])
    plt.show()


def denormalize_images(images):
    return ((images + 1.0) * 127.5).astype('uint8')
