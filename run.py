from pathlib import Path

import yaml

from data import TextDataset
from model import GAN
from utils import display_images


def train(model, config):
    dataset = TextDataset("", config['image_dim'])
    dataset.train = dataset.get_data("data/train/")

    for epoch in range(config['epochs']):
        updates_per_epoch = dataset.train.num_examples // config['batch_size']

        for idx in range(0, updates_per_epoch):
            images, wrong_images, embed, _, _ = dataset.train.next_batch(config['batch_size'], 4, embeddings=True,
                                                                         wrong_img=True)
            discriminator_loss, generator_loss = model(images, embed, wrong_images)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: discriminator loss is {discriminator_loss} and generator loss is {generator_loss}")
            images_generated = model.generate_sample(embed)
            display_images(images_generated)


def main(config):
    model = GAN(config)
    train(model, config)


if __name__ == "__main__":
    config = yaml.load(Path("config.yaml").read_text(), Loader=yaml.SafeLoader)
    main(config)
