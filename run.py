import tensorflow as tf
import yaml
from data import TextDataset
import numpy as np
from model import GAN


def train(model):
    dataset = TextDataset("", 32)

    for epoch in range(10):
        print(f"Epoch nr: {epoch}")
        for images, wrong_images, embed, _, _ in dataset.train.next_batch(16, 4, embeddings=True, wrong_img=True):
            discriminator_loss, generator_loss = model(train_batch)

            if step % 100 == 0:
                print(
                    f"Step {step}: Discriminator loss = {discriminator_loss}, Generator loss: {generator_loss}"
                )

        generated_samples = model.generate_sample()


def main():
    model = GAN()
    train(model)


if __name__ == "__main__":
    main()
