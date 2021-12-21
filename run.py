from data import TextDataset
from model import GAN


def train(model):
    dataset = TextDataset("", 64)
    dataset.train = dataset.get_data("data/train/")

    for epoch in range(10):
        updates_per_epoch = dataset.train.num_examples // 16

        for idx in range(0, updates_per_epoch):
            images, wrong_images, embed, _, _ = dataset.train.next_batch(16, 4, embeddings=True, wrong_img=True)
            discriminator_loss, generator_loss = model(images, embed)


def main():
    model = GAN()
    train(model)


if __name__ == "__main__":
    main()
