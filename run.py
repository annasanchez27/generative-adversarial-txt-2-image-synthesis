import yaml
import tqdm
import wandb

from model import GAN
from model2 import GAN2
from pathlib import Path
from data import TextDataset
from utils import denormalize_images


def train(model, config):
    dataset = TextDataset("", config['image_dim'])
    dataset.train = dataset.get_data("data/train/")
    dataset.test = dataset.get_data("data/test/")

    for epoch in range(config['epochs']):
        updates_per_epoch = dataset.train.num_examples // config['batch_size']

        for _ in tqdm.tqdm(range(0, updates_per_epoch)):
            images, wrong_images, embed, captions, _ = dataset.train.next_batch(config['batch_size'], 4, embeddings=True,
                                                                         wrong_img=True)
            discriminator_loss, generator_loss = model(images, embed, wrong_images)
            wandb.log({"discriminator_loss": discriminator_loss, "generator_loss": generator_loss})

        images, _, embed, captions, _ = dataset.train.next_batch(
            config['batch_size'], 1, embeddings=True,
            wrong_img=False)
        images_generated = model.generate_sample(embed)
        images_generated = denormalize_images(images_generated.numpy())
        wandb.log({"images": [wandb.Image(images_generated[i]) for i in range(4)], "captions": [wandb.Html(captions[i]) for i in range(4)]})


def main(config):
    model = GAN(config)
    train(model, config)


if __name__ == "__main__":
    wandb.init(project="generative-adversarial-txt-2-image", entity="sebastiaan")
    config = yaml.load(Path("config.yaml").read_text(), Loader=yaml.SafeLoader)
    main(config)
