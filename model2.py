import tensorflow as tf
import tensorflow.keras.activations as tfa
import tensorflow.keras.layers as tfkl


class DCGenerator2(tfkl.Layer):
    def __init__(self):
        super(DCGenerator2, self).__init__()
        self.image_size = 64
        self.num_channels = 3
        self.noise_dim = 100
        self.embed_dim = 1024
        self.projected_embed_dim = 128
        self.latent_dim = self.noise_dim + self.projected_embed_dim
        self.ngf = 64

        self.projection = tf.keras.Sequential([
            tfkl.Dense(self.projected_embed_dim),
            tfkl.BatchNormalization(),
            tfkl.LeakyReLU(0.2)
        ]
        )
        # torch in channel, out channel, kernel_size, stride, padding
        # tensorflow filters, kernel_size, stride

        # based on: https://github.com/pytorch/examples/blob/master/dcgan/main.py
        self.netG = tf.keras.Sequential([
            tfkl.Conv2DTranspose(self.ngf * 8, kernel_size=4, strides=1, padding='valid', use_bias=False),
            tfkl.BatchNormalization(),
            tfkl.ReLU(),
            # state size. (ngf*8) x 4 x 4
            tfkl.Conv2DTranspose(self.ngf * 4, kernel_size=4, strides=2, padding='same', use_bias=False),
            tfkl.BatchNormalization(),
            tfkl.ReLU(),
            # state size. (ngf*4) x 8 x 8
            tfkl.Conv2DTranspose(self.ngf * 2, kernel_size=4, strides=2, padding='same', use_bias=False),
            tfkl.BatchNormalization(),
            tfkl.ReLU(),
            # state size. (ngf*2) x 16 x 16
            tfkl.Conv2DTranspose(self.ngf, kernel_size=4, strides=2, padding='same', use_bias=False),
            tfkl.BatchNormalization(),
            tfkl.ReLU(),
            # state size. (ngf) x 32 x 32
            tfkl.Conv2DTranspose(self.num_channels, kernel_size=4, strides=2, padding='same', use_bias=False),
            tfkl.Activation(tfa.tanh)
            # state size. (num_channels) x 64 x 64
        ]
        )

    def call(self, z, embed):
        projected_embed = self.projection(embed)
        projected_embed = tf.expand_dims(tf.expand_dims(projected_embed, 2), 3)
        z = tf.expand_dims(tf.expand_dims(z, 2), 3)
        latent_vector = tf.concat([projected_embed, z], 1)
        output = self.netG(latent_vector)
        return output


class Concat_embed(tfkl.Layer):
    def __init__(self, embed_dim, projected_embed_dim):
        super(Concat_embed, self).__init__()
        self.projection = tf.keras.Sequential([
            tfkl.Dense(projected_embed_dim),
            tfkl.BatchNormalization(),
            tfkl.LeakyReLU(0.2)
        ])

    def call(self, inp, embed):
        print(f"input shape: {inp.shape}, embed shape: {embed.shape}")
        projected_embed = self.projection(embed)
        embed = tf.expand_dims(tf.expand_dims(projected_embed, 1), 1)
        embed = tf.tile(embed, [1, 4, 4, 1])
        hidden_concat = tf.concat([inp, embed], axis=3)
        print(f"hidden_concat shape: {hidden_concat.shape}")
        return hidden_concat


class DCDiscriminator2(tfkl.Layer):
    def __init__(self):
        super(DCDiscriminator2, self).__init__()
        self.image_size = 64
        self.num_channels = 3
        self.embed_dim = 1024
        self.projected_embed_dim = 128
        self.ndf = 64
        self.B_dim = 128
        self.C_dim = 16

        self.netD_1 = tf.keras.Sequential([
            # input is (nc) x 64 x 64
            tfkl.Conv2D(self.ndf, 4, 2, padding='same', use_bias=False),
            tfkl.LeakyReLU(0.2),
            # state size. (ndf) x 32 x 32
            tfkl.Conv2D(self.ndf * 2, 4, 2, padding='same', use_bias=False),
            tfkl.BatchNormalization(),
            tfkl.LeakyReLU(0.2),
            # state size. (ndf*2) x 16 x 16
            tfkl.Conv2D(self.ndf * 4, 4, 2, padding='same', use_bias=False),
            tfkl.BatchNormalization(),
            tfkl.LeakyReLU(0.2),
            # state size. (ndf*4) x 8 x 8
            tfkl.Conv2D(self.ndf * 8, 4, 2, padding='same', use_bias=False),
            tfkl.BatchNormalization(),
            tfkl.LeakyReLU(0.2),
        ])

        self.projector = Concat_embed(self.embed_dim, self.projected_embed_dim)

        self.netD_2 = tf.keras.Sequential([
            # state size. (ndf*8) x 4 x 4
            tfkl.Conv2D(1, 4, 1, padding='valid', use_bias=False),
            tfkl.Activation(tfa.sigmoid)
        ])

    def call(self, x, embed):
        x_intermediate = self.netD_1(x)
        x = self.projector(x_intermediate, embed)
        x = self.netD_2(x)
        return x, x_intermediate


class GAN2(tf.keras.Model):
    def __init__(self, config):
        super(GAN2, self).__init__()

        self.noise_dim = 128
        self.batch_size = config['batch_size']

        self.generator = DCGenerator2()
        self.discriminator = DCDiscriminator2()

        # Set up optimizers for both models.
        self.generator_optimizer = tf.keras.optimizers.Adam(config['learning_rate'], beta_1=config['momentum'])
        self.discriminator_optimizer = tf.keras.optimizers.Adam(config['learning_rate'], beta_1=config['momentum'])

    def discriminator_loss(self, actual_output, generated_output, mismatch_output):
        real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(tf.ones_like(actual_output), actual_output))
        generated_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            tf.zeros_like(generated_output), generated_output
        ))
        mismatch_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            tf.zeros_like(mismatch_output), mismatch_output
        ))
        total_loss = (mismatch_loss + generated_loss) / 2 + real_loss
        return total_loss

    def generator_loss(self, generated_output):
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(tf.ones_like(generated_output), generated_output))

    def generate_sample(self, embed):
        noise = tf.random.normal([self.batch_size, self.noise_dim])
        generated_sample = self.generator(noise, embed)
        return generated_sample

    def train_step(self, x, embed, wrong_images):
        noise = tf.random.normal([self.batch_size, self.noise_dim])

        with tf.GradientTape() as discriminator_tape, tf.GradientTape() as generator_tape:
            generated_samples = self.generator(noise, embed)

            real_output, _ = self.discriminator(x, embed)
            fake_output, _ = self.discriminator(generated_samples, embed)
            mismatch_output, _ = self.discriminator(wrong_images, embed)

            discriminator_loss = self.discriminator_loss(real_output, fake_output, mismatch_output)
            generator_loss = self.generator_loss(fake_output)

        generator_gradients = generator_tape.gradient(
            generator_loss, self.generator.trainable_variables
        )
        self.generator_optimizer.apply_gradients(
            zip(generator_gradients, self.generator.trainable_variables)
        )

        discriminator_gradients = discriminator_tape.gradient(
            discriminator_loss, self.discriminator.trainable_variables
        )
        self.discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients, self.discriminator.trainable_variables)
        )

        return discriminator_loss, generator_loss

    def train_step_new(self, x, embed, wrong_images):
        noise = tf.random.normal([self.batch_size, self.noise_dim])

        with tf.GradientTape() as discriminator_tape:
            generated_samples = self.generator(noise, embed)

            real_output, _ = self.discriminator(x, embed)
            fake_output, _ = self.discriminator(generated_samples, embed)
            mismatch_output, _ = self.discriminator(wrong_images, embed)

            discriminator_loss = self.discriminator_loss(real_output, fake_output, mismatch_output)

        discriminator_gradients = discriminator_tape.gradient(
            discriminator_loss, self.discriminator.trainable_variables
        )
        self.discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients, self.discriminator.trainable_variables)
        )

        with tf.GradientTape() as generator_tape:
            generated_samples = self.generator(noise, embed)
            fake_output, _ = self.discriminator(generated_samples, embed)
            generator_loss = self.generator_loss(fake_output)

        generator_gradients = generator_tape.gradient(
            generator_loss, self.generator.trainable_variables
        )
        self.generator_optimizer.apply_gradients(
            zip(generator_gradients, self.generator.trainable_variables)
        )

        return discriminator_loss, generator_loss

    def call(self, x, embed, wrong_images, train_step_new=False):
        if train_step_new:
            return self.train_step_new(x, embed, wrong_images)
        return self.train_step(x, embed, wrong_images)
