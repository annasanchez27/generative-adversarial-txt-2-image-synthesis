import tensorflow as tf
import tensorflow.keras.activations as tfa
import tensorflow.keras.layers as tfkl


class DCGenerator(tfkl.Layer):
    def __init__(self):
        super(DCGenerator, self).__init__()
        self.output_size = 32
        self.gf_dim = 128
        self.s16 = self.output_size // 16
        self.ReLu = tfkl.Activation(tfa.relu)
        self.embedding_layer = tf.keras.Sequential([
            tfkl.Dense(128, activation=None),
            tfkl.LeakyReLU(),
        ])
        self.input_layer = tf.keras.Sequential([
            tfkl.Dense(self.gf_dim * 8 * 4 * self.s16 * self.s16, use_bias=False),
            tfkl.BatchNormalization(),
            tfkl.ReLU(),
            tfkl.Reshape((4, 4, self.gf_dim * 8)),
        ])
        self.residual_layer1 = tf.keras.Sequential([
            tfkl.Conv2D(
                self.gf_dim * 2, (1, 1), strides=(1, 1), padding="valid", use_bias=False
            ),
            tfkl.BatchNormalization(),
            tfkl.ReLU(),
            tfkl.Conv2D(
                self.gf_dim * 2, (3, 3), strides=(1, 1), padding="same", use_bias=False
            ),
            tfkl.BatchNormalization(),
            tfkl.ReLU(),
            tfkl.Conv2D(
                self.gf_dim * 8, (3, 3), strides=(1, 1), padding="same", use_bias=False
            ),
            tfkl.BatchNormalization(),

        ])
        self.inter_layer = tf.keras.Sequential([
            tfkl.Conv2DTranspose(
                self.gf_dim * 4, (4, 4), strides=(2, 2), padding="same", use_bias=False
            ),
            tfkl.Conv2D(
                self.gf_dim * 4, (3, 3), strides=(1, 1), padding="same", use_bias=False
            ),
            tfkl.BatchNormalization(),
        ])

        self.residual_layer2 = tf.keras.Sequential([
            tfkl.Conv2D(
                self.gf_dim, (1, 1), strides=(1, 1), padding="valid", use_bias=False
            ),
            tfkl.BatchNormalization(),
            tfkl.ReLU(),
            tfkl.Conv2D(
                self.gf_dim, (3, 3), strides=(1, 1), padding="same", use_bias=False
            ),
            tfkl.BatchNormalization(),
            tfkl.ReLU(),
            tfkl.Conv2D(
                self.gf_dim * 4, (3, 3), strides=(1, 1), padding="same", use_bias=False
            ),
            tfkl.BatchNormalization(),
        ])
        self.last_layer = tf.keras.Sequential([
            tfkl.Conv2DTranspose(
                self.gf_dim * 2, (4, 4), strides=(2, 2), padding="same", use_bias=False
            ),
            tfkl.Conv2D(
                self.gf_dim * 2, (3, 3), strides=(1, 1), padding="same", use_bias=False
            ),
            tfkl.BatchNormalization(),
            tfkl.ReLU(),
            tfkl.Conv2DTranspose(
                self.gf_dim, (4, 4), strides=(2, 2), padding="same", use_bias=False
            ),
            tfkl.Conv2D(
                self.gf_dim, (3, 3), strides=(1, 1), padding="same", use_bias=False
            ),
            tfkl.BatchNormalization(),
            tfkl.ReLU(),
            tfkl.Conv2DTranspose(
                3, (4, 4), strides=(2, 2), padding="same", use_bias=False
            ),
            tfkl.Conv2D(
                3, (3, 3), strides=(1, 1), padding="same", use_bias=False
            ),
            tfkl.Activation(tfa.relu),
        ])

    def call(self, z, embed):
        embed = self.embedding_layer(embed)
        x = tf.concat([z, embed], 1)
        out_input = self.input_layer(x)

        x = self.residual_layer1(out_input)
        x = tf.add(out_input, x)
        x = self.ReLu(x)
        out2 = self.inter_layer(x)
        x = self.residual_layer2(out2)
        x = tf.add(out2, x)
        x = self.ReLu(x)
        x = self.last_layer(x)
        return x


class DCDiscriminator(tfkl.Layer):
    def __init__(self):
        super(DCDiscriminator, self).__init__()
        self.output_size = 32
        self.df_dim = 64
        self.s16 = self.output_size // 16
        self.input_layer = tf.keras.Sequential([
            tfkl.Conv2D(
                filters=self.df_dim, kernel_size=(4, 4), strides=(2, 2), padding="same"
            ),
            tfkl.LeakyReLU(0.2),
            tfkl.Conv2D(filters=self.df_dim * 2, kernel_size=(4, 4), strides=(2, 2), padding="same"),
            tfkl.BatchNormalization(),
            tfkl.LeakyReLU(0.2),
            tfkl.Conv2D(filters=self.df_dim * 4, kernel_size=(4, 4), strides=(2, 2), padding="same"),
            tfkl.BatchNormalization(),
            tfkl.LeakyReLU(0.2),
            tfkl.Conv2D(filters=self.df_dim * 8, kernel_size=(4, 4), strides=(2, 2), padding="same"),
            tfkl.BatchNormalization(),
            tfkl.LeakyReLU(0.2),
        ])

        # Residual layer
        self.residual_layer = tf.keras.Sequential([
            tfkl.Conv2D(filters=self.df_dim * 2, kernel_size=(1, 1), strides=(1, 1), padding="same"),
            tfkl.BatchNormalization(),
            tfkl.LeakyReLU(0.2),
            tfkl.Conv2D(filters=self.df_dim * 2, kernel_size=(3, 3), strides=(1, 1), padding="same"),
            tfkl.BatchNormalization(),
            tfkl.LeakyReLU(0.2),
            tfkl.Conv2D(filters=self.df_dim * 8, kernel_size=(3, 3), strides=(1, 1), padding="same"),
            tfkl.BatchNormalization(),
        ])
        self.LeakyRelu = tfkl.LeakyReLU(0.2)
        self.embedding_layer = tf.keras.Sequential([
            tfkl.Dense(128, activation=None),
            tfkl.LeakyReLU(),
        ])
        self.output_layer = tf.keras.Sequential([
            tfkl.Conv2D(filters=self.df_dim * 8, kernel_size=(1, 1), strides=(1, 1), padding="valid"),
            tfkl.BatchNormalization(),
            tfkl.LeakyReLU(0.2),
            tfkl.Conv2D(filters=1, kernel_size=(self.s16, self.s16), strides=(self.s16, self.s16), padding="valid"),
        ])
        self.Sigmoid = tfkl.Activation(tfa.sigmoid)

    def call(self, x, embed):
        out1 = self.input_layer(x)
        x = self.residual_layer(out1)
        x = tf.add(x, out1)
        x = self.LeakyRelu(x)
        embed = self.embedding_layer(embed)
        embed = tf.expand_dims(tf.expand_dims(embed, 1), 1)
        embed = tf.tile(embed, [1, 4, 4, 1])
        x = tf.concat([x, embed], axis=3)
        x = self.output_layer(x)
        return self.Sigmoid(x), x


class GAN(tf.keras.Model):
    def __init__(self, config):
        super(GAN, self).__init__()

        self.noise_dim = 100
        self.batch_size = config['batch_size']

        self.generator = DCGenerator()
        self.discriminator = DCDiscriminator()

        # Set up optimizers for both models.
        self.generator_optimizer = tf.keras.optimizers.Adam(config['learning_rate'])
        self.discriminator_optimizer = tf.keras.optimizers.Adam(config['learning_rate'])

    def discriminator_loss(self, actual_output, generated_output, mismatch_output):
        real_loss = self.sigmoid_cross_entropy_with_logits(tf.ones_like(actual_output), actual_output)
        generated_loss = self.sigmoid_cross_entropy_with_logits(
            tf.zeros_like(generated_output), generated_output
        )
        mismatch_loss = self.sigmoid_cross_entropy_with_logits(
            tf.zeros_like(mismatch_output), mismatch_output
        )
        total_loss = (mismatch_loss + generated_loss) / 2 + real_loss
        return total_loss

    def generator_loss(self, generated_output):
        return self.sigmoid_cross_entropy_with_logits(tf.ones_like(generated_output), generated_output)

    def generate_sample(self, embed):
        noise = tf.random.normal([self.batch_size, self.noise_dim])
        generated_sample = self.generator(noise, embed)
        return generated_sample

    def train_step(self, x, embed, wrong_images):
        noise = tf.random.normal([self.batch_size, self.noise_dim])

        with tf.GradientTape() as discriminator_tape, tf.GradientTape() as generator_tape:
            generated_samples = self.generator(noise, embed)

            real_output = self.discriminator(x, embed)
            fake_output = self.discriminator(generated_samples, embed)
            mismatch_output = self.discriminator(wrong_images, embed)

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

    def call(self, x, embed, wrong_images):
        return self.train_step(x, embed, wrong_images)
