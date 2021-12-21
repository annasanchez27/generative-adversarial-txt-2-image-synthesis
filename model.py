import tensorflow as tf
import tensorflow.keras.layers as tfkl
import tensorflow.keras.activations as tfa


class DCGenerator(tfkl.Layer):
    def __init__(self):
        super(DCGenerator, self).__init__()
        self.embedding_layer = tf.keras.Sequential([
                tfkl.Dense(128, activation=None),
                tfkl.LeakyReLU(),
        ])
        self.input_layer = tf.keras.Sequential([
                tfkl.Dense(2048, use_bias=False),
                tfkl.BatchNormalization(),
                tfkl.ReLU(),
                tfkl.Reshape((-1, 4, 4, 512)),
        ])
        self.residual_layer1 = tf.keras.Sequential([
                tfkl.Conv2D(
                    128, (1, 1), strides=(1, 1), padding="valid", use_bias=False
                ),
                tfkl.BatchNormalization(),
                tfkl.ReLU(),
                tfkl.Conv2D(
                    128, (3, 3), strides=(1, 1), padding="same", use_bias=False
                ),
                tfkl.BatchNormalization(),
                tfkl.ReLU(),
                tfkl.Conv2D(
                    512, (3, 3), strides=(1, 1), padding="same", use_bias=False
                ),
                tfkl.BatchNormalization(),

        ])
        self.inter_layer = tf.keras.Sequential([
                tfkl.Conv2DTranspose(
                    256, (4, 4), strides=(2, 2), padding="same", use_bias=False
                ),
                tfkl.Conv2D(
                    256, (3, 3), strides=(1, 1), padding="same", use_bias=False
                ),
                tfkl.BatchNormalization(),
            ])

        self.residual_layer2 = tf.keras.Sequential([
                tfkl.Conv2D(
                    64, (1, 1), strides=(1, 1), padding="valid", use_bias=False
                ),
                tfkl.BatchNormalization(),
                tfkl.ReLU(),
                tfkl.Conv2D(
                    64, (3, 3), strides=(1, 1), padding="same", use_bias=False
                ),
                tfkl.BatchNormalization(),
                tfkl.ReLU(),
                tfkl.Conv2D(
                    256, (3, 3), strides=(1, 1), padding="same", use_bias=False
                ),
                tfkl.BatchNormalization(),
            ])
        self.last_layer = tf.keras.Sequential([
                tfkl.Conv2DTranspose(
                    128, (4, 4), strides=(2, 2), padding="same", use_bias=False
                ),
                tfkl.Conv2D(
                    128, (3, 3), strides=(1, 1), padding="same", use_bias=False
                ),
                tfkl.BatchNormalization(),
                tfkl.ReLU(),
                tfkl.Conv2DTranspose(
                    64, (4, 4), strides=(2, 2), padding="same", use_bias=False
                ),
                tfkl.Conv2D(
                    64, (3, 3), strides=(1, 1), padding="same", use_bias=False
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
        x = tfkl.relu(x)
        out2 = self.inter_layer(x)
        x = self.residual_layer2(out2)
        x = tf.add(out2, x)
        x = tfkl.relu(x)
        x = self.last_layer(x)
        return x


class DCDiscriminator(tfkl.Layer):
    def __init__(self):
        super(DCDiscriminator, self).__init__()
        self.input_layer = tf.keras.Sequential([
            tfkl.Conv2D(
                filters=64, kernel_size=(4, 4), strides=(2, 2), padding="same"
            ),
            tfkl.LeakyReLU(0.2),
            tfkl.Conv2D(filters=128, kernel_size=(4, 4), strides=(2, 2), padding="same"),
            tfkl.BatchNormalization(),
            tfkl.LeakyReLU(0.2),
            tfkl.Conv2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding="same"),
            tfkl.BatchNormalization(),
            tfkl.LeakyReLU(0.2),
            tfkl.Conv2D(filters=512, kernel_size=(4, 4), strides=(2, 2), padding="same"),
            tfkl.BatchNormalization(),
            tfkl.LeakyReLU(0.2),
        ])

        # Residual layer
        self.residual_layer = tf.keras.Sequential([
            tfkl.Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding="same"),
            tfkl.BatchNormalization(),
            tfkl.LeakyReLU(0.2),
            tfkl.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="same"),
            tfkl.BatchNormalization(),
            tfkl.LeakyReLU(0.2),
            tfkl.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="same"),
            tfkl.BatchNormalization(),
        ])
        self.LeakyRelu = tfkl.LeakyReLU(0.2)
        self.embedding_layer = tf.keras.Sequential([
                tfkl.Dense(128, activation=None),
                tfkl.LeakyReLU(),
        ])
        self.output_layer = tf.keras.Sequential([
                tfkl.Conv2D(filters=512, kernel_size=(1, 1), strides=(1, 1), padding="valid"),
                tfkl.BatchNormalization(),
                tfkl.LeakyReLU(0.2),
                tfkl.Conv2D(filters=1, kernel_size=(2, 2), strides=(2, 2), padding="valid"),
            ])
        self.Sigmoid = tfkl.Activation(tfa.sigmoid),

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
    def __init__(self):
        super(GAN, self).__init__()

        self.noise_dim = 100
        self.batch_size = 16

        self.generator = DCGenerator()
        self.discriminator = DCDiscriminator()

        # Set up optimizers for both models.
        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def discriminator_loss(self, actual_output, generated_output):
        real_loss = self.cross_entropy(tf.ones_like(actual_output), actual_output)
        generated_loss = self.cross_entropy(
            tf.zeros_like(generated_output), generated_output
        )
        total_loss = real_loss + generated_loss

        return total_loss

    def generator_loss(self, generated_output):
        return self.cross_entropy(tf.ones_like(generated_output), generated_output)

    def generate_sample(self, embed):
        noise = tf.random.normal([self.batch_size, self.noise_dim])
        generated_sample = self.generator(noise, embed)
        return generated_sample

    def train_step(self, x, embed):
        noise = tf.random.normal([self.batch_size, self.noise_dim])

        with tf.GradientTape() as discriminator_tape, tf.GradientTape() as generator_tape:
            generated_samples = self.generator(noise, embed)

            real_output = self.discriminator(x, embed)
            fake_output = self.discriminator(generated_samples, embed)

            discriminator_loss = self.discriminator_loss(real_output, fake_output)
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

    def call(self, x, embed):
        return self.train_step(x, embed)
