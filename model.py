import tensorflow as tf
import tensorflow.keras.activations as tfa
import tensorflow.keras.layers as tfkl


class DCGenerator(tfkl.Layer):
    def __init__(self):
        super(DCGenerator, self).__init__()
        self.output_size = 64
        self.gf_dim = 128
        self.s16 = self.output_size // 16
        self.ReLu = tfkl.Activation(tfa.relu)
        self.initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=0.02)
        self.batch_initializer = tf.keras.initializers.RandomNormal(mean=1., stddev=0.02)

        self.embedding_layer = tf.keras.Sequential([
            tfkl.Dense(128, activation=None),
        ])
        self.input_layer = tf.keras.Sequential([
            tfkl.Dense(self.gf_dim * 8 * self.s16 * self.s16, use_bias=False, kernel_initializer=self.initializer),
            tfkl.BatchNormalization(gamma_initializer=self.batch_initializer),
            tfkl.Reshape((4, 4, self.gf_dim * 8)),
        ])
        self.residual_layer1 = tf.keras.Sequential([
            tfkl.Conv2D(
                self.gf_dim * 2, (1, 1), strides=(1, 1), padding="valid", use_bias=False,
                kernel_initializer=self.initializer
            ),
            tfkl.BatchNormalization(gamma_initializer=self.batch_initializer),
            tfkl.ReLU(),
            tfkl.Conv2D(
                self.gf_dim * 2, (3, 3), strides=(1, 1), padding="same", use_bias=False,
                kernel_initializer=self.initializer
            ),
            tfkl.BatchNormalization(gamma_initializer=self.batch_initializer),
            tfkl.ReLU(),
            tfkl.Conv2D(
                self.gf_dim * 8, (3, 3), strides=(1, 1), padding="same", use_bias=False,
                kernel_initializer=self.initializer
            ),
            tfkl.BatchNormalization(gamma_initializer=self.batch_initializer),

        ])
        self.inter_layer = tf.keras.Sequential([
            tfkl.Conv2DTranspose(
                self.gf_dim * 4, (4, 4), strides=(2, 2), padding="same", use_bias=False,
                kernel_initializer=self.initializer
            ),
            tfkl.Conv2D(
                self.gf_dim * 4, (3, 3), strides=(1, 1), padding="same", use_bias=False,
                kernel_initializer=self.initializer
            ),
            tfkl.BatchNormalization(gamma_initializer=self.batch_initializer),
        ])

        self.residual_layer2 = tf.keras.Sequential([
            tfkl.Conv2D(
                self.gf_dim, (1, 1), strides=(1, 1), padding="valid", use_bias=False,
                kernel_initializer=self.initializer
            ),
            tfkl.BatchNormalization(gamma_initializer=self.batch_initializer),
            tfkl.ReLU(),
            tfkl.Conv2D(
                self.gf_dim, (3, 3), strides=(1, 1), padding="same", use_bias=False, kernel_initializer=self.initializer
            ),
            tfkl.BatchNormalization(gamma_initializer=self.batch_initializer),
            tfkl.ReLU(),
            tfkl.Conv2D(
                self.gf_dim * 4, (3, 3), strides=(1, 1), padding="same", use_bias=False,
                kernel_initializer=self.initializer
            ),
            tfkl.BatchNormalization(gamma_initializer=self.batch_initializer),
        ])
        self.last_layer = tf.keras.Sequential([
            tfkl.Conv2DTranspose(
                self.gf_dim * 2, (4, 4), strides=(2, 2), padding="same", use_bias=False,
                kernel_initializer=self.initializer
            ),
            tfkl.Conv2D(
                self.gf_dim * 2, (3, 3), strides=(1, 1), padding="same", use_bias=False,
                kernel_initializer=self.initializer
            ),
            tfkl.BatchNormalization(gamma_initializer=self.batch_initializer),
            tfkl.ReLU(),
            tfkl.Conv2DTranspose(
                self.gf_dim, (4, 4), strides=(2, 2), padding="same", use_bias=False, kernel_initializer=self.initializer
            ),
            tfkl.Conv2D(
                self.gf_dim, (3, 3), strides=(1, 1), padding="same", use_bias=False, kernel_initializer=self.initializer
            ),
            tfkl.BatchNormalization(gamma_initializer=self.batch_initializer),
            tfkl.ReLU(),
            tfkl.Conv2DTranspose(
                3, (4, 4), strides=(2, 2), padding="same", use_bias=False, kernel_initializer=self.initializer
            ),
            tfkl.Conv2D(
                3, (3, 3), strides=(1, 1), padding="same", use_bias=False, kernel_initializer=self.initializer
            ),
            tfkl.Activation(tfa.tanh),
        ])

    def call(self, z, embed, training=True):
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
        self.output_size = 64
        self.df_dim = 64
        self.s16 = self.output_size // 16
        self.initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=0.02)
        self.batch_initializer = tf.keras.initializers.RandomNormal(mean=1., stddev=0.02)
        self.input_layer = tf.keras.Sequential([
            tfkl.Conv2D(
                filters=self.df_dim, kernel_size=(4, 4), strides=(2, 2), padding="same"
            ),
            tfkl.LeakyReLU(0.2),
            tfkl.Conv2D(filters=self.df_dim * 2, kernel_size=(4, 4), strides=(2, 2), padding="same"),
            tfkl.BatchNormalization(gamma_initializer=self.batch_initializer),
            tfkl.LeakyReLU(0.2),
            tfkl.Conv2D(filters=self.df_dim * 4, kernel_size=(4, 4), strides=(2, 2), padding="same"),
            tfkl.BatchNormalization(gamma_initializer=self.batch_initializer),
            tfkl.LeakyReLU(0.2),
            tfkl.Conv2D(filters=self.df_dim * 8, kernel_size=(4, 4), strides=(2, 2), padding="same"),
            tfkl.BatchNormalization(gamma_initializer=self.batch_initializer),
            tfkl.LeakyReLU(0.2),
        ])

        # Residual layer
        self.residual_layer = tf.keras.Sequential([
            tfkl.Conv2D(filters=self.df_dim * 2, kernel_size=(1, 1), strides=(1, 1), padding="same"),
            tfkl.BatchNormalization(gamma_initializer=self.batch_initializer),
            tfkl.LeakyReLU(0.2),
            tfkl.Conv2D(filters=self.df_dim * 2, kernel_size=(3, 3), strides=(1, 1), padding="same"),
            tfkl.BatchNormalization(gamma_initializer=self.batch_initializer),
            tfkl.LeakyReLU(0.2),
            tfkl.Conv2D(filters=self.df_dim * 8, kernel_size=(3, 3), strides=(1, 1), padding="same"),
            tfkl.BatchNormalization(gamma_initializer=self.batch_initializer),
        ])
        self.LeakyRelu = tfkl.LeakyReLU(0.2)
        self.embedding_layer = tf.keras.Sequential([
            tfkl.Dense(128, activation=None),
            tfkl.LeakyReLU(0.2),
        ])
        self.output_layer = tf.keras.Sequential([
            tfkl.Conv2D(filters=self.df_dim * 8, kernel_size=(1, 1), strides=(1, 1), padding="valid"),
            tfkl.BatchNormalization(gamma_initializer=self.batch_initializer),
            tfkl.LeakyReLU(0.2),
            tfkl.Conv2D(filters=1, kernel_size=(self.s16, self.s16), strides=(self.s16, self.s16), padding="valid"),
        ])
        self.Sigmoid = tfkl.Activation(tfa.sigmoid)

    def call(self, x, embed, training=True):
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
        self.generator_optimizer = tf.keras.optimizers.Adam(config['learning_rate'], beta_1=config['momentum'])
        self.discriminator_optimizer = tf.keras.optimizers.Adam(config['learning_rate'], beta_1=config['momentum'])

        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def discriminator_loss(self, actual_output, generated_output, mismatch_output):
        real_loss = self.cross_entropy(tf.fill(dims=tf.shape(actual_output), value=0.9), actual_output)
        generated_loss = self.cross_entropy(
            tf.zeros_like(generated_output), generated_output
        )
        mismatch_loss = self.cross_entropy(
            tf.zeros_like(mismatch_output), mismatch_output
        )
        total_loss = (mismatch_loss + generated_loss) / 2 + real_loss
        return total_loss

    def generator_loss(self, generated_output, fake_int_output):
        generated_loss = self.cross_entropy(tf.ones_like(generated_output), generated_output)
        generated_loss_int = self.cross_entropy(tf.ones_like(fake_int_output), fake_int_output)
        return generated_loss + generated_loss_int

    def generate_sample(self, embed, training=False):
        noise = tf.random.normal([self.batch_size, self.noise_dim])
        generated_sample = self.generator(noise, embed, training=training)
        return generated_sample

    def train_step(self, x, embed, wrong_images, embed_int, training, epoch_int=1):
        noise = tf.random.normal([self.batch_size, self.noise_dim])
        noise_int = tf.random.normal([self.batch_size, self.noise_dim])
        std_dev = 1.0 - epoch_int/600

        x = x + tf.random.normal(tf.shape(x), mean=0.0, stddev=std_dev)

        with tf.GradientTape() as discriminator_tape, tf.GradientTape() as generator_tape:
            generated_samples = self.generator(noise, embed, training=training)
            generated_samples_int = self.generator(noise_int, embed_int, training=training)
            generated_samples = generated_samples + tf.random.normal(tf.shape(generated_samples), mean=0.0, stddev=std_dev)
            generated_samples_int = generated_samples_int + tf.random.normal(tf.shape(generated_samples_int), mean=0.0, stddev=std_dev)

            _, real_output = self.discriminator(x, embed, training=training)
            _, fake_output = self.discriminator(generated_samples, embed, training=training)
            _, fake_int_output = self.discriminator(generated_samples_int, embed_int, training=training)
            _, mismatch_output = self.discriminator(wrong_images, embed, training=training)

            discriminator_loss = self.discriminator_loss(real_output, fake_output, mismatch_output)
            generator_loss = self.generator_loss(fake_output, fake_int_output)

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

    def call(self, x, embed, wrong_images, embed_int, training=True, epoch_int=1):
        return self.train_step(x, embed, wrong_images, embed_int, training=training, epoch_int=epoch_int)
