import tensorflow as tf
from gan import GAN


class Pix2Pix(GAN):
    def __init__(self):
        super().__init__()
        self.generator = self.create_generator()
        self.discriminator = self.create_discriminator()
        self.generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def get_generator(self):
        return self.generator

    def predict(self, input_image):
        return self.generator(input_image)

    def create_generator(self):
        down_stack = [
            self.down_sample(64, 4, apply_batch_norm=False),  # (batch_size, 128, 128, 64)
            self.down_sample(128, 4),  # (batch_size, 64, 64, 128)
            self.down_sample(256, 4),  # (batch_size, 32, 32, 256)
            self.down_sample(512, 4),  # (batch_size, 16, 16, 512)
            self.down_sample(512, 4),  # (batch_size, 8, 8, 512)
            self.down_sample(512, 4),  # (batch_size, 4, 4, 512)
            self.down_sample(512, 4),  # (batch_size, 2, 2, 512)
            self.down_sample(512, 4),  # (batch_size, 1, 1, 512)
        ]

        up_stack = [
            self.up_sample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
            self.up_sample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
            self.up_sample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
            self.up_sample(512, 4),  # (batch_size, 16, 16, 1024)
            self.up_sample(256, 4),  # (batch_size, 32, 32, 512)
            self.up_sample(128, 4),  # (batch_size, 64, 64, 256)
            self.up_sample(64, 4),  # (batch_size, 128, 128, 128)
        ]

        inputs = tf.keras.layers.Input(shape=[256, 256, 3])
        initializer = tf.random_normal_initializer(0., 0.02)
        last = tf.keras.layers.Conv2DTranspose(3, 4,
                                               strides=2,
                                               padding='same',
                                               kernel_initializer=initializer,
                                               activation='tanh')  # (batch_size, 256, 256, 3)

        x = inputs
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])
        for up, skip in zip(up_stack, skips):
            x = up(x)
            x = tf.keras.layers.Concatenate()([x, skip])
        x = last(x)
        return tf.keras.Model(inputs=inputs, outputs=x)

    def create_discriminator(self):
        initializer = tf.random_normal_initializer(0., 0.02)

        inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
        tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')

        x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, channels*2)

        down1 = self.down_sample(64, 4, apply_batch_norm=False)(x)  # (batch_size, 128, 128, 64)
        down2 = self.down_sample(128, 4)(down1)  # (batch_size, 64, 64, 128)
        down3 = self.down_sample(256, 4)(down2)  # (batch_size, 32, 32, 256)

        zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
        conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                      kernel_initializer=initializer,
                                      use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512)

        batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

        leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

        zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)

        last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                      kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)

        return tf.keras.Model(inputs=[inp, tar], outputs=last)

    def generator_loss(self, disc_generated_output, gen_output, target):
        gan_loss = self.loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

        # Mean absolute error
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
        LAMBDA = 100
        total_gen_loss = gan_loss + (LAMBDA * l1_loss)

        return total_gen_loss, gan_loss, l1_loss

    def train(self, input_image, target):
        self.train_step(input_image, target)
