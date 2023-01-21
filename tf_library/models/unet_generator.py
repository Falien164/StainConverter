import tensorflow as tf
from tf_library.models.layers import Layers


class UnetGenerator(Layers):

    def __call__(self):
        return self.create_generator()

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
