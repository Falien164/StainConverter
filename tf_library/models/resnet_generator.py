import tensorflow as tf
from tf_library.models.layers import Layers


class ResnetGenerator(Layers):

    def __call__(self):
        return self.create_generator()

    def create_generator(self):
        inputs = tf.keras.layers.Input(shape=[256, 256, 3])

        down_stack = [
            self.down_sample(filters=64, size=7, strides=1),  # (batch_size, 256x256x64)
            self.down_sample(filters=128, size=3, strides=2),  # (batch_size, 128x128x128)
            self.down_sample(filters=256, size=3, strides=2)  # (batch_size, 256x64x64)
        ]

        res_stack = [
            self.res_block(256) for i in range(0, 9)
        ]

        up_stack = [
            self.up_sample(filters=128, size=3, strides=2),  # (batch_size, 128x128x128)
            self.up_sample(filters=64, size=3, strides=1)  # (batch_size, 256x256x64)
            # upsample(filters=3, size,=7 stride =1, padding = 3), # (batch_size, 256x256x3)
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = tf.keras.layers.Conv2DTranspose(3, 4,
                                               strides=2,
                                               padding='same',
                                               kernel_initializer=initializer,
                                               activation='tanh')  # (batch_size, 256, 256, 3)

        x = inputs

        # Downsampling through the model
        for down in down_stack:
            x = down(x)

        for res in res_stack:
            res_skip = x
            x = res(x)
            x = tf.keras.layers.Concatenate()([x, res_skip])

        # Upsampling and establishing the skip connections
        for up in up_stack:
            x = up(x)

        x = last(x)

        return tf.keras.Model(inputs=inputs, outputs=x)
