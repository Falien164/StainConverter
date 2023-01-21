import tensorflow as tf
from tf_library.models.layers import Layers


class PatchGAN(Layers):

    def __call__(self, type_of_gan: str, size='30x30'):
        # TODO: create method for cyclegan and for pix2pix
        if size == '30x30':
            return self._create_discriminator_30x30(type_of_gan)
        elif size == '16x16':
            return self._create_discriminator_16x16(type_of_gan)

    def _get_discriminator_16x16(self, x):
        initializer = tf.random_normal_initializer(0., 0.02)
        down1 = self.down_sample(filters=64, size=4, strides=2, apply_batch_norm=False)(x)  # (batch_size, 128x128x64)
        down2 = self.down_sample(filters=128, size=4, strides=2)(down1)  # (batch_size, 64x64x128)
        down3 = self.down_sample(filters=256, size=4, strides=2)(down2)  # (batch_size, 32x32x256)
        down4 = self.down_sample(filters=512, size=4, strides=2)(down3)  # (batch_size, 31x31x512)
        down5 = self.down_sample(filters=512, size=4, strides=1)(down4)  # (batch_size, 30x30x1)
        last = tf.keras.layers.Conv2D(1, 4, strides=1, padding="same",
                                      kernel_initializer=initializer)(down5)  # (batch_size, 16, 16, 1)
        return last

    def _get_discriminator_30x30(self, x):
        initializer = tf.random_normal_initializer(0., 0.02)
        down1 = self.down_sample(64, 4, apply_batch_norm=False)(x)  # (batch_size, 128, 128, 64)
        down2 = self.down_sample(128, 4)(down1)  # (batch_size, 64, 64, 128)
        down3 = self.down_sample(256, 4)(down2)  # (batch_size, 32, 32, 256)
        # TODO: zero padding? use down_sample instead?
        zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
        conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                      kernel_initializer=initializer,
                                      use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512)

        batchnorm = tf.keras.layers.BatchNormalization()(conv)

        leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm)

        zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)

        last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                      kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)
        return last

    def _create_discriminator_16x16(self, type_of_gan: str):
        inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
        if 'cyclegan' in type_of_gan:
            x = inp
            last = self._get_discriminator_16x16(x)
            return tf.keras.Model(inputs=inp, outputs=last)
        else:  # 'pix2pix' in type_of_gan:
            tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')
            x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, channels*2)
            last = self._get_discriminator_16x16(x)
            return tf.keras.Model(inputs=[inp, tar], outputs=last)

    def _create_discriminator_30x30(self, type_of_gan: str):
        inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
        if 'cyclegan' in type_of_gan:
            x = inp
            last = self._get_discriminator_30x30(x)
            return tf.keras.Model(inputs=inp, outputs=last)
        else:  # 'pix2pix' in type_of_gan:
            tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')
            x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, channels*2)
            last = self._get_discriminator_30x30(x)
            return tf.keras.Model(inputs=[inp, tar], outputs=last)