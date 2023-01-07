import tensorflow as tf


class GAN:
    def __init__(self):
        self.alpha = 100
        self.loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def down_sample(self, filters, size, stride=2, apply_batch_norm=True):
        initializer = tf.random_normal_initializer(0., 0.02)
        result = tf.keras.Sequential()
        result.add(tf.keras.layers.Conv2D(filters, size, strides=stride, padding='same',
                                          kernel_initializer=initializer, use_bias=False))
        if apply_batch_norm:
            result.add(tf.keras.layers.BatchNormalization())
        result.add(tf.keras.layers.LeakyReLU())
        return result

    def up_sample(self, filters, size, stride=2, apply_dropout=False):
        initializer = tf.random_normal_initializer(0., 0.02)
        result = tf.keras.Sequential()
        result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=stride,
                                                   padding='same',
                                                   kernel_initializer=initializer,
                                                   use_bias=False))
        result.add(tf.keras.layers.BatchNormalization())
        if apply_dropout:
            result.add(tf.keras.layers.Dropout(0.5))
        result.add(tf.keras.layers.ReLU())
        return result

    def res_block(self, filters):
        initializer = tf.random_normal_initializer(0., 0.02)
        result = tf.keras.Sequential()
        result.add(tf.keras.layers.Conv2DTranspose(filters, 3, strides=1,
                                                   padding='same',
                                                   kernel_initializer=initializer,
                                                   use_bias=False))
        result.add(tf.keras.layers.BatchNormalization())
        result.add(tf.keras.layers.ReLU())
        result.add(tf.keras.layers.Conv2DTranspose(filters, 3, strides=1,
                                                   padding='same',
                                                   kernel_initializer=initializer,
                                                   use_bias=False))
        result.add(tf.keras.layers.BatchNormalization())
        return result

    def discriminator_loss(self, real, generated, modifier=1.0):
        real_loss = self.loss_obj(tf.ones_like(real), real)
        generated_loss = self.loss_obj(tf.zeros_like(generated), generated)
        total_disc_loss = real_loss + generated_loss
        return total_disc_loss * modifier

    def calc_cycle_loss(self, real_image, cycled_image):
        return self.alpha * tf.reduce_mean(tf.abs(real_image - cycled_image))

    def identity_loss(self, real_image, same_image):
        return self.alpha * 0.5 * tf.reduce_mean(tf.abs(real_image - same_image))