import tensorflow as tf

class Pix2Pix:
    def __init__(self, generator, discriminator):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator

        self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.alpha = 100

    def predict(self, input_img):
        return self.generator(input_img, training=True)

    @tf.function
    def train_step(self, input_img, target_img):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.generator(input_img, training=True)

            disc_real_output = self.discriminator([input_img, target_img], training=True)
            disc_generated_output = self.discriminator([input_img, gen_output], training=True)

            gen_total_loss, gen_gan_loss, gen_l1_loss = self.generator_loss(disc_generated_output, gen_output,
                                                                            target_img)
            disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(gen_total_loss, self.generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                         self.discriminator.trainable_variables))

    def generator_loss(self, disc_generated_output, gen_output, target):
        gan_loss = self.loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

        # Mean absolute error
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
        total_gen_loss = gan_loss + (self.alpha * l1_loss)

        return total_gen_loss, gan_loss, l1_loss

    def discriminator_loss(self, target_img, generated_img, modifier=1.0):
        real_loss = self.loss_object(tf.ones_like(target_img), target_img)
        generated_loss = self.loss_object(tf.zeros_like(generated_img), generated_img)
        total_disc_loss = real_loss + generated_loss
        return total_disc_loss * modifier
