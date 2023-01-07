from data_loader import DataLoader
from cyclegan import CycleGAN
from pix2pix_old import Pix2Pix
from pix2pix import UnetGenerator

import tensorflow as tf

loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)
def generator_loss( disc_generated_output, gen_output, target):
    gan_loss = loss_obj(tf.ones_like(disc_generated_output), disc_generated_output)

    # Mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    LAMBDA = 100
    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss, gan_loss, l1_loss
@tf.function
def train_step(input_image, target, generator, discriminator, generator_loss,discriminator_loss, generator_optimizer, discriminator_optimizer):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                     discriminator.trainable_variables))


if __name__ == "__main__":
    train_he_images_path = '../output/HE_s0/'
    train_pas_images_path = '../output/PAS_s0/'
    test_he_images_path = '../output/HE_s0/test_HE/'
    test_pas_images_path = '../output/PAS_s0/test_PAS/'

    #TODO: Validate if paths are correct

    data_loader = DataLoader(train_he_images_path=train_he_images_path, train_pas_images_path=train_pas_images_path,
                             test_he_images_path=test_he_images_path, test_pas_images_path=test_pas_images_path)
    train_dataset, test_dataset = data_loader.load_dataset()
    # data_loader.display_sample_pair(train_dataset)

    model = Pix2Pix()
    generator = model.create_generator()
    discriminator = model.create_discriminator()

    def generate_images(model, test_input, target, step, sample):
        import matplotlib.pyplot as plt
        prediction = model.predict(test_input)

        plt.figure(figsize=(12, 12))

        display_list = [test_input[0], target[0], prediction[0]]
        title = ['Input Image', 'Ground Truth', 'Predicted Image']

        for i in range(3):
            plt.subplot(1, 3, i + 1)
            plt.title(title[i])
            # getting the pixel values between [0, 1] to plot it.
            plt.imshow(display_list[i] * 0.5 + 0.5)
            plt.axis('off')
        # plt.savefig(picture_filename + str(step) + "k_" + str(sample) + '.png')
        plt.show()
    # for example_input, example_target in test_dataset.take(1):
    #     generate_images(model.generator_g, example_input, example_target, step=0, sample=0)



    for step, (input_image, target) in train_dataset.repeat().take(10).enumerate():
        train_step(input_image, target, model.generator, model.discriminator, model.discriminator_loss,model.generator_loss, model.generator_optimizer ,model.discriminator_loss)

    for example_input, example_target in test_dataset.take(1):
        generate_images(model, example_input, example_target, step=0, sample=0)

    model_selector()