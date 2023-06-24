import os
import numpy as np

from tf_library.metrics.fid import calculate_fid
import matplotlib.pyplot as plt


class Trainer:

    def __init__(self, results_filename):
        self.results_filename = results_filename

    def train(self, model, train_dataset, test_dataset):
        self.best_fid = np.inf
        for step, (input_img, target_img) in train_dataset.repeat().take(int(os.environ['N_STEPS'])).enumerate():
            model.train_step(input_img, target_img)
            # TODO step %1000
            # TODO za każdym razem bierz ten same obrazki
            fid = calculate_fid(model.generator, test_dataset)
            self.save_fid(fid, step)
            if fid < self.best_fid:
                self.save_model(model.generator)
                self.best_fid = fid

    def save_fid(self, fid, step):
        with open(self.results_filename + 'fid.txt', 'a') as file:
            file.write(f"FID in step {step} = {fid} \n")

    def save_model(self, model):
        model.save_weights(self.results_filename + 'gen.h5')
        # TODO: save all models

    def generate_comparative_images(self, model, test_dataset):
        for idx, (input_img, target_img) in enumerate(test_dataset.take(len(test_dataset))):
            generate_image(model, input_img, target_img, self.results_filename + "/generated_images/", idx)



def generate_image(model, test_input, target, filename, number):
    prediction = model.predict(test_input)

    plt.figure(figsize=(12, 12))

    display_list = [test_input[0], target[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(title[i])
        plt.imshow(display_list[i] * 0.5 + 0.5)  # getting the pixel values between [0, 1] to plot it.
        plt.axis('off')
    plt.savefig(filename + str(number) + '.png')
