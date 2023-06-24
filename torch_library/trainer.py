import numpy as np
import torch

from torch_library.metrics.fid import FID
import matplotlib.pyplot as plt


class Trainer:

    def __init__(self, results_filename):
        self.results_filename = results_filename

    def train(self, model, train_dataset, test_dataset):
        self.best_fid = np.inf
        for epoch in range(2):
            for i, batch in enumerate(train_dataset):
                model.train_step(batch)
                # TODO step %1000
                # TODO za każdym razem bierz ten same obrazki
            fid = FID(cuda=False).calculate_fid(model.generator, test_dataset)
            self.save_fid(fid, epoch)
            if fid < self.best_fid:
                self.save_model(model.generator)
                self.best_fid = fid

    def save_fid(self, fid, step):
        with open(self.results_filename + 'fid.txt', 'a') as file:
            file.write(f"FID in step {step} = {fid} \n")

    def save_model(self, model):
        torch.save(model.state_dict(), self.results_filename + 'gen.h5')
        # TODO: save all models

    def generate_comparative_images(self, model, test_dataset):

        for idx, batch in enumerate(test_dataset):
            real_a = batch['HE'].cpu().numpy()
            real_b = batch['PAS'].cpu().numpy()
            generate_image(model, batch['HE'], batch['PAS'], self.results_filename + "/generated_images/", idx)


def generate_image(model, test_input, target, filename, number):
    prediction = model.predict(test_input.cpu().numpy())
    prediction = np.transpose(prediction, (0, 2, 3, 1))
    test_input = np.transpose(test_input, (0, 2, 3, 1))
    target = np.transpose(target, (0, 2, 3, 1))

    plt.figure(figsize=(12, 12))

    display_list = [test_input[0], target[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(title[i])
        plt.imshow(display_list[i] * 0.5 + 0.5)  # getting the pixel values between [0, 1] to plot it.
        plt.axis('off')
    plt.savefig(filename + str(number) + '.png')
