import matplotlib.pyplot as plt
import csv


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


