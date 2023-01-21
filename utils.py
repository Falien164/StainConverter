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
        plt.imshow(display_list[i] * 0.5 + 0.5) # getting the pixel values between [0, 1] to plot it.
        plt.axis('off')
    # if save:
    # TODO: read_from_env_file
    plt.savefig(filename + str(number) + '.png')
    # plt.show()


def save_model(model, filename):
    model.save_weights(filename)


def load_model(model, path):
    model.load_weights(path)

def create_metric_file():
    with open(metrics_result_filename, 'w') as f:
        header = ['mse', 'rmse', 'psnr', 'ssim', 'uqi', 'ergas', 'scc', 'rase', 'sam', 'vifp','euc_dist_LBP', 'cosine_dist_LBP','jaccard_dist_LBP', 'step]']
        writer = csv.writer(f)
        writer.writerow(header)

def save_metrics_to_file(step):
    with open(metrics_result_filename, 'a') as f:
        results = list(countMetrics())
        results.append(step)
        writer = csv.writer(f)
        writer.writerow(results)