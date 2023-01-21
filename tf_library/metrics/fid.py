import numpy as np
import tensorflow as tf
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input


def calculate_fid(generator, dataset):
    model_inception_v3 = InceptionV3(include_top=False, pooling='avg', input_shape=(256, 256, 3))
    test_input = []
    test_output = []
    for example_input, example_target in dataset.take(len(list(dataset))):
        prediction = generator(example_input, training=True)
        test_input.append(example_target)
        test_output.append(prediction)

    test_input = tf.concat(test_input, 0)
    test_output = tf.concat(test_output, 0)

    images1 = preprocess_input(test_input)
    images2 = preprocess_input(test_output)

    act1 = model_inception_v3.predict(images1)
    act2 = model_inception_v3.predict(images2)
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid