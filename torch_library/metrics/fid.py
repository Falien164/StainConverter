from torch.nn.functional import adaptive_avg_pool2d
from torch_library.metrics.inception_model import InceptionV3
from torch.autograd import Variable
from scipy import linalg

import numpy as np
import torch


class FID:
    def __init__(self, cuda=True):
        self.model = InceptionV3()
        self.cuda = cuda
        if cuda:
            self.model = self.model.cuda()

    def calculate_fid(self, generator, train_dataset):
        images_fake = []
        images_real = []
        for image in train_dataset:
            images_fake.append(generator(Variable(image["HE"]).cuda()).cpu())
            images_real.append(image["PAS"])

        mu_1, std_1 = self.calculate_activation_statistics(torch.concat(images_real), self.model, cuda=self.cuda)
        mu_2, std_2 = self.calculate_activation_statistics(torch.concat(images_fake), self.model, cuda=self.cuda)

        """get fretched distance"""
        fid_value = self.calculate_frechet_distance(mu_1, std_1, mu_2, std_2)
        return fid_value

    @staticmethod
    def calculate_activation_statistics(images, model, dims=2048, cuda=True):
        model.eval()
        act = np.empty((len(images), dims))

        batch = images.cuda() if cuda else images

        pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        act = pred.cpu().data.numpy().reshape(pred.size(0), -1)

        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)
        return mu, sigma

    @staticmethod
    def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
        """

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1) +
                np.trace(sigma2) - 2 * tr_covmean)
