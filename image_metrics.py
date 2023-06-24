from sewar.full_ref import mse, rmse, uqi, sam, vifp
from skimage.metrics import structural_similarity as ssim

from skimage.feature import local_binary_pattern
from skimage.color import rgb2gray

import numpy as np
import pandas as pd


class ImageMetrics:

    def __init__(self, n_images):
        self.MSE = 0
        self.RMSE = 0
        self.PSNR = 0
        self.SSIM = 0
        self.UQI = 0
        self.SAM = 0
        self.VIFP = 0

        self.euc_dist_LBP = 0
        self.cosine_dist_LBP = 0
        self.n_images = n_images

    def calculate_metrics_for_one_img(self, inp, target):
        self.MSE += mse(target, inp)
        self.RMSE += rmse(target, inp)
        self.PSNR += self.calculate_psnr(target, inp)
        self.SSIM += ssim(target, inp, multichannel=True)
        self.UQI += uqi(target, inp)
        self.SAM += sam(target, inp)
        self.VIFP += vifp(target, inp)

        target_LBP, pred_LBP = self.calculate_lbp(target, inp)
        target_LBP, pred_LBP = self.flatten_images(target_LBP, pred_LBP)

        self.euc_dist_LBP += np.sqrt(self.calculate_distance(target_LBP, pred_LBP))
        self.cosine_dist_LBP += np.dot(target_LBP, pred_LBP) / (np.linalg.norm(target_LBP) * np.linalg.norm(pred_LBP))

    def normalize_results(self):
        MSE = self.MSE / self.n_images
        RMSE = self.RMSE / self.n_images
        PSNR = self.PSNR / self.n_images
        SSIM = self.SSIM / self.n_images
        UQI = self.UQI / self.n_images
        SAM = self.SAM / self.n_images
        VIFP = self.VIFP / self.n_images

        euc_dist_LBP = self.euc_dist_LBP / self.n_images
        cosine_dist_LBP = self.cosine_dist_LBP / self.n_images

        return MSE, RMSE, PSNR, SSIM, UQI, SAM, VIFP, euc_dist_LBP, cosine_dist_LBP

    def count_metrics_for_tensorflow(self, model, test_dataset):

        for example_input, example_target in test_dataset.take(self.n_images):
            prediction = model.predict(example_input)
            target_np = example_target.numpy()[0]
            pred_np = prediction.numpy()[0]
            self.calculate_metrics_for_one_img(pred_np, target_np)

        MSE, RMSE, PSNR, SSIM, UQI, SAM, VIFP, euc_dist_LBP, cosine_dist_LBP = self.normalize_results()

        return MSE, RMSE, PSNR, SSIM, UQI, SAM, VIFP, euc_dist_LBP, cosine_dist_LBP

    def count_metrics_for_torch(self, model, test_dataset):

        for batch in test_dataset:
            prediction = model.predict(batch['HE'])

            prediction = np.transpose(prediction, (0, 2, 3, 1))
            target = np.transpose(batch['PAS'].cpu().numpy(), (0, 2, 3, 1))

            self.calculate_metrics_for_one_img(prediction[0], target[0])

        MSE, RMSE, PSNR, SSIM, UQI, SAM, VIFP, euc_dist_LBP, cosine_dist_LBP = self.normalize_results()

        return MSE, RMSE, PSNR, SSIM, UQI, SAM, VIFP, euc_dist_LBP, cosine_dist_LBP

    @staticmethod
    def calculate_distance(i1, i2):
        return np.sum((i1 - i2) ** 2)

    @staticmethod
    def flatten_images(target_img, predicted_img):
        flat_target_img = target_img.flatten()
        flat_predicted_img = predicted_img.flatten()
        return flat_target_img, flat_predicted_img

    @staticmethod
    def calculate_psnr(original, compressed):
        mse = np.mean((original - compressed) ** 2)
        if mse == 0:  # MSE is zero means no noise is present in the signal .
            # Therefore PSNR have no importance.
            return 100
        max_pixel = 255.0
        psnr = 20 * np.log10(np.max(max_pixel / np.sqrt(mse)))
        return psnr

    @staticmethod
    def calculate_lbp(target_img, generated_img):
        radius = 4
        n_points = 14
        target_gray = rgb2gray(target_img)
        pred_gray = rgb2gray(generated_img)
        target_lbp = local_binary_pattern(target_gray, n_points, radius)
        generated_lbp = local_binary_pattern(pred_gray, n_points, radius)
        return target_lbp, generated_lbp

    @staticmethod
    def export_metrics_to_file(metrics, metrics_result_filename):
        header = ['mse', 'rmse', 'psnr', 'ssim', 'uqi', 'sam', 'vifp', 'euc_dist_LBP', 'cosine_dist_LBP']
        results = [{'metric': name, 'value': value} for name, value in zip(header, metrics)]
        pd.DataFrame(results).to_csv(metrics_result_filename, index=False)
