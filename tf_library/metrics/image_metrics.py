from sewar.full_ref import mse, rmse, uqi, sam, vifp
from skimage.metrics import structural_similarity as ssim

from skimage.feature import local_binary_pattern
from skimage.color import rgb2gray

import numpy as np
import pandas as pd


class ImageMetrics:

    def count_metrics(self, model, test_dataset):
        n_images = len(test_dataset)

        MSE = 0
        RMSE = 0
        PSNR = 0
        SSIM = 0
        UQI = 0
        SAM = 0
        VIFP = 0

        euc_dist_LBP = 0
        cosine_dist_LBP = 0
        # TODO: count len of test_dataset
        for example_input, example_target in test_dataset.take(n_images):
            prediction = model.predict(example_input)

            target_np = example_target.numpy()[0]
            pred_np = prediction.numpy()[0]
            MSE += mse(target_np, pred_np)
            RMSE += rmse(target_np, pred_np)
            PSNR += self.calculate_psnr(target_np, pred_np)
            SSIM += ssim(target_np, pred_np, multichannel=True)
            UQI += uqi(target_np, pred_np)
            SAM += sam(target_np, pred_np)
            VIFP += vifp(target_np, pred_np)

            target_LBP, pred_LBP = self.calculate_lbp(target_np, pred_np)
            target_LBP, pred_LBP = self.flatten_images(target_LBP, pred_LBP)

            euc_dist_LBP += np.sqrt(self.calculate_distance(target_LBP, pred_LBP))
            cosine_dist_LBP += np.dot(target_LBP, pred_LBP) / (np.linalg.norm(target_LBP) * np.linalg.norm(pred_LBP))

        MSE = MSE / n_images
        RMSE = RMSE / n_images
        PSNR = PSNR / n_images
        SSIM = SSIM / n_images
        UQI = UQI / n_images
        SAM = SAM / n_images
        VIFP = VIFP / n_images

        euc_dist_LBP = euc_dist_LBP / n_images
        cosine_dist_LBP = cosine_dist_LBP / n_images

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
