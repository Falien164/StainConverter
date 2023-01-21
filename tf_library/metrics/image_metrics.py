from sewar.full_ref import mse, rmse, uqi, ergas, scc, rase, sam, vifp
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
from scipy.spatial import distance
from scipy.spatial.distance import cdist
from skimage.feature import local_binary_pattern
from skimage.color import rgb2gray

import numpy as np


class ImageMetrics:

    def count_metrics(self, generator, test_dataset):
        MSE = 0
        RMSE = 0
        psnr = 0
        SSIM = 0
        UQI = 0
        ERGAS = 0
        SCC = 0
        RASE = 0
        SAM = 0
        VIFP = 0

        euc_dist_LBP = 0
        cosine_dist_LBP = 0
        jaccard_dist_LBP = 0
        # TODO: count len of test_dataset
        for example_input, example_target in test_dataset.take(FID_NUMBER_OF_PHOTOS):
            prediction = generator(example_input, training=True)

            target_np = example_target.numpy()[0]
            pred_np = prediction.numpy()[0]
            MSE += mse(target_np, pred_np)
            RMSE += rmse(target_np, pred_np)
            psnr += self.PSNR(target_np, pred_np)
            SSIM += ssim(target_np, pred_np, multichannel=True)
            UQI += uqi(target_np, pred_np)
            ERGAS += ergas(target_np, pred_np)
            SCC += scc(target_np, pred_np)
            RASE += rase(target_np, pred_np)
            SAM += sam(target_np, pred_np)
            VIFP += vifp(target_np, pred_np)

            target_LBP, pred_LBP = self.LBP(target_np, pred_np)
            target_LBP, pred_LBP = self.flatten_images(target_LBP, pred_LBP)

            euc_dist_LBP += np.sqrt(self.calculate_distance(target_LBP, pred_LBP))
            cosine_dist_LBP += np.dot(target_LBP, pred_LBP) / (np.linalg.norm(target_LBP) * np.linalg.norm(pred_LBP))
            jaccard_dist_LBP += distance.jaccard(target_LBP, pred_LBP)

        MSE = MSE / FID_NUMBER_OF_PHOTOS
        RMSE = RMSE / FID_NUMBER_OF_PHOTOS
        psnr = psnr / FID_NUMBER_OF_PHOTOS
        SSIM = SSIM / FID_NUMBER_OF_PHOTOS
        UQI = UQI / FID_NUMBER_OF_PHOTOS
        ERGAS = ERGAS / FID_NUMBER_OF_PHOTOS
        SCC = SCC / FID_NUMBER_OF_PHOTOS
        RASE = RASE / FID_NUMBER_OF_PHOTOS
        SAM = SAM / FID_NUMBER_OF_PHOTOS
        VIFP = VIFP / FID_NUMBER_OF_PHOTOS

        euc_dist_LBP = euc_dist_LBP / FID_NUMBER_OF_PHOTOS
        cosine_dist_LBP = cosine_dist_LBP / FID_NUMBER_OF_PHOTOS
        jaccard_dist_LBP = jaccard_dist_LBP / FID_NUMBER_OF_PHOTOS

        return MSE, RMSE, psnr, SSIM, UQI, ERGAS, SCC, RASE, SAM, VIFP, euc_dist_LBP, cosine_dist_LBP, jaccard_dist_LBP

    @staticmethod
    def calculate_distance(i1, i2):
        return np.sum((i1 - i2) ** 2)

    @staticmethod
    def flatten_images(target_img, predicted_img):
        flat_target_img = target_img.flatten()
        flat_predicted_img = predicted_img.flatten()
        return flat_target_img, flat_predicted_img

    @staticmethod
    def PSNR(original, compressed):
        mse = np.mean((original - compressed) ** 2)
        if (mse == 0):  # MSE is zero means no noise is present in the signal .
            # Therefore PSNR have no importance.
            return 100
        max_pixel = 255.0
        psnr = 20 * np.log10(np.max(max_pixel / np.sqrt(mse)))
        return psnr

    @staticmethod
    def LBP(target_img, generated_img):
        radius = 4
        n_points = 14
        target_gray = rgb2gray(target_img)
        pred_gray = rgb2gray(generated_img)
        target_lbp = local_binary_pattern(target_gray, n_points, radius)
        generated_lbp = local_binary_pattern(pred_gray, n_points, radius)
        return target_lbp, generated_lbp
