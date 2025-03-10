import os
import math
import numpy as np
import cv2
import glob
import pandas as pd
from natsort import natsorted


def main():
    # Configurations

    # GT - Ground-truth;
    # Gen: Generated / Restored / Recovered images
    #folder_GT = "/home/u2308283114/udh/training/g02n2_2024-10-13-10_03_37/testPics/cover/"
    #folder_Gen = "/home/u2308283114/udh/training/g02n2_2024-10-13-10_03_37/testPics/stego/"
    folder_GT = "/home/u2308283114/udh/training/g02n2_2024-10-13-10_03_37/testPics/secret/"
    folder_Gen = "/home/u2308283114/udh/training/g02n2_2024-10-13-10_03_37/testPics/revSec/"
    crop_border = 1
    suffix = '_secret_rev'  # suffix for Gen images
    test_Y = True  # True: test Y channel only; False: test RGB channels

    PSNR_all = []
    SSIM_all = []
    MAE_all = []
    RMSE_all = []
    MSE_all = []  # Add MSE list
    img_list = sorted(glob.glob(folder_GT + '/*'))
    img_list = natsorted(img_list)

    if test_Y:
        print('Testing Y channel.')
    else:
        print('Testing RGB channels.')

    results = []
    for i, img_path in enumerate(img_list):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        im_GT = cv2.imread(img_path) / 255.
        #im_Gen = cv2.imread(os.path.join(folder_Gen, 'stego%d.png' % i)) / 255.
        im_Gen = cv2.imread(os.path.join(folder_Gen, 'revSec%d.png' % i)) / 255.

        if test_Y and im_GT.shape[2] == 3:  # evaluate on Y channel in YCbCr color space
            im_GT_in = bgr2ycbcr(im_GT)
            im_Gen_in = bgr2ycbcr(im_Gen)
        else:
            im_GT_in = im_GT
            im_Gen_in = im_Gen

        # calculate PSNR, SSIM, MAE, RMSE, and MSE
        PSNR = calculate_psnr(im_GT_in * 255, im_Gen_in * 255)
        SSIM = calculate_ssim(im_GT_in * 255, im_Gen_in * 255)
        MAE = calculate_mae(im_GT_in, im_Gen_in)
        RMSE = calculate_rmse(im_GT_in, im_Gen_in)
        MSE = calculate_mse(im_GT_in, im_Gen_in)  # MSE calculation

        print('{:3d} - {:10}. \tPSNR: {:.6f} dB, \tSSIM: {:.6f}, \tMAE: {:.6f}, \tRMSE: {:.6f}, \tMSE: {:.6f}'.format(
            i + 1, base_name, PSNR, SSIM, MAE, RMSE, MSE))

        PSNR_all.append(PSNR)
        SSIM_all.append(SSIM)
        MAE_all.append(MAE)
        RMSE_all.append(RMSE)
        MSE_all.append(MSE)

        results.append([base_name, PSNR, SSIM, MAE, RMSE, MSE])

    avg_psnr = sum(PSNR_all) / len(PSNR_all)
    avg_ssim = sum(SSIM_all) / len(SSIM_all)
    avg_mae = sum(MAE_all) / len(MAE_all)
    avg_rmse = sum(RMSE_all) / len(RMSE_all)
    avg_mse = sum(MSE_all) / len(MSE_all)  # Average MSE

    print('Average: PSNR: {:.6f} dB, SSIM: {:.6f}, MAE: {:.6f}, RMSE: {:.6f}, MSE: {:.6f}'.format(
        avg_psnr, avg_ssim, avg_mae, avg_rmse, avg_mse))

    # Save results to a CSV file
    df = pd.DataFrame(results, columns=['Image', 'PSNR', 'SSIM', 'MAE', 'RMSE', 'MSE'])
    df.to_csv('results.csv', index=False)

    with open('1.txt', 'w') as f:
        f.write(str(PSNR_all))


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def calculate_rmse(img1, img2):
    test1 = np.array(img1).astype(np.float64) * 255
    test2 = np.array(img2).astype(np.float64) * 255
    rmse = np.sqrt(np.mean((test1 - test2) ** 2))
    return rmse


def calculate_mae(img1, img2):
    test1 = np.array(img1).astype(np.float64) * 255
    test2 = np.array(img2).astype(np.float64) * 255
    mae = np.mean(np.abs(test1 - test2))
    return mae


def calculate_mse(img1, img2):
    # img1 and img2 should have range [0, 1]
    mse = np.mean((img1 - img2) ** 2)
    return mse


def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def bgr2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


if __name__ == '__main__':
    main()
