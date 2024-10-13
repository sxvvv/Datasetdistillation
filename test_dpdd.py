import numpy as np
import os
import cv2
import math
from skimage import metrics, img_as_ubyte
from sklearn.metrics import mean_absolute_error
import argparse
from tqdm import tqdm
import torch.nn as nn
import torch
from restormer import Restormer
from natsort import natsorted
from glob import glob
from pdb import set_trace as stx
import matplotlib.pyplot as plt
import lpips

def MAE(img1, img2):
    mae_0=mean_absolute_error(img1[:,:,0], img2[:,:,0],
                              multioutput='uniform_average')
    mae_1=mean_absolute_error(img1[:,:,1], img2[:,:,1],
                              multioutput='uniform_average')
    mae_2=mean_absolute_error(img1[:,:,2], img2[:,:,2],
                              multioutput='uniform_average')
    return np.mean([mae_0,mae_1,mae_2])

def PSNR(img1, img2):
    mse_ = np.mean( (img1 - img2) ** 2 )
    if mse_ == 0:
        return 100
    return 10 * math.log10(1 / mse_)

def SSIM(img1, img2):
    return metrics.structural_similarity(img1, img2, data_range=1, multichannel= False, channel_axis=-1)

def load_img(filepath):
    return cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)

def load_img16(filepath):
    return cv2.cvtColor(cv2.imread(filepath, -1), cv2.COLOR_BGR2RGB)

def save_img(filepath, img):
    cv2.imwrite(filepath,cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

alex = lpips.LPIPS(net='alex').cuda()
parser = argparse.ArgumentParser(description='Dual Pixel Defocus Deblurring using Restormer')

parser.add_argument('--input_dir', default='/data/SDA/suxin/SNN/DPDD/test', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='/data/SDA/suxin/kd-data/results/restormer-dpdd/', type=str, help='Directory for results')
parser.add_argument('--weights', default='/data/SDA/suxin/kd-data/model_dpdd/best_student_model.pth', type=str, help='Path to weights')
parser.add_argument('--save_images', action='store_true', help='Save denoised images in result directory')

args = parser.parse_args()


##########################
def splitimage(imgtensor, crop_size=720, overlap_size=8):
    _, C, H, W = imgtensor.shape
    hstarts = [x for x in range(0, H, crop_size - overlap_size)]
    while hstarts[-1] + crop_size >= H:
        hstarts.pop()
    hstarts.append(H - crop_size)
    wstarts = [x for x in range(0, W, crop_size - overlap_size)]
    while wstarts[-1] + crop_size >= W:
        wstarts.pop()
    wstarts.append(W - crop_size)
    starts = []
    split_data = []
    for hs in hstarts:
        for ws in wstarts:
            cimgdata = imgtensor[:, :, hs:hs + crop_size, ws:ws + crop_size]
            starts.append((hs, ws))
            split_data.append(cimgdata)
    return split_data, starts


def get_scoremap(H, W, C, B=1, is_mean=True):
    center_h = H / 2
    center_w = W / 2

    score = torch.ones((B, C, H, W))
    if not is_mean:
        for h in range(H):
            for w in range(W):
                score[:, :, h, w] = 1.0 / (math.sqrt((h - center_h) ** 2 + (w - center_w) ** 2 + 1e-3))
    return score


def mergeimage(split_data, starts, crop_size=720, resolution=(1, 3, 720, 720)):
    B, C, H, W = resolution[0], resolution[1], resolution[2], resolution[3]
    tot_score = torch.zeros((B, C, H, W))
    merge_img = torch.zeros((B, C, H, W))
    scoremap = get_scoremap(crop_size, crop_size, C, B=B, is_mean=False)
    for simg, cstart in zip(split_data, starts):
        hs, ws = cstart
        merge_img[:, :, hs:hs + crop_size, ws:ws + crop_size] += scoremap * simg
        tot_score[:, :, hs:hs + crop_size, ws:ws + crop_size] += scoremap
    merge_img = merge_img / tot_score
    return merge_img
            
def load_pretrained_weights(model, path):
    if os.path.exists(path):
        state_dict = torch.load(path, map_location='cuda:0')
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded weights from {path}")
    else:
        print(f"Pretrained weights not found at {path}")
##########################
net = Restormer()
load_pretrained_weights(net, args.weights)

net.cuda()
result_dir = args.result_dir
if args.save_images:
    os.makedirs(result_dir, exist_ok=True)

filesC = natsorted(glob(os.path.join(args.input_dir, 'gt', '*.png')))
file = natsorted(glob(os.path.join(args.input_dir, 'input', '*.png')))

indoor_labels  = np.load('/data/SDA/suxin/SNN/DPDD/test/indoor_labels.npy')
outdoor_labels = np.load('/data/SDA/suxin/SNN/DPDD/test/outdoor_labels.npy')

psnr, mae, ssim, pips = [], [], [], []
with torch.no_grad():
    for file, fileC in tqdm(zip(file, filesC), total=len(filesC)):
        
        imgC = np.float32(load_img16(fileC))/65535.
        img = np.float32(load_img16(file))/65535.

        patchC = torch.from_numpy(imgC).unsqueeze(0).permute(0,3,1,2).cuda()
        patch = torch.from_numpy(img).unsqueeze(0).permute(0,3,1,2).cuda()
        B, C, H, W = patchC.shape
        
        input_ = patch
        split_data, starts = splitimage(input_)
        for i, data in enumerate(split_data):
            split_data[i] = net(data).cuda()
            functional.reset_net(net)
            split_data[i] = split_data[i].cpu()
        restored = mergeimage(split_data, starts, resolution=(B, C, H, W))
        # restored = net(input_)
        restored = torch.clamp(restored,0,1)
        pips.append(alex(patchC.cuda(), restored.cuda(), normalize=True).item())

        restored = restored.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

        psnr.append(PSNR(imgC, restored))
        mae.append(MAE(imgC, restored))
        ssim.append(SSIM(imgC, restored))
        
        save_file = os.path.join(result_dir, os.path.split(fileC)[-1])
        restored = np.uint16((restored*65535).round())
        save_img(save_file, restored)

psnr, mae, ssim, pips = np.array(psnr), np.array(mae), np.array(ssim), np.array(pips)

psnr_indoor, mae_indoor, ssim_indoor, pips_indoor = psnr[indoor_labels-1], mae[indoor_labels-1], ssim[indoor_labels-1], pips[indoor_labels-1]
psnr_outdoor, mae_outdoor, ssim_outdoor, pips_outdoor = psnr[outdoor_labels-1], mae[outdoor_labels-1], ssim[outdoor_labels-1], pips[outdoor_labels-1]

print("Overall: PSNR {:4f} SSIM {:4f} MAE {:4f} LPIPS {:4f}".format(np.mean(psnr), np.mean(ssim), np.mean(mae), np.mean(pips)))
print("Indoor:  PSNR {:4f} SSIM {:4f} MAE {:4f} LPIPS {:4f}".format(np.mean(psnr_indoor), np.mean(ssim_indoor), np.mean(mae_indoor), np.mean(pips_indoor)))
print("Outdoor: PSNR {:4f} SSIM {:4f} MAE {:4f} LPIPS {:4f}".format(np.mean(psnr_outdoor), np.mean(ssim_outdoor), np.mean(mae_outdoor), np.mean(pips_outdoor)))
