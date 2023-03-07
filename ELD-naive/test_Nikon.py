import os,time,scipy.io
import cv2

import numpy as np
import rawpy
import glob
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import process
from process import postprocess_bayer_v2, postprocess_bayer

from model import *
from utils import *

host = 'F:/datasets'
input_dir = f'{host}/SID/Sony/short/'
gt_dir = f'{host}/SID/Sony/long/'
data_dir = f'{host}/ELD/'
result_dir = './Nikon_gen1870_test/'
model_dir = './saved_model/'
# cameras information
cameras = ['NikonD850',]
suffixes = ['.nef',]
scenes = list(range(1, 10+1))
img_ids_set = [[4, 9, 14], [5, 10, 15]]# [[3,8,13], [2,7,12]]

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# checkpoint number
model_name = 'nikon' 
lastepoch = 1960

model = UNetSeeInDark()
try:
    model = load_weights(model, path= model_dir + f'checkpoint_{model_name}_e{lastepoch:04d}.pth')
except FileExistsError as e:
    log(f'{e}')

model_eval = EvalModel(model, chop=True)
model_eval = model_eval.to(device)
model_eval.netG = model_eval.netG.to(device)

# for scene in scenes:
for i, img_ids in enumerate(img_ids_set):
    log(f"img_ids: {img_ids}")
    dst = process.ELDEvalDataset(data_dir, scenes=scenes, img_ids=img_ids)
    eval_datasets = [process.ELDEvalDataset(data_dir, camera_suffix, scenes=scenes, 
                    img_ids=img_ids) for camera_suffix in zip(cameras, suffixes)]

    eval_dataloaders = [torch.utils.data.DataLoader(
        eval_dataset, batch_size=1, shuffle=False,
        num_workers=0, pin_memory=True) for eval_dataset in eval_datasets]

    correct = True
    for camera, dataloader in zip(cameras, eval_dataloaders):
        log('Eval camera {}'.format(camera))
        psnrs = []
        ssims = []
        # we evaluate PSNR/SSIM on full size images
        with torch.no_grad():
            with tqdm(total=len(dataloader)) as t:
                for i, data in enumerate(dataloader):
                    inputs, target = data['input'].to(device), data['target'].to(device)
                    rawpath_in = data['fn'][0]
                    rawpath_gt = data['rawpath'][0]
                    cfa = data['cfa'][0] if 'cfa' in data else 'bayer'
                    aligned = False if 'unaligned' in data else True

                    output = model_eval(inputs)

                    if correct:
                        output = model_eval.corrector(output, target)
                    
                    output_np = tensor2im(output)
                    target_np = tensor2im(target)   
                    inputs_np = tensor2im(inputs)

                    res = quality_assess(output_np, target_np, data_range=255)
                    res_in = quality_assess(inputs_np, target_np, data_range=255)  

                    t.set_description(f'{camera}')
                    t.set_postfix(PSNR=float(f"{res['PSNR']:.2f}"), SSIM=float(f"{res['SSIM']:.4f}"))
                    t.update(1)

                    # save_pic
                    output = postprocess_bayer(rawpath_gt, output)
                    target = postprocess_bayer(rawpath_gt, target)
                    inputs = postprocess_bayer(rawpath_gt, inputs)

                    scene_name = rawpath_in.split('\\')[-2]
                    name = os.path.splitext(os.path.basename(rawpath_in))[0]
                    if not os.path.exists(os.path.join(result_dir, scene_name)):
                        os.makedirs(os.path.join(result_dir, scene_name))

                    Image.fromarray(output.astype(np.uint8)).save(os.path.join(result_dir, 
                                    scene_name,'{}_{:.2f}_{:.4f}_{}.png'.format(name, res['PSNR'], res['SSIM'], i)))
                    Image.fromarray(inputs.astype(np.uint8)).save(os.path.join(result_dir, 
                                    scene_name, 'input_{:.2f}_{:.4f}_{}.png'.format(res_in['PSNR'], res_in['SSIM'], i)))
                    Image.fromarray(target.astype(np.uint8)).save(os.path.join(result_dir, scene_name, 't_label.png'))

                    psnrs.append(res['PSNR'])
                    ssims.append(res['SSIM'])
    
        print(np.mean(psnrs))
        print(np.mean(ssims))

