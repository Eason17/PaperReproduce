import os,time,scipy.io
import cv2
from matplotlib import pyplot as plt

import numpy as np
import rawpy
import glob
from tqdm import tqdm

import h5py
import torch
import torch.nn as nn
import torch.optim as optim
import process
from process import postprocess_bayer_v2, generate_noisy_obs

from model import *
from utils import *
from losses import *

host = '/data'
input_dir = f'{host}/SID/Sony/short/'
gt_dir = f'{host}/SID/Sony/long/'
result_dir = './NikonD850_dpvs_train/'
model_dir = './saved_model/'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# get train and test IDs
train_ids = get_data_ids(input_dir, rule='0*.ARW')
test_ids = get_data_ids(gt_dir, rule='1*.ARW')
# patch size for training
patch_size = 512
crop_per_image = 4
save_freq = 10
plot_freq = 1
lastepoch = 470
learning_rate = 1e-4
use_dpvs = True # use deep supervision
read_h5_dataset = True
h5_dataset_name = f'{host}/SID/SID_train_GT.h5'
f = None

if read_h5_dataset:
    f = h5py.File(h5_dataset_name, "r")

if use_dpvs:
    model = DeepUnet()
else:
    model = UNetSeeInDark()

if lastepoch == 0:
    model._initialize_weights()
else:
    if use_dpvs:
        model = load_weights(model, path=model_dir+'checkpoint_nikon_dpvs_e%04d.pth'%lastepoch)
    else:
        model = load_weights(model, path=model_dir+'checkpoint_nikon_e%04d.pth'%lastepoch)
model = model.to(device)

if use_dpvs:
    loss_fn = Unet_dpsv_Loss()
else:
    loss_fn = Unet_Loss()

optimizer = optim.Adam(model.parameters(), lr = learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, gamma=0.5, step_size=100)

for epoch in range(lastepoch+1,3001):
    cnt = 0
    total_loss = 0
    random_inds = np.random.permutation(len(train_ids))
    if read_h5_dataset:
        ratio = f["data"].attrs["ratio"][random_inds[-1]]
        name = f["data"].attrs["name"][random_inds[-1]]
        ccm = f["data"].attrs["ccm"][random_inds[-1]]
        wb = f["data"].attrs["wb"][random_inds[-1]]

    with tqdm(total=len(train_ids)) as t:
        for i, ind in enumerate(random_inds):
            # if i > 10: break
            # get the path from image id
            if read_h5_dataset:
                train_id = ind
                ratio = f["data"].attrs["ratio"][ind]
                name = f["data"].attrs["name"][ind]
                ccm = f["data"].attrs["ccm"][ind]
                wb = f["data"].attrs["wb"][ind]
            else:
                train_id = train_ids[ind]
                in_files = glob.glob(input_dir + '%05d_00*.ARW'%train_id)
                in_path = in_files[np.random.randint(0,len(in_files))]
                _, in_fn = os.path.split(in_path)

                gt_files = glob.glob(gt_dir + '%05d_00*.ARW'%train_id)
                gt_path = gt_files[0]
                _, gt_fn = os.path.split(gt_path)
                in_exposure =  float(in_fn[9:-5])
                gt_exposure =  float(gt_fn[9:-5])
                ratio = min(gt_exposure/in_exposure,300)
            
            st=time.time()
            cnt+=1

            # 读取raw图，这部分后续可以包装成dataset（dataloader）
            if read_h5_dataset:
                gt_images = f["data"][ind]
                gt_images = gt_images[np.newaxis,:,:,:]
            else:
                gt_raw = rawpy.imread(gt_path)
                gt_images = np.expand_dims(pack_raw_bayer(gt_raw),axis = 0)
                raw = gt_raw

            input_images = gt_images.copy()
            # 随机裁剪成crop_per_image份
            input_patch, gt_patch = random_crop(input_images, gt_images, crop_per_image=crop_per_image)
            for i in range(len(input_images)):
                input_patch[i] = generate_noisy_obs(input_patch[i], camera_type='NikonD850')
            # 图像增强（翻转，旋转会破坏行噪声假设）
            choice = np.random.randint(2,size=2)
            input_patch = data_aug(input_patch, choice)
            gt_patch = data_aug(gt_patch, choice)
            # 送入cuda显存
            input_patch = np.clip(input_patch, 0, 1)
            gt_patch = np.clip(gt_patch, 0, 1)
            in_img = torch.from_numpy(input_patch).type(torch.FloatTensor).to(device)
            gt_img = torch.from_numpy(gt_patch).type(torch.FloatTensor).to(device)
            
            # 模型推导与计算
            model.zero_grad()
            if use_dpvs:
                out_img, o2, o4, o8 = model(in_img, mode='train')
                loss = loss_fn(out_img, gt_img, o2, o4, o8)
            else:
                out_img = model(in_img)
                loss = loss_fn(out_img, gt_img)

            loss.backward()
            # 优化器优化
            optimizer.step()

            # 更新tqdm的参数
            total_loss += loss.item()
            t.set_description(f'Epoch {epoch}')
            t.set_postfix(loss=float(f"{total_loss/cnt:.6f}"))
            t.update(1)
            # plot test
            # if read_h5_dataset and ind % 10==0:
            #     # inputs = process.raw2rgb_v2(in_img[0], wb=wb, ccm=ccm)
            #     output = process.raw2rgb_v2(out_img[0], wb=wb, ccm=ccm)
            #     # target = process.raw2rgb_v2(gt_img[0], wb=wb, ccm=ccm)
            #     # plt.imshow(inputs)
            #     # plt.imshow(target)
            #     plt.imshow(output)
            #     plt.show()
            #     # cv2.imshow('inputs', inputs[:,:,::-1])
            #     # cv2.imshow('target', target[:,:,::-1])
            #     # cv2.waitKey(0)
            #     # cv2.destroyAllWindows()
        
        # 更新学习率
        scheduler.step()

        if epoch % plot_freq == 0:
            if not os.path.isdir(result_dir):# + '%04d'%epoch):
                os.makedirs(result_dir)# + '%04d'%epoch)

            if read_h5_dataset:
                inputs = process.raw2rgb_v2(in_img[0], wb=wb, ccm=ccm)
                output = process.raw2rgb_v2(out_img[0], wb=wb, ccm=ccm)
                target = process.raw2rgb_v2(gt_img[0], wb=wb, ccm=ccm)
                if use_dpvs:
                    o2 = process.raw2rgb_v2(o2[0], wb=wb, ccm=ccm)
                    o4 = process.raw2rgb_v2(o4[0], wb=wb, ccm=ccm)
                    o8 = process.raw2rgb_v2(o8[0], wb=wb, ccm=ccm)
                    # 图像增强恢复（转回来）
                    o2 = data_aug(o2, choice, bias=-2)
                    o4 = data_aug(o4, choice, bias=-2)
                    o8 = data_aug(o8, choice, bias=-2)
                    ps = patch_size // 2
                    temp = np.zeros((ps,ps*3//2,3), dtype=np.uint8)
                    temp[:ps, 0:ps, :] = o2 * 255
                    temp[ps//2:ps, ps:ps*3//2, :] = o4 * 255
                    temp[ps//4:ps//2, ps:ps*5//4, :] = o8 * 255
                    filename = result_dir + '%04d_others.png'%(epoch)
                    plt.imsave(filename, temp)
            else:
                inputs = tensor2im(postprocess_bayer_v2(gt_path, in_img))
                output = tensor2im(postprocess_bayer_v2(in_path, out_img))
                target = tensor2im(postprocess_bayer_v2(in_path, gt_img))

            # 图像增强恢复（转回来）
            inputs = data_aug(inputs, choice, bias=-2)
            output = data_aug(output, choice, bias=-2)
            target = data_aug(target, choice, bias=-2)

            temp = np.concatenate((inputs, output, target),axis=1)
            filename = result_dir + '%04d_%05d_nikon_train.png'%(epoch,train_id)
            plt.imsave(filename, temp)
            # cv2.imwrite(filename, np.uint8(temp[...,::-1]*255))
        
        if epoch % save_freq == 0:
            log(f"learning_rate: {scheduler.get_last_lr()[0]:.6f}")
            torch.save(model.state_dict(), model_dir+'checkpoint_nikon_e%04d.pth'%epoch)

