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
from process import postprocess_bayer_v2

from model import *
from utils import *

host = '/data'
input_dir = f'{host}/SID/Sony/short/'
gt_dir = f'{host}/SID/Sony/long/'
result_dir = './sidpair_train/'
model_dir = './saved_model/'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# get train and test IDs
train_ids = get_data_ids(input_dir, rule='0*.ARW')
test_ids = get_data_ids(gt_dir, rule='1*.ARW')
# patch size for training
patch_size = 512
crop_per_image = 2
save_freq = 10
plot_freq = 1
lastepoch = 690
learning_rate = 4e-5

def reduce_mean(out_im, gt_im):
    return torch.abs(out_im - gt_im).mean()

allfolders = glob.glob('./result/*0')
for folder in allfolders:
    lastepoch = np.maximum(lastepoch, int(folder[-4:]))

model = UNetSeeInDark()
if lastepoch == 0:
    model._initialize_weights()
else:
    model = load_weights(model, path=model_dir+'checkpoint_sony_e%04d.pth'%lastepoch)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr = learning_rate)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.997)
for epoch in range(lastepoch+1,2001):
    cnt = 0
    total_loss = 0
  
    with tqdm(total=len(train_ids)) as t:
        for ind in np.random.permutation(len(train_ids)):
            # get the path from image id
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
            raw = rawpy.imread(in_path)
            input_images = np.expand_dims(pack_raw_bayer(raw),axis = 0) * ratio
            gt_raw = rawpy.imread(gt_path)
            gt_images = np.expand_dims(pack_raw_bayer(gt_raw),axis = 0)
            # 随机裁剪成crop_per_image份
            input_patch, gt_patch = random_crop(input_images, gt_images, crop_per_image=crop_per_image)
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
            out_img = model(in_img)
            # 计算损失，反向传播
            loss = reduce_mean(out_img, gt_img)
            loss.backward()
            # 优化器优化
            optimizer.step()

            # 更新tqdm的参数
            total_loss += loss.item()
            t.set_description(f'Epoch {epoch}')
            t.set_postfix(loss=float(f"{total_loss/cnt:.6f}"))
            t.update(1)
        
        # 更新学习率
        scheduler.step()

        if epoch % plot_freq == 0:
            if not os.path.isdir(result_dir):# + '%04d'%epoch):
                os.makedirs(result_dir)# + '%04d'%epoch)

            inputs = tensor2im(postprocess_bayer_v2(gt_path, in_img))
            output = tensor2im(postprocess_bayer_v2(in_path, out_img))
            target = tensor2im(postprocess_bayer_v2(in_path, gt_img))

            # 图像增强恢复（转回来）
            inputs = data_aug(inputs, choice, bias=-2)
            output = data_aug(output, choice, bias=-2)
            target = data_aug(target, choice, bias=-2)

            temp = np.concatenate((inputs, output, target),axis=1)
            filename = result_dir + '%04d_%05d_00_train_%d.png'%(epoch,train_id,ratio)
            cv2.imwrite(filename, temp[...,::-1])
        
        if epoch % save_freq == 0:
            log(f"learning_rate: {scheduler.get_last_lr()[0]:.6f}")
            torch.save(model.state_dict(), model_dir+'checkpoint_sony_e%04d.pth'%epoch)

