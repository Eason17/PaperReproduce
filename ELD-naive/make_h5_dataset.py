import os,time,scipy.io
import cv2

import numpy as np
from matplotlib import pyplot as plt
import rawpy
import glob
from tqdm import tqdm

import h5py
import process
from process import postprocess_bayer_v2

from model import *
from utils import *

host = 'F:/datasets'
input_dir = f'{host}/SID/Sony/short/'
gt_dir = f'{host}/SID/Sony/long/'
result_dir = './sidpair_train/'
model_dir = './saved_model/'


def fix_MIT5K_h5_dataset(data_dir, crop_size=1000, 
                        dataset_name='MIT5K_train.h5', 
                        result_dir=f'H:\DeepLearning\datasets'):
    # get IDs
    rawpaths = []
    for root, dirs, files in os.walk(data_dir):
        for name in files:
            if '.dng' in name.lower():
                rawpaths.append(os.path.join(root, name))
    length = len(rawpaths)
    log(f'Successful load {len(rawpaths)} raw files from "{data_dir}"...')

    # 创建数据集
    f = h5py.File(os.path.join(result_dir, dataset_name), "r+")
    wbs = []
    ccms = []
    names = []
    isos = []
    expos = []
    error = 0
    total_img = 0
    with tqdm(total=length) as t:
        for idx in range(length):
            # get the path from image id
            rawpath = rawpaths[idx]
            # 读取raw图，这部分后续可以包装成dataset（dataloader）
            raw = rawpy.imread(rawpath)
            iso, expo = metainfo(rawpath)
            wb, ccm = process.read_wb_ccm(raw)
            name = rawpath.split('\\')[-1]
            try:
                input_images = pack_raw_bayer(raw)
            except:
                log(f'Something wrong happened in reading process')
                error += 1
                continue

            c,h,w = input_images.shape
            nh = h // crop_size
            nw = w // crop_size

            if nh*nw == 0:
                log(f'This picture is too small to crop')
                error += 1
                continue

            if wb[0] == 0:
                wb = np.array([1.,1.,1.,0.], dtype=np.float32)

            total_img += nh*nw
            for i in range(nh):
                for j in range(nw):
                    if i+j>0: error -= 1
                    wbs.append(wb)
                    ccms.append(ccm)
                    isos.append(iso)
                    expos.append(expo)
                    names.append(int(name[1:5]))
                    if error == 0: break
                if error == 0: break
            # 更新tqdm的参数
            t.update(1)
            t.set_postfix({'errors':error})

    f.__delitem__('wb')
    f.__delitem__('ccm')
    f.create_dataset('wb', data=np.array(wbs))
    f.create_dataset('ccm', data=np.array(ccms))
    log(f'total image num:{total_img}, error:{error}')


def create_MIT5K_h5_dataset(data_dir, crop_size=1000, 
                            dataset_name='MIT5K_train.h5', 
                            result_dir=f'H:\DeepLearning\datasets'):
    # get IDs
    rawpaths = []
    for root, dirs, files in os.walk(data_dir):
        for name in files:
            if '.dng' in name.lower():
                rawpaths.append(os.path.join(root, name))
    length = len(rawpaths)
    log(f'Successful load {len(rawpaths)} raw files from "{data_dir}"...')

    # 创建数据集
    f = h5py.File(os.path.join(result_dir, dataset_name), "w")
    dst = f.create_dataset("data", (length, 4, crop_size, crop_size), compression='gzip',
            maxshape=(None, 4, crop_size, crop_size), chunks=(1, 4, crop_size, crop_size))

    wbs = []
    ccms = []
    names = []
    isos = []
    expos = []
    total_img = 0
    error = 0
    with tqdm(total=length) as t:
        for idx in range(length):
            # get the path from image id
            rawpath = rawpaths[idx]
            # 读取raw图，这部分后续可以包装成dataset（dataloader）
            raw = rawpy.imread(rawpath)
            iso, expo = metainfo(rawpath)
            wb, ccm = process.read_wb_ccm(raw)
            name = rawpath.split('\\')[-1]
            try:
                input_images = pack_raw_bayer(raw)
            except:
                log(f'Something wrong happened in reading process')
                error += 1
                continue

            c,h,w = input_images.shape
            nh = h // crop_size
            nw = w // crop_size
            total_img += nh*nw

            if nh*nw == 0:
                log(f'This picture is too small to crop')
                error += 1
                continue

            if wb[0] == 0:
                wb = np.array([1.,1.,1.,0.], dtype=np.float32)

            for i in range(nh):
                for j in range(nw):
                    if i+j>0: error -= 1
                    dst[idx-error] = input_images[...,i*crop_size:(i+1)*crop_size,
                                            j*crop_size:(j+1)*crop_size]
                    wbs.append(wb)
                    ccms.append(ccm)
                    isos.append(iso)
                    expos.append(expo)
                    names.append(int(name[1:5]))
                    if error == 0: break
                if error == 0: break

            # 更新tqdm的参数
            t.update(1)
            t.set_postfix({'errors':error})

    f['data'].attrs['iso'] = np.array(isos)
    f['data'].attrs['name'] = np.array(names)
    f['data'].attrs['expo'] = np.array(expos)
    f['data'].attrs['wb'] = np.array(wbs)
    f['data'].attrs['ccm'] = np.array(ccms)
    log(f'total image num:{total_img}, error:{error}')


def create_SID_h5_dataset(data_dir, pair_file):
    # get train and test IDs
    train_fns = read_paired_fns(pair_file)
    train_fns_input = [fn[0] for fn in train_fns]
    train_fsn_gt = [fn[1] for fn in train_fns]
    train_ids = get_data_ids(data_dir)
    length = len(train_fns_input)

    # 创建数据集    
    f = h5py.File("SID_train_noisy.h5", "w")
    dst = f.create_dataset("data", (length, 4, 1424, 2128), chunks=(1, 4, 1424, 2128))

    wbs = [None]*length
    ccms = [None]*length
    names = train_fns_input
    ratios = [None]*length

    with tqdm(total=length) as t:
        for ind in range(length):
            # get the path from image id
            
            train_id = train_ids[ind]
            in_files = glob.glob(input_dir + '%05d_00*.ARW'%train_id)
            in_path = in_files[np.random.randint(0,len(in_files))]
            _, in_fn = os.path.split(in_path)

            gt_files = glob.glob(data_dir + '%05d_00*.ARW'%train_id)
            gt_path = gt_files[0]
            _, gt_fn = os.path.split(gt_path)
            in_exposure =  float(in_fn[9:-5])
            gt_exposure =  float(gt_fn[9:-5])
            ratio = min(gt_exposure/in_exposure,300)

            # 读取raw图，这部分后续可以包装成dataset（dataloader）
            # raw = rawpy.imread(in_path)
            # input_images = np.expand_dims(pack_raw_bayer(raw),axis = 0) * ratio
            gt_raw = rawpy.imread(in_path)
            gt_images = pack_raw_bayer(gt_raw),# .astype(np.float16)
            dst[ind] = gt_images
            wbs[ind], ccms[ind] = process.read_wb_ccm(gt_raw)
            # names[ind] = in_fn
            ratios[ind] = 0 #np.array(int(ratio))

            # 更新tqdm的参数
            t.update(1)

    dst.attrs['wb'] = wbs
    dst.attrs['ccm'] = ccms
    dst.attrs['ratio'] = ratios
    dst.attrs['name'] = names


def read_h5_dataset(dataset_name, data_dir='F:/datasets/fivek_dataset'):
    # get IDs
    # rawpaths = []
    # for root, dirs, files in os.walk(data_dir):
    #     for name in files:
    #         if '.dng' in name.lower():
    #             rawpaths.append(os.path.join(root, name))
    # length = len(rawpaths)
    # log(f'Successful load {len(rawpaths)} raw files from "{data_dir}"...')
    # 创建数据集    
    f = h5py.File(dataset_name, "r")
    length = len(f['data'])

    with tqdm(total=length) as t:
        for ind in range(length):
            # get the path from image id
            gt_raw = f["data"][ind].transpose(1,2,0)
            wb = f['data'].attrs['wb'][ind]
            wb = np.array([1,1,1,1])#wb[1]
            ccm = np.eye(3)#f['data'].attrs['ccm'][ind]
            # rgb = process.postprocess_bayer(rawpaths[ind], gt_raw)
            # gt_rgb = process.raw2rgb_v2(gt_raw, wb=wb, ccm=ccm)
            r = gt_raw[:,:,0]#gt_rgb[:,:,0]
            g = gt_raw[:,:,1]/2+ gt_raw[:,:,2]/2#gt_rgb[:,:,1]
            b = gt_raw[:,:,2]#gt_rgb[:,:,2]
            rgb = np.stack([r,g,b], axis=2)
            plt.imshow(rgb)
            plt.show()
            # cv2.imshow('rgb', rgb[:,:,::-1])
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            # print(gt_raw.shape)
            # print(f["data"].attrs['wb'])
            # print(f["data"].attrs['ccm'])
            # print(f["data"].attrs['ratio'])
            # print(f["data"].attrs['name'])

            # 更新tqdm的参数
            t.update(1)

if __name__ == "__main__":
    # create_MIT5K_h5_dataset(r'F:\datasets\fivek_dataset\raw_photos')
    # fix_MIT5K_h5_dataset(r'F:\datasets\fivek_dataset\raw_photos')
    # read_h5_dataset("F:/datasets/MIT5K/MIT5K_train.h5")
    # create_h5_dataset(input_dir, "Sony_train.txt")
    read_h5_dataset("F:/datasets/SID/SID_train_GT.h5")