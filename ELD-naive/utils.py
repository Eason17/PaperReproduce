import os
import numpy as np
import random
import torch
import glob
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import time
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
# from skimage.measure import compare_psnr, compare_ssim
import exifread

def log(string, log=None):
    log_string = f'{time.strftime("%H:%M:%S")} >>  {string}'
    print(log_string)
    if log is not None:
        with open(log,'a+') as f:
            f.write(log_string+'\n')

# 作为装饰器函数
def no_grad(fn):
    with torch.no_grad():
        def transfer(*args,**kwargs):
            fn(*args,**kwargs)
        return fn


def load_weights(model, path):
    pretrained_dict=torch.load(path)
    model_dict=model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # k1 = 'conv1.doubleconv.0.conv_relu.0.weight'
    # k2 = 'conv1.doubleconv.0.conv_relu.0.bias'
    # if k1 in pretrained_dict:
    #     del pretrained_dict[k1], pretrained_dict[k2]
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model


def metainfo(rawpath):
    with open(rawpath, 'rb') as f:
        tags = exifread.process_file(f)
        _, suffix = os.path.splitext(os.path.basename(rawpath))

        try: #suffix == '.dng':
            expo = eval(str(tags['Image ExposureTime']))
            iso = eval(str(tags['Image ISOSpeedRatings']))
        except:
            expo = eval(str(tags['EXIF ExposureTime']))
            iso = eval(str(tags['EXIF ISOSpeedRatings']))
        else:
            log("Reading metainfo error...")

        # print('ISO: {}, ExposureTime: {}'.format(iso, expo))
    return iso, expo


def set_input(data):
    inputs, target = data['input'], data['target']
    rawpath_in = data['fn'][0]
    rawpath_gt = data['rawpath'][0]
    cfa = data['cfa'][0] if 'cfa' in data else 'bayer'
    aligned = False if 'unaligned' in data else True
    return inputs, target, rawpath_in, rawpath_gt, cfa, aligned


def read_paired_fns(filename):
    with open(filename) as f:
        fns = f.readlines()
        fns = [tuple(fn.strip().split(' ')) for fn in fns]
    return fns


def get_data_ids(dir, rule='0*.ARW'):
    data_fns = glob.glob(dir + rule)
    data_ids = []
    ids_set = set()
    for i in range(len(data_fns)):
        _, data_fn = os.path.split(data_fns[i])
        ids = int(data_fn[0:5])
        if ids in ids_set: continue
        data_ids.append(ids)
        ids_set.add(ids)
    return data_ids


def tensor2im(image_tensor, visualize=False, video=False):    
    image_tensor = image_tensor.detach()

    if visualize:                
        image_tensor = image_tensor[:, 0:3, ...]

    if not video: 
        image_numpy = image_tensor[0].cpu().float().numpy()
        image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
    else:
        image_numpy = image_tensor.cpu().float().numpy()
        image_numpy = (np.transpose(image_numpy, (0, 2, 3, 1))) * 255.0

    image_numpy = np.clip(image_numpy, 0, 255)

    return image_numpy


def compute_expo_ratio(input_fn, target_fn):        
    in_exposure = float(input_fn.split('_')[-1][:-5])
    gt_exposure = float(target_fn.split('_')[-1][:-5])
    ratio = min(gt_exposure / in_exposure, 300)
    return ratio


def pack_raw_bayer(raw, wp=16383):
    #pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    raw_pattern = raw.raw_pattern if raw.raw_pattern is not None else np.array([[0,1],[3,2]])
    R = np.where(raw_pattern==0)
    G1 = np.where(raw_pattern==1)
    B = np.where(raw_pattern==2)
    G2 = np.where(raw_pattern==3)
    
    white_point = wp
    img_shape = im.shape
    H = img_shape[0] //2 * 2
    W = img_shape[1] //2 * 2
    if len(img_shape) == 3:
        C = img_shape[2]
        out = im.transpose(2,0,1)
    else:
        out = np.stack((im[R[0][0]:H:2, R[1][0]:W:2], #RGBG
                        im[G1[0][0]:H:2, G1[1][0]:W:2],
                        im[B[0][0]:H:2, B[1][0]:W:2],
                        im[G2[0][0]:H:2, G2[1][0]:W:2]), axis=0).astype(np.float32)

    black_level = np.array(raw.black_level_per_channel)[:,None,None].astype(np.float32)

    out = (out - black_level) / (white_point - black_level)
    out = np.clip(out, 0.0, 1.0)
    
    return out

    
def random_crop(lr_img, hr_img, crop_size=512, crop_per_image=8, aug=False):
    # 本函数用于将numpy随机裁剪成以crop_size为边长的方形crop_per_image等份
    is_tensor = torch.is_tensor(lr_img)
    device = 'cpu'
    dtype = lr_img.dtype
    if is_tensor:
        device = lr_img.device
        if device != 'cpu':
            lr_img = lr_img.cpu()
            hr_img = hr_img.cpu()
        lr_img = lr_img.numpy()
        hr_img = hr_img.numpy()

    b, c, h, w = lr_img.shape
    # 创建空numpy做画布
    lr_crops = np.zeros((b*crop_per_image,c,crop_size,crop_size))
    hr_crops = np.zeros((b*crop_per_image,c,crop_size,crop_size))

    # 往空tensor的通道上贴patchs
    for i in range(crop_per_image):
        h_start = np.random.randint(0, h - crop_size)
        w_start = np.random.randint(0, w - crop_size)
        h_end = h_start + crop_size
        w_end = w_start + crop_size 

        lr_crop = lr_img[..., h_start:h_end, w_start:w_end]
        hr_crop = hr_img[..., h_start:h_end, w_start:w_end]

        lr_crops[i:i+1, ...] = lr_crop
        hr_crops[i:i+1, ...] = hr_crop

    if is_tensor:
        lr_crops = torch.from_numpy(lr_crops).to(device).type(dtype)
        hr_crops = torch.from_numpy(hr_crops).to(device).type(dtype)

    return lr_crops, hr_crops

def data_aug(data, choice, bias=0, rot=False):
    if choice[0] == 1:
        data = np.flip(data, axis=2+bias)
    if choice[1] == 1:
        data = np.flip(data, axis=3+bias)
    return data


class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            self.next_input = self.next_input.float()


class EMA():
    '''
    # 转自知乎https://zhuanlan.zhihu.com/p/68748778
    # 初始化
    ema = EMA(model, 0.999)
    ema.register()

    # 训练过程中，更新完参数后，同步update shadow weights
    def train():
        optimizer.step()
        ema.update()

    # eval前，apply shadow weights；eval之后，恢复原来模型的参数
    def evaluate():
        ema.apply_shadow()
        # evaluate
        ema.restore()
    '''
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def quality_assess(X, Y, data_range=255):
    # Y: correct; X: estimate
    if X.ndim == 3:
        psnr = compare_psnr(Y, X, data_range=data_range)
        ssim = compare_ssim(Y, X, data_range=data_range, multichannel=True)
        return {'PSNR':psnr, 'SSIM': ssim}

    elif X.ndim == 4:

        vpsnr = np.mean(compare_psnr_video(Y/data_range*255, X/data_range*255))
        vssim = np.mean(compare_ssim_video(Y/data_range*255, X/data_range*255))

        if X.shape[0] != 1:
            _, _strred, _strredsn = strred(raw2gray(Y)/data_range, raw2gray(X)/data_range)
        else:
            _strred = 0
            _strredsn = 0

        return {'PSNR': vpsnr, 'SSIM': vssim, 'STRRED': _strred, 'STRREDSN':_strredsn}
    else:
        raise NotImplementedError


def sample(imgs, split=None ,figure_size=(1, 1), img_dim=(400, 600), path=None, num=0, metrics=False):
    if type(img_dim) is int:
        img_dim = (img_dim, img_dim)
    img_dim = tuple(img_dim)
    if len(img_dim) == 1:
        h_dim = img_dim
        w_dim = img_dim
    elif len(img_dim) == 2:
        h_dim, w_dim = img_dim
    h, w = figure_size
    num_of_imgs = figure_size[0] * figure_size[1]
    gap = len(imgs) // num_of_imgs
    colormap = True# if gap > 1 else False
    if split is None:
        split = list(range(0, len(imgs)+1, gap))
    figure = np.zeros((h_dim*h, w_dim*w, 3))
    for i in range(h):
        if metrics:
            img_hr = imgs[ split[i*w+w-1] : split[i*w+w] ].transpose(1,2,0)
        for j in range(w):
            idx = i*w+j
            if idx >= len(split)-1: break
            digit = imgs[ split[idx] : split[idx+1] ]
            if len(digit) == 1:
                for k in range(3):
                    figure[i*h_dim: (i+1)*h_dim,
                        j*w_dim: (j+1)*w_dim, k] = digit
            elif len(digit) == 3:
                for k in range(3):
                    figure[i*h_dim: (i+1)*h_dim,
                        j*w_dim: (j+1)*w_dim, k] = digit[2-k]
            if metrics:
                if j < w-1: 
                    img_lr = digit.transpose(1,2,0)
                    psnr = compare_psnr(img_hr, img_lr)
                    ssim = compare_ssim(img_hr, img_lr, multichannel=colormap)
                    text = f'psnr:{psnr:.4f} - ssim:{ssim:.4f}'
                    log(text, log='./logs.txt')
                    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
                    cv2.putText(figure, text, (j*w_dim, i*h_dim+20), font, 1, (255,255,255))
                else: 
                    text = 'reference'
                    log(text)
                    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
                    cv2.putText(figure, text, (j*w_dim, i*h_dim+20), font, 1, (255,255,255))

    if path is None:
        cv2.imshow('Figure%d'%num, figure)
        cv2.waitKey()
    else:
        figure *= 255
        filename1 = path.split('\\')[-1]
        filename2 = path.split('/')[-1]
        if len(filename1) < len(filename2):
            filename = filename1
        else:
            filename = filename2
        root_path = path[:-len(filename)]
        if not os.path.exists(root_path):
            os.makedirs(root_path)
        log("Saving Image at {}".format(path), log='./logs.txt')
        cv2.imwrite(path, figure)
    