import cv2, scipy
import torch
import time
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt

def log(string, log=None):
    log_string = f'{time.strftime("%H:%M:%S")} >>  {string}'
    print(log_string)
    if log is not None:
        with open(log,'a+') as f:
            f.write(log_string+'\n')

def plt_show(img, name="img", figsize=(10,6)):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.imshow(img)
    plt.title(name)
    plt.show()


def get_camera_noisy_params(camera_type=None):
    cam_noisy_params = {}
    cam_noisy_params['NikonD850'] = {
        'Kmin':1.2, 'Kmax':2.4828, 'lam':-0.26, 'q':1/(2**14),
        'sigTLk':0.906, 'sigTLb':-0.6754,   'sigTLsig':0.035165/3,
        'sigRk':0.8322,  'sigRb':-2.3326,   'sigRsig':0.301333/3,
    }
    if camera_type in cam_noisy_params:
        return cam_noisy_params['NikonD850']
    else:
        log(f'''Warning: we have not test the noisy parameters of camera \
            "{camera_type}". Now we use NikonD850's parameters to test.''')
        return cam_noisy_params['NikonD850']

def generate_noisy_obs(y, camera_type=None, w_max=16383, G=True, P=True):
    # 噪声参数采样
    def sample_params(camera_type='NikonD850'):
        # 获取已经测算好的相机噪声参数
        params = get_camera_noisy_params(camera_type=camera_type)
        # 根据最小二乘法得到的噪声参数回归模型采样噪声参数
        log_K = np.random.uniform(low=params['Kmin'], high=params['Kmax'])
        mu_TL = params['sigTLk']*log_K + params['sigTLb']
        log_sigTL = np.random.normal(loc=mu_TL, scale=params['sigTLsig'])
        mu_R = params['sigRk']*log_K + params['sigRb']
        log_sigR = np.random.normal(loc=mu_R, scale=params['sigRsig'])
        # 去掉log
        lam = params['lam']
        q = params['q']
        K = np.exp(log_K)
        sigTL = np.exp(log_sigTL)
        sigR = np.exp(log_sigR)
        
        return {'K':K, 'sigTL':sigTL, 'sigR':sigR, 'lam':lam, 'q':q}

    # # Burst denoising
    # sig_read = 10. ** np.random.uniform(low=-3., high=-1.5)
    # sig_shot = 10. ** np.random.uniform(low=-2., high=-1.)
    # shot = np.random.randn(*y.shape).astype(np.float32) * np.sqrt(np.maximum(y, 1e-10)) * sig_shot
    # read = np.random.randn(*y.shape).astype(np.float32) * sig_read
    # z = y + shot + read

    # 模拟曝光衰减的系数
    ratio = np.random.uniform(low=100, high=300)
    # 采样一组噪声参数
    p = sample_params()
    
    y = y / ratio

    log(f"ratio={ratio}, {p}")
    if P:
        noisy_ps = np.random.poisson(y/p['K']).astype(np.float32) * p['K']
    else:
        noisy_ps = 0
    if G:
        noisy_TL = stats.tukeylambda.rvs(p['lam'], scale=p['sigTL'], size=y.shape).astype(np.float32)
    noisy_row = np.random.randn(y.shape[0], 1, 1).astype(np.float32) * p['sigR']
    noisy_q = np.random.uniform(low=-0.5*p['q'], high=0.5*p['q']) * w_max
    z = (noisy_ps + noisy_TL + noisy_row + noisy_q) * ratio / w_max
    z = np.clip(z, 0, 1).astype(np.float32)
    # plt_show((noisy_TL + noisy_row + noisy_q) * ratio / 16384, 'black_noisy')

    return z


def split_crop(tensor, refer=None, crop_size=512, crop_per_image=None):
    # 本函数用于将tensor图均匀split成以crop_size为边长的方形crop_per_image等份
    # 如果crop_per_image超过图像极限，则以图像极限数量为准
    b, c, h, w = tensor.size()
    nh = h // crop_size
    nw = w // crop_size
    starth = (h - crop_size * nh) // 2
    startw = (w - crop_size * nw) // 2
    # 实际裁剪数量以crop_per_image为准，超过则以极限为准
    if crop_per_image is None:
        num_of_crops = nh*nw
    else:
        num_of_crops = min(crop_per_image, nh*nw)
    # 创建空Tensor做画布
    lr_crops = torch.empty((b,c*num_of_crops,crop_size,crop_size))
    hr_crops = None
    if refer is not None:
        hr_crops = torch.empty((b,c*num_of_crops,crop_size,crop_size))
    # 往空tensor的通道上贴patchs
    for i in range(nw):
        for j in range(nh):
            cnt = i*nh+j
            if cnt >= num_of_crops: break
            h_start = j*crop_size + starth
            w_start = i*crop_size + startw
            h_end = h_start + crop_size
            w_end = w_start + crop_size

            lr_crops[:, (i*nh+j)*c:(i*nh+j+1)*c, :, :] = tensor[:, :, h_start:h_end, w_start:w_end].contiguous()
            if refer is not None:
                hr_crops[:, (i*nh+j)*c:(i*nh+j+1)*c, :, :] = refer[:, :, h_start:h_end, w_start:w_end].contiguous()

    if refer is not None:
        return lr_crops, hr_crops
    
    return lr_crops
        

def data_augmentation(image, mode):
    if mode == 0:
        # original
        return image
    elif mode == 1:
        # flip up and down
        return np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        return np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        image = np.rot90(image)
        return np.flipud(image)
    elif mode == 4:
        # rotate 180 degree
        return np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        image = np.rot90(image, k=2)
        return np.flipud(image)
    elif mode == 6:
        # rotate 270 degree
        return np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        image = np.rot90(image, k=3)
        return np.flipud(image)


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

        lr_crop = lr_img[:, :, h_start:h_end, w_start:w_end]
        hr_crop = hr_img[:, :, h_start:h_end, w_start:w_end]

        lr_crops[i:i+1, :, :, :] = lr_crop
        hr_crops[i:i+1, :, :, :] = hr_crop

    lr_crops = torch.from_numpy(lr_crops).to(device).type(dtype)
    hr_crops = torch.from_numpy(hr_crops).to(device).type(dtype)

    return lr_crops, hr_crops


if __name__ == "__main__":
    # 读取图像
    img = cv2.imread('../test/778.png')[np.newaxis,:,:,::-1] /255.
    noisy = cv2.imread('../test/778.png')[:,:,::-1] /255. * 16383
    noisy = generate_noisy_obs(noisy)
    tensor = torch.from_numpy(img).permute(0,3,1,2)
    refer = torch.from_numpy(noisy[np.newaxis,...]).permute(0,3,1,2)
    cpi = 12
    lr, hr = random_crop(tensor, refer, crop_size=128, crop_per_image=cpi)
    lr_np = lr.numpy()
    lr_np = lr_np.transpose(0,2,3,1)
    hr_np = hr.numpy()
    hr_np = hr_np.transpose(0,2,3,1)
    for i in range(cpi):
        plt_show(lr_np[i],"lr_np")
        plt_show(hr_np[i],"hr_np")
    # plt_show(img,"z_final")