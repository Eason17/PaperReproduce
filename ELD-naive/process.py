"""Forward processing of raw data to sRGB images.

Unprocessing Images for Learned Raw Denoising
http://timothybrooks.com/tech/unprocessing
"""

import rawpy
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as tdist
from scipy import stats
from utils import *

def postprocess_bayer(rawpath, img4c):
    if torch.is_tensor(img4c):
        img4c = img4c.detach()
        img4c = img4c[0].cpu().float().numpy()
    img4c = np.clip(img4c, 0, 1)

    #unpack 4 channels to Bayer image
    raw = rawpy.imread(rawpath)
    raw_pattern = raw.raw_pattern
    R = np.where(raw_pattern==0)
    G1 = np.where(raw_pattern==1)
    G2 = np.where(raw_pattern==3)
    B = np.where(raw_pattern==2)
    
    black_level = np.array(raw.black_level_per_channel)[:,None,None]

    white_point = 16383

    img4c = img4c * (white_point - black_level) + black_level
    
    img_shape = img4c.shape
    H = img_shape[1] * 2
    W = img_shape[2] * 2

    raw.raw_image_visible[R[0][0]:H:2, R[1][0]:W:2] = img4c[0, :,:]
    raw.raw_image_visible[G1[0][0]:H:2,G1[1][0]:W:2] = img4c[1, :,:]
    raw.raw_image_visible[B[0][0]:H:2,B[1][0]:W:2] = img4c[2, :,:]
    raw.raw_image_visible[G2[0][0]:H:2,G2[1][0]:W:2] = img4c[3, :,:]
    
    # out = raw.postprocess(use_camera_wb=False, user_wb=[1,1,1,1], half_size=True, no_auto_bright=True, output_bps=8, bright=1, user_black=None, user_sat=None)
    # out = raw.postprocess(use_camera_wb=False, user_wb=[1.96875, 1, 1.444, 1], half_size=True, no_auto_bright=True, output_bps=8, bright=1, user_black=None, user_sat=None)    
    out = raw.postprocess(use_camera_wb=True, half_size=True, no_auto_bright=True, output_bps=8, bright=1, user_black=None, user_sat=None)
    return out

def postprocess_bayer_v2(rawpath, img4c):    
    with rawpy.imread(rawpath) as raw:
        out_srgb = raw2rgb_postprocess(img4c.detach(), raw)        
    
    return out_srgb

def apply_gains(bayer_images, wbs):
    """Applies white balance to a batch of Bayer images."""
    N, C, _, _ = bayer_images.shape
    wbs = wbs.repeat((N,1)).view(N, C, 1, 1)
    outs = bayer_images * wbs
    return outs


def apply_ccms(images, ccms):
    """Applies color correction matrices."""
    images = images.permute(
        0, 2, 3, 1)  # Permute the image tensor to BxHxWxC format from BxCxHxW format
    images = images[:, :, :, None, :]
    ccms = ccms[:, None, None, :, :]
    outs = torch.sum(images * ccms, dim=-1)
    # Re-Permute the tensor back to BxCxHxW format
    outs = outs.permute(0, 3, 1, 2)
    return outs


def gamma_compression(images, gamma=2.2):
    """Converts from linear to gamma space."""
    outs = torch.clamp(images, min=1e-8) ** (1 / gamma)
    # outs = (1 + gamma[0]) * np.power(images, 1.0/gamma[1]) - gamma[0] + gamma[2]*images
    outs = torch.clamp((outs*255).int(), min=0, max=255).float() / 255
    return outs


def raw2LRGB(bayer_images): 
    """RGBG -> linear RGB"""
    lin_rgb = torch.stack([
        bayer_images[:,0,...], 
        torch.mean(bayer_images[:, [1,3], ...], dim=1), 
        bayer_images[:,2,...]], dim=1)

    return lin_rgb


def process(bayer_images, wbs, cam2rgbs, gamma=2.2):
    """Processes a batch of Bayer RGBG images into sRGB images."""
    # White balance.
    bayer_images = apply_gains(bayer_images, wbs)
    # Binning
    bayer_images = torch.clamp(bayer_images, min=0.0, max=1.0)
    images = raw2LRGB(bayer_images)
    # Color correction.
    images = apply_ccms(images, cam2rgbs)
    # Gamma compression.
    images = torch.clamp(images, min=0.0, max=1.0)
    images = gamma_compression(images, gamma)
    
    return images


def raw2rgb(packed_raw, raw):
    """Raw2RGB pipeline (preprocess version)"""
    wb = np.array(raw.camera_whitebalance) 
    wb /= wb[1]
    cam2rgb = raw.color_matrix[:3, :3]
    if cam2rgb[0,0] == 0:
        cam2rgb = np.eye(3, dtype=np.float32)

    if isinstance(packed_raw, np.ndarray):
        packed_raw = torch.from_numpy(packed_raw).float()

    wb = torch.from_numpy(wb).float().to(packed_raw.device)
    cam2rgb = torch.from_numpy(cam2rgb).float().to(packed_raw.device)
    out = process(packed_raw[None], wbs=wb[None], cam2rgbs=cam2rgb[None], gamma=2.2)[0, ...].numpy()
    
    return out


def raw2rgb_v2(packed_raw, wb, ccm):
    if torch.is_tensor(packed_raw):
        packed_raw = packed_raw.detach().cpu().float()
    else:
        packed_raw = torch.from_numpy(packed_raw).float()
    wb = torch.from_numpy(wb).float()
    cam2rgb = torch.from_numpy(ccm).float()
    out = process(packed_raw[None], wbs=wb[None], cam2rgbs=cam2rgb[None], gamma=1)[0, ...].numpy()
    return out.transpose(1,2,0)


def raw2rgb_postprocess(packed_raw, raw):
    """Raw2RGB pipeline (postprocess version)"""
    assert packed_raw.ndimension() == 4
    wb = np.array(raw.camera_whitebalance) 
    wb /= wb[1]
    cam2rgb = raw.color_matrix[:3, :3]
    if cam2rgb[0,0] == 0:
        cam2rgb = np.eye(3, dtype=np.float32)

    wb = torch.from_numpy(wb[None]).float().to(packed_raw.device)
    cam2rgb = torch.from_numpy(cam2rgb[None]).float().to(packed_raw.device)
    out = process(packed_raw, wbs=wb, cam2rgbs=cam2rgb, gamma=2.2)
    # out = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
    return out


def read_wb_ccm(raw):
    wb = np.array(raw.camera_whitebalance) 
    wb /= wb[1]
    wb = wb.astype(np.float32)
    ccm = raw.color_matrix[:3, :3].astype(np.float32)
    if ccm[0,0] == 0:
        ccm = np.eye(3, dtype=np.float32)
    return wb, ccm


def get_camera_noisy_params(camera_type=None):
    cam_noisy_params = {}
    cam_noisy_params['NikonD850'] = {
        'Kmin':1.2, 'Kmax':2.4828, 'lam':-0.26, 'q':1/(2**14),
        'sigTLk':0.906, 'sigTLb':-0.6754,   'sigTLsig':0.035165,
        'sigRk':0.8322,  'sigRb':-2.3326,   'sigRsig':0.301333,
    }
    cam_noisy_params['SonyA7S2_lowISO'] = {
        'Kmin':-0.2734, 'Kmax':0.64185, 'lam':-0.05, 'q':1/(2**14),
        'sigTLk':0.75004, 'sigTLb':0.88237,   'sigTLsig':0.02526,
        'sigRk':0.73954,  'sigRb':-0.32404,   'sigRsig':0.03596,
    }
    cam_noisy_params['SonyA7S2_highISO'] = {
        'Kmin':0.41878, 'Kmax':1.1234, 'lam':-0.05, 'q':1/(2**14),
        'sigTLk':0.55284, 'sigTLb':0.12758,   'sigTLsig':0.00733,
        'sigRk':0.50505,  'sigRb':-1.39476,   'sigRsig':0.02262,
    }
    # Mi10 K=0.0014328*ISO+0.0011364
    cam_noisy_params['Mi10_lowISO'] = {# ISO<=1600 Kmin对应ISO400
        'Kmin':-0.55476, 'Kmax':0.83154, 'lam':0.02, 'q':1/(2**14),
        'sigTLk':0.72256, 'sigTLb':0.32613,   'sigTLsig':0.02762,
        'sigRk':0.80423,  'sigRb':-1.98576,   'sigRsig':0.03674,
    }
    cam_noisy_params['Mi10_highISO'] = {# ISO>1600 几乎是平的，当做平的也行
        'Kmin':0.83154, 'Kmax':2.21605, 'lam':0.012, 'q':1/(2**14),
        'sigTLk':0.00304, 'sigTLb':0.65304,   'sigTLsig':0.00088,
        'sigRk':-0.03177,  'sigRb':-1.57920,   'sigRsig':0.02277,
    }
    if camera_type in cam_noisy_params:
        return cam_noisy_params[camera_type]
    else:
        log(f'''Warning: we have not test the noisy parameters of camera \
            "{camera_type}". Now we use NikonD850's parameters to test.''')
        return cam_noisy_params['NikonD850']


def generate_noisy_obs(y, camera_type=None, w_max=16383, G=True, P=True, param=None):
    # 噪声参数采样
    def sample_params(camera_type='NikonD850'):
        if camera_type == 'SonyA7S2':
            choice = np.random.randint(2)
            camera_type += '_lowISO' if choice<1 else '_highISO'
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

    y = y * w_max
    # 模拟曝光衰减的系数
    ratio = np.random.uniform(low=90, high=300)
    # 采样一组噪声参数
    p = sample_params(camera_type)
    p['ratio'] = ratio
    
    if param is not None:
        p = param
    y = y / p['ratio']

    if P:
        noisy_ps = np.random.poisson(y/p['K']).astype(np.float32) * p['K']
    else:
        noisy_ps = 0
    if G:
        noisy_TL = stats.tukeylambda.rvs(p['lam'], scale=p['sigTL'], size=y.shape).astype(np.float32)
    noisy_row = np.random.randn(y.shape[-2],1).astype(np.float32) * p['sigR']
    noisy_q = np.random.uniform(low=-0.5*p['q'], high=0.5*p['q']) * w_max

    # 归一化回[0, 1]
    z = (noisy_ps + noisy_TL + noisy_row + noisy_q) * p['ratio'] / w_max
    z = np.clip(z, 0, 1).astype(np.float32)
    # plt.imshow((noisy_TL[0] + noisy_row[0] + noisy_q) * ratio / 16383)

    return z#, p


class ELDEvalDataset(torch.utils.data.Dataset):
    def __init__(self, basedir, camera_suffix=('NikonD850','.nef'), scenes=None, img_ids=None):
        super(ELDEvalDataset, self).__init__()
        self.basedir = basedir
        self.camera_suffix = camera_suffix # ('Canon', '.CR2')
        self.scenes = scenes
        self.img_ids = img_ids
        # self.input_dict = {}
        # self.target_dict = {}
        
    def __getitem__(self, i):
        camera, suffix = self.camera_suffix
        
        scene_id = i // len(self.img_ids)
        img_id = i % len(self.img_ids)

        scene = 'scene-{}'.format(self.scenes[scene_id])

        datadir = os.path.join(self.basedir, camera, scene)

        input_path = os.path.join(datadir, 'IMG_{:04d}{}'.format(self.img_ids[img_id], suffix))

        gt_ids = np.array([1, 6, 11, 16])
        ind = np.argmin(np.abs(self.img_ids[img_id] - gt_ids))
        
        target_path = os.path.join(datadir, 'IMG_{:04d}{}'.format(gt_ids[ind], suffix))

        iso, expo = metainfo(target_path)
        target_expo = iso * expo
        iso, expo = metainfo(input_path)

        ratio = target_expo / (iso * expo)
        
        with rawpy.imread(input_path) as raw:
            input = pack_raw_bayer(raw) * ratio            

        with rawpy.imread(target_path) as raw:
            target = pack_raw_bayer(raw)

        input = np.maximum(np.minimum(input, 1.0), 0)
        target = np.maximum(np.minimum(target, 1.0), 0)
        input = np.ascontiguousarray(input)
        target = np.ascontiguousarray(target)        

        data = {'input': input, 'target': target, 'fn':input_path, 'rawpath': target_path}
        
        return data

    def __len__(self):
        return len(self.scenes) * len(self.img_ids)

if __name__ == '__main__':
    path = r'F:\datasets\fivek_dataset\raw_photos\HQa4201to5000\photos'
    files = [os.path.join(path, name) for name in os.listdir(path) if '.dng' in name]
    for name in files:
        print(name)
        raw = rawpy.imread(name)
        img = raw.raw_image_visible.astype(np.float32)[np.newaxis,:,:]
        black_level = np.array(raw.black_level_per_channel)[0]
        # img = img[:, 1000:1500, 2200:2700]
        fig = plt.figure(figsize=(16,10))
        img = np.clip((img-black_level) / (16383-black_level), 0, 1)
        p = {'K': 1.67, 'sigTL': 2.81169081937262, 
            'sigR': 0.6393283265693124, 'lam': -0.05, 'q': 6.103515625e-05, 
            'ratio': 10}

        noisy = generate_noisy_obs(img,camera_type='SonyA7S2', param=p)
        # refer_path = path+'\\'+'DSC02750.ARW'
        # raw_refer = rawpy.imread(refer_path)
        # print(np.min(raw_refer.raw_image_visible), np.max(raw_refer.raw_image_visible), np.mean(raw_refer.raw_image_visible))
        # raw_refer.raw_image_visible[:,:] = np.clip((raw_refer.raw_image_visible.astype(np.float32)-512) / (16383-512)*200, 0, 1)*16383
        # print(np.min(raw_refer.raw_image_visible), np.max(raw_refer.raw_image_visible), np.mean(raw_refer.raw_image_visible))
        # out1 = raw_refer.postprocess(use_camera_wb=True, no_auto_bright=True)
        # print(np.min(out1), np.max(out1), np.mean(out1))
        # plt.imsave('real.png', out1)
        # plt.imshow(out1)
        # plt.show()
        raw.raw_image_visible[:,:] = noisy[0,:,:]*16383
        out = raw.postprocess(use_camera_wb=True, half_size=True)
        plt.imshow(out)
        plt.show()
        plt.imsave('gen.png', out)
        print('test')
        print("")