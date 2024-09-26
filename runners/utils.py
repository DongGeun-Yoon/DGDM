import os
import numpy as np
import scipy

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from datetime import datetime
from torchvision.utils import make_grid, save_image
import math
from Register import Registers
from datasets.custom import CustomAlignedDataset
from skimage.metrics import structural_similarity as cal_ssim

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def MAE(pred, true):
    return np.mean(np.abs(pred-true),axis=(0,1)).sum()

def MSE(pred, true):
    return np.mean((pred-true)**2,axis=(0,1)).sum()

def PSNR(pred, true):
    mse = np.mean((pred-true)**2)
    return 20 * np.log10(1) - 10 * np.log10(mse)

def metric(pred, true, mean, std, return_ssim_psnr=False, clip_range=[0, 1]):
    pred = pred*std + mean
    true = true*std + mean
    mae = MAE(pred, true)
    mse = MSE(pred, true)

    if return_ssim_psnr:
        pred = np.maximum(pred, clip_range[0])
        pred = np.minimum(pred, clip_range[1])
        ssim, psnr = 0, 0
        for b in range(pred.shape[0]):
            for f in range(pred.shape[1]):
                ssim += cal_ssim(pred[b, f].swapaxes(0, 2), true[b, f].swapaxes(0, 2), multichannel=True)
                psnr += PSNR(pred[b, f], true[b, f])
        ssim = ssim / (pred.shape[0] * pred.shape[1])
        psnr = psnr / (pred.shape[0] * pred.shape[1])
        return mse, mae, ssim, psnr
    else:
        return mse, mae

# FVD
i3D_WEIGHTS_URL = "https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt"

def load_i3d_pretrained(pretrain_root='./'):
    filepath = os.path.join(pretrain_root, 'i3d_torchscript.pt')
    if not os.path.exists(filepath):
        os.system(f"wget {i3D_WEIGHTS_URL} -O {pretrain_root}/{filepath}")
    i3d = torch.jit.load(filepath).eval()
    return i3d

def preprocess_single(video, resolution=224, sequence_length=None):
    # video: CTHW, [0, 1]
    c, t, h, w = video.shape

    # temporal crop
    if sequence_length is not None:
        assert sequence_length <= t
        video = video[:, :sequence_length]

    # scale shorter side to resolution
    scale = resolution / min(h, w)
    if h < w:
        target_size = (resolution, math.ceil(w * scale))
    else:
        target_size = (math.ceil(h * scale), resolution)
    video = F.interpolate(video, size=target_size, mode='bilinear', align_corners=False)

    # center crop
    c, t, h, w = video.shape
    w_start = (w - resolution) // 2
    h_start = (h - resolution) // 2
    video = video[:, :, h_start:h_start + resolution, w_start:w_start + resolution]

    # [0, 1] -> [-1, 1]
    video = (video - 0.5) * 2

    return video.contiguous()

def get_feats(videos, detector, bs=10):
    # videos : torch.tensor BCTHW [0, 1]
    detector_kwargs = dict(rescale=False, resize=False, return_features=True) # Return raw features before the softmax layer.
    feats = np.empty((0, 400))
    device = videos.device
    # device = torch.device("cuda:0") #if device is not torch.device("cpu") else device
    with torch.no_grad():
        for i in range((len(videos)-1)//bs + 1):
            feats = np.vstack([feats, detector(torch.stack([preprocess_single(video) for video in videos[i*bs:(i+1)*bs]]).to(device), **detector_kwargs).detach().cpu().numpy()])
    return feats

def compute_stats(feats: np.ndarray):
    mu = feats.mean(axis=0) # [d]
    sigma = np.cov(feats, rowvar=False) # [d, d]
    return mu, sigma

def compute_fvd(feats_fake: np.ndarray, feats_real: np.ndarray) -> float:
    mu_gen, sigma_gen = compute_stats(feats_fake)
    mu_real, sigma_real = compute_stats(feats_real)
    m = np.square(mu_gen - mu_real).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False) # pylint: disable=no-member
    fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
    return float(fid)

# 
def remove_prefix_state_dict(state_dict, prefix="module"):
    """
    remove prefix from the key of pretrained state dict for Data-Parallel
    """
    new_state_dict = {}
    first_state_name = list(state_dict.keys())[0]
    if not first_state_name.startswith(prefix):
        for key, value in state_dict.items():
            new_state_dict[key] = state_dict[key].float()
    else:
        for key, value in state_dict.items():
            new_state_dict[key[len(prefix)+1:]] = state_dict[key].float()
    return new_state_dict

def remove_file(fpath):
    if os.path.exists(fpath):
        os.remove(fpath)

def make_dir(dir):
    os.makedirs(dir, exist_ok=True)
    return dir


def make_save_dirs(args, prefix, suffix=None, with_time=False):
    time_str = datetime.now().strftime("%Y-%m-%dT%H-%M-%S") if with_time else ""
    suffix = suffix if suffix is not None else ""

    result_path = make_dir(os.path.join(args.result_path, prefix, suffix, time_str))
    image_path = make_dir(os.path.join(result_path, "image"))
    log_path = make_dir(os.path.join(result_path, "log"))
    checkpoint_path = make_dir(os.path.join(result_path, "checkpoint"))
    sample_path = make_dir(os.path.join(result_path, "samples"))
    sample_to_eval_path = make_dir(os.path.join(result_path, "sample_to_eval"))
    print("create output path " + result_path)
    return image_path, checkpoint_path, log_path, sample_path, sample_to_eval_path


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Attention') != -1:
        pass
    elif classname.find('Conv2d') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Parameter') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def get_dataset(data_config, test=False):
    train_dataset = Registers.datasets[data_config.dataset_type](data_config.dataset_config, stage='train')
    val_dataset = Registers.datasets[data_config.dataset_type](data_config.dataset_config, stage='val')
    if test:
        return train_dataset, val_dataset, val_dataset
    return train_dataset, val_dataset

@torch.no_grad()
def save_single_image(image, save_path, file_name, to_normal=True):
    image = image.detach().clone()
    if to_normal:
        image = image.mul_(0.5).add_(0.5).clamp_(0, 1.)
    image = image.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(image)
    im.save(os.path.join(save_path, file_name))

@torch.no_grad()
def save_single_video(image, save_path, file_name, grid_size=10, to_normal=True):
    image = image.detach().clone()
    image_grid = get_image_grid(image, grid_size=grid_size, to_normal=to_normal)
    im = Image.fromarray(image_grid)
    im.save(os.path.join(save_path, file_name))

@torch.no_grad()
def save_single_numpy(image, save_path, file_name, grid_size=10, to_normal=True):
    image = image.cpu().detach().clone().numpy()
    np.save(os.path.join(save_path, file_name[:-4]), image)
    
@torch.no_grad()
def get_image_grid(batch, grid_size=4, to_normal=True):
    batch = batch.detach().clone()
    image_grid = make_grid(batch, nrow=grid_size)
    if to_normal:
        image_grid = image_grid.mul_(0.5).add_(0.5).clamp_(0, 1.)
    image_grid = image_grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return image_grid

@torch.no_grad()
def save_frames(image, save_path, file_name, grid_size=10, to_normal=True):
    image = image.detach().clone() # b t c h w 
    if to_normal:
        image = image.mul_(0.5).add_(0.5).clamp_(0, 1.)
    image = image.permute([0,2,3,1]).mul_(255).to('cpu', torch.uint8).numpy()
    for i in range(grid_size):
        frame = image[i,:,:,0] if image.shape[3] == 1 else image[i]
        im = Image.fromarray(frame)
        save_name = file_name[:-4] + f"_{i}" + file_name[-4:]
        im.save(os.path.join(save_path, save_name))