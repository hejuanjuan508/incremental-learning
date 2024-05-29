from copy import deepcopy
from pathlib import Path
import random
import pywt
import cv2
import torch
import numpy as np
import imgaug
from cv2 import blur
from imgaug.augmenters import RandAugment
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import os
import numpy as np
from glob import glob
from PIL import Image, ImageFilter
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
MIN_BOUND = -100.0
MAX_BOUND = 400.0

PIXEL_MEAN = 0.1021
PIXEL_STD = 0.19177



def low_freq_mutate_np( amp_src, amp_trg, L):
    a_src = np.fft.fftshift(amp_src )
    a_trg = np.fft.fftshift(amp_trg)

    h, w = a_src.shape
    b = (np.floor(np.amin((h,w))*L)  ).astype(int)
    c_h = np.floor(h/2.0).astype(int)
    c_w = np.floor(w/2.0).astype(int)

    h1 = c_h-b
    h2 = c_h+b+1
    w1 = c_w-b
    w2 = c_w+b+1

    a_src[h1:h2,w1:w2] = a_trg[h1:h2,w1:w2]
    a_src = np.fft.ifftshift( a_src)
    return a_src

def blur(img, p=0.5):
    if random.random() < p:
        sigma = np.random.uniform(0.1, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return img

def FDA_source_to_target_np( src_img, trg_img, L=0.1):
    # exchange magnitude
    # input: src_img, trg_img

    src_img_np = src_img #.cpu().numpy()
    trg_img_np = trg_img #.cpu().numpy()

    # get fft of both source and target
    fft_src_np = np.fft.fft2( src_img_np)
    fft_trg_np = np.fft.fft2( trg_img_np)
    fshift = np.fft.fftshift(fft_src_np)
    fshift1 = np.fft.fftshift(fft_trg_np)
    # extract amplitude and phase of both ffts
    amp_src, pha_src = np.abs(fshift), np.angle(fshift)
    amp_trg, pha_trg = np.abs(fshift1), np.angle(fshift1)

    # mutate the amplitude part of source with target
    amp_src_ = low_freq_mutate_np( amp_src, amp_trg, L=L )

    # mutated fft of source
    fft_src_ = amp_src_ * np.exp( 1j * pha_src )

    # get the mutated image
    src_in_trg = np.fft.ifft2( fft_src_)
    src_in_trg = np.abs(src_in_trg)

    return src_in_trg
def setBounds(image, MIN_BOUND, MAX_BOUND):
    """
    Clip image to lower bound MIN_BOUND, upper bound MAX_BOUND.
    """
    return np.clip(image, MIN_BOUND, MAX_BOUND)


def normalize(image):
    """
    Perform standardization/normalization, i.e. zero_centering and setting
    the data to unit variance.
    """
    #image = setBounds(image, MIN_BOUND, MAX_BOUND)

    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    # image = np.clip(image, 0., 1.)
    # image = image - PIXEL_MEAN
    # image = image/PIXEL_STD
    return image
def magnitude_phase_split(img):
    # 分离幅度谱与相位谱
    dft = np.fft.fft2(img)
    #原始频谱
    dft_shift = np.fft.fftshift(dft)
    # 幅度谱
    magnitude_spectrum = np.log(np.abs(dft_shift))
    # 相位谱
    phase_spectrum = np.log(np.angle(dft_shift))
    return magnitude_spectrum,phase_spectrum

def magnitude_phase_combine(img_m,img_p):
    # 不同图像幅度谱与相位谱结合
    img_mandp = img_m*np.e**(1j*img_p)
    # 图像重构
    img_mandp = np.uint8(np.abs(np.fft.ifft2(img_mandp)))
    img_mandp =img_mandp/np.max(img_mandp)*255
    return img_mandp

def splitChannels(gt):
    c0 = np.copy(gt)
    c1 = np.copy(gt)

    c0[c0 == 2] = 0  # np.unique(c0) = [0, 1]
    c1[c1 == 1] = 0  # np.unique(c1) = [0, 2]
    c1[c1 == 2] = 1  # np.unique(c1) = [0, 1]

    return c0, c1


class LungDataset(torch.utils.data.Dataset):
    def __init__(self, augment_params,model,root, flag, size,flag1,weak=True,strong=True, transform=True):
        self.all_files = self.extract_files(root)
        self.size = size
        self.flag=flag
        self.flag1=flag1
        self.augment_params = False
        if model == 'mixFedGAN':
            self.augment_params = augment_params

        if (transform == True):
            self.transform = transforms.Compose([
                transforms.Resize(self.size),
                transforms.ToTensor()

            ])
        else:
            self.transform = transforms.Compose([transforms.ToTensor()])
        if(weak==True):
            self.weak = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(size=self.size,
                                      padding=int(self.size * 0.125),
                                      padding_mode='reflect'),

                transforms.ToTensor()])


    @staticmethod
    def extract_files(root):
        """
        Extract the paths to all slices given the root path (ends with train or val)
        """
        files = []
        for subject in root.glob("*"):  # Iterate over the subjects
            slice_path = subject / "data"  # Get the slices for current subject
            for slice in slice_path.glob("*"):
                files.append(slice)
        return files

    @staticmethod
    def change_img_to_label_path(path):
        """
        Replace data with mask to get the masks
        """
        parts = list(path.parts)
        parts[parts.index("data")] = "masks"
        return Path(*parts)


    def augment(self, img, gt):
        """
        Augments slice and segmentation mask in the exact same way
        Note the manual seed initialization
        """
        ###################IMPORTANT###################
        # Fix for https://discuss.pytorch.org/t/dataloader-workers-generate-the-same-random-augmentations/28830/2
        random_seed = torch.randint(0, 1000000, (1,))[0].item()
        imgaug.seed(random_seed)
        #####################################################
        mask = SegmentationMapsOnImage(gt, gt.shape)

        #print(gt.shape,mask.shape)
        slice_aug, mask_aug = self.augment_params(image=img, segmentation_maps=mask)
        mask_aug = mask_aug.get_arr()
        #print(slice_aug.shape,mask_aug.shape)
        return slice_aug, mask_aug

    def __len__(self):
        """
        Return the length of the dataset (length of all files)
        """
        return len(self.all_files)

    def __getitem__(self, idx):
        """
        Given an index return the (augmented) slice and corresponding mask
        Add another dimension for pytorch
        """
        file_path = self.all_files[idx]
        if (self.flag==True):
            mask_path = self.change_img_to_label_path(file_path)
        else:
            mask_path = file_path
        #print(file_path)
        slice = np.load(file_path)
        gt = np.load(mask_path)

        #img=slice
        #match()
        img = normalize(slice)  #标准化和去除肺部阴影操作

        #img = self.transform(Image.fromarray(img)) 这种方法不保证图片转的都是正确的尺寸
        #gt = self.transform(Image.fromarray(gt))
        #调整图像大小


        img = cv2.resize(img, (self.size, self.size))

        gt = cv2.resize(gt, (256,256), interpolation=cv2.INTER_NEAREST)
        targetL, targetT = splitChannels(gt)
        targetL = self.transform(Image.fromarray(targetL))
        targetT = self.transform(Image.fromarray(targetT))
        mask = np.vstack((np.asarray(targetL), np.asarray(targetT)))
        # downsampled_mask = F.interpolate(torch.from_numpy(mask).unsqueeze(0), scale_factor=0.5, mode='bilinear', align_corners=False)
        # mask = downsampled_mask.squeeze(0)
        #src_in_trg = FDA_source_to_target_np(img, img1, L=0.1)


        #img_strong = self.strong(Image.fromarray(img))  # 给数据做弱增广

        #image = np.asanyarray(img)

        #gt = np.asanyarray(gt)

        # img = torch.from_numpy(img).float()
        # img=img.unsqueeze(0).unsqueeze(0)
        coeffs = pywt.dwt2(img, 'haar')
        # 获取低频和高频分量
        LL, (LH, HL, HH) = coeffs
        # 阈值处理高频分量以去噪
        threshold = 20
        LH[LH < threshold] = 0
        HL[HL < threshold] = 0
        HH[HH < threshold] = 0
        # 执行逆小波变换以重建去噪后的图像
        reconstructed_img = pywt.idwt2((LL, (LH, HL, HH)), 'haar')

        # mask = torch.from_numpy(mask).float()
        # mask = mask.unsqueeze(0)
        # xfm1 = pywt.dwt2(mask, 'haar', mode='zero')
        # Yl_m, Yh_m = xfm1
        # gt = torch.from_numpy(gt).float()
        # gt = gt.unsqueeze(0).unsqueeze(0)
        # xfm2 = pywt.dwt2(gt, 'haar', mode='zero')
        # Yl_g, Yh_g = xfm2
        return np.expand_dims(img, 0),mask,np.expand_dims(gt, 0)