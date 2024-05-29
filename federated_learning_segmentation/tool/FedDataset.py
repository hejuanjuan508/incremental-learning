from copy import deepcopy

import torch
from PIL import Image

from torch.utils.data import Dataset
from torchvision.transforms import transforms


import numpy as np
import imgaug
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

from tool.Arguments import Arguments
from tool.LungDataset import splitChannels


class FedDataset(torch.utils.data.Dataset):
    def __init__(self, augment_params,train_image,train_label,size,flag,weak):
        self.train_image = train_image
        self.train_label=train_label
        self.size=size
        self.flag=flag
        self.transform = transforms.Compose([
            transforms.Resize(self.size),
            transforms.ToTensor()

        ])
        if weak==True:
            self.weak = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(size=self.size,
                                      padding=int(self.size * 0.125),
                                      padding_mode='reflect'),

                transforms.ToTensor()])


        self.augment_params = augment_params

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

        # print(gt.shape,mask.shape)
        slice_aug, mask_aug = self.augment_params(image=img, segmentation_maps=mask)
        mask_aug = mask_aug.get_arr()
        # print(slice_aug.shape,mask_aug.shape)
        return slice_aug, mask_aug

    def __len__(self):
        return len(self.train_image)

    def __getitem__(self, item):
        # print(self.dataset[i][0].shape, self.dataset[i][1].shape)
        img = self.train_image[item, :, :]
        gt = self.train_label[item, :, :]
        targetL, targetT = splitChannels(gt)

        targetL = self.transform(Image.fromarray(targetL))
        targetT = self.transform(Image.fromarray(targetT))

        mask = np.vstack((np.asarray(targetL), np.asarray(targetT)))

        args = Arguments()
        if(args.type == 'mixFedGAN'):
            img = self.train_image[item, :, :]
            gt = self.train_label[item, :, :]
            slice, target = self.augment(img, gt)
            targetL1, targetT1 = splitChannels(target)
            targetL1 = self.transform(Image.fromarray(targetL1))
            targetT1 = self.transform(Image.fromarray(targetT1))
            mask1 = np.vstack((np.asarray(targetL1), np.asarray(targetT1)))
            return np.expand_dims(img, 0),mask, np.expand_dims(gt, 0),np.expand_dims(slice, 0), mask1, np.expand_dims(target, 0)
        else:
            return np.expand_dims(img, 0), mask, np.expand_dims(gt, 0)





