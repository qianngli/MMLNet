import torch.utils.data as Data
import torchvision.transforms as transforms
from PIL import Image, ImageOps, ImageFilter
import os.path as osp
import sys
import random
import cv2
from albumentations import (
    RandomRotate90, Transpose, ShiftScaleRotate, Blur,
    OpticalDistortion, CLAHE, GaussNoise, MotionBlur,
    GridDistortion, HueSaturationValue,ToGray,
    MedianBlur, PiecewiseAffine, Sharpen, Emboss, RandomBrightnessContrast, Flip, OneOf, Compose
)
import numpy as np
from os import listdir

def Img_AUG(p=0.5):
    return Compose([
        RandomRotate90(),
        Flip(),
        Transpose(),
        ToGray(),
        OneOf([
            GaussNoise(),
        ], p=0.2),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        OneOf([
            OpticalDistortion(p=0.3),
            GridDistortion(p=0.1),
            PiecewiseAffine(p=0.3),
        ], p=0.2),
        OneOf([
            CLAHE(clip_limit=2),
            Sharpen(),
            Emboss(),
            RandomBrightnessContrast()
        ], p=0.5),
        HueSaturationValue(p=0.5),
    ], p=p)

class DatasetLoader(Data.Dataset):

    def __init__(self, args, mode='train'):

        self.dataset = args.dataset
        base_dir = '/data/liqiang/' + self.dataset  
        
        if self.dataset == 'IRSTD-1k':
            if mode == 'train':
                txtfile = 'trainval.txt'
            elif mode == 'test':
                txtfile = 'test.txt'
            else:
                raise ValueError("Unkown mode")

            self.list_dir = osp.join(base_dir, txtfile)
            self.imgs_dir = osp.join(base_dir, 'images')
            self.label_dir = osp.join(base_dir, 'masks')

            self.names_single = []
            with open(self.list_dir, 'r') as f:
                self.names_single += [line.strip() for line in f.readlines()]
        
            if mode == 'train':
                self.names = []
                for i in range(5):
                    for j in range(len(self.names_single)):              
                        self.names.append(self.names_single[j])
                random.shuffle(self.names)
            elif mode == 'test':
                self.names = self.names_single
            else:
                raise ValueError("Unkown mode")

            self.size = 480

        if self.dataset == 'sirst_aug' or self.dataset == 'MDvsFA_cGAN':

            if mode == 'train':
                self.names  = listdir(base_dir + '/train/images/') 
                self.imgs_dir = base_dir + '/train/images/'
                self.label_dir = base_dir + '/train/masks/'

            elif mode == 'test':
                self.names = listdir(base_dir + '/test/images/') 
                self.imgs_dir = base_dir + '/test/images/'
                self.label_dir = base_dir + '/test/masks/'

            else:
                raise ValueError("Unkown mode")

            if self.dataset == 'sirst_aug':
                self.size = 256
            else:
                self.size = 128

        self.mode = mode
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])

    def __getitem__(self, i):
       
        if self.dataset == 'IRSTD-1k':
            name = self.names[i] + '.png'
        else:
            name = self.names[i] 

        img_path = osp.join(self.imgs_dir, name)
        label_path = osp.join(self.label_dir, name)
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(label_path)

        if self.mode == 'train':
            img, mask = self.train_transform(img, mask)

        img, mask = self.transform(img), transforms.ToTensor()(mask)

        return img, mask, name

    def __len__(self):
        return len(self.names)

    def train_transform(self, img, mask):
        # random mirror

        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
       
        # random scale (short edge)
        if random.random() < 0.2: 

            long_size = random.randint(int(self.size  * 0.75), int(self.size  * 1.5))
            w, h = img.size
            if h > w:
                oh = long_size
                ow = int(1.0 * w * long_size / h + 0.5)
                short_size = ow
            else:
                ow = long_size
                oh = int(1.0 * h * long_size / w + 0.5)
                short_size = oh
            img = img.resize((ow, oh), Image.BILINEAR)
            mask = mask.resize((ow, oh), Image.NEAREST)
            # pad crop
            if short_size < self.size :
                padh = self.size  - oh if oh < self.size  else 0
                padw = self.size  - ow if ow < self.size  else 0
                img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
                mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
       
        # random crop crop_size
        w, h = img.size
        if w > self.size :
            x1 = random.randint(0, w - self.size )
            y1 = random.randint(0, h - self.size )
            img = img.crop((x1, y1, x1 + self.size , y1 + self.size ))
            mask = mask.crop((x1, y1, x1 + self.size , y1 + self.size ))

        # gaussian blur as in PSP
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
        #Albu aug
        img_1 = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
        mask_1 = cv2.cvtColor(np.asarray(mask),cv2.COLOR_RGB2BGR)
        data = {"image": img_1, "mask": mask_1}
        augmentation = Img_AUG(p=1.0)
        augmented = augmentation(**data)  
        img_1, mask_1 = augmented["image"], augmented["mask"]
        img = Image.fromarray(cv2.cvtColor(img_1,cv2.COLOR_BGR2RGB))
        mask = Image.fromarray(cv2.cvtColor(mask_1, cv2.COLOR_BGR2RGB))

        return img, mask

    def test_transform(self, img, mask):
        img = img.resize((self.size , self.size), Image.BILINEAR)
        mask = mask.resize((self.size , self.size), Image.BILINEAR)
        return img, mask
