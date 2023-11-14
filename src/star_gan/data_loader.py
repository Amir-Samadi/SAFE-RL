from typing import Tuple, Any

import numpy as np
from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder, DatasetFolder
from PIL import Image
import torch
import os
import random
import json
from matplotlib import pyplot as plt

class Atari(torch.utils.data.Dataset):
    def __init__(self, imageRoot, obsRoot, gtRoot, transform, augment=False):
        self.transform = transform
        self.gtRoot = gtRoot
        self.imageRoot = imageRoot
        self.obsRoot = obsRoot
        with open(gtRoot) as json_file:
            self.data = json.load(json_file)
        

        # self.data['image_names'] = sorted(self.data['image_names'], key=lambda k: k['file_name'])
        # self.data['label'] = sorted(self.data['images'], key=lambda k: k['file_name'])
        # self.data['obs_names'] = 
        
        self.count = len(self.data['image_names'])
        print("number of samples in dataset:{}".format(self.count))

    def __len__(self):
        return self.count

    def __getitem__(self, ind):
        
        imgdir = os.path.join(self.imageRoot, self.data['image_names'][ind])
        obsdir = os.path.join(self.obsRoot, self.data['obs_names'][ind])
        img = self.transform(Image.open(imgdir).convert('RGB'))
        # obs = self.transform(np.load(obsdir))
        obs = np.load(obsdir)/255.
        # obs = self.transform(np.load(obsdir).transpose(2, 1, 0))
        target = torch.tensor(self.data['label'][ind], dtype=torch.int64)

        return img, obs, target

class Atari_obj_list(torch.utils.data.Dataset):
    def __init__(self, objRoot, gtRoot, transform, augment=False):
        self.transform = transform
        self.gtRoot = gtRoot
        self.objRoot = objRoot
        with open(gtRoot) as json_file:
            self.data = json.load(json_file)
        # self.data['image_names'] = sorted(self.data['image_names'], key=lambda k: k['file_name'])
        # self.data['label'] = sorted(self.data['images'], key=lambda k: k['file_name'])
        # self.data['obs_names'] = 
        self.count = len(self.data['obj_names'])
        print("number of samples in dataset:{}".format(self.count))

    def __len__(self):
        return self.count

    def __getitem__(self, ind):
        
        objdir = os.path.join(self.objRoot, self.data['obj_names'][ind])
        # obs = self.transform(np.load(obsdir))
        obj = np.load(objdir)
        # obs = self.transform(np.load(obsdir).transpose(2, 1, 0))
        return obj


def get_loader(PDCF_config, starGAN_config, mode):
    
    transform = []
    # transform.append(T.CenterCrop(crop_size))
    transform.append(T.ToTensor())
    # transform.append(T.Resize(PDCF_config.obs_size))
    # transform.append(T.Normalize(mean=(0.5,), std=(0.5,)))
    # if image_channels == 1 or image_channels == 4:
    # else:
    #     transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    if PDCF_config.CF_method in ["PDCF"]:
        dataset_atari_objList = Atari_obj_list(
            objRoot=os.path.join(PDCF_config.dataset_dir, PDCF_config.env, PDCF_config.teacher_alg, "obj"),\
            gtRoot=os.path.join(PDCF_config.dataset_dir, PDCF_config.env, PDCF_config.teacher_alg,"objList_"+mode+".json"),
            transform=transform)
        
        data_loader = data.DataLoader(dataset=dataset_atari_objList,
                                    batch_size=starGAN_config.batch_size,
                                    shuffle=(mode=='train'),
                                    num_workers=PDCF_config.num_workers,
                                    drop_last=True)
    
    else:
        dataset_atari = Atari(imageRoot=os.path.join(PDCF_config.dataset_dir, PDCF_config.env, PDCF_config.teacher_alg, "img"),\
            obsRoot=os.path.join(PDCF_config.dataset_dir, PDCF_config.env, PDCF_config.teacher_alg, "obs"),\
            gtRoot=os.path.join(PDCF_config.dataset_dir, PDCF_config.env, PDCF_config.teacher_alg,"labels_"+mode+".json"),
            transform=transform)
        
        data_loader = data.DataLoader(dataset=dataset_atari,
                                    batch_size=starGAN_config.batch_size,
                                    shuffle=(mode=='train'),
                                    num_workers=starGAN_config.num_workers,
                                    drop_last=True)
    return data_loader 



def get_star_gan_transform(crop_size, image_size, image_channels):
    transform = []
    transform.append(T.CenterCrop(crop_size))
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    if image_channels == 1 or image_channels == 4:
        transform.append(T.Normalize(mean=(0.5,), std=(0.5,)))
    else:
        transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)
    return transform