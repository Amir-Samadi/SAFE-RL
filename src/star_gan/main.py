import os
import argparse
from src.star_gan.solver import Solver
from torch.backends import cudnn
import csv

from src.olson.main import olson_CF 
import src.olson.main as olson_main
import src.olson.model as olson_model
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.inception import InceptionScore
# from olson_main import main
# from src.olson.main import ablate_screen,zero_grads,train,array_to_pil_format,\
    # autoencoder_step,denorm,disc_step,model_step,save_models,prepro_dataset_batch,restrict_tf_memory
import argparse
import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import src.olson.model as model
from src.star_gan.data_loader import get_loader
from src.util import restrict_tf_memory, array_to_pil_format, denorm
import os
import time
from src.olson.atari_data import MultiEnvironment, ablate_screen, prepro_dataset_batch
os.environ['OMP_NUM_THREADS'] = '1'
from collections import deque
#ts = logutil.TimeSeries('Atari Distentangled Auto-Encoder')


restrict_tf_memory()

def str2bool(v):
    return v.lower() in ('true')

def basemodels(starGAN_config, PDCF_config, data_loader, teacher_model, Olson_config=None):
    config = starGAN_config
    print(config)
    # For fast training.
    cudnn.benchmark = True

    # Create directories if not exist.
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

    
    if(config.agent_type in ['SAFE_RL_starGAN', 'SAFE_RL_attGAN', 'GANterfactual']):
        # Solver for training and testing StarGAN.
        solver = Solver(data_loader, starGAN_config, PDCF_config, teacher_model)

        if PDCF_config.mode == 'train':
            solver.train()
        elif PDCF_config.mode == 'test':
            if "FID" in config.metrics or "KID" in config.metrics:
                temp_data_loader = get_loader(PDCF_config=PDCF_config, starGAN_config=starGAN_config, mode='train')
                for i, (x_real, obs_real, c_org) in enumerate(temp_data_loader):
                    x_real = x_real.to(PDCF_config.device)
                    x_real_normal = ((x_real-x_real.min()) / (x_real.max()-x_real.min()))
                    if "FID" in config.metrics:
                        solver.fid.update(x_real_normal, real=True)
                    if "KID" in config.metrics:
                        solver.kid.update(x_real_normal, real=True)
                    if (i == 300):
                        break
                del temp_data_loader
  
            solver.test()
    elif config.agent_type in ['Olson']:
        if PDCF_config.mode == 'train':
            olson_CF(Olson_config, teacher_model, data_loader,\
                KID=None,FID=None,LPIPS=None,IS=None)
        
        elif PDCF_config.mode == 'test':
            if "KID" in Olson_config.metrics:
                # If argument normalize is True images are expected to be dtype float and have values in the [0, 1] range
                kid = KernelInceptionDistance(subset_size=Olson_config.batch_size , reset_real_features=True, normalize=True).to(Olson_config.device)
            if "FID" in Olson_config.metrics:
                # If argument normalize is True images are expected to be dtype float and have values in the [0, 1] range
                fid = FrechetInceptionDistance(feature=64, reset_real_features=True, normalize=True).to(Olson_config.device)
                fid.reset()
            if "LPIPS" in Olson_config.metrics:
                # If set to True will instead expect input to be in the [0,1] range.
                lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).to(Olson_config.device)
            if "IS" in Olson_config.metrics:
                # If argument normalize is True images are expected to be dtype float and have values in the [0, 1] range
                inception = InceptionScore(normalize=True).to(Olson_config.device)
            if "FID" in config.metrics or "KID" in config.metrics:
                temp_data_loader = get_loader(PDCF_config=PDCF_config, starGAN_config=starGAN_config, mode='train')
                for i, (x_real, obs_real, c_org) in enumerate(temp_data_loader):
                    x_real = x_real.to(PDCF_config.device)
                    x_real_normal = ((x_real-x_real.min()) / (x_real.max()-x_real.min()))
                    if "FID" in config.metrics:
                        fid.update(x_real_normal, real=True)
                    if "KID" in config.metrics:
                        kid.update(x_real_normal, real=True)
                    if (i == 300):
                        break
                del temp_data_loader

            olson_CF(Olson_config, teacher_model, data_loader,\
                KID=kid,FID=fid,LPIPS=lpips,IS=inception)
        

    else:
        raise("the CF_method is out of [PDCF, SAFE_RL_starGAN, SAFE_RL_attGAN, Olson, GANterfactual]")

