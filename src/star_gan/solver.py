
from pytorch_grad_cam import DeepFeatureFactorization
from pytorch_grad_cam.utils.image import show_factorization_on_image
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad, LayerCAM, EigenGradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, SoftmaxOutputTarget,FasterRCNNBoxScoreTarget,ClassifierOutputSoftmaxTarget,RawScoresOutputTarget,BinaryClassifierOutputTarget\
    ,SemanticSegmentationTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torch import topk

import wandb

import datetime
import os
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import keras
from torchvision.utils import save_image

from src.atari_wrapper import AtariWrapper
from src.star_gan.model import Discriminator
from src.star_gan.model import Generator
from src.util import restrict_tf_memory, get_agent_prediction, load_baselines_model
import src.olson.model as olson_model
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('TkAgg')
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.inception import InceptionScore
from baselines.common.tf_util import adjust_shape
import csv

# matplotlib.rcParams['backend'] = 'TkAgg'
class Solver(object):
    """Solver for training and testing StarGAN."""

    def __init__(self, data_loader, config, PDCF_config, teacher_model):
        """Initialize configurations."""

        # Data loader.
        self.PDCF_config = PDCF_config
        self.celeba_loader = None
        self.rafd_loader = None
        self.data_loader = data_loader
        # Model configurations.
        self.c_dim = config.c_dim
        self.c2_dim = config.c2_dim
        self.image_size = config.image_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.g_lambda_cls = config.g_lambda_cls
        self.d_lambda_cls = config.d_lambda_cls
        self.lambda_rec_x = config.lambda_rec_x
        self.lambda_rec_sal = config.lambda_rec_sal
        self.lambda_sal_fuse = config.lambda_sal_fuse
        self.lambda_gp = config.lambda_gp
        self.g_loss_cls_of_d = config.g_loss_cls_of_d
        self.wandb_log = PDCF_config.wandb_log
        
        # Training configurations.
        self.dataset = config.dataset
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters

        # Test configurations.
        self.test_iters = config.test_iters

        self.device = PDCF_config.device

        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        # Extensions
        torch.cuda.empty_cache()
        
        # if config.agent_path is None:
        #     self.agent = None
        # elif self.agent_type == "deepq":
        #     restrict_tf_memory()
        #     self.pacman = True
        #     self.agent = keras.models.load_model(config.agent_path)
        # elif self.agent_type == "olson":
        #     restrict_tf_memory()
        #     self.pacman = False
        #     self.agent = olson_model.Agent(config.c_dim, 32).cuda()
        #     self.agent.load_state_dict(torch.load(config.agent_path, map_location=lambda storage, loc: storage))
        # elif self.agent_type == "acer":
        #     restrict_tf_memory()
        #     self.pacman = True
        #     self.agent = load_baselines_model(config.agent_path, num_actions=5, num_env=self.batch_size)
        # else:
        #     raise NotImplementedError("Known agent-types are: deepq, olson and acer")
        self.metrics = config.metrics
        if self.PDCF_config.mode=='test':
            if "KID" in self.metrics:
                # If argument normalize is True images are expected to be dtype float and have values in the [0, 1] range
                self.kid = KernelInceptionDistance(subset_size=self.batch_size , reset_real_features=True, normalize=True).to(self.device)
            if "FID" in self.metrics:
                # If argument normalize is True images are expected to be dtype float and have values in the [0, 1] range
                self.fid = FrechetInceptionDistance(feature=64, reset_real_features=True, normalize=True).to(self.device)
                self.fid.reset()
            if "LPIPS" in self.metrics:
                # If set to True will instead expect input to be in the [0,1] range.
                self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).to(self.device)
            if "IS" in self.metrics:
                # If argument normalize is True images are expected to be dtype float and have values in the [0, 1] range
                self.inception = InceptionScore(normalize=True).to(self.device)
            
        self.agent = teacher_model
        self.agent.policy = self.agent.policy.to(self.device)
        self.image_channels = config.image_channels
        self.agent_type = config.agent_type
        self.agent_algo = config.agent_algo
        self.saliency_method = config.saliency_method
        
        self.lambda_counter = config.lambda_counter
        self.counter_mode = config.counter_mode
        self.selective_counter = config.selective_counter
        self.ablate_agent = config.ablate_agent

        # Build the model and tensorboard.
        self.build_model()

    def saliency_generator(self, x, obs, org_class, desired_class):
        # obs_reg=obs
        # from util_highway import high_dimention_env_train
        # high_dimention_env = high_dimention_env_train(self.PDCF_config)
        # mean_reward = np.zeros(100)
        # j=0
        # for episode in range(10):
        #     obs, done = high_dimention_env.reset()
        #     done = False
        #     truncated = False
        #     high_dimention_env.render()
        #     while not (done or truncated):
        #         action,_ = np.array(self.agent.predict(obs,deterministic=True))
        #         obs, reward, done, truncated, info = high_dimention_env.step(action.item())
        #         high_dimention_env.render()
        #         mean_reward[j] += reward
        #     j+=1
        # high_dimention_env.close()
        # print ("teacher rewards:", mean_reward)
        # print ("teacher mean rewards:", np.average(mean_reward))
        if self.agent_algo in ["DQN"]:
            target_layers = [self.agent.policy.q_net.features_extractor.cnn[4]]
            model = self.agent.policy.q_net.features_extractor
        elif self.agent_algo in ["PPO", "A2C"]:
            target_layers = [self.agent.policy.features_extractor.cnn[4]]
            model = self.agent.policy.features_extractor

        # elif self.agent_algo in ["A2C"]:
            
        
        # target_layers = [decision_model.classifier]
        # target_layers = [decision_model.feat_extract.feat_extract.features]
        # target_layers = [decision_model.feat_extract.feat_extract.features.denseblock4.denselayer16.conv2]
        if (self.saliency_method == 'GradCAM'):
            cam = GradCAM(model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available())
        elif (self.saliency_method == 'HiResCAM'):
            cam = HiResCAM(model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available())
        elif (self.saliency_method == 'AblationCAM'):
            cam = AblationCAM(model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available())
        elif (self.saliency_method == 'ScoreCAM'):
            cam = ScoreCAM(model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available())
        elif (self.saliency_method == 'LayerCAM'):
            cam = LayerCAM(model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available())
        elif (self.saliency_method == 'FullGrad'):
            cam = FullGrad(model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available())
        elif (self.saliency_method == 'DeepFeatureFactorization'):
            cam = DeepFeatureFactorization(model=model, target_layer=target_layers, computation_on_concepts=None)
        elif (self.saliency_method == 'EigenCAM'):
            cam = EigenCAM(model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available())
        elif (self.saliency_method == 'EigenGradCAM'):
            cam = EigenGradCAM(model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available())
        elif (self.saliency_method == 'GradCAMPlusPlus'):
            cam = GradCAMPlusPlus(model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available())
        
        sal = torch.from_numpy(cam(input_tensor=obs, targets=[ClassifierOutputTarget(0)])).to(self.device)
        
        if (self.saliency_method == 'EigenCAM'):
            sal = sal.unsqueeze(1)
        
        # fig, ax = plt.subplots(3,5)
        # ax[0,0].imshow(obs[0,3].cpu())
        
        # ax[0,1].imshow(EigenCAM(model=self.agent.policy, target_layers=target_layers, use_cuda=torch.cuda.is_available())\
        #     (input_tensor=obs, targets=[ClassifierOutputTarget(0)])[0], cmap='jet')
        # ax[1,0].imshow(cam(input_tensor=obs, targets=[ClassifierOutputTarget(0)])[0], cmap='jet')
        # ax[1,1].imshow(cam(input_tensor=obs, targets=[ClassifierOutputTarget(1)])[0], cmap='jet')
        # ax[1,2].imshow(cam(input_tensor=obs, targets=[ClassifierOutputTarget(2)])[0], cmap='jet')
        # ax[1,3].imshow(cam(input_tensor=obs, targets=[ClassifierOutputTarget(3)])[0], cmap='jet')
        # ax[1,4].imshow(cam(input_tensor=obs, targets=[ClassifierOutputTarget(4)])[0], cmap='jet')
        # ax[2,0].imshow(cam(input_tensor=obs, targets=[ClassifierOutputSoftmaxTarget(0)])[0], cmap='jet')
        # ax[2,1].imshow(cam(input_tensor=obs, targets=[ClassifierOutputSoftmaxTarget(1)])[0], cmap='jet')
        # ax[2,2].imshow(cam(input_tensor=obs, targets=[ClassifierOutputSoftmaxTarget(2)])[0], cmap='jet')
        # ax[2,3].imshow(cam(input_tensor=obs, targets=[ClassifierOutputSoftmaxTarget(3)])[0], cmap='jet')
        # ax[2,4].imshow(cam(input_tensor=obs, targets=[ClassifierOutputSoftmaxTarget(4)])[0], cmap='jet')
        return sal
    
    def build_model(self):
        """Create a generator and a discriminator."""
        if self.agent_type in ["SAFE_RL_starGAN", "SAFE_RL_attGAN"]:
            self.G = Generator(self.agent_type, self.g_conv_dim, self.image_channels, self.PDCF_config.num_of_stack, self.PDCF_config.saliency_dim, self.c_dim, self.g_repeat_num)
        else:
            self.G = Generator(self.agent_type, self.g_conv_dim, self.image_channels, self.PDCF_config.num_of_stack, 0, self.c_dim, self.g_repeat_num)
        self.D = Discriminator(self.image_size, self.d_conv_dim, self.image_channels, self.PDCF_config.num_of_stack, self.c_dim, self.d_repeat_num)
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')
            
        self.G.to(self.device)
        self.D.to(self.device)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        for i in np.arange(batch_size):
            out[i, labels.long()[i]] = 1
        return out

    def create_labels(self, c_org, c_dim=5):
        """Generate target domain labels for debugging and testing."""
        # Get hair color indices.
        c_trg_list=[]
        for i in range(c_dim):
            c_trg = self.label2onehot(torch.ones(c_org.size(0))*i, c_dim)
            c_trg_list.append(c_trg.to(self.device))
        return c_trg_list

    def classification_loss(self, logit, target):
        return F.cross_entropy(logit, target)
   
    def get_validity(self, fake, c_trg):
        with torch.no_grad():            
            if self.agent_algo in ["DQN"]:
                self.agent.policy.to(self.device)
                fake_labels = self.agent.policy.q_net(fake*255).argmax(dim=1)
            elif self.agent_algo in ["PPO", "A2C"]:
                self.agent.policy.to(self.device)
                fake_labels = self.agent.policy.action_net(self.agent.policy.features_extractor(fake)).argmax(dim=1)
            # fake_labels = self.agent.policy.action_net(self.agent.policy.features_extractor(fake))
            valid_targets = torch.eq(fake_labels, c_trg.argmax(dim=1)).sum() 
            return (valid_targets/fake_labels.size(0)).tolist()

    def get_sparcity(self, x_real, fake):
        return ((fake*255).type(torch.int32) != (x_real*255).type(torch.int32)).type(torch.float32).mean()
    def get_mean_dis(self, x_real, fake):
        return ((fake*255).type(torch.int32) - (x_real*255).type(torch.int32)).abs().type(torch.float32).mean()
        # return F.l1_loss((fake*255).type(torch.int32), (x_real*255).type(torch.int32)).item()
   
    def preprocess_batch_for_agent(self, batch):
        batch = self.denorm(batch)
        batch = batch.detach().permute(0, 2, 3, 1).cpu().numpy()

        preprocessed_batch = []
        for i, frame in enumerate(batch):
            frame = (frame * 255).astype(np.uint8)
            if self.agent_type == "deepq":
                frame = AtariWrapper.preprocess_frame(frame)
            elif self.agent_type == "acer":
                frame = AtariWrapper.preprocess_frame_ACER(frame)
            else:
                frame = AtariWrapper.preprocess_space_invaders_frame(frame, ablate_agent=self.ablate_agent)

            frame = np.squeeze(frame)
            stacked_frames = np.stack([frame for _ in range(4)], axis=-1)
            if not self.pacman:
                stacked_frames = AtariWrapper.to_channels_first(stacked_frames)
            preprocessed_batch.append(stacked_frames)
        return np.array(preprocessed_batch)

    def train(self):
        """Train StarGAN within a single dataset."""
        # Set data loader.
        # Fetch fixed inputs for debugging.
        data_iter = iter(self.data_loader)
        x_fixed, obs_fixed, c_org = next(data_iter)
        x_fixed = x_fixed.to(self.device)
        if self.PDCF_config.env in ["highway", "roundabout"]:
            # obs_fixed = obs_fixed.to(self.device).permute(0,1,3,2).type(torch.float32)
            obs_fixed = obs_fixed.to(self.device).type(torch.float32)
            x_fixed = x_fixed.permute(0,1,3,2)
        elif self.PDCF_config.env in ["pong"]:
            obs_fixed = obs_fixed.to(self.device).permute(0,3,1,2).type(torch.float32)
        c_fixed_list = self.create_labels(c_org, self.c_dim)
        if self.agent_type in ["SAFE_RL_starGAN", "SAFE_RL_attGAN"]:
            sal_fixed = self.saliency_generator(x=x_fixed, obs=obs_fixed,  org_class=c_org, desired_class=c_fixed_list)
        
        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)





        # Start training.
        print('Start models...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch real images and labels.
            try:
                x_real, obs_real, label_org = next(data_iter)
            except:
                data_iter = iter(self.data_loader)
                x_real, obs_real, label_org = next(data_iter)
            # Generate target domain labels randomly.
            rand_idx = torch.randperm(label_org.size(0))
            label_trg = label_org[rand_idx]

            c_org = self.label2onehot(label_org, self.c_dim)
            c_trg = self.label2onehot(label_trg, self.c_dim)

            x_real = x_real.to(self.device)           # Input images.
            if self.PDCF_config.env in ["highway", "roundabout"]:
                # obs_real = obs_real.to(self.device).permute(0,1,3,2).type(torch.float32)
                obs_real = obs_real.to(self.device).type(torch.float32)
                x_real = x_real.permute(0,1,3,2)
            elif self.PDCF_config.env in ["pong"]:
                obs_real = obs_real.to(self.device).permute(0,3,1,2).type(torch.float32)
            c_org = c_org.to(self.device)             # Original domain labels.
            c_trg = c_trg.to(self.device)             # Target domain labels.
            label_org = label_org.to(self.device)     # Labels for computing classification loss.
            label_trg = label_trg.to(self.device)     # Labels for computing classification loss.

            if self.agent_type in ["SAFE_RL_starGAN", "SAFE_RL_attGAN"]:
                sal_real = self.saliency_generator(x=x_real, obs=obs_real,  org_class=label_org, desired_class=label_trg)


            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

            # Compute loss with real images.
            out_src, out_cls = self.D(torch.concatenate([x_real, obs_real], dim=1))
            # out_src, out_cls = self.D(x_real)
            d_loss_real = - torch.mean(out_src)
            d_loss_cls = self.classification_loss(out_cls, label_org)

            # Compute loss with fake images.
            if self.agent_type in ["SAFE_RL_starGAN", "SAFE_RL_attGAN"]:
                x_fake, obs_fake, sal_fake = self.G(torch.concatenate([x_real, obs_real], dim=1), c_trg, sal_real)
            else:
                x_fake, obs_fake, _ = self.G(torch.concatenate([x_real, obs_real], dim=1), c_trg, None)
            out_src, out_cls = self.D(torch.concatenate([x_fake.detach(), obs_fake.detach()], dim=1))
            # out_src, out_cls = self.D(x_fake.detach())
            d_loss_fake = torch.mean(out_src)

            # Compute loss for gradient penalty.
            alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
            x_hat =     (alpha * torch.concatenate([x_real.data, obs_real.data], dim=1)\
                    +   (1 - alpha) * torch.concatenate([x_fake.data, obs_fake.data], dim=1)).requires_grad_(True)

            # x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
            
            out_src, _ = self.D(x_hat)
            d_loss_gp = self.gradient_penalty(out_src, x_hat)

            # Backward and optimize.
            d_loss = d_loss_real + d_loss_fake + self.d_lambda_cls * d_loss_cls + self.lambda_gp * d_loss_gp
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # Logging.
            loss = {}
            loss['D/loss_real'] = d_loss_real.item()
            loss['D/loss_fake'] = d_loss_fake.item()
            loss['D/loss_cls'] = d_loss_cls.item()
            loss['D/loss_gp'] = d_loss_gp.item()
            
            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #
            
            if (i+1) % self.n_critic == 0:
                # Original-to-target domain.
                if self.agent_type in ["SAFE_RL_starGAN", "SAFE_RL_attGAN"]:
                    x_fake, obs_fake, sal_fake = self.G(torch.concatenate([x_real, obs_real], dim=1), c_trg, sal_real)
                else:
                    x_fake, obs_fake, _ = self.G(torch.concatenate([x_real, obs_real], dim=1), c_trg, None)
                out_src, out_cls = self.D(torch.concatenate([x_fake, obs_fake], dim=1))
                # out_src, out_cls = self.D(x_fake)

                g_loss_fake = - torch.mean(out_src)
                if (self.g_loss_cls_of_d):
                    g_loss_cls = self.classification_loss(out_cls, label_trg)
                else:
                    # agent_pred = self.agent.predict(obs_fake.detach().cpu().numpy(), deterministic=True)
                    # agent_pred = torch.from_numpy(agent_pred[0]).to(self.device).type(torch.float32)
                    # agent_pred = self.agent.policy.action_net(self.agent.policy.features_extractor(obs_fake))
                    if self.agent_algo in ["DQN"]:
                        self.agent.policy.to(self.device)
                        agent_pred = self.agent.policy.q_net(obs_fake*255)
                    elif self.agent_algo in ["PPO", "A2C"]:
                        self.agent.policy.to(self.device)
                        agent_pred = self.agent.policy.action_net(self.agent.policy.features_extractor(obs_fake))
                    # agent_pred = self.agent.policy.action_net(self.agent.policy.extract_features(obs_fake))
                    # g_loss_cls = self.classification_loss(agent_pred, label_trg) + self.classification_loss(out_cls, label_trg)
                    g_loss_cls = self.g_lambda_cls*self.classification_loss(agent_pred, label_trg) + self.d_lambda_cls*self.classification_loss(out_cls, label_trg)


                # Target-to-original domain.
                if self.agent_type in ["SAFE_RL_starGAN", "SAFE_RL_attGAN"]:
                    x_reconst, obs_reconst, sal_reconst = self.G(torch.concatenate([x_fake, obs_fake], dim=1), c_org, sal_fake)
                    g_loss_rec_x = torch.mean(torch.abs(x_real - x_reconst))
                    g_loss_rec_obs = torch.mean(torch.abs(obs_real - obs_reconst))
                    g_loss_rec_sal = torch.mean(torch.abs(sal_real - sal_reconst))
                
                    ##loss for fusing saliency
                    g_loss_sal_fuse = F.l1_loss   (x_fake * (1-sal_real.mean(dim=1).unsqueeze(1)),\
                                                (x_real * (1-sal_real.mean(dim=1).unsqueeze(1)))) +\
                                    F.l1_loss   (obs_fake * (1-sal_real.mean(dim=1).unsqueeze(1)),\
                                                (obs_real * (1-sal_real.mean(dim=1).unsqueeze(1)))) 
                         
                        
                    g_loss_rec = self.lambda_rec_x*(g_loss_rec_x+g_loss_rec_obs) + self.lambda_rec_sal*g_loss_rec_sal \
                        + self.lambda_sal_fuse * g_loss_sal_fuse
                
                
                else:
                    x_reconst, obs_reconst, _ = self.G(torch.concatenate([x_fake, obs_fake], dim=1), c_org, None)
                    g_loss_rec_x = torch.mean(torch.abs(x_real - x_reconst))
                    g_loss_rec_obs = torch.mean(torch.abs(obs_real - obs_reconst))
                    g_loss_rec = self.lambda_rec_x*(g_loss_rec_x+g_loss_rec_obs)
                
                # Backward and optimize.
                g_loss = g_loss_fake + g_loss_rec + g_loss_cls

                # Counter loss - only calculate and add if the target domain is not the original domain, otherwise
                # it would counteract the reconstruction loss in these cases
                if self.agent is not None: # and label_org[0] != label_trg[0]:
                    # if self.PDCF_config.env in ["highway", "roundabout"]:
                        # obs_fake = obs_fake.permute(0,1,3,2)
                        
                    # elif self.PDCF_config.env in ["pong"]:
                    #     obs_fake = obs_fake.permute(0,3,1,2)
                    if self.agent_algo in ["DQN"]:
                        self.agent.policy.to(self.device)
                        agent_prediction = self.agent.policy.q_net(obs_fake*255).argmax(dim=1)
                    elif self.agent_algo in ["PPO", "A2C"]:
                        self.agent.policy.to(self.device)
                        agent_prediction = self.agent.policy.action_net(self.agent.policy.features_extractor(obs_fake)).argmax(dim=1)
   
                    # # Permute to get channels last
                    # # x_fake_keras = self.preprocess_batch_for_agent(x_fake)
                    # if self.agent_type == "deepq":
                    #     # agent_prediction = self.agent.predict(obs_fake.detach().cpu().numpy(), deterministic=True)
                    #     agent_prediction = self.agent.policy.action_net(self.agent.policy.features_extractor(obs_fake.detach())).argmax(dim=1)
                    
                    # elif self.agent_type == "acer":
                    #     agent_prediction = self.agent.policy.action_net(self.agent.policy.features_extractor(obs_fake.detach())).argmax(dim=1)
                    #     # agent_prediction = self.agent.predict(obs_fake.detach().cpu().numpy(), deterministic=True)
                    #     # sess = self.agent.step_model.sess
                    #     # feed_dict = {self.agent.step_model.X: adjust_shape(self.agent.step_model.X, x_fake_keras)}
                    #     # agent_prediction = sess.run(self.agent.step_model.pi, feed_dict)
                    # else:
                    #     # torch_state = torch.Tensor(x_fake_keras).cuda()
                    #     # agent_prediction = self.agent.predict(obs_fake.detach().cpu().numpy(), deterministic=True)
                    #     agent_prediction = self.agent.policy.action_net(self.agent.policy.features_extractor(obs_fake.detach())).argmax(dim=1)
                    #     # agent_prediction = torch.softmax(self.agent.pi(self.agent(torch_state)).detach(), dim=-1)
                    #     # agent_prediction = agent_prediction.cpu().numpy()
                    if isinstance(agent_prediction, tuple):
                        # extract action distribution in case the agent is a dueling DQN
                        agent_prediction = agent_prediction[0]
                    # agent_prediction = torch.from_numpy(agent_prediction).to(self.device)

                    if self.selective_counter:
                        # filter samples in order to only calculate the counter-loss on samples
                        # where label_trg != label_org
                        relevant_samples = (label_trg != label_org).nonzero(as_tuple=True)[0].to(self.device)
                        relevant_agent_prediction = agent_prediction[relevant_samples].to(self.device).type(torch.float32)
                        relevant_agent_prediction = self.label2onehot(relevant_agent_prediction, self.c_dim).to(self.device)
                        relevant_c_trg = c_trg[relevant_samples].to(self.device).type(torch.float32)
                        relevant_label_trg = label_trg[relevant_samples].to(self.device).type(torch.int64)
                        
                    else:
                        relevant_agent_prediction = agent_prediction.to(self.device).type(torch.float32)
                        relevant_agent_prediction = self.label2onehot(relevant_agent_prediction, self.c_dim).to(self.device)
                        relevant_c_trg = c_trg.to(self.device).type(torch.float32)
                        relevant_label_trg = label_trg.to(self.device).type(torch.int64)

                    # using advantage/softmax since the DQN output is not softmax-distributed
                    if self.counter_mode == "cross_entropy":
                        g_loss_counter = self.classification_loss(relevant_agent_prediction, relevant_label_trg)
                    elif self.counter_mode == "raw":
                        # mse
                        g_loss_counter = torch.mean(torch.square(relevant_agent_prediction - relevant_c_trg))
                    elif self.counter_mode == "softmax":
                        fake_action_softmax = torch.softmax(relevant_agent_prediction, dim=-1)
                        # mse
                        g_loss_counter = torch.mean(torch.square(fake_action_softmax - relevant_c_trg))
                    elif self.counter_mode == "advantage":
                        # convert Q-values to advantage values
                        mean_q_values = torch.mean(relevant_agent_prediction, dim=-1)
                        advantages = torch.empty(relevant_agent_prediction.size())
                        for action in range(self.c_dim):
                            action_q_values = relevant_agent_prediction[:, action]
                            advantages[:, action] = action_q_values - mean_q_values

                        # perform softmax counter loss on advantage
                        advantage_softmax = torch.softmax(advantages, dim=-1)
                        g_loss_counter = torch.mean(torch.square(advantage_softmax - relevant_c_trg))
                    elif self.counter_mode == "z-score":
                        trg_action_q_values = torch.gather(relevant_agent_prediction, 1,
                                                           torch.unsqueeze(relevant_label_trg, dim=-1))
                        fake_action_z_score = (trg_action_q_values - torch.mean(relevant_agent_prediction, dim=-1)) / \
                                              torch.std(relevant_agent_prediction, dim=-1)
                        g_loss_counter = -torch.mean(fake_action_z_score)
                    else:
                        raise NotImplementedError("Known counter-modes are: 'raw', 'softmax', 'advantage' and"
                                                  "'z-score'")

                    g_loss += self.lambda_counter * g_loss_counter

                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # Logging.
                loss['G/loss_fake'] = g_loss_fake.item()
                loss['G/loss_rec_x'] = g_loss_rec_x.item()
                loss['G/loss_rec_obs'] = g_loss_rec_obs.item()
                if self.agent_type in ["SAFE_RL_starGAN", "SAFE_RL_attGAN"]:
                    loss['G/loss_rec_sal'] = g_loss_rec_sal.item()
                    loss['G/loss_sal_fuse'] = g_loss_sal_fuse.item()
                loss['G/loss_cls'] = g_loss_cls.item()
                if self.agent is not None: # and label_org[0] != label_trg[0]:
                    loss['G/loss_counter'] = g_loss_counter.item()

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)
                

            # Translate fixed images for debugging.
            if (i+1) % self.sample_step == 0:
                if self.agent_type in ["SAFE_RL_starGAN", "SAFE_RL_attGAN"]:
                    with torch.no_grad():
                        x_fake_list = [x_fixed]
                        obs_fake_list = [obs_fixed]
                        sal_fake_list = [sal_fixed]
                        for c_fixed in c_fixed_list:
                            x_fake, obs_fake, sal_fake = \
                                self.G(torch.concatenate([x_fixed, obs_fixed], dim=1), c_fixed, sal_fixed)
                            sal_fake_list.append(sal_fake)
                            obs_fake_list.append(obs_fake) 
                            x_fake_list.append(x_fake)
                    x_concat = torch.cat(x_fake_list, dim=3)
                    obs_concat = torch.cat(obs_fake_list, dim=3)
                    sal_concat = torch.cat(sal_fake_list, dim=3)
                    sample_path = os.path.join(self.sample_dir, '{}'.format(i+1))
                    save_image(x_concat.data.cpu(), sample_path+'-images.jpg', nrow=1, padding=0)
                    save_image(obs_concat[:,3:4].data.cpu(), sample_path+'-obs_3.jpg', nrow=1, padding=0, normalize=True)
                    save_image(obs_concat[:,2:3].data.cpu(), sample_path+'-obs_2.jpg', nrow=1, padding=0, normalize=True)
                    save_image(obs_concat[:,1:2].data.cpu(), sample_path+'-obs_1.jpg', nrow=1, padding=0, normalize=True)
                    save_image(obs_concat[:,0:1].data.cpu(), sample_path+'-obs_0.jpg', nrow=1, padding=0, normalize=True)
                    save_image(sal_concat.data.cpu(), sample_path+'-sal.jpg', nrow=1, padding=0, normalize=True)

                else:
                    with torch.no_grad():
                        x_fake_list = [x_fixed]
                        obs_fake_list = [obs_fixed]
                        for c_fixed in c_fixed_list:
                            x_fake, obs_fake, _ = \
                                self.G(torch.concatenate([x_fixed, obs_fixed], dim=1), c_fixed, None)
                            obs_fake_list.append(obs_fake) 
                            x_fake_list.append(x_fake)
                    x_concat = torch.cat(x_fake_list, dim=3)
                    obs_concat = torch.cat(obs_fake_list, dim=3)
                    sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(i+1))
                    sample_path = os.path.join(self.sample_dir, '{}'.format(i+1))
                    save_image(x_concat.data.cpu(), sample_path+'-images.jpg', nrow=1, padding=0)
                    save_image(obs_concat[:,3:4].data.cpu(), sample_path+'-obs_3.jpg', nrow=1, padding=0, normalize=True)
                    save_image(obs_concat[:,2:3].data.cpu(), sample_path+'-obs_2.jpg', nrow=1, padding=0, normalize=True)
                    save_image(obs_concat[:,1:2].data.cpu(), sample_path+'-obs_1.jpg', nrow=1, padding=0, normalize=True)
                    save_image(obs_concat[:,0:1].data.cpu(), sample_path+'-obs_0.jpg', nrow=1, padding=0, normalize=True)
                print('Saved real and fake images into {}...'.format(sample_path))
                if self.wandb_log:
                    wandb.log(loss)

                
                
            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i+1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # Decay learning rates.
            if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

    def test(self):
        metrics_dict = {}
        wandb_metrics_dict = {}
        if "KID" in self.metrics:
            metrics_dict['metric/KID'] = []
            metrics_dict['metric/KID_std'] = []
        if "FID" in self.metrics:
            metrics_dict['metric/FID'] = []
        if "LPIPS" in self.metrics:
            metrics_dict['metric/LPIPS'] = []
        if "IS" in self.metrics:
            metrics_dict['metric/IS'] = []
            metrics_dict['metric/IS_std'] = []
        if "sparsity" in self.metrics:
            metrics_dict['metric/sparsity'] = []
        if "mean_dis" in self.metrics:
            metrics_dict['metric/mean_dis'] = []
        if "validity" in self.metrics:
            metrics_dict['metric/validity'] = []

        """Translate images using StarGAN trained on a single dataset."""
        # Load the trained generator.
        self.restore_model(self.test_iters)
               
        with torch.no_grad():
            for i, (x_real, obs_real, c_org) in enumerate(self.data_loader):
                if i==50:
                    break
                x_real = x_real.to(self.device)
                c_trg_list = self.create_labels(c_org, self.c_dim)
                if self.PDCF_config.env in ["highway", "roundabout"]:
                    # obs_real = obs_real.to(self.device).permute(0,1,3,2).type(torch.float32)
                    obs_real = obs_real.to(self.device).type(torch.float32)
                    x_real = x_real.permute(0,1,3,2)
                elif self.PDCF_config.env in ["pong"]:
                    obs_real = obs_real.to(self.device).permute(0,3,1,2).type(torch.float32)
                x_fake_list = [x_real]
                obs_fake_list = [obs_real]
                if self.agent_type in ["SAFE_RL_starGAN", "SAFE_RL_attGAN"]:
                    sal_fake_list = []
                for c_fixed in c_trg_list:
                    if self.agent_type in ["SAFE_RL_starGAN", "SAFE_RL_attGAN"]:
                        sal_real = self.saliency_generator(x=x_real, obs=obs_real,  org_class=c_org, desired_class=c_fixed)
                        x_fake, obs_fake, sal_fake = \
                            self.G(torch.concatenate([x_real, obs_real], dim=1), c_fixed, sal_real)
                        sal_fake_list.append(sal_fake)
                        obs_fake_list.append(obs_fake) 
                        x_fake_list.append(x_fake)
                    else:
                        x_fake, obs_fake, _ = \
                            self.G(torch.concatenate([x_real, obs_real], dim=1), c_fixed, None)
                        obs_fake_list.append(obs_fake) 
                        x_fake_list.append(x_fake)
################ =================================================================================== ################
################                                   metrics_value                                     ################
################ =================================================================================== ################
                    if "KID" in self.metrics or "FID" in self.metrics or "LPIPS" in self.metrics or "IS" in self.metrics:
                        x_real_normal = ((x_real-x_real.min()) / (x_real.max()-x_real.min()))
                        x_fake_normal = ((x_fake-x_fake.min()) / (x_fake.max()-x_fake.min()))
                        obs_real_normal = ((obs_real-obs_real.min()) / (obs_real.max()-obs_real.min()))
                        obs_fake_normal = ((obs_fake-obs_fake.min()) / (obs_fake.max()-obs_fake.min()))

                    if "KID" in self.metrics:
                        # self.kid.update(x_real_normal, real=True)
                        self.kid.update(x_fake_normal, real=False)
                        kidValue = self.kid.compute()
                        metrics_dict['metric/KID'].append(kidValue[0].item())
                        metrics_dict['metric/KID_std'].append(kidValue[1].item())
                        wandb_metrics_dict['metric/KID'] = metrics_dict['metric/KID'][-1]
                        wandb_metrics_dict['metric/KID_std'] = metrics_dict['metric/KID_std'][-1]
                
                    if "FID" in self.metrics:
                        # self.fid.update(x_real_normal, real=True)
                        self.fid.update(x_fake_normal, real=False)
                        metrics_dict['metric/FID'].append(self.fid.compute().item())
                        wandb_metrics_dict['metric/FID'] = metrics_dict['metric/FID'][-1]
                    
                    if "LPIPS" in self.metrics:
                        metrics_dict['metric/LPIPS'].append(self.lpips(x_real_normal, x_fake_normal).item()) 
                        wandb_metrics_dict['metric/LPIPS'] = metrics_dict['metric/LPIPS'][-1]
                    
                    if "IS" in self.metrics:
                        self.inception.update(x_fake_normal)
                        inceptionScore = self.inception.compute()
                        metrics_dict['metric/IS'].append(inceptionScore[0].item()) 
                        metrics_dict['metric/IS_std'].append(inceptionScore[1].item()) 
                        wandb_metrics_dict['metric/IS'] = metrics_dict['metric/IS'][-1] 
                        wandb_metrics_dict['metric/IS_std'] = metrics_dict['metric/IS_std'][-1] 
                    
                    if "sparsity" in self.metrics:
                        #sparsity should see the actual generated image not normalized one
                        metrics_dict['metric/sparsity'].append(self.get_sparcity(obs_real, obs_fake))               
                        wandb_metrics_dict['metric/sparsity'+str(c_fixed[0].nonzero().item())] = metrics_dict['metric/sparsity'][-1]             
                    
                    if "mean_dis" in self.metrics:
                        #sparsity should see the actual generated image not normalized one
                        metrics_dict['metric/mean_dis'].append(self.get_mean_dis(obs_real, obs_fake))               
                        wandb_metrics_dict['metric/mean_dis'+str(c_fixed[0].nonzero().item())] = metrics_dict['metric/mean_dis'][-1]  

                    if "validity" in self.metrics:
                        #decision_model should see the actual generated image not normalized one
                        metrics_dict['metric/validity'].append(self.get_validity(obs_fake, c_fixed))
                        wandb_metrics_dict['metric/validity'+str(c_fixed[0].nonzero().item())] = metrics_dict['metric/validity'][-1]
                        
                    if(self.wandb_log):
                        wandb.log(wandb_metrics_dict)
                
################ =================================================================================== ################
################                                 save the results                                    ################
################ =================================================================================== ################                
                x_concat = torch.cat(x_fake_list, dim=3)
                obs_concat = torch.cat(obs_fake_list, dim=3)
                result_path = os.path.join(self.result_dir, '{}'.format(i+1))
                save_image(x_concat.data.cpu(), result_path+'-images.jpg', nrow=1, padding=0)
                save_image(obs_concat[:,3:4].data.cpu(), result_path+'-obs_3.jpg', nrow=1, padding=0, normalize=True)
                save_image(obs_concat[:,2:3].data.cpu(), result_path+'-obs_2.jpg', nrow=1, padding=0, normalize=True)
                save_image(obs_concat[:,1:2].data.cpu(), result_path+'-obs_1.jpg', nrow=1, padding=0, normalize=True)
                save_image(obs_concat[:,0:1].data.cpu(), result_path+'-obs_0.jpg', nrow=1, padding=0, normalize=True)
                if self.agent_type in ["SAFE_RL_starGAN", "SAFE_RL_attGAN"]:
                    sal_fake_list.insert(0, sal_real)
                    sal_concat = torch.cat(sal_fake_list, dim=3)
                    save_image(sal_concat.data.cpu(), result_path+'-sal.jpg', nrow=1, padding=0, normalize=True)
                print('Saved real and fake images into {}...'.format(result_path))
        
        results = dict()
        for key in metrics_dict.keys(): 
            results[key] = sum(metrics_dict[key])/len(metrics_dict[key])
            if torch.is_tensor(results[key]):
                results[key] = results[key].tolist()
        results['FID'] = metrics_dict['metric/FID']
        results['KID'] = metrics_dict['metric/KID']
        with open(self.result_dir+'Models_results.csv', 'w') as f:
            w = csv.DictWriter(f, fieldnames=results.keys())
            w.writeheader()
            w.writerows([results])
        # print("average value of metric/KID: {}".format(sum(metrics_dict['metric/KID'])/len(metrics_dict['metric/KID']))) 
        # print("average value of metric/KID_std: {}".format(sum(metrics_dict['metric/KID_std'])/len(metrics_dict['metric/KID_std']))) 
        # print("average value of metric/LPIPS: {}".format(sum(metrics_dict['metric/LPIPS'])/len(metrics_dict['metric/LPIPS']))) 
        # print("average value of metric/IS: {}".format(sum(metrics_dict['metric/IS'])/len(metrics_dict['metric/IS']))) 
        # print("average value of metric/IS_std: {}".format(sum(metrics_dict['metric/IS_std'])/len(metrics_dict['metric/IS_std']))) 
        # print("average value of metric/sparsity: {}".format(sum(metrics_dict['metric/sparsity'])/len(metrics_dict['metric/sparsity']))) 
        # print("average value of metric/mean_dis: {}".format(sum(metrics_dict['metric/mean_dis'])/len(metrics_dict['metric/mean_dis']))) 
        # print("average value of metric/validity: {}".format(sum(metrics_dict['metric/validity'])/len(metrics_dict['metric/validity']))) 