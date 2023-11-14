import argparse
from torchvision.utils import save_image
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from PIL import Image
import src.olson.top_entropy_counterfactual as olson_tec

import csv
import src.olson.model as model
from src.star_gan.data_loader import get_loader
from src.util import restrict_tf_memory, array_to_pil_format, denorm

import os
#import logutil
import time
from src.olson.atari_data import MultiEnvironment, ablate_screen, prepro_dataset_batch

os.environ['OMP_NUM_THREADS'] = '1'

from collections import deque

#ts = logutil.TimeSeries('Atari Distentangled Auto-Encoder')


# restrict_tf_memory()


# print('Parsing arguments')
# parser = argparse.ArgumentParser()
# parser.add_argument('--batch_size', type=int, default=64)
# parser.add_argument('--epsilon', type=float, default=.2)
# parser.add_argument('--lr', type=float, default=1e-4)
# # Output directory of the model that creates counterfactual states
# parser.add_argument('--checkpoint_dir', type=str, default='../../res/models/PacMan_FearGhost2_3_Olson')
# parser.add_argument('--epochs', type=int, default=300)

# parser.add_argument('--latent', type=int, default=16)
# parser.add_argument('--wae_latent', type=int, default=128)
# parser.add_argument('--agent_latent', type=int, default=512)
# parser.add_argument('--seed', type=int, default=13)
# parser.add_argument('--env', type=str, default="MsPacmanNoFrameskip-v4")
# # Directory of the used wasserstein encoder
# parser.add_argument('--Q', type=str, default="../../res/models/PacMan_FearGhost2_3_Olson_wae/Q")
# parser.add_argument('--P', type=str, default="../../res/models/PacMan_FearGhost2_3_Olson_wae/P")
# parser.add_argument('--missing', type=str, default="none")
# # Directory of the RL-Agent that should be explained
# parser.add_argument('--agent_file', type=str, default="../../res/agents/ACER_PacMan_FearGhost2_cropped_5actions_40M_3.pt")
# parser.add_argument('--enc_lam', type=float, default=5)
# parser.add_argument('--clip', type=float, default=.0001)
# parser.add_argument('--gen_lam', type=float, default=.5)
# parser.add_argument('--starting_epoch', type=int, default=0)
# parser.add_argument('--info', type=str, default="")
# parser.add_argument('--cf_loss', type=str, default="None")
# parser.add_argument('--m_frames', type=int, default=40)
# parser.add_argument('--fskip', type=int, default=4)
# parser.add_argument('--use_agent', type=int, default=1)
# parser.add_argument('--gpu', type=int, default=7)

# parser.add_argument('--use_dataset', type=bool, default=True)
# # The dataset used for training the model
# parser.add_argument('--dataset_dir', type=str, default="../../res/datasets/ACER_PacMan_FearGhost2_cropped_5actions_40M_3_Unique")
# parser.add_argument('--img_size', type=str, default=176)
# parser.add_argument('--img_channels', type=int, default=3)
# parser.add_argument('--action_size', type=int, default=5)
# parser.add_argument('--is_pacman', type=bool, default=True)


# args = parser.parse_args()

map_loc = {
        'cuda:0': 'cuda:0',
        'cuda:1': 'cuda:1',
}

def olson_CF(args, agent, data_loader, KID=None,FID=None,LPIPS=None,IS=None):
    def zero_grads():
        optim_enc.zero_grad()
        optim_gen.zero_grad()
        optim_disc.zero_grad()

    def model_step(state, p):
        #get variables
        z = encoder(state)
        reconstructed = generator(z, p)
        disc_pi, _ = discriminator(z)

        #different loss functions
        ae_loss, enc_loss_pi = autoencoder_step(p, disc_pi + TINY, reconstructed, state)
        disc_loss_pi = disc_step(z.detach(), p)

        return ae_loss, enc_loss_pi, disc_loss_pi

    def autoencoder_step(p, disc_pi, reconstructed, state):
        zero_grads()

        #disentanglement loss and L2 loss
        #enc_loss =  enc_lambda * (1 -((p, value - disc_labels)**2).mean())
        #disc_pi, disc_v = disc_labels
        #enc_loss_v = (((disc_v - running_average)/value)**2).mean()
        enc_loss_pi = 1 + (disc_pi * torch.log(disc_pi)).mean()


        ae_loss = .5*torch.sum(torch.max((reconstructed - state)**2, clip) ) / bs
        enc_loss = enc_lambda*(enc_loss_pi)
        (enc_loss + ae_loss).backward()

        optim_enc.step()
        optim_gen.step()

        return ae_loss.item(), enc_loss_pi.item()

    def disc_step(z, p):
        zero_grads()

        disc_pi, _ = discriminator(z)

        #disc_loss_v  = (((real_v - disc_v)/real_v)**2).mean()
        disc_loss_pi = ((p - disc_pi) **2).mean()

        (disc_loss_pi).backward()
        optim_disc.step()

        return disc_loss_pi.item()

    def train(epoch, data_loader):
        #import pdb; pdb.set_trace()
        data_iter = iter(data_loader)
        if args.use_dataset:
            x_real, obs_real, label_org = next(data_iter)
            # batch = array_to_pil_format(denorm(x_real).detach().permute(0, 2, 3, 1).cpu().numpy())
            # new_frame_rgb, new_frame_bw = prepro_dataset_batch(batch, pacman=args.is_pacman)
            done = False
        else:
            new_frame_rgb, new_frame_bw = envs.reset()

        if args.env in ["highway", "roundabout"]:
            obs_real = obs_real.to(args.device).type(torch.float32)
            x_real = x_real.to(args.device).type(torch.float32)
        elif args.env in ["pong"]:
            obs_real = obs_real.to(args.device).permute(0,3,1,2).type(torch.float32)
            x_real = x_real.to(args.device).type(torch.float32)
        #agent_state = Variable(torch.Tensor(new_frame.mean(3)).unsqueeze(1)).cuda()
        #agent_state_history = deque([agent_state, agent_state.clone(), agent_state.clone(),agent_state.clone()], maxlen=4)

        # state = Variable(torch.Tensor(new_frame_rgb).permute(0,3,1,2)).cuda()
        state = x_real
        agent_state = obs_real
        # agent_state = Variable(torch.Tensor(ablate_screen(new_frame_bw, args.missing)).cuda())
        agent_state_history = deque([agent_state, agent_state.clone(), agent_state.clone(),agent_state.clone()], maxlen=4)

        actions_size = args.action_size
        greedy = np.ones(args.batch_size).astype(int)

        #global running_average
        #running_average = agent.value(agent(torch.cat(list(agent_state_history), dim=1))).mean().detach() if running_average.mean().item() == 0 else running_average

        fs = 0
        for i in range(int( mil / bs)):
            # agent_state = torch.cat(list(agent_state_history), dim=1)#torch.cat(list(agent_state_history), dim=1)

            
            # z_a = agent(agent_state)
            # # value = agent.value(z_a).detach()
            # p = F.softmax(agent.pi(z_a), dim=1)
            # real_actions = p.max(1)[1].data.cpu().numpy()
            ###############samir
            if args.agent_algo in ["DQN"]:
                z_a = agent.policy.q_net.features_extractor(agent_state)
                p = agent.policy.q_net.q_net(z_a)
            elif args.agent_algo in ["PPO", "A2C"]:
                z_a = agent.policy.features_extractor(agent_state)
                p = agent.policy.action_net(z_a)

            # z_a = agent.policy.extract_features(agent_state)
            # p = agent.policy.action_net(agent.policy.extract_features(agent_state))
            real_actions = p.argmax(dim=1).data.cpu().numpy()
            #value = agent.policy.value_net(agent.policy.extract_features(agent_state)).detach()
            # action = agent.policy.action_net(agent.policy.extract_features(agent_state.to('cuda'))).argmax(dim=1)
            # action = agent.policy(agent_state, deterministic=True)[0]
            # value = agent.policy(agent_state, deterministic=True)[1]
            ###############
            
            #running_average = (running_average *.999) + (value.mean() * .001)
            #if fs >= 10000:
            #    import pdb; pdb.set_trace()

            #loss functions
            ae_loss, enc_loss_pi, disc_loss_pi = model_step(state, p.detach())

            #import pdb; pdb.set_trace()
            if args.use_dataset:
                # getting new images and creating a new iter if the last iteration is complete
                try:
                    x_real, obs_real, label_org = next(data_iter)
                except:
                    data_iter = iter(data_loader)
                    x_real, obs_real, label_org = next(data_iter)
                if len(x_real) != bs:
                    data_iter = iter(data_loader)
                    x_real, label_org = next(data_iter)
                # batch = array_to_pil_format(denorm(x_real).detach().permute(0, 2, 3, 1).cpu().numpy())
                # new_frame_rgb, new_frame_bw = prepro_dataset_batch(batch, pacman=args.is_pacman)
                if args.env in ["highway", "roundabout"]:
                    obs_real = obs_real.to(args.device).type(torch.float32)
                    x_real = x_real.to(args.device).type(torch.float32)
                elif args.env in ["pong"]:
                    obs_real = obs_real.to(args.device).permute(0,3,1,2).type(torch.float32)
                    x_real = x_real.to(args.device).type(torch.float32)
                state = x_real
                agent_state = obs_real

            else:
                if np.random.random_sample() < args.epsilon:
                    actions = np.random.randint(actions_size, size=bs)
                    actions = (real_actions * greedy) + (actions * (1 - greedy))
                else:
                    actions = real_actions
                new_frame_rgb, new_frame_bw, _, done, _ = envs.step(actions)

            # agent_state_history.append(Variable(torch.Tensor(ablate_screen(new_frame_bw, args.missing)).cuda()))
            # state = Variable(torch.Tensor(new_frame_rgb).permute(0,3,1,2)).cuda()

            if np.sum(done) > 0:
                for j, d in enumerate(done):
                    if d:
                        greedy[j] = (np.random.rand(1)[0] > (1 - args.epsilon)).astype(int)


            if i % 50 == 0:
                print("Recon: {:.3f} --Enc entropy: {:.3f} --disc pi: {:.3f}".format(ae_loss, enc_loss_pi, disc_loss_pi))
                if i % 500 == 0:
                    fs = (i * args.batch_size) + (epoch * mil)
                    #print("running running_average is: {}".format(running_average.item()))
                    print("{} frames processed. {:.2f}% complete".format(fs , 100* (fs / (args.m_frames * mil))))

    def save_models(epoch, args):
        torch.save(generator.state_dict(), os.path.join(args.model_save_dir, 'gen{}'.format(epoch)))
        torch.save(encoder.state_dict(), os.path.join(args.model_save_dir, 'enc{}'.format(epoch)))
        torch.save(discriminator.state_dict(), os.path.join(args.model_save_dir, 'disc{}'.format(epoch)))
        print('models are saved into {}...'.format(args.model_save_dir))
    
    def load_models(args):
        encoder.load_state_dict(torch.load(os.path.join(args.model_save_dir, 'enc{}'.format(args.load_epoch)), map_location=args.device))
        generator.load_state_dict(torch.load(os.path.join(args.model_save_dir, 'gen{}'.format(args.load_epoch)), map_location=args.device))
        discriminator.load_state_dict(torch.load(os.path.join(args.model_save_dir, 'disc{}'.format(args.load_epoch)), map_location=args.device))
        print("model from epoch {} is loaded".format(args.load_epoch))
    def test(data_loader, args):
        metrics_dict = {}
        wandb_metrics_dict = {}
        if "KID" in args.metrics:
            metrics_dict['metric/KID'] = []
            metrics_dict['metric/KID_std'] = []
        if "FID" in args.metrics:
            metrics_dict['metric/FID'] = []
        if "LPIPS" in args.metrics:
            metrics_dict['metric/LPIPS'] = []
        if "IS" in args.metrics:
            metrics_dict['metric/IS'] = []
            metrics_dict['metric/IS_std'] = []
        if "sparsity" in args.metrics:
            metrics_dict['metric/sparsity'] = []
        if "mean_dis" in args.metrics:
            metrics_dict['metric/mean_dis'] = []
        if "validity" in args.metrics:
            metrics_dict['metric/validity'] = []
        load_models(args)
        for i, (x_real, obs_real, c_org) in enumerate(data_loader):

            if args.env in ["highway", "roundabout"]:
                obs_real = obs_real.to(args.device).type(torch.float32)
                x_real = x_real.to(args.device).type(torch.float32)
            elif args.env in ["pong"]:
                obs_real = obs_real.to(args.device).permute(0,3,1,2).type(torch.float32)
                x_real = x_real.to(args.device).type(torch.float32)

            x_fake_list = [x_real]
            obs_fake_list = [obs_real]
            CF_list = []
            state_real = x_real
            agent_state_real = obs_real

            encoder.eval()
            generator.eval()

            for target_domain in range(args.action_size):
                obs_fake, x_fake = generate_olson_counterfactual(agent_state_real, state_real, \
                                                             target_domain, agent, encoder, generator,\
                                                            None, None, None, None, max_iters=5000)
                x_fake_list.append(x_fake)
                obs_fake_list.append(obs_fake)
            
            
            ################ =================================================================================== ################
################                                   metrics_value                                     ################
################ =================================================================================== ################
                if "KID" in args.metrics or "FID" in args.metrics or "LPIPS" in args.metrics or "IS" in args.metrics:
                    x_real_normal = ((x_real-x_real.min()) / (x_real.max()-x_real.min()))
                    x_fake_normal = ((x_fake-x_fake.min()) / (x_fake.max()-x_fake.min()))
                    # obs_real_normal = ((obs_real-obs_real.min()) / (obs_real.max()-obs_real.min()))
                    # obs_fake_normal = ((obs_fake-obs_fake.min()) / (obs_fake.max()-obs_fake.min()))

                if "KID" in args.metrics:
                    KID.update(x_fake_normal, real=False)
                    kidValue = KID.compute()
                    metrics_dict['metric/KID'].append(kidValue[0].item())
                    metrics_dict['metric/KID_std'].append(kidValue[1].item())
                    wandb_metrics_dict['metric/KID'] = metrics_dict['metric/KID'][-1]
                    wandb_metrics_dict['metric/KID_std'] = metrics_dict['metric/KID_std'][-1]
            
                if "FID" in args.metrics:
                    FID.update(x_fake_normal, real=False)
                    metrics_dict['metric/FID'].append(FID.compute().item())
                    wandb_metrics_dict['metric/FID'] = metrics_dict['metric/FID'][-1]
                
                if "LPIPS" in args.metrics:
                    metrics_dict['metric/LPIPS'].append(LPIPS(x_real_normal, x_fake_normal).item()) 
                    wandb_metrics_dict['metric/LPIPS'] = metrics_dict['metric/LPIPS'][-1]
                
                if "IS" in args.metrics:
                    IS.update(x_fake_normal)
                    inceptionScore = IS.compute()
                    metrics_dict['metric/IS'].append(inceptionScore[0].item()) 
                    metrics_dict['metric/IS_std'].append(inceptionScore[1].item()) 
                    wandb_metrics_dict['metric/IS'] = metrics_dict['metric/IS'][-1] 
                    wandb_metrics_dict['metric/IS_std'] = metrics_dict['metric/IS_std'][-1] 
                
                if "sparsity" in args.metrics:
                    #sparsity should see the actual generated image not normalized one
                    metrics_dict['metric/sparsity'].append(get_sparcity(x_real_normal, x_fake_normal))               
                    wandb_metrics_dict['metric/sparsity'+str(target_domain)] = metrics_dict['metric/sparsity'][-1]             
                
                if "mean_dis" in args.metrics:
                    #sparsity should see the actual generated image not normalized one
                    metrics_dict['metric/mean_dis'].append(get_mean_dis(x_real_normal, x_fake_normal))               
                    wandb_metrics_dict['metric/mean_dis'+str(target_domain)] = metrics_dict['metric/mean_dis'][-1]  

                if "validity" in args.metrics:
                    #decision_model should see the actual generated image not normalized one

                    metrics_dict['metric/validity'].append(get_validity(obs_fake, target_domain))
                    wandb_metrics_dict['metric/validity'+str(target_domain)] = metrics_dict['metric/validity'][-1]
            
            x_fake_cat = torch.cat(x_fake_list, dim=3)
            # obs_fake_cat = torch.cat(obs_fake, dim=3)
            # agent_state_fixed_cat = torch.cat([agent_state_fixed, reconstructed], dim=3)
            save_image(x_fake_cat.data.cpu(), args.result_dir+'/'+str(i)+'-images.png', nrow=1, padding=1, normalize=True)
            # save_image(obs_fake_cat.data.cpu()[:,0:1], args.result_dir+'/'+str(i)+'-obs0.png', nrow=1, padding=0, normalize=True)
            # save_image(obs_fake_cat.data.cpu()[:,1:2], args.result_dir+'/'+str(i)+'-obs1.png', nrow=1, padding=0, normalize=True)
            # save_image(obs_fake_cat.data.cpu()[:,2:3], args.result_dir+'/'+str(i)+'-obs2.png', nrow=1, padding=0, normalize=True)
            # save_image(obs_fake_cat.data.cpu()[:,3:4], args.result_dir+'/'+str(i)+'-obs3.png', nrow=1, padding=0, normalize=True)
            print('Saved real and fake images into {}...'.format(args.result_dir))
        
        results = dict()
        for key in metrics_dict.keys(): 
            results[key] = sum(metrics_dict[key])/len(metrics_dict[key])
            if torch.is_tensor(results[key]):
                results[key] = results[key].tolist()
        results['FID'] = metrics_dict['metric/FID']
        results['KID'] = metrics_dict['metric/KID']
        with open(args.result_dir+'Models_results.csv', 'w') as f:
            w = csv.DictWriter(f, fieldnames=results.keys())
            w.writeheader()
            w.writerows([results])
    
    def get_validity(fake, c_trg):
        with torch.no_grad():
            if args.agent_algo in ["DQN"]:
                p_cf = F.softmax(agent.policy.q_net.q_net(fake))
            elif args.agent_algo in ["PPO", "A2C"]:
                p_cf = F.softmax(agent.policy.action_net(fake))            
            
            fake_labels = p_cf.argmax(dim=1)
            # fake_labels = self.agent.policy.action_net(self.agent.policy.features_extractor(fake))
            valid_targets = torch.eq(fake_labels, c_trg).sum() 
            return (valid_targets/fake_labels.size(0)).tolist()

    def get_sparcity(x_real, fake):
        return ((fake*255).type(torch.int32) != (x_real*255).type(torch.int32)).type(torch.float32).mean()
    
    def get_mean_dis(x_real, fake):
        return ((fake*255).type(torch.int32) - (x_real*255).type(torch.int32)).abs().type(torch.float32).mean()
    
    def generate_olson_counterfactual(agent_state_fixed, state_fixed\
        , target_domain, agent, encoder, generator, Q, P, is_pacman, ablate_agent,
                                    max_iters=5000):
        """
        Generates a counterfactual frame for the given image with a trained approach of Olson et al.

        :param image: The PIL image to generate a counterfactual for.
        :param target_domain: The integer encoded target action/domain.
        :param agent: The agent that was used to train the explainability approach.
        :param encoder: The trained encoder.
        :param generator: The trained generator.
        :param Q: The trained encoder Q from the Wasserstein Autoencoder.
        :param P: The trained decoder P from the Wasserstein Autoencoder.
        :param is_pacman: Whether the target environment is Pac-Man or Space Invaders.
        :Param ablate_agent: Whether the laser canon should be removed from the frame that is passed to the agent.
        :param max_iters: Maximum amount of iterations for the gradient descent in the agents latent space via the
            Wasserstein Autoencoder.
        :return: (counterfactual, generation_time) - The counterfactual is a PIL image and the generation time is the pure
            time spent for the forward call (without pre- or postprocessing).
        """
        # state_rgb, state_bw = prepro(np.array(image), pacman=is_pacman)
        # state = Variable(torch.Tensor(np.expand_dims(state_rgb, axis=0)).permute(0, 3, 1, 2)).cuda()
        # if ablate_agent:
        #     agent_state = Variable(torch.Tensor(np.expand_dims(ablate_screen(state_bw, "agent"), axis=0)).cuda())
        # else:
        #     agent_state = Variable(torch.Tensor(np.expand_dims(state_bw, axis=0)).cuda())
        # agent_state = torch.cat([agent_state, agent_state.clone(), agent_state.clone(), agent_state.clone()], dim=1)
        # np.set_printoptions(precision=4)
        state = state_fixed
        agent_state = agent_state_fixed
        start_time = time.time()
        # get latent state representations
        z = encoder(state)

        if args.agent_algo in ["DQN"]:
            z_a = agent.policy.q_net.features_extractor(agent_state)
            p = agent.policy.q_net.q_net(z_a)
        elif args.agent_algo in ["PPO", "A2C"]:
            z_a = agent.policy.features_extractor(agent_state)
            p = agent.policy.action_net(z_a)

        # z_a = agent.policy.extract_features(agent_state)
        # z_n = Q(z_a)
        z_n = z_a

        # generate the counterfactual image
        counterfactual_z_n = olson_tec.generate_counterfactual(z_n, target_domain, agent, P, args, MAX_ITERS=max_iters)
        # p_cf = F.softmax(agent.pi(P(counterfactual_z_n)), dim=1)
        if args.agent_algo in ["DQN"]:
            p_cf = F.softmax(agent.policy.q_net.q_net(counterfactual_z_n))
        elif args.agent_algo in ["PPO", "A2C"]:
            p_cf = F.softmax(agent.policy.action_net(counterfactual_z_n))
        # p_cf = F.softmax(agent.policy.action_net(counterfactual_z_n), dim=1)
        counterfactual = generator(z, p_cf)
        # generation_time = time.time() - start_time

        # fig, ax = plt.subplots(4,2)
        # ax[0,0].imshow(state_fixed[0].cpu().permute(1,2,0))
        # ax[0,1].imshow(counterfactual[0].detach().cpu().permute(1,2,0))
        # ax[1,0].imshow(state_fixed[1].cpu().permute(1,2,0))
        # ax[1,1].imshow(counterfactual[1].detach().cpu().permute(1,2,0))
        # ax[2,0].imshow(state_fixed[2].cpu().permute(1,2,0))
        # ax[2,1].imshow(counterfactual[2].detach().cpu().permute(1,2,0))
        # ax[3,0].imshow(state_fixed[3].cpu().permute(1,2,0))
        # ax[3,1].imshow(counterfactual[3].detach().cpu().permute(1,2,0))

        # plt.show()
        
        
        # counterfactual = Image.fromarray((counterfactual[0].permute(1, 2, 0).cpu().data.numpy() * 255).astype(np.uint8))
        return counterfactual_z_n, counterfactual
    
    def main():
        if args.mode in ["train"]:
            if(args.load_epoch):
                load_models(args)
                
            try:
                x_fixed, obs_fixed, label_fixed = next(data_iter)
            except:
                data_iter = iter(data_loader)
                x_fixed, obs_fixed, label_fixed = next(data_iter)
            
            if args.env in ["highway", "roundabout"]:
                obs_fixed = obs_fixed.to(args.device).type(torch.float32)
                x_fixed = x_fixed.to(args.device).type(torch.float32)
            elif args.env in ["pong"]:
                obs_fixed = obs_fixed.to(args.device).permute(0,3,1,2).type(torch.float32)
                x_fixed = x_fixed.to(args.device).type(torch.float32)
            
            state_fixed = x_fixed
            agent_state_fixed = obs_fixed

            for i in range(args.m_frames):
                encoder.train()
                generator.train()
                train(i, data_loader)

                if i % 5 == 4 or i == args.m_frames -1:
                    save_models(epoch=i,args=args)#args.m_frames)
                if (i+1)%2 == 0:
                    CF_list = []
                    # load_models(args)
                    for target_domain in range(args.action_size):
                        CF_list.append(generate_olson_counterfactual(agent_state_fixed, state_fixed, target_domain, agent, encoder, generator,\
                            None, None, None, None, max_iters=5000)[1])
                    CF_list.insert(0, state_fixed)
                    state_fixed_cat = torch.cat(CF_list, dim=3)
                    # agent_state_fixed_cat = torch.cat([agent_state_fixed, reconstructed], dim=3)
                    save_image(state_fixed_cat.data.cpu(), args.sample_dir+'/'+str(i)+'-images.jpg', nrow=1, padding=0, normalize=True)
                    # save_image(denorm(agent_state_fixed_cat[:,3:4].data.cpu()), args.sample_dir+'/'+str(epoch)+'-obs_3.jpg', nrow=1, padding=0)
                    # save_image(denorm(agent_state_fixed_cat[:,2:3].data.cpu()), args.sample_dir+'/'+str(epoch)+'-obs_2.jpg', nrow=1, padding=0)
                    # save_image(denorm(agent_state_fixed_cat[:,1:2].data.cpu()), args.sample_dir+'/'+str(epoch)+'-obs_1.jpg', nrow=1, padding=0)
                    # save_image(denorm(agent_state_fixed_cat[:,0:1].data.cpu()), args.sample_dir+'/'+str(epoch)+'-obs_0.jpg', nrow=1, padding=0)
                    print('Saved real and fake images into {}...'.format(args.sample_dir))




                # save_images(epoch=i, args=args, agent_state_fixed=agent_state_fixed,state_fixed=state_fixed)#args.m_frames)
        elif args.mode in ["test"]:
            test(args=args, data_loader=data_loader)
        else:
            raise("set mode to train or test")
    
    action_size = args.action_size
    if not args.use_dataset:
        print('Initializing OpenAI environment...')
        if args.fskip % 2 == 0 and args.env == 'SpaceInvaders-v0':
            print("SpaceInvaders needs odd frameskip due to bullet alternations")
            args.fskip = args.fskip -1

        envs = MultiEnvironment(args.env, args.batch_size, args.fskip)

    print('Building models...')
    # if not (os.path.isfile(args.agent_file) and  os.path.isfile(args.agent_file) and  os.path.isfile(args.agent_file)):
    #     print("need an agent file")
    #     exit()
    #     args.agent_file = args.env + ".model.80.tar"

    # if args.agent_file.endswith(".h5"):
    #     agent = model.KerasAgent(args.agent_file, num_actions=action_size, latent_size=args.agent_latent)
    # elif args.agent_file.endswith(".pt"):
    #     agent = model.ACER_Agent(num_actions=action_size, latent_size=args.agent_latent).cuda()
    #     agent.load_state_dict(torch.load(args.agent_file))
    # else:
    #     agent = model.Agent(action_size, args.agent_latent).cuda()
    #     agent.load_state_dict(torch.load(args.agent_file, map_location=map_loc))

    Z_dim = args.latent
    wae_z_dim = args.wae_latent

    encoder = model.Encoder(Z_dim, env=args.env).to(args.device)
    generator = model.Generator(Z_dim, action_size, env=args.env).to(args.device)
    discriminator = model.Discriminator(Z_dim, action_size).to(args.device)
    # Q = model.Q_net(args.wae_latent, agent_latent=args.agent_latent).cuda()
    # P = model.P_net(args.wae_latent, agent_latent=args.agent_latent).cuda()


    # Q.load_state_dict(torch.load(args.Q, map_location="cuda:0"))
    # P.load_state_dict(torch.load(args.P, map_location="cuda:0"))
    encoder.train()
    generator.train()
    discriminator.train()
    # Q.eval()
    # P.eval()

    optim_gen = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.0,0.9))
    optim_enc = optim.Adam(filter(lambda p: p.requires_grad, encoder.parameters()), lr=args.lr, betas=(0.0,0.9))
    optim_disc = optim.Adam(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=args.lr, betas=(0.0,0.9))

    print('finished building model')

    bs =  args.batch_size
    enc_lambda = args.enc_lam

    clip = torch.tensor(args.clip, dtype=torch.float).to(args.device)

    TINY = 1e-9
    loss = nn.NLLLoss()
    mil = 1000000
    start = time.time()
    main()
    print((time.time() - start) / 60)
