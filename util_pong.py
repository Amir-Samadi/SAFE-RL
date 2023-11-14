# import gymnasium as gym
import gym
from stable_baselines3 import DQN,A2C,SAC,DDPG,PPO
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
import random
import json
import pandas as pd
from tkinter import Y
import numpy as np
from sklearn.utils import class_weight
import csv
import torch
import torch.nn as nn
from IPython.display import clear_output
# import matplotlib 
# matplotlib.use("Qt5Agg")
# from matplotlib import pyplot as plt
import highway_env
from matplotlib import pyplot as plt
from scipy.signal import convolve, gaussian
from copy import deepcopy
import os
import io
import base64
import time
import glob
from IPython.display import HTML
import local_lib as myLib
# import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)
from itertools import count
from time import time, strftime, localtime
import scipy.optimize
from tensorboardX import SummaryWriter
from core.models import *
from core.agent_ray_pd import AgentCollection
import ray
import envs
from trpo import trpo
import os
import pickle
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
import argparse
from policy_distillation import main as policy_distillation 
from student import Student
from teacher import Teacher
from utils.utils import *
from PIL import Image, ImageOps

def student_model_play(config, student_model):
    env = create_high_dim_env(config=config)
    obs,done = env.reset(), False
    student_score = 0 
    totall_score_avg=0
    cntr = 0
    for _ in range(500):
        action = []
        # low_dim_obs = []
        for i in range(config.num_Of_agents):
            ram = env.envs[i].env.unwrapped.ale.getRAM()
            cpu_paddle_y = ram[21] # Y coordinate of computer paddle
            player_paddle_y = ram[51] # Y coordinate of your paddle
            ball_x = ram[49] # X coordinate of ball
            ball_y = ram[54]  # Y coordinate of ball
            low_dim_obs=torch.tensor([cpu_paddle_y, player_paddle_y, ball_x, ball_y], dtype=torch.float, device=config.device)
            # tensor_low_dim_obs = torch.vstack(low_dim_obs).flatten()
            action.append(student_model.policy(low_dim_obs.repeat(2,1)).loc[0].argmax().item())
        # action = [item for item in action]
        # action = [action[:6].argmax().item(),action[6:12].argmax().item(),
        #           action[12:18].argmax().item(),action[18:24].argmax().item()]   
        env.render() 
        env.render()               
        obs, reward, done, info =  env.step(action) #[Noop, Fire, Right, Left, RightFire, LeftFire]
        student_score += reward
        if done.any() == 1:
            cntr += sum(done)
            totall_score_avg += sum(student_score[np.where(done==True)[0].tolist()])
            student_score[np.where(done==True)[0].tolist()] = 0 
        env.render() 
        env.render()
    env.close()
    # print('student score: ', totall_score_avg/cntr)
    # return totall_score_avg/cntr

# # ### environment setup
def create_high_dim_env(config):
    num_Of_agents = config.num_Of_agents
    num_Of_attributes = config.num_Of_attributes
    num_of_actions = config.num_of_actions
    # seed = random.randint(1, 1000) 
    seed = 0 
    # env_config = {}
    # env_config["full_action_space"] = False
    # # env_config["frameskip"] = 100
    env_config = {"render_mode": "rgb_array"}
    # # env_config["obs_type"]='rgb'
    # env = make_atari_env(config.env_name, n_envs=num_Of_agents, seed=seed, env_kwargs=env_config)
    env = make_atari_env(config.env_name, n_envs=num_Of_agents, seed=seed, env_kwargs=env_config)
    obs = env.reset()
    env = VecFrameStack(env, n_stack=4)
    obs, reward, done, info = env.step([1,2,3,0])
    ram = env.envs[0].env.unwrapped.ale.getRAM()  # get emulator RAM state
    return env 
# ## teacher training phase
def train_teacher(config):
    env = create_high_dim_env(config)
    fileName = config.teacher_models_dir +  config.teacher_alg + "_" + config.env + "_teacher_model"
    
    if(config.teacher_alg == 'DQN'):
        model = DQN('CnnPolicy', env,
                    # gamma=0.9,
                    verbose=1,
                    tensorboard_log="Pong_cnn/DQN",device=config.device)
    elif(config.teacher_alg == 'A2C'):
        model = A2C('CnnPolicy', env,
                # gamma=0.9,
                verbose=1,
                tensorboard_log="Pong_cnn/A2C",device=config.device)
    elif(config.teacher_alg == 'PPO'):
        model = PPO('CnnPolicy', env,
                # gamma=0.99,
                verbose=1,
                tensorboard_log="Pong_cnn/PPO",device=config.device)
    model.learn(total_timesteps=int(1e6))
    model.save(fileName)

def load_teacher(config):
    fileName = config.teacher_models_dir +  config.teacher_alg + "_" + config.env + "_teacher_model"
    if config.teacher_alg=='DQN':
        teacher_model = DQN.load(fileName)
    elif config.teacher_alg=='PPO':
        import gym
        teacher_model = PPO.load(fileName)
    elif config.teacher_alg=='A2C':
        teacher_model = A2C.load(fileName)
    else: 
        raise NameError('not implemented'+ config.teacher_alg +'yet')
    import gymnasium as gym
    return teacher_model

def teacher_model_play(config, teacher_model):
    env = create_high_dim_env(config)
    teacher_score = 0
    totall_score_avg=0
    cntr = 0
    # import gym
    # from atariari.benchmark.wrapper import AtariARIWrapper
    # env = AtariARIWrapper(gym.make('PongNoFrameskip-v4',  render_mode='human'))
    # obs = env.reset()
    # obs, reward, done, info = env.step(1)
    # env = make_atari_env('PongNoFrameskip-v4', n_envs=4, seed=0)
    # env = VecFrameStack(env, n_stack=4)
    # obs = env.reset()
    obs, done = env.reset(), False
    env.render()

    for episode in range(5000):
        action,_ = teacher_model.predict(obs)
        obs, reward, done, info = env.step(action)
        # time.sleep(0.01)
        env.render()
        teacher_score += reward
        if done.any() == 1:
            cntr += sum(done)
            totall_score_avg += sum(teacher_score[np.where(done==True)[0].tolist()])
            teacher_score[np.where(done==True)[0].tolist()] = 0 
        # ram = env.envs[0].env.unwrapped.ale.getRAM()
        # cpu_score = ram[13]  # computer/ai opponent score 
        # player_score = ram[14]  # your score
        # cpu_paddle_y = ram[21]  # Y coordinate of computer paddle
        # player_paddle_y = ram[51]  # Y coordinate of your paddle
        # ball_x = ram[49]  # X coordinate of ball
        # ball_y = ram[54]  # Y coordinate of ball      
    env.close()
    print('teacher score: ', totall_score_avg/cntr)

    return totall_score_avg/cntr

class dataType:
        def __init__(self):
            self.cpu_paddle_y, self.player_paddle_y , self.ball_x , self.ball_y = ([] for i in range(4))
            
def create_pong_dataset(config, teacher_model):
    # class Observation:
    #     cpu_paddle_y=np.array([])
    #     player_paddle_y=np.array([])
    #     ball_x=np.array([])
    #     ball_y=np.array([])

    observation = [dataType() for i in range(config.num_Of_agents)]
    actions = []
    actions_2 = []
    q_values = []
    distribution = []
    
    env = make_atari_env('PongNoFrameskip-v4', n_envs=config.num_Of_agents, seed=random.randint(1, 1000))
    env = VecFrameStack(env, n_stack=4)
    obs, done = env.reset(), False
    
    for episode in range(100000):
        if (episode+1%1000 == 0):
            env.close()
            env = make_atari_env('PongNoFrameskip-v4', n_envs=config.num_Of_agents, seed=random.randint(1, 1000))
            env = VecFrameStack(env, n_stack=4)
            obs, done = env.reset(), False
        if config.teacher_alg=='DQN':
            q_values.append(teacher_model.policy.q_net(teacher_model.q_net.obs_to_tensor(obs)[0]).detach().cpu())
        elif config.teacher_alg=='PPO':
            # (teacher_model.policy.obs_to_tensor(obs)[0]).distribution.logits.detach().cpu())
            # for i in range(100):
            #     teacher_model.policy.get_distribution((teacher_model.policy.obs_to_tensor(obs)[0])).distribution.sample_n(1000)
            #     temp_prob = teacher_model.policy.get_distribution\
            #         (teacher_model.policy.obs_to_tensor(obs)[0]).\
            #             distribution.probs.detach().cpu()
            #     temp_probs = torch.cat((temp_probs, temp_prob.unsqueeze(0)), 0)
            # std.append(torch.std(temp_probs, dim=1, keepdim=True))
            # q_values_prob.append(teacher_model.policy.get_distribution
            #                 (teacher_model.policy.obs_to_tensor(obs)[0]).distribution.probs.
            #                 detach().cpu())
            q_values.append(teacher_model.policy.get_distribution(teacher_model.policy.obs_to_tensor(obs)[0]).distribution.probs.detach().cpu())
        elif config.teacher_alg=='A2C':
            q_values.append(teacher_model.policy.action_net(teacher_model.policy.features_extractor(
                torch.tensor(teacher_model.policy.obs_to_tensor(obs)[0],dtype=torch.float))).detach().cpu())
        else:
            raise NameError('not implemented'+ config.teacher_alg +'yet')
        action = q_values[-1].argmax(dim=-1).tolist()
        
        actions.append(action)
        for i in range(config.num_Of_agents):
            ram = env.envs[i].env.unwrapped.ale.getRAM()
            observation[i].cpu_paddle_y = np.append(observation[i].cpu_paddle_y, ram[21])# Y coordinate of computer paddle
            observation[i].player_paddle_y = np.append(observation[i].player_paddle_y, ram[51])# Y coordinate of your paddle
            observation[i].ball_x = np.append(observation[i].ball_x, ram[49])# X coordinate of ball
            observation[i].ball_y = np.append(observation[i].ball_y, ram[54]) # Y coordinate of ball                 
        obs, reward, done, info =  env.step(action)
        # env.render()    
            
    # q_values = [x.detach().cpu().numpy() for x in q_values]
    # q_values = np.array(q_values)
    # actions = np.array(actions)
    # cpu_paddle_y = np.concatenate((observation[0].cpu_paddle_y, observation[1].cpu_paddle_y, observation[2].cpu_paddle_y, observation[3].cpu_paddle_y),axis=0)
    # player_paddle_y = np.concatenate((observation[0].player_paddle_y, observation[1].player_paddle_y, observation[2].player_paddle_y, observation[3].player_paddle_y),axis=0)
    # ball_x = np.concatenate((observation[0].ball_x, observation[1].ball_x, observation[2].ball_x, observation[3].ball_x),axis=0)
    # ball_y = np.concatenate((observation[0].ball_y, observation[1].ball_y, observation[2].ball_y, observation[3].ball_y),axis=0)
    # ball_y = np.concatenate((observation[0].ball_y, observation[1].ball_y, observation[2].ball_y, observation[3].ball_y),axis=0)
    # q_value_0 = np.concatenate((q_values[:, 0, 0], q_values[:, 1, 0], q_values[:, 2, 0], q_values[:, 3, 0]),axis=0)
    # q_value_1 = np.concatenate((q_values[:, 0, 1], q_values[:, 1, 1], q_values[:, 2, 1], q_values[:, 3, 1]),axis=0)
    # q_value_2 = np.concatenate((q_values[:, 0, 2], q_values[:, 1, 2], q_values[:, 2, 2], q_values[:, 3, 2]),axis=0)
    # q_value_3 = np.concatenate((q_values[:, 0, 3], q_values[:, 1, 3], q_values[:, 2, 3], q_values[:, 3, 3]),axis=0)
    # q_value_4 = np.concatenate((q_values[:, 0, 4], q_values[:, 1, 4], q_values[:, 2, 4], q_values[:, 3, 4]),axis=0)
    # q_value_5 = np.concatenate((q_values[:, 0, 5], q_values[:, 1, 5], q_values[:, 2, 5], q_values[:, 3, 5]),axis=0)
    # actions = np.concatenate((actions[:,0],actions[:,1],actions[:,2],actions[:,3]))
    # actions_2 = np.concatenate((actions_2[:,0],actions_2[:,1],actions_2[:,2],actions_2[:,3]))
    # d = {   'obs_cpu_paddle_y':cpu_paddle_y, 'obs_player_paddle_y':player_paddle_y, 'obs_ball_x':ball_x, 'obs_ball_y':ball_y,  
    #         'q_value_0':q_value_0,'q_value_1':q_value_1,'q_value_2':q_value_2, 'q_value_3':q_value_3, 'q_value_4':q_value_4, 'q_value_5':q_value_5,
    #         'action':actions, 
    #         'action_2':actions_2, 
    #     }


    q_values = [x.numpy() for x in q_values]
    q_values = np.array(q_values)
    actions = np.array(actions)
    d = {   'obs_cpu_paddle_y_1':observation[0].cpu_paddle_y, 'obs_player_paddle_y_1':observation[0].player_paddle_y, 'obs_ball_x_1':observation[0].ball_x, 'obs_ball_y_1':observation[0].ball_y,
            'obs_cpu_paddle_y_2':observation[1].cpu_paddle_y, 'obs_player_paddle_y_2':observation[1].player_paddle_y, 'obs_ball_x_2':observation[1].ball_x, 'obs_ball_y_2':observation[1].ball_y,
            'obs_cpu_paddle_y_3':observation[2].cpu_paddle_y, 'obs_player_paddle_y_3':observation[2].player_paddle_y, 'obs_ball_x_3':observation[2].ball_x, 'obs_ball_y_3':observation[2].ball_y,
            'obs_cpu_paddle_y_4':observation[3].cpu_paddle_y, 'obs_player_paddle_y_4':observation[3].player_paddle_y, 'obs_ball_x_4':observation[3].ball_x, 'obs_ball_y_4':observation[3].ball_y,
            'q_value_1_0':q_values[:,0,0],'q_value_1_1':q_values[:,0,1],'q_value_1_2':q_values[:,0,2], 'q_value_1_3':q_values[:,0,3], 'q_value_1_4':q_values[:,0,4], 'q_value_1_5':q_values[:,0,5],
            'q_value_2_0':q_values[:,1,0],'q_value_2_1':q_values[:,1,1],'q_value_2_2':q_values[:,1,2], 'q_value_2_3':q_values[:,1,3], 'q_value_2_4':q_values[:,1,4], 'q_value_2_5':q_values[:,1,5],
            'q_value_3_0':q_values[:,2,0],'q_value_3_1':q_values[:,2,1],'q_value_3_2':q_values[:,2,2], 'q_value_3_3':q_values[:,2,3], 'q_value_3_4':q_values[:,2,4], 'q_value_3_5':q_values[:,2,5],
            'q_value_4_0':q_values[:,3,0],'q_value_4_1':q_values[:,3,1],'q_value_4_2':q_values[:,3,2], 'q_value_4_3':q_values[:,3,3], 'q_value_4_4':q_values[:,3,4], 'q_value_4_5':q_values[:,3,5],
            'action_1':actions[:,0],
            'action_2':actions[:,1],
            'action_3':actions[:,2],
            'action_4':actions[:,3]  
        }

    df = pd.DataFrame(data=d)
    df.to_excel(config.teacher_alg+"_pong_dataset.xlsx")
    env.close()    

def load_pong_dataset(config):
    dataset_df = pd.read_excel(config.teacher_alg+"_pong_dataset.xlsx", index_col=0)
    numeric_feature_names = [   
            'obs_cpu_paddle_y_1', 'obs_player_paddle_y_1', 'obs_ball_x_1', 'obs_ball_y_1',
            # 'obs_cpu_paddle_y_2', 'obs_player_paddle_y_2', 'obs_ball_x_2', 'obs_ball_y_2',
            # 'obs_cpu_paddle_y_3', 'obs_player_paddle_y_3', 'obs_ball_x_3', 'obs_ball_y_3',
            # 'obs_cpu_paddle_y_4', 'obs_player_paddle_y_4', 'obs_ball_x_4', 'obs_ball_y_4',
            # 'markov1_q_value_1_0', 'markov1_q_value_1_1','markov1_q_value_1_2', 'markov1_q_value_1_3', 'markov1_q_value_1_4', 'markov1_q_value_1_5',
            # 'markov1_q_value_2_0', 'markov1_q_value_2_1','markov1_q_value_2_2', 'markov1_q_value_2_3', 'markov1_q_value_2_4', 'markov1_q_value_2_5',
            # 'markov1_q_value_3_0', 'markov1_q_value_3_1','markov1_q_value_3_2', 'markov1_q_value_3_3', 'markov1_q_value_3_4', 'markov1_q_value_3_5',
            # 'markov1_q_value_4_0', 'markov1_q_value_4_1','markov1_q_value_4_2', 'markov1_q_value_4_3', 'markov1_q_value_4_4', 'markov1_q_value_4_5',
            'q_value_1_0', 'q_value_1_1','q_value_1_2', 'q_value_1_3', 'q_value_1_4', 'q_value_1_5',
            # 'q_value_2_0', 'q_value_2_1','q_value_2_2', 'q_value_2_3', 'q_value_2_4', 'q_value_2_5',
            # 'q_value_3_0', 'q_value_3_1','q_value_3_2', 'q_value_3_3', 'q_value_3_4', 'q_value_3_5',
            # 'q_value_4_0', 'q_value_4_1','q_value_4_2', 'q_value_4_3', 'q_value_4_4', 'q_value_4_5',
            'action_1',
            # 'action_2',
            # 'action_3',
            # 'action_4',
    ]
    numeric_features = dataset_df[numeric_feature_names]
    return numeric_features

def compute_class_weight(config, numeric_features):
    if config.env=='Pong':
        class_weights = class_weight.compute_class_weight(class_weight='balanced',
            classes=np.unique(numeric_features.action_1),
            y=numeric_features.action_1.values)
        # class_weights=np.repeat(class_weights/np.sum(class_weights),4)
        class_weights = class_weights/np.sum(class_weights)
        
    else:
        class_weights = class_weight.compute_class_weight(class_weight='balanced',
            classes=np.unique(numeric_features.action),
            y=numeric_features.action.values)
        class_weights = class_weights/np.sum(class_weights)

    # class_weights = np.array([0.1,0.1,0.1,0.1,0.1,0.1])
    # class_weights = np.ones(config.num_Of_agents * config.num_of_actions)
    print("class_weights:", class_weights)
    return class_weights

def low_dim_env(config, student_model):
    env = create_high_dim_env(config)
    class Observation:
        cpu_paddle_y=np.array([])
        player_paddle_y=np.array([])
        ball_x=np.array([])
        ball_y=np.array([])
    obs_image,done = env.reset(), False
    for jj in range(40):
        action = []
        # observation = [Observation() for i in range(config.num_Of_agents)]
        observation = Observation()
        # for i in range(config.num_Of_agents):
        ram = env.envs[0].env.unwrapped.ale.getRAM()
        # observation.cpu_paddle_y = np.append(observation[i].cpu_paddle_y, ram[21])# Y coordinate of computer paddle
        # observation.player_paddle_y = np.append(observation[i].player_paddle_y, ram[51])# Y coordinate of your paddle
        # observation.ball_x = np.append(observation[i].ball_x, ram[49])# X coordinate of ball
        # observation.ball_y = np.append(observation[i].ball_y, ram[54]) # Y coordinate of ball
        low_dim_obs = [ram[21], ram[51], ram[49], ram[54]]
        low_dim_obs_tensor = torch.tensor(low_dim_obs, dtype=torch.float, device=config.device)
            # low_dim_obs = np.array([observation[0].cpu_paddle_y.item(), observation[0].player_paddle_y.item(),
            #                      observation[0].ball_x.item(), observation[0].ball_y.item()])
        # low_dim_obs = np.array([observation[0].cpu_paddle_y.item(), observation[0].player_paddle_y.item(), observation[0].ball_x.item(), observation[0].ball_y.item(),
        #                         observation[1].cpu_paddle_y.item(), observation[1].player_paddle_y.item(), observation[1].ball_x.item(), observation[1].ball_y.item(),
        #                         observation[2].cpu_paddle_y.item(), observation[2].player_paddle_y.item(), observation[2].ball_x.item(), observation[2].ball_y.item(),
        #                         observation[3].cpu_paddle_y.item(), observation[3].player_paddle_y.item(), observation[3].ball_x.item(), observation[3].ball_y.item()])                                
        action = student_model.policy(low_dim_obs_tensor).loc.argmax().item()                
        obs_image, reward, done, info =  env.step([action,0,0,0])
        env.render()
    ram = env.envs[0].env.unwrapped.ale.getRAM()
    low_dim_obs = [ram[21], ram[51], ram[49], ram[54]]
    low_dim_obs_tensor = torch.tensor(low_dim_obs, dtype=torch.float, device=config.device)
    

     

    obs_image_original = deepcopy(obs_image)
    low_dim_obs_original = deepcopy(low_dim_obs)
    loss_fn = nn.CrossEntropyLoss()
    high_dim_obs_tensor = torch.tensor(obs_image, dtype=torch.float, device=config.device, requires_grad=True)
    exp_is_valid = []
    selected_action = student_model.policy(low_dim_obs_tensor).loc.argmax().item()
    print("selected action: ", selected_action)
    return selected_action, low_dim_obs_tensor, obs_image_original, env

def CF_generation(config, student_model):
    from tqdm import trange
    from torch.optim import Adam, SGD
    trials=1000
    epochs = 20000
    loss_fn = nn.CrossEntropyLoss()
    mu = 0.999
    number_of_orig_states = 10
    teacher_student_same_output_totall = 0
    # y_prim = student_model.policy(torch.tensor(obs, dtype=torch.float, device=device)).loc
    for iters in range(trials):

        selected_action, low_dim_obs_tensor, obs_image_original, env = low_dim_env(config, student_model)


        teacher_action = teacher_model.predict(obs_image_original, deterministic=True)[0][0]

        if teacher_action in [0,1] and selected_action in [0,1]:#[Noop, Fire, Right, Left, RightFire, LeftFire]
            print ('teacher_action:', teacher_action, 'student_action: ', selected_action)
        elif teacher_action in [2,4] and selected_action in [2,4]:#[Noop, Fire, Right, Left, RightFire, LeftFire]
            print ('teacher_action:', teacher_action, 'student_action: ', selected_action)
        elif teacher_action in [3,5] and selected_action in [3,5]:#[Noop, Fire, Right, Left, RightFire, LeftFire]
            print ('teacher_action:', teacher_action, 'student_action: ', selected_action)
        else:
            continue


        indx=np.array([1, 2, 3])
        msk = torch.zeros_like(low_dim_obs_tensor).bool()
        msk[indx]=True
        availableActions = [[i] for i in range(config.num_of_actions) if i!=selected_action]
        #[Noop, Fire, Right, Left, RightFire, LeftFire]

        desireAction, candidate_CFs_label_totall = [], []




        candidate_CFs_label, changed, CFs, loss_history=[], [], [], []    
        for CounterActions in availableActions:
            x = torch.zeros_like(low_dim_obs_tensor, dtype=torch.float, device=config.device, requires_grad=True)
            # optimizer = SGD([x], lr=0.01, momentum=0.9)
            for epoch in trange(epochs):
                y_prim = student_model.policy(x+low_dim_obs_tensor).loc
                loss_1 = loss_fn (y_prim.unsqueeze(0), torch.tensor(CounterActions, device=config.device, dtype=torch.long))
                loss_2 = torch.linalg.norm(x, dim=0, ord=1)
                loss = mu*loss_1 + (1-mu)*loss_2
                x.grad = torch.zeros_like(x)
                loss.backward()
                x = x.detach() - 0.99 * x.grad * msk
                x.requires_grad = True
                
                # if epoch % 900 == 0:
                #     loss_history.append(loss.item())
            if y_prim.argmax().item() == CounterActions[0]:            
                CFs.append((low_dim_obs_tensor.detach().cpu().numpy() + x.detach().cpu().numpy())) 
                candidate_CFs_label.append(CounterActions)
                print("found CF actions:", x)
                changed.append(x)
                # break
        print("found CF actions:", candidate_CFs_label)
        
        teacher_student_same_output, candidate_CFs_label = check_validity(config, env, 
                        teacher_model, candidate_CFs_label, CFs, obs_image_original)
        
        teacher_student_same_output_totall += teacher_student_same_output
        candidate_CFs_label_totall.append(candidate_CFs_label)
    print("validity performance: ", teacher_student_same_output/len(candidate_CFs_label_totall))
    return teacher_student_same_output/len(candidate_CFs_label_totall), indx.shape[0]

def check_validity(config, env, teacher_model, candidate_CFs_label, CFs, obs_image_original):

    teacher_student_same_output=0
    
    teacher_action_prev = teacher_model.predict(obs_image_original, deterministic=True)[0][0]

    # fig, ax = plt.subplots(4,2)
    # ax[0,0].imshow(env.envs[0].env.unwrapped.ale.getScreenRGB())
    ram = env.envs[0].unwrapped.ale.getRAM()
    # ax[0,0].title.set_text('original state \n computer_P:{}, player_P:{},\n ball_X:{}, ball_Y:{}'\
    #                         .format(ram[21], ram[51], ram[49], ram[54]))
    for i, cf in enumerate(CFs):
        print (cf)
        try:
            env.envs[0].unwrapped.ale.setRAM(21,int(cf[0].item())) #ram 21 Y coordinate of computer paddle
            env.envs[0].unwrapped.ale.setRAM(51,int(cf[1].item())) #ram 51 Y coordinate of your paddle
            env.envs[0].unwrapped.ale.setRAM(49,int(cf[2].item())) #ram 49 X coordinate of ball
            env.envs[0].unwrapped.ale.setRAM(54,int(cf[3].item())) #ram 54 Y coordinate of ball
        except:
            continue
        env.render()
        env.render()
        
        obs = env.step([0,0,0,0])[0]
        env.render()
        env.render()

        ram = env.envs[0].unwrapped.ale.getRAM()
        # ax[1,0].imshow(env.envs[0].env.unwrapped.ale.getScreenRGB())
        # ax[1,0].title.set_text('CF state \n computer_P:{}, player_P:{},\n ball_X:{}, ball_Y:{}'\
        #                      .format(ram[21], ram[51], ram[49], ram[54]))

        env.render()
        # if config.teacher_alg=='DQN':
        #     q_values = teacher_model.policy.q_net(teacher_model.q_net.obs_to_tensor(obs)[0]).detach().cpu()
        # elif config.teacher_alg=='PPO':
        teacher_action = teacher_model.predict(obs, deterministic=True)[0][0]
        # elif config.teacher_alg=='A2C':
        #     q_values = teacher_model.policy.action_net(teacher_model.policy.features_extractor(
        #         torch.tensor(teacher_model.policy.obs_to_tensor(obs)[0],dtype=torch.float))).detach().cpu()
        # plt.show()    
        if teacher_action != teacher_action_prev: 
            if teacher_action in [0,1] and candidate_CFs_label[i][0] in [0,1]:#[Noop, Fire, Right, Left, RightFire, LeftFire]
                teacher_student_same_output += 1
            elif teacher_action in [2,4] and candidate_CFs_label[i][0] in [2,4]:#[Noop, Fire, Right, Left, RightFire, LeftFire]
                teacher_student_same_output += 1
            elif teacher_action in [3,5] and candidate_CFs_label[i][0] in [3,5]:#[Noop, Fire, Right, Left, RightFire, LeftFire]
                teacher_student_same_output += 1
        
    return teacher_student_same_output, candidate_CFs_label
      
def save_student_model(config, student_model):
    torch.save(student_model, config.teacher_alg + "_pong"+"_student_model_" + config.loss_metric + "_256")

def load_student_model(config):
    return torch.load(config.teacher_alg+ "_pong"+"_student_model_" + config.loss_metric + "_256")

def create_img_dataset(config, teacher_model):
    
    if not os.path.exists(config.dataset_dir):
        os.mkdir(config.dataset_dir)
    
    dataset_dir = os.path.join(config.dataset_dir, config.env) 
    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)
    dataset_dir = os.path.join(dataset_dir, config.teacher_alg) 
    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)
    dataset_dir_img = os.path.join(dataset_dir, "img")
    if not os.path.exists(dataset_dir_img):
        os.mkdir(dataset_dir_img)
    dataset_dir_obs = os.path.join(dataset_dir, "obs")
    if not os.path.exists(dataset_dir_obs):
        os.mkdir(dataset_dir_obs)
    
    
    env = create_high_dim_env(config)
    obs = env.reset()
    cntr=0
    image_names=[]
    labels={"label":[], "image_names":[], "obs_names":[]}
    for episode in range(10000):
    # for episode in range(60):
        if episode<60: #skip first 60 frames
            action =  random.sample(range(0, config.num_of_actions), 4)
            # action,_ = teacher_model.predict(obs, deterministic=True)
            obs,_,_,_ = env.step(action)
            env.render("human")
            continue
        action,_ = teacher_model.predict(obs, deterministic=True)
        # env.render()
        
        for indx, img  in enumerate(env.get_images()):
            labels["obs_names"].append(str(cntr)+".npy")
            labels["image_names"].append(str(cntr)+".jpg")
            labels["label"].append(action[indx])
            np.save(os.path.join(dataset_dir_obs, str(cntr)), obs[indx, :, :, :])
            Image.fromarray(img).resize(config.image_size).save(os.path.join(dataset_dir_img, labels["image_names"][-1]))    
            cntr += 1 
        obs,_,_,_ = env.step(action)
    env.close()

    rndNumbers = [i for i in range(len(labels["image_names"]))]
    random.shuffle(rndNumbers)
    trian_episodes = rndNumbers[0:int(len(rndNumbers)*0.9)]  
    test_episodes = rndNumbers[int(len(rndNumbers)*0.9):]
    with open(dataset_dir + "/labels_train.json", 'w') as f:
        json.dump({"label":np.array(labels["label"])[[trian_episodes]][0],\
            "image_names":np.array(labels["image_names"])[[trian_episodes]][0],\
                "obs_names":np.array(labels["obs_names"])[[trian_episodes]][0]}, f,cls=NumpyEncoder)
    with open(dataset_dir + "/labels_test.json", 'w') as f:
        json.dump({"label":np.array(labels["label"])[[test_episodes]][0],\
            "image_names":np.array(labels["image_names"])[[test_episodes]][0],\
                "obs_names":np.array(labels["obs_names"])[[test_episodes]][0]}, f,cls=NumpyEncoder)

def create_objList_dataset(config, teacher_model):
    
    if not os.path.exists(config.dataset_dir):
        os.mkdir(config.dataset_dir)
    
    dataset_dir = os.path.join(config.dataset_dir, config.env) 
    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)
    dataset_dir = os.path.join(dataset_dir, config.teacher_alg) 
    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)
    dataset_dir_obj = os.path.join(dataset_dir, "obj")
    if not os.path.exists(dataset_dir_obj):
        os.mkdir(dataset_dir_obj)
        
    env = create_high_dim_env(config)
    obs = env.reset()
    cntr=0
    obj_names=[]
    teacher_model.policy.eval()
    for episode in range(10000):
    # for episode in range(60):
        if episode<60: #skip first 60 frames
            action =  random.sample(range(0, config.num_of_actions), 4)
            # action,_ = teacher_model.predict(obs, deterministic=True)
            obs,_,_,_ = env.step(action)
            env.render("human")
            continue
        action,_ = teacher_model.predict(obs, deterministic=True)

        if config.teacher_alg in ["DQN"]:
            q_values = teacher_model.policy.q_net(teacher_model.policy.obs_to_tensor(obs)[0].type(torch.float32))
        elif config.teacher_alg in ["PPO", "A2C"]:
            q_values = teacher_model.policy.action_net(teacher_model.policy.features_extractor(teacher_model.policy.obs_to_tensor(obs)[0].type(torch.float32)))
        
        for indx, inner_env  in enumerate(env.envs):
            obj_list = dict()
            inner_env.render()
            ram = inner_env.env.unwrapped.ale.getRAM()
            obj_list["opp_paddle_pos"] = ram[21]
            obj_list["our_paddle_pos"] = ram[51]
            obj_list["ball_x_pos"] = ram[49]
            obj_list["ball_y_pos"] = ram[54]
            obj_list["action"] = action[indx]
            q_values[indx] = q_values[indx] + q_values[indx][q_values[indx].argmin().tolist()].abs()
            q_values[indx] = q_values[indx]/q_values[indx].sum()
            obj_list["q_value"] = q_values.detach().cpu().tolist()
            obj_names.append(str(cntr)+".npy")
            np.save(os.path.join(dataset_dir_obj, str(cntr)), obj_list)
            cntr += 1 
        obs,_,_,_ = env.step(action)
    env.close()

    rndNumbers = list(range(cntr))
    random.shuffle(rndNumbers)
    trian_episodes = rndNumbers[0:int(len(rndNumbers)*0.9)]  
    test_episodes = rndNumbers[int(len(rndNumbers)*0.9):]
    with open(dataset_dir + "/objList_train.json", 'w') as f:
        json.dump({"obj_names":np.array(obj_names)[[trian_episodes]][0]}, f,cls=NumpyEncoder)
    with open(dataset_dir + "/objList_test.json", 'w') as f:
        json.dump({"obj_names":np.array(obj_names)[[test_episodes]][0]}, f,cls=NumpyEncoder)


def PDCF_main(args, teacher_model):
    if args.mode in ['test']:
        models_results = {  'environment':[], 'teacher alg': [], 'distillation loss':[],
                   'teacher score':[], 'student score':[], 'CF validity':[], 'CF sparsity':[], 'PD loss':[] }
        # numeric_features = load_pong_dataset(args)
        # class_weights = compute_class_weight(args,numeric_features)
        teacher_score = teacher_model_play(args,teacher_model)
        for loss_metrics in ['kl', 'nll', 'wasserstein', 'kl_cross']:
            args.loss_metric = loss_metrics 
            student_model = load_student_model(config=args)
            student_score = student_model_play(config=args, student_model=student_model)
            student_model_play(config=args, student_model=student_model)
            validity, sparsity = CF_generation(args, student_model)

            models_results['environment'].append(args.env)
            models_results['teacher alg'].append(args.teacher_alg)
            models_results['distillation loss'].append(args.loss_metric)
            models_results['teacher score'].append(teacher_score)
            models_results['student score'].append(student_score)
            models_results['CF validity'].append(validity)
            models_results['CF sparsity'].append(sparsity)
            models_results['PD loss'].append(PD_loss)

        with open(args.env+'_results.csv', 'w') as f:
            w = csv.DictWriter(f, models_results.keys())
            w.writeheader()
            for idx in range(len(models_results['environment'])): 
                w.writerow({
                    'environment':models_results['environment'][idx],
                    'teacher alg':models_results['teacher alg'][idx],
                    'distillation loss':models_results['distillation loss'][idx],
                    'teacher score':models_results['teacher score'][idx],
                    'student score':models_results['student score'][idx],
                    'CF validity':models_results['CF validity'][idx],
                    'CF sparsity':models_results['CF sparsity'][idx],
                    'PD loss':models_results['PD loss'][idx]}
                    )
    elif args.mode in ['train']:
        # create_objList_dataset(config=args, teacher_model=teacher_model)
        # create_pong_dataset(config=args, teacher_model=teacher_model)
        data_loader = get_loader(PDCF_config=args, starGAN_config=starGAN_config, mode=args.mode)
        student_model, PD_loss = policy_distillation(args, data_loader)
        # student_model = save_student_model(config=args,student_model=student_model)
        # student_model = load_student_model(config=args)
        

    
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
