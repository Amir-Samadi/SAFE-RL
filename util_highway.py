import gymnasium as gym
# import gym
from stable_baselines3 import DQN,A2C,SAC,DDPG,PPO
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
import random
import pandas as pd
from tkinter import Y
import numpy as np
from sklearn.utils import class_weight
from tqdm import trange
from torch.optim import Adam, SGD
import json
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
import pandas as pd

import os
import io
import base64
import time
import glob
from IPython.display import HTML
import local_lib as myLib
# import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)
from stable_baselines3 import SAC,A2C,DDPG, DQN
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


def low_dimention_env_train(config):

    env_name = config.env_name
    num_Of_vehicles_Under_vision = config.num_Of_attributes
    num_of_actions = config.num_of_actions
    vehicle_attr = config.num_Of_attributes
    seed = random.randint(1, 1000) 

    low_dimention_env = gym.make(env_name, render_mode="rgb_array")
    low_dimention_env.configure({
        "observation": {
            "type": "Kinematics",
            "vehicles_count": num_Of_vehicles_Under_vision,
            "features": ["x", "y", "vx", "vy"],
            "normalize": False,
            "absolute": True,
            # "order": "sorted"
        }
    })
    low_dimention_env.reset()
    return low_dimention_env

def high_dimention_env_train(config):
    env_name = config.env_name

    env = gym.make(env_name, render_mode="rgb_array")
    env.configure({
        "observation": {
            "type": "GrayscaleObservation",
            "observation_shape": (config.obs_size[1], config.obs_size[0]),
            "stack_size": 4,
            "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
            "scaling": config.scaling,
        },  
            "scaling": config.scaling,
            # "lanes_count": 4,
            "screen_width": config.image_size[1],
            "screen_height": config.image_size[0],
            'centering_position': config.centering_position
    })
    env.reset()
    return env

def student_model_play(config, student_model):
    mean_reward = np.zeros(100)
    low_dimention_env = low_dimention_env_train(config)
    obs,done = low_dimention_env.reset(), False
    j=0
    for i in range(100):
        while done == False:      
            action = student_model.policy(torch.tensor(obs.reshape(obs.size), device=config.device)).loc.argmax().tolist()
            obs, reward, done, info = low_dimention_env.step(action)
            low_dimention_env.render()
            mean_reward[j] += reward
            # time.sleep(0.01)
        obs, done = low_dimention_env.reset(),False
        j+=1
    print ("student rewards:", mean_reward)
    print ("student mean rewards:", np.average(mean_reward))
    low_dimention_env.close()

    return np.average(mean_reward)

def teacher_model_play(config, teacher_model):
    high_dimention_env = high_dimention_env_train(config)
    mean_reward = np.zeros(100)
    j=0
    for episode in range(10):
        obs, done = high_dimention_env.reset()
        done = False
        truncated = False
        high_dimention_env.render()
        while not (done or truncated):
            action,_ = np.array(teacher_model.predict(obs,deterministic=True))
            obs, reward, done, truncated, info = high_dimention_env.step(action.item())
            high_dimention_env.render()
            mean_reward[j] += reward
        j+=1
    high_dimention_env.close()
    print ("teacher rewards:", mean_reward)
    print ("teacher mean rewards:", np.average(mean_reward))

    return np.average(mean_reward)

def train_teacher(config):
    def env():
        return high_dimention_env_train(config)
    
    fileName = config.teacher_models_dir +  config.teacher_alg + "_" + config.env + "_teacher_model"
    if(config.teacher_alg == 'DQN'):
        model = DQN('CnnPolicy', DummyVecEnv([env]),
                    gamma=0.9,
                    verbose=1,
                    tensorboard_log="highway_cnn/DQN",
                    device=config.device,
                    buffer_size=100000)
    elif(config.teacher_alg == 'A2C'):
        model = A2C('CnnPolicy', DummyVecEnv([env]),
                    gamma=0.9,
                    verbose=1,
                    tensorboard_log="highway_cnn/A2C",
                    device=config.device)
    elif(config.teacher_alg == 'PPO'):
        model = PPO('CnnPolicy', DummyVecEnv([env]),
                    gamma=0.9,
                    verbose=1,
                    tensorboard_log="highway_cnn/PPO",
                    device=config.device)
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

def create_pong_dataset(config, teacher_model):
    class Observation:
        x=np.array([])
        y=np.array([])
        vx=np.array([])
        vy=np.array([])
    actions = []
    q_values = []
    distribution = []
    observation = [Observation() for i in range(config.num_Of_attributes)]
    high_dimention_env = high_dimention_env_train(config)
    for episode in range(3000):
        obs, done = high_dimention_env.reset(), False
        while not done:
            if teacher_alg=='DQN':
                q_values.append(teacher_model.policy.q_net(teacher_model.q_net.obs_to_tensor(obs)[0]).detach().cpu())
                # q_values.append(teacher_model.q_net(torch.tensor(obs, device=device, dtype=torch.float)).cpu().detach().numpy()[0])
            elif teacher_alg=='PPO':
                q_values.append(teacher_model.policy.get_distribution(teacher_model.policy.obs_to_tensor(obs)[0]).distribution.probs.detach().cpu())
                # q_values.append(teacher_model.policy.action_net(teacher_model.policy.features_extractor                (torch.tensor(teacher_model.policy.obs_to_tensor(obs)[0],dtype=torch.float))))
                # q_values.append(teacher_model.policy.action_net(teacher_model.policy.mlp_extractor.forward_actor\
                #     (teacher_model.policy.extract_features(teacher_model.policy.obs_to_tensor(obs)[0]))))
            
            elif teacher_alg=='A2C':
                q_values.append(teacher_model.policy.action_net(teacher_model.policy.features_extractor(
                    torch.tensor(teacher_model.policy.obs_to_tensor(obs)[0],dtype=torch.float))).detach().cpu())
            else:
                raise NameError('not implemented'+ teacher_alg +'yet')
            action = q_values[-1].detach().cpu().argmax().item()
            actions.append(action)
            for i in range(num_Of_vehicles_Under_vision):
                observation[i].x = np.append(observation[i].x, high_dimention_env.env.road.vehicles[i].position[0])
                observation[i].y = np.append(observation[i].y, high_dimention_env.env.road.vehicles[i].position[1])
                observation[i].vx = np.append(observation[i].vx, high_dimention_env.env.road.vehicles[i].velocity[0])
                observation[i].vy = np.append(observation[i].vy, high_dimention_env.env.road.vehicles[i].velocity[1])                 
            obs, reward, done, info =  high_dimention_env.step(action)

            
    q_values = [x.numpy() for x in q_values]
    q_values = np.array(q_values).squeeze(axis=1)
    actions = np.array(actions)
    d = {   'obs_0_x':observation[0].x, 'obs_0_y':observation[0].y, 'obs_0_vx':observation[0].vx, 'obs_0_vy':observation[0].vy,  
            'obs_1_x':observation[1].x, 'obs_1_y':observation[1].y, 'obs_1_vx':observation[1].vx, 'obs_1_vy':observation[1].vy,  
            'obs_2_x':observation[2].x, 'obs_2_y':observation[2].y, 'obs_2_vx':observation[2].vx, 'obs_2_vy':observation[2].vy,  
            'obs_3_x':observation[3].x, 'obs_3_y':observation[3].y, 'obs_3_vx':observation[3].vx, 'obs_3_vy':observation[3].vy,  
            'obs_4_x':observation[4].x, 'obs_4_y':observation[4].y, 'obs_4_vx':observation[4].vx, 'obs_4_vy':observation[4].vy,      
            'obs_5_x':observation[5].x, 'obs_5_y':observation[5].y, 'obs_5_vx':observation[5].vx, 'obs_5_vy':observation[5].vy,      
            'q_value_0':q_values[:,0],'q_value_1':q_values[:,1],'q_value_2':q_values[:,2],'q_value_3':q_values[:,3], 'q_value_4':q_values[:,4],
            'action':actions
        }
    df = pd.DataFrame(data=d)
    df.to_excel(config.teacher_alg+"_highway_dataset.xlsx")
    high_dimention_env.close()

def load_pong_dataset(config):
    dataset_df = pd.read_excel(config.teacher_alg+"_highway_dataset.xlsx", index_col=0)
    numeric_feature_names = ['obs_0_x', 'obs_0_y', 'obs_0_vx',  'obs_0_vy',
                                'obs_1_x', 'obs_1_y', 'obs_1_vx',  'obs_1_vy',
                                'obs_2_x', 'obs_2_y', 'obs_2_vx',  'obs_2_vy',
                                'obs_3_x', 'obs_3_y', 'obs_3_vx',  'obs_3_vy',
                                'obs_4_x', 'obs_4_y', 'obs_4_vx',  'obs_4_vy',
                                'obs_5_x', 'obs_5_y', 'obs_5_vx',  'obs_5_vy',
                                'q_value_0','q_value_1','q_value_2','q_value_3','q_value_4']
        
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

def CF_generation(config, student_model):
    high_dimention_env = high_dimention_env_train()
    obs_image = high_dimention_env.reset()
    high_dimention_env.render()
    obs_image_original = deepcopy(obs_image)
    low_dimention_obs = np.zeros((config.num_Of_vehicles_Under_vision, config.vehicle_attr))
    #calculating low dimensional obs for feeding the student network
    for j, vehicles in enumerate(high_dimention_env.env.road.vehicles[:]):
        if (j == config.num_Of_vehicles_Under_vision):
            break
        else :
            low_dimention_obs[j, 0:2] = torch.tensor(vehicles.position)
            low_dimention_obs[j, 2:4] = torch.tensor(vehicles.velocity)

    low_dimention_obs = torch.tensor(low_dimention_obs.reshape(low_dimention_obs.size), dtype=torch.float, device=device, requires_grad=True)
    loss_fn = nn.CrossEntropyLoss()

    epochs = 10000

    indx=np.array([4, 5, 8, 9])
    msk = torch.zeros_like(low_dimention_obs).bool()
    msk[indx]=True
    selected_action = student_model.policy(low_dimention_obs).loc.argmax().tolist()
    print("original student actions is:", selected_action)
    availableActions = [x for x in high_dimention_env.get_available_actions() if x!=selected_action]
    desireAction, loss_history, CFs, y_prims, changed =[], [], [], [] , []
    mu = 0.99
    for CounterActions in availableActions:
        x = torch.zeros_like(low_dimention_obs, dtype=torch.float, device=device, requires_grad=True)
        for epoch in trange(epochs):
            y_prim = student_model.policy(x+low_dimention_obs).loc
            loss_1 = loss_fn (y_prim, torch.tensor(CounterActions, device=device, dtype=torch.long))
            loss_2 = torch.linalg.norm(x, dim=0, ord=2)
            loss = mu*loss_1 + (1-mu)*loss_2
            x.grad = torch.zeros_like(x)
            loss.backward()
            x = x.detach() - 0.1 * x.grad * msk
            x.requires_grad = True
            
            if epoch % 900 == 0:
                loss_history.append(loss.item())
            
        if y_prim.argmax().item()  == CounterActions :
            CFs.append((low_dimention_obs.detach().cpu().numpy() + x.detach().cpu().numpy())) 
            y_prims.append(CounterActions)
            changed.append(x)
    print("found CF actions:", y_prims)

def check_validity(config, high_dimention_env, teacher_model, candidate_CFs_label, CFs, obs_image_original):
    exp_is_valid = []
    plt.imshow(obs_image_original[3].T, cmap=plt.get_cmap('gray'))
    plt.show()
    print ("original teacher action", teacher_model.predict(obs_image_original))
    for i, cf in enumerate(CFs):
        cf = cf.reshape((config.num_Of_vehicles_Under_vision, config.vehicle_attr))
        #applying the CFs to environments
        for j, vehicles in enumerate(high_dimention_env.env.road.vehicles[:]):
            if (j == config.num_Of_vehicles_Under_vision):
                break
            else :
                vehicles.position=cf[j,0:2]
                # vehicles.velocity=cf[j,2:4]
        high_dimention_env.env.render()
        high_dimention_env.env.render()
        high_dimention_env.env.render()
        obs_image=high_dimention_env.env.observation_type.observe() 
        ##prepare data
        plt.imshow(obs_image[3].T, cmap=plt.get_cmap('gray'))
        plt.show()
        action,_ = teacher_model.predict(obs_image)
        action_prim = student_model.policy(torch.tensor(CFs[i].reshape(config.num_Of_vehicles_Under_vision*config.vehicle_attr)
           , device=device)).loc.argmax().item()
        # if action_prim==action_desire then the exp is valide
        if(action == action_prim):
            exp_is_valid.append(1)
        else:
            exp_is_valid.append(0)
    print(exp_is_valid)
      
def save_student_model(config, student_model):
    torch.save(student_model, config.teacher_alg + "_pong"+"_student_model_" + config.loss_metric + "_256")

def load_student_model(config):
    return torch.load(config.teacher_alg+ "_pong"+"_student_model_" + config.loss_metric + "_256")

def rules_extraction():
    from tqdm import trange

    high_dimention_env = high_dimention_env_train()
    loss_fn = nn.CrossEntropyLoss()
    class TTC:
            front=[]
            left=[]
            right=[]
            y=[]
            y_prim = []
    time2coll = TTC
    for i in trange(1000):
        obs_image = high_dimention_env.reset()
        # high_dimention_env.render()
        obs_image_original = deepcopy(obs_image)
        low_dimention_obs = np.zeros((num_Of_vehicles_Under_vision, vehicle_attr))
        #calculating low dimensional obs for feeding the student network
        for j, vehicles in enumerate(high_dimention_env.env.road.vehicles[:]):
            if (j == num_Of_vehicles_Under_vision):
                break
            else :
                low_dimention_obs[j, 0:2] = torch.tensor(vehicles.position)
                low_dimention_obs[j, 2:4] = torch.tensor(vehicles.velocity)

        low_dimention_obs = torch.tensor(low_dimention_obs.reshape(low_dimention_obs.size), dtype=torch.float, device=device, requires_grad=True)
        
        epochs = 10000
        indx=np.array([4, 5, 8, 9])
        msk = torch.zeros_like(low_dimention_obs).bool()
        msk[indx]=True
        student_selected_action = student_model.policy(low_dimention_obs).loc.argmax().tolist()
        
        if teacher_alg=='DQN':
            teacher_selected_action = teacher_model.policy.q_net(teacher_model.q_net.obs_to_tensor(obs_image)[0]).detach().cpu().argmax()
        elif teacher_alg=='PPO':
            teacher_selected_action = teacher_model.policy.get_distribution(teacher_model.policy.obs_to_tensor(obs_image)[0]).distribution.probs.detach().cpu().argmax()    
        elif teacher_alg=='A2C':
            teacher_selected_action = teacher_model.policy.action_net(teacher_model.policy.features_extractor(
                    torch.tensor(teacher_model.policy.obs_to_tensor(obs_image)[0],dtype=torch.float))).detach().cpu().argmax()
        
        if student_selected_action != teacher_selected_action.item():
            continue 
        

        availableActions = [x for x in high_dimention_env.get_available_actions() if x!=student_selected_action]
        desireAction, loss_history, CFs, y_prims, changed =[], [], [], [] , []
        mu = 0.99
        for CounterActions in availableActions:
            x = torch.zeros_like(low_dimention_obs, dtype=torch.float, device=device, requires_grad=True)
            for epoch in range(epochs):
                y_prim = student_model.policy(x+low_dimention_obs).loc
                loss_1 = loss_fn (y_prim, torch.tensor(CounterActions, device=device, dtype=torch.long))
                loss_2 = torch.linalg.norm(x, dim=0, ord=2)
                loss = mu*loss_1 + (1-mu)*loss_2
                x.grad = torch.zeros_like(x)
                loss.backward()
                x = x.detach() - 0.1 * x.grad * msk
                x.requires_grad = True
                
                if epoch % 900 == 0:
                    loss_history.append(loss.item())
                
            if y_prim.argmax().item()  == CounterActions :
                CFs.append((low_dimention_obs.detach().cpu().numpy() + x.detach().cpu().numpy())) 
                y_prims.append(CounterActions)
                changed.append(x)
        print("found CF actions:", y_prims)

        


        exp_is_valid = []
        for i, cf in enumerate(CFs):
            cf = cf.reshape((num_Of_vehicles_Under_vision, vehicle_attr))
            #applying the CFs to environments
            for j, vehicles in enumerate(high_dimention_env.env.road.vehicles[:]):
                if (j == num_Of_vehicles_Under_vision):
                    break
                else :
                    vehicles.position=cf[j,0:2]
                    # vehicles.velocity=cf[j,2:4]
            high_dimention_env.env.render()
            high_dimention_env.env.render()
            high_dimention_env.env.render()
            obs_image=high_dimention_env.env.observation_type.observe() 
            ##prepare data
            # plt.imshow(obs_image[3].T, cmap=plt.get_cmap('gray'))
            # plt.show()
            action,_ = teacher_model.predict(obs_image)
            action_prim = student_model.policy(torch.tensor(CFs[i].reshape(num_Of_vehicles_Under_vision*vehicle_attr)
            , device=device)).loc.argmax().item()
            # if action_prim==action_desire then the exp is valide
            if(action == action_prim):
                exp_is_valid.append(1)
                for lanes in ["left","front","right"]:
                    if  (lanes=="left" and cf[0,1]<1): 
                        time2coll.left.append(100000)
                    elif (lanes=="right" and cf[0,1]>7):
                        time2coll.right.append(100000)          
                    elif lanes=="left":
                        tempObs = cf.copy()
                        tempObs[0,1]-=4
                        time2coll.left.append(np.nanmin(myLib.Time2Coll(tempObs)))
                    elif lanes=="front":
                        tempObs = cf.copy()
                        time2coll.front.append(np.nanmin(myLib.Time2Coll(tempObs)))
                    else:
                        tempObs = cf.copy()
                        tempObs[0,1]+=4
                        time2coll.right.append(np.nanmin(myLib.Time2Coll(tempObs)))
                time2coll.y.append (teacher_selected_action.item())
                time2coll.y_prim.append(action)
            else:
                exp_is_valid.append(0)
        print(exp_is_valid)

    # time2coll.left = np.nan_to_num(time2coll.left,nan=1000)
    # time2coll.front = np.nan_to_num(time2coll.front,nan=1000)
    # time2coll.right = np.nan_to_num(time2coll.right,nan=1000)
    import pandas as pd
    d = {'TTC_left': time2coll.left, 'TTC_front':time2coll.front, 'TTC_right': time2coll.right,
         'action':time2coll.y, 'action_prim':time2coll.y_prim,
         }
    df = pd.DataFrame(data=d)
    df.to_excel(teacher_alg + "if_then_rules_data.xlsx") 

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
    
    
    env = high_dimention_env_train(config)
    obs,done = env.reset()
    cntr=0
    image_names=[]
    labels={"label":[], "image_names":[], "obs_names":[]}
    for episode in range(30000):
    # for episode in range(60):
        if episode<10: #skip first 100 frames
            action =  random.sample(range(0, config.num_of_actions), 1)
            obs, reward, done, truncated, info = env.step(action[-1])
            if(done or truncated):
                obs = env.reset()
            continue 
        action,_ = teacher_model.predict(obs, deterministic=True)

        labels["obs_names"].append(str(cntr)+".npy")
        labels["image_names"].append(str(cntr)+".jpg")
        labels["label"].append(int(action))
        np.save(os.path.join(dataset_dir_obs, str(cntr)), obs)
        Image.fromarray(env.render()).save(os.path.join(dataset_dir_img, labels["image_names"][-1]))   

        obs, reward, done, truncated, info = env.step(action)
        if(done or truncated):
            obs, _ = env.reset()
        env.render()
        
        cntr += 1 
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

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
