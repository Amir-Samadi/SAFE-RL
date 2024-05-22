# from bdb import effective
# from select import select
# from scipy.signal import convolve, gaussian
# import torch.nn as nn
# import numpy as np
# import torch
# from stable_baselines3 import DQN
# from tqdm import trange
# import matplotlib.pyplot as plt
# import copy
# from copy import deepcopy
# import random
# from IPython.display import clear_output
# # from six import with_metaclass
# #####################################################################
# #####################################################################
# #####################################################################
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# class DQNAgent(nn.Module):
#     def __init__(self,  model, epsilon=0):

#         super().__init__()
#         self.epsilon = epsilon
#         # and self.n_actions as output vector of logits of q(s, a)
#         self.network = model
#         # self.network.add_module('softmax', nn.Softmax(dim=0))
#         # 
# #         self.parameters = self.network.parameters
        
#     def forward(self, points):
#         # pass the state at time t through the newrok to get Q(s,a)
#         qvalues = self.network(points)
#         return qvalues
        
#     def get_qvalues(self, states):
#         # input is an array of states in numpy and outout is Qvals as numpy array
# #         states = [torch.tensor(states, device=device, dtype=torch.float32) for t in states]
#         states = torch.tensor(states, device=device, dtype=torch.float32)
#         qvalues = self.forward(states)
#         return qvalues.data.cpu().numpy()

#     def sample_actions(self, qvalues):
#         # sample actions from a batch of q_values using epsilon greedy policy
#         epsilon = self.epsilon
#         batch_size, n_actions = qvalues.shape
#         random_actions = np.random.choice(n_actions, size=batch_size)
#         best_actions = qvalues.argmax(axis=-1)
#         should_explore = np.random.choice(
#             [0, 1], batch_size, p=[1-epsilon, epsilon])
#         return np.where(should_explore, random_actions, best_actions)
    
    
# #####################################################################
# #####################################################################
# #####################################################################
# def compute_td_loss(indx, presence, x, y, vx, vy, cos_h, sin_h, new_model, obs, desire_actions, device):

#         # convert numpy array to torch tensors
#         obs = torch.tensor(obs, device=device, dtype=torch.float)
        
#         msk_presence = torch.zeros_like(obs)
#         msk_x = torch.zeros_like(obs)
#         msk_y = torch.zeros_like(obs)
#         msk_vx = torch.zeros_like(obs)
#         msk_vy = torch.zeros_like(obs)
#         msk_cos_h = torch.zeros_like(obs)
#         msk_sin_h = torch.zeros_like(obs)
        
#         msk_presence[indx,0] = 1.0
#         msk_x[indx,1] = 1.0
#         msk_y[indx,2] = 1.0
#         msk_vx[indx,3] = 1.0
#         msk_vy[indx,4] = 1.0
#         msk_cos_h[indx,5] = 1.0
#         msk_sin_h[indx,6] = 1.0
        
#         obs_temp = deepcopy(obs)
        
#         obs[indx,0] = obs[indx,0] + presence * msk_presence[indx,0]
#         obs[indx,1] = obs[indx,1] + x * msk_x[indx,1]
#         obs[indx,2] = obs[indx,2] + y * msk_y[indx,2]
#         obs[indx,3] = obs[indx,3] + vx * msk_vx[indx,3]
#         obs[indx,4] = obs[indx,4] + vy * msk_vy[indx,4]
#         obs[indx,5] = obs[indx,5] + cos_h * msk_cos_h[indx,5]
#         obs[indx,6] = obs[indx,6] + sin_h * msk_sin_h[indx,6]
        
#         # action = agent(obs).argmax(axis=-1)#???????????
#         predicted_qvalues = new_model(torch.reshape(obs, (1,35)))#???????????
        
#     #     CrossEntropyLoss
#     # from torch.nn import functional as F
#     #  F.smooth_l1_loss(current_q_values, target_q_values)
#         loss = nn.CrossEntropyLoss()
#         target = torch.empty(1, dtype=torch.long,device='cuda').random_(desire_actions,desire_actions+1)#???????????
#         l1 = loss(predicted_qvalues, target)
#         l2 = torch.sum((obs-obs_temp)**2)
#         # l2 = torch.sum((presence+x+x)**2)
#         output = 100*l1 + l2
        
#         return output
# #####################################################################
# #####################################################################
# #####################################################################       
# def DQN_creation(policy, env, policy_kwargs, learning_rate,
#                 buffer_size, learning_starts, batch_size, gamma, train_freq, gradient_steps,
#                 target_update_interval, exploration_fraction, verbose, tensorboard_log):
#     model = DQN(policy=policy, env=env,
#                 policy_kwargs=policy_kwargs, learning_rate=learning_rate,
#                 buffer_size=buffer_size,
#                 learning_starts=learning_starts,
#                 batch_size=batch_size,
#                 gamma=gamma,
#                 train_freq=train_freq,
#                 gradient_steps=gradient_steps,
#                 target_update_interval=target_update_interval,
#                 exploration_fraction=exploration_fraction,
#                 verbose=verbose,
#                 tensorboard_log=tensorboard_log)
#     return  model
# #####################################################################
# #####################################################################
# #####################################################################
# def Save_DQN_model (model, fileName):
#     model.save(fileName)
# #####################################################################
# #####################################################################
# #####################################################################
# def Load_DQN_model (fileName):
#     return DQN.load(fileName)
# #####################################################################
# #####################################################################
# #####################################################################
# def CF_find(model, env, obs, device, indx):
#     selected_action = model(torch.tensor(obs.reshape(obs.size), dtype=torch.float, device=device)).argmax(axis=-1)
#     availableActions = [x for x in env.get_available_actions() if x!=selected_action]
#     CFs = []
#     for CounterActions in availableActions:
#         CFs.append(learn_CF(model, obs, env, indx, device, CounterActions))
#     return CFs
# #####################################################################
# #####################################################################
# #####################################################################

# def learn_CF(model, obs, env, indx, device, CounterActions):
    
#     total_steps = 5 * 10**3
#     # setup spme frequency for loggind and updating target network
#     loss_freq = 20
#     eval_freq = 3000
#     td_loss_history = []
    
#     presence = torch.tensor(10**-10*np.ones_like(indx), dtype=torch.float32, requires_grad=True).to(device)
#     x = torch.tensor(10**-10*np.ones_like(indx), dtype=torch.float32, requires_grad=True).to(device)
#     y = torch.tensor(10**-10*np.ones_like(indx), dtype=torch.float32, requires_grad=True).to(device)
#     vx = torch.tensor(10**-10*np.ones_like(indx), dtype=torch.float32, requires_grad=True).to(device)
#     vy = torch.tensor(10**-10*np.ones_like(indx), dtype=torch.float32, requires_grad=True).to(device)
#     cos_h = torch.tensor(10**-10*np.ones_like(indx), dtype=torch.float32, requires_grad=True).to(device)
#     sin_h = torch.tensor(10**-10*np.ones_like(indx), dtype=torch.float32, requires_grad=True).to(device)

    

#     print(presence)

#     obs_prim = torch.tensor(obs, device=device, dtype=torch.float)


   

#     for step in trange(total_steps + 1):

#         loss = compute_td_loss(indx, presence, x, y, vx, vy,cos_h, sin_h, model, obs, CounterActions, device)
        
#         presence.grad = torch.zeros_like(presence)
#         x.grad = torch.zeros_like(x)
#         y.grad = torch.zeros_like(y)
#         vx.grad = torch.zeros_like(vx)
#         vy.grad = torch.zeros_like(vy)
#         cos_h.grad = torch.zeros_like(cos_h)
#         sin_h.grad = torch.zeros_like(sin_h)
        
#         loss.backward(inputs=[y])
#         presence = presence.detach() - 0.1 * presence.grad
#         x = x.detach() - 0.1 * x.grad
#         y = y.detach() - 0.1 * y.grad
#         vx = vx.detach() - 0.1 * vx.grad
#         vy = vy.detach() - 0.1 * vy.grad
#         cos_h = cos_h.detach() - 0.1 * cos_h.grad
#         sin_h = sin_h.detach() - 0.1 * sin_h.grad
        
#         presence.requires_grad = True
#         x.requires_grad = True
#         y.requires_grad = True
#         vx.requires_grad = True
#         vy.requires_grad = True
#         cos_h.requires_grad = True
#         sin_h.requires_grad = True
        
#         if step % loss_freq == 0:
#             td_loss_history.append(loss.data.cpu().item())
            
#         if step % eval_freq == 0:
#             clear_output(True)
#             assert not np.isnan(td_loss_history[-1])
#             # plt.subplot(1, 2, 2)
#             # plt.title("TD loss history (smoothened)")
#             # plt.plot(smoothen(td_loss_history))
#             # plt.grid()
#             # plt.show()
        
#     msk_presence = torch.zeros_like(obs_prim)
#     msk_x = torch.zeros_like(obs_prim)
#     msk_y = torch.zeros_like(obs_prim)
#     msk_vx = torch.zeros_like(obs_prim)
#     msk_vy = torch.zeros_like(obs_prim)
#     msk_cos_h = torch.zeros_like(obs_prim)
#     msk_sin_h = torch.zeros_like(obs_prim)
#     msk_presence[indx,0] = 1.0
#     msk_x[indx,1] = 1.0
#     msk_y[indx,2] = 1.0
#     msk_vx[indx,3] = 1.0
#     msk_vy[indx,4] = 1.0
#     msk_cos_h[indx,5] = 1.0
#     msk_sin_h[indx,6] = 1.0
#     obs_prim[indx,0] = obs_prim[indx,0] + presence * msk_presence[indx,0]
#     obs_prim[indx,1] = obs_prim[indx,1] + x * msk_x[indx,1]
#     obs_prim[indx,2] = obs_prim[indx,2] + y * msk_y[indx,2]
#     obs_prim[indx,3] = obs_prim[indx,3] + vx * msk_vx[indx,3]
#     obs_prim[indx,4] = obs_prim[indx,4] + vy * msk_vy[indx,4]
#     obs_prim[indx,5] = obs_prim[indx,5] + cos_h * msk_cos_h[indx,5]
#     obs_prim[indx,6] = obs_prim[indx,6] + sin_h * msk_sin_h[indx,6]
#     # print("obs_prim", obs_prim)
#     # print("obs", obs)
#     if(model(obs_prim.reshape(35)).argmax(axis=-1)==CounterActions):
#         print("CF finded")


#     # print("q value with changes:", model(obs_prim.reshape(1,35)))
#     # print("q value without changes:", model(torch.tensor(obs.reshape(obs.size), device=device, dtype=torch.float)))
#     return obs_prim
# #####################################################################
# #####################################################################
# #####################################################################
# def CF2Env(obs, CF, indx, env):
#     env_prim = copy.deepcopy(env)
#     for indices in indx:
#         for vehicles in env_prim.env.road.vehicles[1:]:
#             if (all(vehicles.position.astype(int)==obs[indices,1:3].astype(int))):
#                 vehicles.position=CF[indices,1:3].cpu().detach().numpy()
#                 break
#     return env_prim
# #####################################################################
# #####################################################################
# #####################################################################
# from numpy import linalg as LA
# from dataclasses import dataclass



# def Time2Coll(obs):
#     time2coll = np.full(obs.shape[0],np.inf)
#     @dataclass
#     class vehicle:
#         Radius:float
#         X:float
#         Y:float
#         Xvel:float
#         Yvel:float
#     ego_vehicle = vehicle(1.5, obs[0,1], obs[0,2], obs[0,3], obs[0,4])     
#     for j in range(obs.shape[0]):
#         participant = vehicle(1, obs[j,1], obs[j,2], obs[j,3], obs[j,4])     
#         time2coll[j] = Time2Coll2(ego_vehicle, participant)
#     return time2coll


# def Time2Coll2(ego_vehicle, participant):
#     distance = (ego_vehicle.Radius + participant.Radius) * (ego_vehicle.Radius + participant.Radius)
#     a = (ego_vehicle.Xvel - participant.Xvel) * (ego_vehicle.Xvel - participant.Xvel) + (ego_vehicle.Yvel - participant.Yvel) * (ego_vehicle.Yvel - participant.Yvel);
#     b = 2 * ((ego_vehicle.X - participant.X) * (ego_vehicle.Xvel - participant.Xvel) + (ego_vehicle.Y - participant.Y) * (ego_vehicle.Yvel - participant.Yvel));
#     c = (ego_vehicle.X - participant.X) * (ego_vehicle.X - participant.X) + (ego_vehicle.Y - participant.Y) * (ego_vehicle.Y - participant.Y) - distance
#     d = b * b - 4 * a * c
#     ## Ignore glancing collisions that may not cause a response due to limited precision and lead to an infinite loop
#     if (b > -1e-6 or d <= 0):
#         return np.nan
#     e = np.sqrt(d)
#     t1 = (-b - e) / (2 * a);    ## Collison time, +ve or -ve
#     t2 = (-b + e) / (2 * a);    ## Exit time, +ve or -ve
#     ## b < 0 => Getting closer
#     ## If we are overlapping and moving closer, collide now
#     if (t1 < 0 and t2 > 0 and b <= -1e-6):
#         return 0
#     return t1


    
# def compute_td_loss_CF_CNN(indx, x, y, vx, vy, new_model, obs_kinematics, obs_bev, desire_actions, device):
#     # convert numpy array to torch tensors
#     obs_kinematics = torch.tensor(obs_kinematics, device=device, dtype=torch.float)
    
#     msk_x = torch.zeros_like(obs_kinematics)
#     msk_y = torch.zeros_like(obs_kinematics)
#     msk_vx = torch.zeros_like(obs_kinematics)
#     msk_vy = torch.zeros_like(obs_kinematics)
    
#     msk_x[indx,0] = 1.0
#     msk_y[indx,1] = 1.0
#     msk_vx[indx,2] = 1.0
#     msk_vy[indx,3] = 1.0
    
#     obs_temp = deepcopy(obs_kinematics)

#     obs_kinematics[indx,0] = obs_kinematics[indx,0] + x * msk_x[indx,0]
#     obs_kinematics[indx,1] = obs_kinematics[indx,1] + y * msk_y[indx,1]
#     obs_kinematics[indx,2] = obs_kinematics[indx,2] + vx * msk_vx[indx,2]
#     obs_kinematics[indx,3] = obs_kinematics[indx,3] + vy * msk_vy[indx,3]
    
#     # action = agent(obs).argmax(axis=-1)#???????????
#     predicted_qvalues = new_model.q_net.q_net(new_model.q_net.features_extractor
#                 (torch.tensor(obs_bev, device=device, dtype=torch.float)))
    
# #     CrossEntropyLoss
# #   from torch.nn import functional as F
# #   F.smooth_l1_loss(current_q_values, target_q_values)
#     loss = nn.CrossEntropyLoss()
#     target = torch.empty(1, dtype=torch.long,device=device).random_(desire_actions,desire_actions+1)#???????????
#     l1 = loss(predicted_qvalues, target)
#     l2 = torch.sum((obs_kinematics-obs_temp)**2)
#     # l2 = torch.sum((presence+x+x)**2)
#     # output = 100*l1 + l2
#     output = l1 
    
#     return output,obs_kinematics

# def CF2Env2(obs, CF, indx, env):
#     env_prim = copy.deepcopy(env)
#     for indices in indx:
#         for vehicles in env_prim.env.road.vehicles[1:]:
#             if (all(vehicles.position.astype(int)==obs[indices,0:2].astype(int))):
#                 vehicles.position=CF[indices,0:2].cpu().detach().numpy()
#                 vehicles.velocity=CF[indices,2:-1].cpu().detach().numpy()
#                 break
#     return env_prim


# def smoothen(values):
#     kernel = gaussian(100, std=100)
#     kernel = kernel / np.sum(kernel)
#     return convolve(values, kernel, 'valid')

# ##for the learning process in "Teacher_student_CNN_Linear_gradientCF"
# def CF_find_2(student_model, env, obs, device, indx, num_Of_vehicles_Under_vision, vehicle_attr):
#     selected_action = student_model(torch.tensor(obs.reshape(num_Of_vehicles_Under_vision*vehicle_attr), dtype=torch.float, device=device)).argmax(axis=-1)
#     availableActions = [x for x in env.get_available_actions() if x!=selected_action]
#     CFs = []
#     desireAction=[]
#     for CounterActions in availableActions:
#         potentialCF = learn_CF2(student_model, obs, env, indx, device, CounterActions, num_Of_vehicles_Under_vision, vehicle_attr)
#         if potentialCF is not None:
#             print("CF finded")
#             CFs.append(potentialCF)
#             desireAction.append(CounterActions)
#     return CFs, desireAction
# ##for the learning process in "Teacher_student_CNN_Linear_gradientCF"
# def learn_CF2(student_model,obs,env,indx,device,CounterActions, num_Of_vehicles_Under_vision, vehicle_attr):
#     loss_freq=1000
#     eval_freq = 1000
#     td_loss_history =[]
#     total_steps=int(1e4)
#     desire_actions = CounterActions

#     x = torch.tensor(10**-10*np.ones_like(indx), dtype=torch.float32, requires_grad=True).to(device)
#     y = torch.tensor(10**-10*np.ones_like(indx), dtype=torch.float32, requires_grad=True).to(device)
#     vx = torch.tensor(10**-10*np.ones_like(indx), dtype=torch.float32, requires_grad=True).to(device)
#     vy = torch.tensor(10**-10*np.ones_like(indx), dtype=torch.float32, requires_grad=True).to(device)
#     def smoothen(values):
#         kernel = gaussian(100, std=100)
#         kernel = kernel / np.sum(kernel)
#         return convolve(values, kernel, 'valid')

#     from tqdm import trange
#     from IPython.display import clear_output
#     import matplotlib.pyplot as plt


#     for step in trange(total_steps + 1):
    
        
#         loss = compute_td_loss2(indx, x, y, vx, vy, student_model, obs, desire_actions, device, num_Of_vehicles_Under_vision, vehicle_attr )
        
#         x.grad = torch.zeros_like(x)
#         y.grad = torch.zeros_like(y)
#         vx.grad = torch.zeros_like(vx)
#         vy.grad = torch.zeros_like(vy)
        
#         loss.backward(inputs=[x,y])

#         x = x.detach() - 0.001 * x.grad
#         y = y.detach() - 0.001 * y.grad
#         vx = vx.detach() - 0.001 * vx.grad
#         vy = vy.detach() - 0.001 * vy.grad
        
#         x.requires_grad = True
#         y.requires_grad = True
#         vx.requires_grad = True
#         vy.requires_grad = True
        
#         if step % loss_freq == 0:
#             td_loss_history.append(loss.data.cpu().item())
            
#         if step % eval_freq == 0:
#             clear_output(True)
#             assert not np.isnan(td_loss_history[-1])
#             plt.subplot(1, 2, 2)
#             plt.title("TD loss history (smoothened)")
#             # plt.plot(smoothen(td_loss_history))
#             plt.plot(td_loss_history)
#             plt.grid()
#             plt.show()

#     obs_prim = torch.tensor(deepcopy(obs), device=device, dtype=torch.float)
#     msk_x = torch.zeros_like(obs_prim)
#     msk_y = torch.zeros_like(obs_prim)
#     msk_vx = torch.zeros_like(obs_prim)
#     msk_vy = torch.zeros_like(obs_prim)
#     msk_x[indx,0] = 1.0
#     msk_y[indx,1] = 1.0
#     msk_vx[indx,2] = 1.0
#     msk_vy[indx,3] = 1.0
#     obs_prim[indx,0] = obs_prim[indx,0] + x * msk_x[indx,0]
#     obs_prim[indx,1] = obs_prim[indx,1] + y * msk_y[indx,1]
#     obs_prim[indx,2] = obs_prim[indx,2] + vx * msk_vx[indx,2]
#     obs_prim[indx,3] = obs_prim[indx,3] + vy * msk_vy[indx,3]
#     # print("obs_prim", obs_prim)
#     # print("obs", obs)
#     if(student_model(obs_prim.reshape(num_Of_vehicles_Under_vision*vehicle_attr)).argmax(axis=-1).item()==desire_actions):
#         print("CF finded")
#         return obs_prim.cpu().detach().numpy()


#     # print("q value with changes:", model(obs_prim.reshape(1,35)))
#     # print("q value without changes:", model(torch.tensor(obs.reshape(obs.size), device=device, dtype=torch.float)))
    
# def compute_td_loss2(indx, x, y, vx, vy, new_model, obs, desire_actions, device, num_Of_vehicles_Under_vision, vehicle_attr):
#     # convert numpy array to torch tensors
#     obs = torch.tensor(obs, device=device, dtype=torch.float)
    
#     msk_x = torch.zeros_like(obs)
#     msk_y = torch.zeros_like(obs)
#     msk_vx = torch.zeros_like(obs)
#     msk_vy = torch.zeros_like(obs)
    
#     msk_x[indx,0] = 1.0
#     msk_y[indx,1] = 1.0
#     msk_vx[indx,2] = 1.0
#     msk_vy[indx,3] = 1.0
    
#     obs_temp = deepcopy(obs)
    
#     obs[indx,0] = obs[indx,0] + x * msk_x[indx,0]
#     obs[indx,1] = obs[indx,1] + y * msk_y[indx,1]
#     obs[indx,2] = obs[indx,2] + vx * msk_vx[indx,2]
#     obs[indx,3] = obs[indx,3] + vy * msk_vy[indx,3]
    
#     predicted_qvalues = new_model(torch.reshape(obs, (1,num_Of_vehicles_Under_vision*vehicle_attr)))#???????????
#     #     CrossEntropyLoss
#     crossloss = nn.CrossEntropyLoss()
#     target = torch.empty(1, dtype=torch.long,device='cuda').random_(desire_actions,desire_actions+1)#???????????
#     l1 = crossloss(predicted_qvalues, target)
#     l2 = torch.sum((obs-obs_temp)**2)
#     # l2 = torch.sum((presence+x+x)**2)
#     output = 10*l1 
    
#     return output
#     # loss1 = torch.relu(target - predicted_qvalues.argmax(-1))
#     # loss2 = torch.sum((obs - obs_temp)**2)
#     # return 10 * loss1 + loss2
# # negetive log liklihood loss
# #     target = torch.tensor([desire_actions])
# #     output = nn.functional.nll_loss(nn.functional.log_softmax(predicted_qvalues), target)

# #     print("target",target)
# #     output = (predicted_qvalues_for_actions - predicted_qvalues_prim_for_actions.detach()) ** 2 
# def CF2Env3(obs, CF, indx, env):
#     env_prim = copy.deepcopy(env)
#     for indices in indx:
#         for vehicles in env_prim.env.road.vehicles[1:]:
#             if (all(vehicles.position.astype(int)==obs[indices,0:2].astype(int))):
#                 vehicles.position=CF[indices,0:2]
#                 # vehicles.velocity=CF[indices,2:4]
#                 break
#     return env_prim



















from bdb import effective
from select import select
from scipy.signal import convolve, gaussian
import torch.nn as nn
import numpy as np
import torch
from stable_baselines3 import DQN
from tqdm import trange
import matplotlib.pyplot as plt
import copy
from copy import deepcopy
import random
from IPython.display import clear_output
# from six import with_metaclass
#####################################################################
#####################################################################
#####################################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class DQNAgent(nn.Module):
    def __init__(self,  model, epsilon=0):

        super().__init__()
        self.epsilon = epsilon
        # and self.n_actions as output vector of logits of q(s, a)
        self.network = model
        # self.network.add_module('softmax', nn.Softmax(dim=0))
        # 
#         self.parameters = self.network.parameters
        
    def forward(self, points):
        # pass the state at time t through the newrok to get Q(s,a)
        qvalues = self.network(points)
        return qvalues
        
    def get_qvalues(self, states):
        # input is an array of states in numpy and outout is Qvals as numpy array
#         states = [torch.tensor(states, device=device, dtype=torch.float32) for t in states]
        states = torch.tensor(states, device=device, dtype=torch.float32)
        qvalues = self.forward(states)
        return qvalues.data.cpu().numpy()

    def sample_actions(self, qvalues):
        # sample actions from a batch of q_values using epsilon greedy policy
        epsilon = self.epsilon
        batch_size, n_actions = qvalues.shape
        random_actions = np.random.choice(n_actions, size=batch_size)
        best_actions = qvalues.argmax(axis=-1)
        should_explore = np.random.choice(
            [0, 1], batch_size, p=[1-epsilon, epsilon])
        return np.where(should_explore, random_actions, best_actions)
    
    
#####################################################################
#####################################################################
#####################################################################
def compute_td_loss(indx, presence, x, y, vx, vy, cos_h, sin_h, new_model, obs, desire_actions, device):

        # convert numpy array to torch tensors
        obs = torch.tensor(obs, device=device, dtype=torch.float)
        
        msk_presence = torch.zeros_like(obs)
        msk_x = torch.zeros_like(obs)
        msk_y = torch.zeros_like(obs)
        msk_vx = torch.zeros_like(obs)
        msk_vy = torch.zeros_like(obs)
        msk_cos_h = torch.zeros_like(obs)
        msk_sin_h = torch.zeros_like(obs)
        
        msk_presence[indx,0] = 1.0
        msk_x[indx,1] = 1.0
        msk_y[indx,2] = 1.0
        msk_vx[indx,3] = 1.0
        msk_vy[indx,4] = 1.0
        msk_cos_h[indx,5] = 1.0
        msk_sin_h[indx,6] = 1.0
        
        obs_temp = deepcopy(obs)
        
        obs[indx,0] = obs[indx,0] + presence * msk_presence[indx,0]
        obs[indx,1] = obs[indx,1] + x * msk_x[indx,1]
        obs[indx,2] = obs[indx,2] + y * msk_y[indx,2]
        obs[indx,3] = obs[indx,3] + vx * msk_vx[indx,3]
        obs[indx,4] = obs[indx,4] + vy * msk_vy[indx,4]
        obs[indx,5] = obs[indx,5] + cos_h * msk_cos_h[indx,5]
        obs[indx,6] = obs[indx,6] + sin_h * msk_sin_h[indx,6]
        
        # action = agent(obs).argmax(axis=-1)#???????????
        predicted_qvalues = new_model(torch.reshape(obs, (1,35)))#???????????
        
    #     CrossEntropyLoss
    # from torch.nn import functional as F
    #  F.smooth_l1_loss(current_q_values, target_q_values)
        loss = nn.CrossEntropyLoss()
        target = torch.empty(1, dtype=torch.long,device='cuda').random_(desire_actions,desire_actions+1)#???????????
        l1 = loss(predicted_qvalues, target)
        l2 = torch.sum((obs-obs_temp)**2)
        # l2 = torch.sum((presence+x+x)**2)
        output = 100*l1 + l2
        
        return output
#####################################################################
#####################################################################
#####################################################################       
def DQN_creation(policy, env, policy_kwargs, learning_rate,
                buffer_size, learning_starts, batch_size, gamma, train_freq, gradient_steps,
                target_update_interval, exploration_fraction, verbose, tensorboard_log):
    model = DQN(policy=policy, env=env,
                policy_kwargs=policy_kwargs, learning_rate=learning_rate,
                buffer_size=buffer_size,
                learning_starts=learning_starts,
                batch_size=batch_size,
                gamma=gamma,
                train_freq=train_freq,
                gradient_steps=gradient_steps,
                target_update_interval=target_update_interval,
                exploration_fraction=exploration_fraction,
                verbose=verbose,
                tensorboard_log=tensorboard_log)
    return  model
#####################################################################
#####################################################################
#####################################################################
def Save_DQN_model (model, fileName):
    model.save(fileName)
#####################################################################
#####################################################################
#####################################################################
def Load_DQN_model (fileName):
    return DQN.load(fileName)
#####################################################################
#####################################################################
#####################################################################
def CF_find(model, env, obs, device, indx):
    selected_action = model(torch.tensor(obs.reshape(obs.size), dtype=torch.float, device=device)).argmax(axis=-1)
    availableActions = [x for x in env.get_available_actions() if x!=selected_action]
    CFs = []
    for CounterActions in availableActions:
        CFs.append(learn_CF(model, obs, env, indx, device, CounterActions))
    return CFs
#####################################################################
#####################################################################
#####################################################################

def learn_CF(model, obs, env, indx, device, CounterActions):
    
    total_steps = 5 * 10**3
    # setup spme frequency for loggind and updating target network
    loss_freq = 20
    eval_freq = 3000
    td_loss_history = []
    
    presence = torch.tensor(10**-10*np.ones_like(indx), dtype=torch.float32, requires_grad=True).to(device)
    x = torch.tensor(10**-10*np.ones_like(indx), dtype=torch.float32, requires_grad=True).to(device)
    y = torch.tensor(10**-10*np.ones_like(indx), dtype=torch.float32, requires_grad=True).to(device)
    vx = torch.tensor(10**-10*np.ones_like(indx), dtype=torch.float32, requires_grad=True).to(device)
    vy = torch.tensor(10**-10*np.ones_like(indx), dtype=torch.float32, requires_grad=True).to(device)
    cos_h = torch.tensor(10**-10*np.ones_like(indx), dtype=torch.float32, requires_grad=True).to(device)
    sin_h = torch.tensor(10**-10*np.ones_like(indx), dtype=torch.float32, requires_grad=True).to(device)

    

    print(presence)

    obs_prim = torch.tensor(obs, device=device, dtype=torch.float)


   

    for step in trange(total_steps + 1):

        loss = compute_td_loss(indx, presence, x, y, vx, vy,cos_h, sin_h, model, obs, CounterActions, device)
        
        presence.grad = torch.zeros_like(presence)
        x.grad = torch.zeros_like(x)
        y.grad = torch.zeros_like(y)
        vx.grad = torch.zeros_like(vx)
        vy.grad = torch.zeros_like(vy)
        cos_h.grad = torch.zeros_like(cos_h)
        sin_h.grad = torch.zeros_like(sin_h)
        
        loss.backward(inputs=[y])
        presence = presence.detach() - 0.1 * presence.grad
        x = x.detach() - 0.1 * x.grad
        y = y.detach() - 0.1 * y.grad
        vx = vx.detach() - 0.1 * vx.grad
        vy = vy.detach() - 0.1 * vy.grad
        cos_h = cos_h.detach() - 0.1 * cos_h.grad
        sin_h = sin_h.detach() - 0.1 * sin_h.grad
        
        presence.requires_grad = True
        x.requires_grad = True
        y.requires_grad = True
        vx.requires_grad = True
        vy.requires_grad = True
        cos_h.requires_grad = True
        sin_h.requires_grad = True
        
        if step % loss_freq == 0:
            td_loss_history.append(loss.data.cpu().item())
            
        if step % eval_freq == 0:
            clear_output(True)
            assert not np.isnan(td_loss_history[-1])
            # plt.subplot(1, 2, 2)
            # plt.title("TD loss history (smoothened)")
            # plt.plot(smoothen(td_loss_history))
            # plt.grid()
            # plt.show()
        
    msk_presence = torch.zeros_like(obs_prim)
    msk_x = torch.zeros_like(obs_prim)
    msk_y = torch.zeros_like(obs_prim)
    msk_vx = torch.zeros_like(obs_prim)
    msk_vy = torch.zeros_like(obs_prim)
    msk_cos_h = torch.zeros_like(obs_prim)
    msk_sin_h = torch.zeros_like(obs_prim)
    msk_presence[indx,0] = 1.0
    msk_x[indx,1] = 1.0
    msk_y[indx,2] = 1.0
    msk_vx[indx,3] = 1.0
    msk_vy[indx,4] = 1.0
    msk_cos_h[indx,5] = 1.0
    msk_sin_h[indx,6] = 1.0
    obs_prim[indx,0] = obs_prim[indx,0] + presence * msk_presence[indx,0]
    obs_prim[indx,1] = obs_prim[indx,1] + x * msk_x[indx,1]
    obs_prim[indx,2] = obs_prim[indx,2] + y * msk_y[indx,2]
    obs_prim[indx,3] = obs_prim[indx,3] + vx * msk_vx[indx,3]
    obs_prim[indx,4] = obs_prim[indx,4] + vy * msk_vy[indx,4]
    obs_prim[indx,5] = obs_prim[indx,5] + cos_h * msk_cos_h[indx,5]
    obs_prim[indx,6] = obs_prim[indx,6] + sin_h * msk_sin_h[indx,6]
    # print("obs_prim", obs_prim)
    # print("obs", obs)
    if(model(obs_prim.reshape(35)).argmax(axis=-1)==CounterActions):
        print("CF finded")


    # print("q value with changes:", model(obs_prim.reshape(1,35)))
    # print("q value without changes:", model(torch.tensor(obs.reshape(obs.size), device=device, dtype=torch.float)))
    return obs_prim
#####################################################################
#####################################################################
#####################################################################
def CF2Env(obs, CF, indx, env):
    env_prim = copy.deepcopy(env)
    for indices in indx:
        for vehicles in env_prim.env.road.vehicles[1:]:
            if (all(vehicles.position.astype(int)==obs[indices,1:3].astype(int))):
                vehicles.position=CF[indices,1:3].cpu().detach().numpy()
                break
    return env_prim
#####################################################################
#####################################################################
#####################################################################
from numpy import linalg as LA
from dataclasses import dataclass



def Time2Coll(obs):
    time2coll = np.full(obs.shape[0],np.inf)
    @dataclass
    class vehicle:
        Radius:float
        X:float
        Y:float
        Xvel:float
        Yvel:float
    ego_vehicle = vehicle(1.5, obs[0,0], obs[0,1], obs[0,2], obs[0,3])     
    for j in range(1, obs.shape[0]):
        participant = vehicle(1, obs[j,0], obs[j,1], obs[j,2], obs[j,3])     
        time2coll[j] = Time2Coll2(ego_vehicle, participant)
    return time2coll


def Time2Coll2(ego_vehicle, participant):
    distance = (ego_vehicle.Radius + participant.Radius) * (ego_vehicle.Radius + participant.Radius)
    a = (ego_vehicle.Xvel - participant.Xvel) * (ego_vehicle.Xvel - participant.Xvel) + (ego_vehicle.Yvel - participant.Yvel) * (ego_vehicle.Yvel - participant.Yvel);
    b = 2 * ((ego_vehicle.X - participant.X) * (ego_vehicle.Xvel - participant.Xvel) + (ego_vehicle.Y - participant.Y) * (ego_vehicle.Yvel - participant.Yvel));
    c = (ego_vehicle.X - participant.X) * (ego_vehicle.X - participant.X) + (ego_vehicle.Y - participant.Y) * (ego_vehicle.Y - participant.Y) - distance
    d = b * b - 4 * a * c
    ## Ignore glancing collisions that may not cause a response due to limited precision and lead to an infinite loop
    if (b > -1e-6 or d <= 0):
        return np.nan
    e = np.sqrt(d)
    t1 = (-b - e) / (2 * a);    ## Collison time, +ve or -ve
    t2 = (-b + e) / (2 * a);    ## Exit time, +ve or -ve
    ## b < 0 => Getting closer
    ## If we are overlapping and moving closer, collide now
    if (t1 < 0 and t2 > 0 and b <= -1e-6):
        return 0
    return t1


    
def compute_td_loss_CF_CNN(indx, x, y, vx, vy, new_model, obs_kinematics, obs_bev, desire_actions, device):
    # convert numpy array to torch tensors
    obs_kinematics = torch.tensor(obs_kinematics, device=device, dtype=torch.float)
    
    msk_x = torch.zeros_like(obs_kinematics)
    msk_y = torch.zeros_like(obs_kinematics)
    msk_vx = torch.zeros_like(obs_kinematics)
    msk_vy = torch.zeros_like(obs_kinematics)
    
    msk_x[indx,0] = 1.0
    msk_y[indx,1] = 1.0
    msk_vx[indx,2] = 1.0
    msk_vy[indx,3] = 1.0
    
    obs_temp = deepcopy(obs_kinematics)

    obs_kinematics[indx,0] = obs_kinematics[indx,0] + x * msk_x[indx,0]
    obs_kinematics[indx,1] = obs_kinematics[indx,1] + y * msk_y[indx,1]
    obs_kinematics[indx,2] = obs_kinematics[indx,2] + vx * msk_vx[indx,2]
    obs_kinematics[indx,3] = obs_kinematics[indx,3] + vy * msk_vy[indx,3]
    
    # action = agent(obs).argmax(axis=-1)#???????????
    predicted_qvalues = new_model.q_net.q_net(new_model.q_net.features_extractor
                (torch.tensor(obs_bev, device=device, dtype=torch.float)))
    
#     CrossEntropyLoss
#   from torch.nn import functional as F
#   F.smooth_l1_loss(current_q_values, target_q_values)
    loss = nn.CrossEntropyLoss()
    target = torch.empty(1, dtype=torch.long,device=device).random_(desire_actions,desire_actions+1)#???????????
    l1 = loss(predicted_qvalues, target)
    l2 = torch.sum((obs_kinematics-obs_temp)**2)
    # l2 = torch.sum((presence+x+x)**2)
    # output = 100*l1 + l2
    output = l1 
    
    return output,obs_kinematics

def CF2Env2(obs, CF, indx, env):
    env_prim = copy.deepcopy(env)
    for indices in indx:
        for vehicles in env_prim.env.road.vehicles[1:]:
            if (all(vehicles.position.astype(int)==obs[indices,0:2].astype(int))):
                vehicles.position=CF[indices,0:2].cpu().detach().numpy()
                vehicles.velocity=CF[indices,2:-1].cpu().detach().numpy()
                break
    return env_prim


def smoothen(values):
    kernel = gaussian(100, std=100)
    kernel = kernel / np.sum(kernel)
    return convolve(values, kernel, 'valid')

##for the learning process in "Teacher_student_CNN_Linear_gradientCF"
def CF_find_2(student_model, env, obs, device, indx, num_Of_vehicles_Under_vision, vehicle_attr):
    selected_action = student_model(torch.tensor(obs.reshape(num_Of_vehicles_Under_vision*vehicle_attr), dtype=torch.float, device=device)).argmax(axis=-1)
    availableActions = [x for x in env.get_available_actions() if x!=selected_action]
    CFs = []
    desireAction=[]
    for CounterActions in availableActions:
        potentialCF = learn_CF2(student_model, obs, env, indx, device, CounterActions, num_Of_vehicles_Under_vision, vehicle_attr)
        if potentialCF is not None:
            print("CF finded")
            CFs.append(potentialCF)
            desireAction.append(CounterActions)
    return CFs, desireAction
##for the learning process in "Teacher_student_CNN_Linear_gradientCF"
def learn_CF2(student_model,obs,env,indx,device,CounterActions, num_Of_vehicles_Under_vision, vehicle_attr):
    loss_freq=1000
    eval_freq = 1000
    td_loss_history =[]
    total_steps=int(1e4)
    desire_actions = CounterActions

    x = torch.tensor(10**-10*np.ones_like(indx), dtype=torch.float32, requires_grad=True).to(device)
    y = torch.tensor(10**-10*np.ones_like(indx), dtype=torch.float32, requires_grad=True).to(device)
    vx = torch.tensor(10**-10*np.ones_like(indx), dtype=torch.float32, requires_grad=True).to(device)
    vy = torch.tensor(10**-10*np.ones_like(indx), dtype=torch.float32, requires_grad=True).to(device)
    def smoothen(values):
        kernel = gaussian(100, std=100)
        kernel = kernel / np.sum(kernel)
        return convolve(values, kernel, 'valid')

    from tqdm import trange
    from IPython.display import clear_output
    import matplotlib.pyplot as plt


    for step in trange(total_steps + 1):
    
        
        loss = compute_td_loss2(indx, x, y, vx, vy, student_model, obs, desire_actions, device, num_Of_vehicles_Under_vision, vehicle_attr )
        
        x.grad = torch.zeros_like(x)
        y.grad = torch.zeros_like(y)
        vx.grad = torch.zeros_like(vx)
        vy.grad = torch.zeros_like(vy)
        
        loss.backward(inputs=[x,y])

        x = x.detach() - 0.001 * x.grad
        y = y.detach() - 0.001 * y.grad
        vx = vx.detach() - 0.001 * vx.grad
        vy = vy.detach() - 0.001 * vy.grad
        
        x.requires_grad = True
        y.requires_grad = True
        vx.requires_grad = True
        vy.requires_grad = True
        
        if step % loss_freq == 0:
            td_loss_history.append(loss.data.cpu().item())
            
        if step % eval_freq == 0:
            clear_output(True)
            assert not np.isnan(td_loss_history[-1])
            plt.subplot(1, 2, 2)
            plt.title("TD loss history (smoothened)")
            # plt.plot(smoothen(td_loss_history))
            plt.plot(td_loss_history)
            plt.grid()
            plt.show()

    obs_prim = torch.tensor(deepcopy(obs), device=device, dtype=torch.float)
    msk_x = torch.zeros_like(obs_prim)
    msk_y = torch.zeros_like(obs_prim)
    msk_vx = torch.zeros_like(obs_prim)
    msk_vy = torch.zeros_like(obs_prim)
    msk_x[indx,0] = 1.0
    msk_y[indx,1] = 1.0
    msk_vx[indx,2] = 1.0
    msk_vy[indx,3] = 1.0
    obs_prim[indx,0] = obs_prim[indx,0] + x * msk_x[indx,0]
    obs_prim[indx,1] = obs_prim[indx,1] + y * msk_y[indx,1]
    obs_prim[indx,2] = obs_prim[indx,2] + vx * msk_vx[indx,2]
    obs_prim[indx,3] = obs_prim[indx,3] + vy * msk_vy[indx,3]
    # print("obs_prim", obs_prim)
    # print("obs", obs)
    if(student_model(obs_prim.reshape(num_Of_vehicles_Under_vision*vehicle_attr)).argmax(axis=-1).item()==desire_actions):
        print("CF finded")
        return obs_prim.cpu().detach().numpy()


    # print("q value with changes:", model(obs_prim.reshape(1,35)))
    # print("q value without changes:", model(torch.tensor(obs.reshape(obs.size), device=device, dtype=torch.float)))
    
def compute_td_loss2(indx, x, y, vx, vy, new_model, obs, desire_actions, device, num_Of_vehicles_Under_vision, vehicle_attr):
    # convert numpy array to torch tensors
    obs = torch.tensor(obs, device=device, dtype=torch.float)
    
    msk_x = torch.zeros_like(obs)
    msk_y = torch.zeros_like(obs)
    msk_vx = torch.zeros_like(obs)
    msk_vy = torch.zeros_like(obs)
    
    msk_x[indx,0] = 1.0
    msk_y[indx,1] = 1.0
    msk_vx[indx,2] = 1.0
    msk_vy[indx,3] = 1.0
    
    obs_temp = deepcopy(obs)
    
    obs[indx,0] = obs[indx,0] + x * msk_x[indx,0]
    obs[indx,1] = obs[indx,1] + y * msk_y[indx,1]
    obs[indx,2] = obs[indx,2] + vx * msk_vx[indx,2]
    obs[indx,3] = obs[indx,3] + vy * msk_vy[indx,3]
    
    predicted_qvalues = new_model(torch.reshape(obs, (1,num_Of_vehicles_Under_vision*vehicle_attr)))#???????????
    #     CrossEntropyLoss
    crossloss = nn.CrossEntropyLoss()
    target = torch.empty(1, dtype=torch.long,device='cuda').random_(desire_actions,desire_actions+1)#???????????
    l1 = crossloss(predicted_qvalues, target)
    l2 = torch.sum((obs-obs_temp)**2)
    # l2 = torch.sum((presence+x+x)**2)
    output = 10*l1 
    
    return output
    # loss1 = torch.relu(target - predicted_qvalues.argmax(-1))
    # loss2 = torch.sum((obs - obs_temp)**2)
    # return 10 * loss1 + loss2
# negetive log liklihood loss
#     target = torch.tensor([desire_actions])
#     output = nn.functional.nll_loss(nn.functional.log_softmax(predicted_qvalues), target)

#     print("target",target)
#     output = (predicted_qvalues_for_actions - predicted_qvalues_prim_for_actions.detach()) ** 2 
def CF2Env3(obs, CF, indx, env):
    env_prim = copy.deepcopy(env)
    for indices in indx:
        for vehicles in env_prim.env.road.vehicles[1:]:
            if (all(vehicles.position.astype(int)==obs[indices,0:2].astype(int))):
                vehicles.position=CF[indices,0:2]
                # vehicles.velocity=CF[indices,2:4]
                break
    return env_prim



def learn_CF3(student_model,obs,env,indx,device,CounterActions, num_Of_vehicles_Under_vision, vehicle_attr,lambdaa):
    loss_freq=1000
    eval_freq = 1000
    td_loss_history =[]
    total_steps=int(1e4)
    desire_actions = CounterActions

    x = torch.tensor(10**-10*np.ones_like(indx), dtype=torch.float32, requires_grad=True).to(device)
    y = torch.tensor(10**-10*np.ones_like(indx), dtype=torch.float32, requires_grad=True).to(device)
    vx = torch.tensor(10**-10*np.ones_like(indx), dtype=torch.float32, requires_grad=True).to(device)
    vy = torch.tensor(10**-10*np.ones_like(indx), dtype=torch.float32, requires_grad=True).to(device)
    def smoothen(values):
        kernel = gaussian(100, std=100)
        kernel = kernel / np.sum(kernel)
        return convolve(values, kernel, 'valid')

    from tqdm import trange
    from IPython.display import clear_output
    import matplotlib.pyplot as plt


    for step in trange(total_steps + 1):
    
        
        loss = compute_td_loss3(indx, x, y, vx, vy, student_model, obs, desire_actions, device, num_Of_vehicles_Under_vision, vehicle_attr,lambdaa )
        
        x.grad = torch.zeros_like(x)
        y.grad = torch.zeros_like(y)
        vx.grad = torch.zeros_like(vx)
        vy.grad = torch.zeros_like(vy)
        
        loss.backward(inputs=[x,y])

        x = x.detach() - 0.001 * x.grad
        y = y.detach() - 0.001 * y.grad
        vx = vx.detach() - 0.001 * vx.grad
        vy = vy.detach() - 0.001 * vy.grad
        
        x.requires_grad = True
        y.requires_grad = True
        vx.requires_grad = True
        vy.requires_grad = True
        
        if step % loss_freq == 0:
            td_loss_history.append(loss.data.cpu().item())
            
        if step % eval_freq == 0:
            clear_output(True)
            assert not np.isnan(td_loss_history[-1])
            plt.subplot(1, 2, 2)
            plt.title("TD loss history (smoothened)")
            # plt.plot(smoothen(td_loss_history))
            plt.plot(td_loss_history)
            plt.grid()
            plt.show()

    obs_prim = torch.tensor(deepcopy(obs), device=device, dtype=torch.float)
    msk_x = torch.zeros_like(obs_prim)
    msk_y = torch.zeros_like(obs_prim)
    msk_vx = torch.zeros_like(obs_prim)
    msk_vy = torch.zeros_like(obs_prim)
    msk_x[indx,0] = 1.0
    msk_y[indx,1] = 1.0
    msk_vx[indx,2] = 1.0
    msk_vy[indx,3] = 1.0
    obs_prim[indx,0] = obs_prim[indx,0] + x * msk_x[indx,0]
    obs_prim[indx,1] = obs_prim[indx,1] + y * msk_y[indx,1]
    obs_prim[indx,2] = obs_prim[indx,2] + vx * msk_vx[indx,2]
    obs_prim[indx,3] = obs_prim[indx,3] + vy * msk_vy[indx,3]
    # print("obs_prim", obs_prim)
    # print("obs", obs)
    if(student_model.policy(obs_prim.reshape(num_Of_vehicles_Under_vision*vehicle_attr)).loc.argmax().item()==desire_actions):
        print("CF finded")
        return obs_prim.cpu().detach().numpy()


    # print("q value with changes:", model(obs_prim.reshape(1,35)))
    # print("q value without changes:", model(torch.tensor(obs.reshape(obs.size), device=device, dtype=torch.float)))
    
def compute_td_loss3(indx, x, y, vx, vy, new_model, obs, desire_actions, device, num_Of_vehicles_Under_vision, vehicle_attr, lambdaa):
    # convert numpy array to torch tensors
    obs = torch.tensor(obs, device=device, dtype=torch.float)
    
    msk_x = torch.zeros_like(obs)
    msk_y = torch.zeros_like(obs)
    msk_vx = torch.zeros_like(obs)
    msk_vy = torch.zeros_like(obs)
    
    msk_x[indx,0] = 1.0
    msk_y[indx,1] = 1.0
    msk_vx[indx,2] = 1.0
    msk_vy[indx,3] = 1.0
    
    obs_temp = deepcopy(obs)
    
    obs[indx,0] = obs[indx,0] + x * msk_x[indx,0]
    obs[indx,1] = obs[indx,1] + y * msk_y[indx,1]
    obs[indx,2] = obs[indx,2] + vx * msk_vx[indx,2]
    obs[indx,3] = obs[indx,3] + vy * msk_vy[indx,3]
    
    predicted_qvalues = new_model.policy(torch.reshape(obs, (1,num_Of_vehicles_Under_vision*vehicle_attr))).loc
    #     CrossEntropyLoss
    crossloss = nn.CrossEntropyLoss()
    target = torch.empty(1, dtype=torch.long,device='cuda').random_(desire_actions,desire_actions+1)#???????????
    l1 = crossloss(predicted_qvalues, target)
    l2 = torch.sum((obs-obs_temp)**2)
    # l2 = torch.sum((presence+x+x)**2)
    output = lambdaa*l1+(1-lambdaa)*l2 
    
    return output

