from ast import arg
from itertools import count
from time import time, strftime, localtime
import gym
import scipy.optimize
from tensorboardX import SummaryWriter
from core.models import *
from core.agent_ray_pd import AgentCollection
from utils.utils import *
import numpy as np
import ray
import envs
from trpo import trpo
from student import Student
from teacher import Teacher
import os
import pickle
from stable_baselines3 import DQN
import highway_env
import local_lib as myLib
import pandas as pd
import warnings
from torch.utils.data import TensorDataset, DataLoader
import argparse
# warnings.simplefilter(action='ignore', category=FutureWarning)
from statistics import mean

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True
# torch.set_default_tensor_type('torch.DoubleTensor')
torch.set_default_tensor_type('torch.FloatTensor')
dtype = torch.float
torch.set_default_dtype(dtype)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



'''
1. train single or multiple teacher policies
2. collect samples from teacher policy
3. use KL or W2 distance as metric to train student policy
4. test student policy
'''

def main(args, dataloader):
    # policy and envs for sampling
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # exp_date = strftime('%Y.%m.%d', localtime(time()))
    # writer = SummaryWriter(log_dir='./exp_data/{}/{}_{}'.format(exp_date, args.env_name, time()))

    # expert_data = []
    # time_beigin = time()
    # tensor_numeric_features = torch.tensor(np.array(numeric_features.values),
    #                 dtype=torch.float, device=args.device)
    # # tensor_numeric_features = torch.tensor(numeric_features.values, dtype=torch.float, device=args.device)
    # # for i,item in enumerate(tensor_numeric_features):
    # #     expert_data.append([])
    # #     expert_data[-1].append(item[0 : args.env_input_size])
    # #     expert_data[-1].append(item[args.env_input_size : args.env_input_size+args.env_output_size])
    # #     expert_data[-1].append(torch.tensor([0.01 for i in range(args.env_output_size)], dtype=torch.float, device=args.device))
    # # numeric_features = np.array(numeric_features.values)
    # input_tensor = tensor_numeric_features[:,0 : args.env_input_size]
    # output_tensor = tensor_numeric_features[:,
    #                 args.env_input_size : args.env_input_size+args.env_output_size]
    # output_sigma_tensor = torch.ones([tensor_numeric_features.size(0),
    #                 args.env_output_size], device=args.device)*0.01
    
    # dataset = TensorDataset(input_tensor,output_tensor,output_sigma_tensor)
    # train_length=int(1*len(dataset))
    # test_length=len(dataset)-train_length
    # tr_dataset = torch.utils.data.Subset(dataset, range(train_length))
    # val_dataset = torch.utils.data.Subset(dataset, range(train_length, train_length+test_length))
                
    # if (args.batch_type=='random'):
    #     tr_dataloader = DataLoader(tr_dataset, shuffle=True, batch_size = args.student_batch_size,
    #                              drop_last=False)
    #     val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size = args.testing_batch_size,
    #                              drop_last=True)
    # elif (args.batch_type=='respectively'):
    #     tr_dataloader = DataLoader(tr_dataset, shuffle=False, batch_size = args.student_batch_size,
    #                              drop_last=False)
    #     val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size = args.testing_batch_size,
    #                              drop_last=True)
    # time_data_set_prep = time() - time_beigin
    # print("time_data_set_prep", time_data_set_prep) 
    tr_dataloader = dataloader
    print("dataset length: ", len(tr_dataloader))
    student = Student(args)
    print('Training student policy...')
    time_beigin = time()
    # train student policy

    # import wandb
    # wandb.init(project="ppo_student")
    # wandb.config = {
    #     "learning_rate": args.lr,
    #     "epochs": args.num_student_episodes,
    #     "batch_size": args.student_batch_size
    #     }
    for iter in count(1):
        tr_loss_history, val_loss_history = [], []
        for tr_batch_idx, tr_batch_data in enumerate(tr_dataloader):
            # objList = 
            qvalues = tr_batch_data["q_value"]
            tr_loss_history.append(student.train(tr_batch_data, qvalues).tolist())
                

        # if args.algo == 'npg':
        #     loss = student.npg_train(expert_data,class_weights)
        # elif args.algo == 'storm':
        #     if iter == 1:
        #         loss, prev_params, prev_grad, direction = student.storm_train(None, None, None, expert_data, iter)
        #     else:
        #         loss, prev_params, prev_grad, direction = student.storm_train(prev_params, prev_grad, direction, expert_data, iter)
        # else:
        #     loss = student.train(expert_data, tensor_numeric_features ,class_weights)
        # wandb.log({"loss": loss})
            writer.add_scalar('{} loss'.format(args.loss_metric), mean(tr_loss_history), iter)
            if ((tr_batch_idx+1)%args.report_step==0):
                # for val_batch_idx, val_batch_data in enumerate(val_dataloader):
                #     val_loss_history.append(student.validation(val_batch_data, class_weights).tolist())
                print('Itr {} Batch {}: {} train loss: {:.3f}'\
                      .format(iter, tr_batch_idx+1, args.loss_metric, mean(tr_loss_history)))

                # print('Itr {} Batch {}: {} train loss: {:.3f} validation loss: {:.3f}'\
                #       .format(iter, tr_batch_idx+1, args.loss_metric, mean(tr_loss_history), mean(val_loss_history)))
        if iter >= args.num_student_episodes:
            break
    time_train = time() - time_beigin
    print('Training student policy finished, using time {}'.format(time_train))
    return student, mean(tr_loss_history)
