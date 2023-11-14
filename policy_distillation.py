# # from itertools import count
# # from time import time, strftime, localtime
# # import gym
# # import scipy.optimize
# # from tensorboardX import SummaryWriter
# # from core.models import *
# # from core.agent_ray_pd import AgentCollection
# # from utils.utils import *
# # import numpy as np
# # import ray
# # import envs
# # from trpo import trpo
# # from student import Student
# # from teacher import Teacher
# # import os
# # import pickle
# # from stable_baselines3 import DQN
# # import highway_env
# # import local_lib as myLib
# # import pandas as pd
# # import warnings
# # warnings.simplefilter(action='ignore', category=FutureWarning)


# # torch.utils.backcompat.broadcast_warning.enabled = True
# # torch.utils.backcompat.keepdim_warning.enabled = True
# # # torch.set_default_tensor_type('torch.DoubleTensor')
# # torch.set_default_tensor_type('torch.FloatTensor')
# # dtype = torch.float
# # torch.set_default_dtype(dtype)


# # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# # '''
# # 1. train single or multiple teacher policies
# # 2. collect samples from teacher policy
# # 3. use KL or W2 distance as metric to train student policy
# # 4. test student policy
# # '''

# # def train_teachers():
# #     envs = []
# #     teacher_policies = []
# #     time_begin = time()
# #     print('Training {} teacher policies...'.format(args.num_teachers))
# #     for i in range(args.num_teachers):
# #         print('Training no.{} teacher policy...'.format(i + 1))
# #         env = gym.make(args.env_name)
# #         envs.append(env)
# #         teacher_policies.append(trpo(env, args))
# #     time_pretrain = time() - time_begin
# #     print('Training teacher is done, using time {}'.format(time_pretrain))
# #     return envs, teacher_policies

# # def main(args, env):
# #     ray.init(num_cpus=args.num_workers, num_gpus=1)
# #     # policy and envs for sampling
# #     np.random.seed(args.seed)
# #     torch.manual_seed(args.seed)
# #     exp_date = strftime('%Y.%m.%d', localtime(time()))
# #     writer = SummaryWriter(log_dir='./exp_data/{}/{}_{}'.format(exp_date, args.env_name, time()))
# #     # load saved models if args.load_models
# #     if args.load_models:
# #         envs = []
# #         teacher_policies = []
# #         dummy_env = env
# #         dummy_env.env.reset()
# #         envs.append(dummy_env)
# #     else:
# #         envs, teacher_policies = train_teachers()

# #     dataset_df = pd.read_excel('DQN_CNN_regression.xlsx', index_col=0)
# #     numeric_feature_names = ['obs_0_x', 'obs_0_y', 'obs_0_vx',  'obs_0_vy',
# #                             'obs_1_x', 'obs_1_y', 'obs_1_vx',  'obs_1_vy',
# #                             'obs_2_x', 'obs_2_y', 'obs_2_vx',  'obs_2_vy',
# #                             'obs_3_x', 'obs_3_y', 'obs_3_vx',  'obs_3_vy',
# #                             'q_value_0','q_value_1','q_value_2','q_value_3','q_value_4']
# #     expert_data = []
# #     numeric_features = dataset_df[numeric_feature_names]
# #     tensor_numeric_features = torch.tensor(numeric_features.values, dtype=torch.float)
# #     for i,item in enumerate(tensor_numeric_features):
# #         expert_data.append([])
# #         expert_data[-1].append(item[0:16])
# #         expert_data[-1].append(item[16:21])
# #         expert_data[-1].append(torch.tensor([0.001,0.1,0.001,0.1,0.1], dtype=torch.float, device=args.device))


# #     student = Student(args)
# #     print('Training student policy...')
# #     time_beigin = time()
# #     # train student policy
# #     for iter in count(1):
# #         if args.algo == 'npg':
# #             loss = student.npg_train(expert_data)
# #         elif args.algo == 'storm':
# #             if iter == 1:
# #                 loss, prev_params, prev_grad, direction = student.storm_train(None, None, None, expert_data, iter)
# #             else:
# #                 loss, prev_params, prev_grad, direction = student.storm_train(prev_params, prev_grad, direction, expert_data, iter)
# #         else:
# #             loss = student.train(expert_data)
# #         writer.add_scalar('{} loss'.format(args.loss_metric), loss.data, iter)
# #         print('Itr {} {} loss: {:.2f}'.format(iter, args.loss_metric, loss.data))
# #         if iter > args.num_student_episodes:
# #             break
# #     time_train = time() - time_beigin
# #     print('Training student policy finished, using time {}'.format(time_train))
# #     return student



# # if __name__ == '__main__':
# #     import argparse

# #     parser = argparse.ArgumentParser(description='Policy distillation')
# #     # Network, env, MDP, seed
# #     parser.add_argument('--hidden-size', type=int, default=256,
# #                         help='number of hidden units per layer')
# #     parser.add_argument('--num-layers', type=int, default=2,
# #                         help='number of hidden layers')
# #     parser.add_argument('--env-name', default="highway-fast-v0", metavar='G',
# #                         help='name of the environment to run')
# #     parser.add_argument('--gamma', type=float, default=0.995, metavar='G',
# #                         help='discount factor (default: 0.995)')
# #     parser.add_argument('--tau', type=float, default=0.97, metavar='G',
# #                         help='gae (default: 0.97)')
# #     parser.add_argument('--seed', type=int, default=1, metavar='N',
# #                         help='random seed (default: 1)')
# #     parser.add_argument('--load-models', default=True, action='store_true',
# #                         help='load_pretrained_models')

# #     # Teacher policy training
# #     parser.add_argument('--agent-count', type=int, default=10, metavar='N',
# #                         help='number of agents (default: 100)')
# #     parser.add_argument('--num-teachers', type=int, default=1, metavar='N',
# #                         help='number of teacher policies (default: 1)')
# #     parser.add_argument('--max-kl', type=float, default=1e-2, metavar='G',
# #                         help='max kl value (default: 1e-2)')
# #     parser.add_argument('--cg-damping', type=float, default=1e-2, metavar='G',
# #                         help='damping for conjugate gradient (default: 1e-2)')
# #     parser.add_argument('--cg-iter', type=int, default=10, metavar='G',
# #                         help='maximum iteration of conjugate gradient (default: 1e-1)')
# #     parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
# #                         help='l2 regularization parameter for critics (default: 1e-3)')
# #     parser.add_argument('--teacher-batch-size', type=int, default=1000, metavar='N',
# #                         help='per-iteration batch size for each agent (default: 1000)')
# #     parser.add_argument('--sample-batch-size', type=int, default=10000, metavar='N',
# #                         help='expert batch size for each teacher (default: 10000)')
# #     parser.add_argument('--render', action='store_true',
# #                         help='render the environment')
# #     parser.add_argument('--log-interval', type=int, default=1, metavar='N',
# #                         help='interval between training status logs (default: 10)')
# #     parser.add_argument('--device', type=str, default='cpu',
# #                         help='set the device (cpu or cuda)')
# #     parser.add_argument('--num-workers', type=int, default=10,
# #                         help='number of workers for parallel computing')
# #     parser.add_argument('--num-teacher-episodes', type=int, default=10, metavar='N',
# #                         help='num of teacher training episodes (default: 100)')

# #     # Student policy training
# #     parser.add_argument('--lr', type=float, default=1e-5, metavar='G',
# #                         help='adam learnig rate (default: 1e-3)')
# #     parser.add_argument('--test-interval', type=int, default=10, metavar='N',
# #                         help='interval between training status logs (default: 10)')
# #     parser.add_argument('--student-batch-size', type=int, default=1000, metavar='N',
# #                         help='per-iteration batch size for student (default: 1000)')
# #     parser.add_argument('--sample-interval', type=int, default=10, metavar='N',
# #                         help='frequency to update expert data (default: 10)')
# #     parser.add_argument('--testing-batch-size', type=int, default=10000, metavar='N',
# #                         help='batch size for testing student policy (default: 10000)')
# #     parser.add_argument('--num-student-episodes', type=int, default=1000, metavar='N',
# #                         help='num of teacher training episodes (default: 1000)')
# #     parser.add_argument('--loss-metric', type=str, default='kl',
# #                         help='metric to build student objective')
# #     parser.add_argument('--algo', type=str, default='sgd',
# #                         help='update method')
# #     parser.add_argument('--storm-interval', type=int, default=10, metavar='N',
# #                         help='frequency of storm (default: 10)')
# #     parser.add_argument('--init-alpha', type=float, default=1.0, metavar='G',
# #                         help='storm init alpha (default: 1.0)')
# #     args = parser.parse_args()

# #     main(args)



# from ast import arg
# from itertools import count
# from time import time, strftime, localtime
# import gym
# import scipy.optimize
# from tensorboardX import SummaryWriter
# from core.models import *
# from core.agent_ray_pd import AgentCollection
# from utils.utils import *
# import numpy as np
# import ray
# import envs
# from trpo import trpo
# from student import Student
# from teacher import Teacher
# import os
# import pickle
# from stable_baselines3 import DQN
# import highway_env
# import local_lib as myLib
# import pandas as pd
# import warnings
# import argparse
# warnings.simplefilter(action='ignore', category=FutureWarning)


# torch.utils.backcompat.broadcast_warning.enabled = True
# torch.utils.backcompat.keepdim_warning.enabled = True
# # torch.set_default_tensor_type('torch.DoubleTensor')
# torch.set_default_tensor_type('torch.FloatTensor')
# dtype = torch.float
# torch.set_default_dtype(dtype)


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# '''
# 1. train single or multiple teacher policies
# 2. collect samples from teacher policy
# 3. use KL or W2 distance as metric to train student policy
# 4. test student policy
# '''

# def main(args, numeric_features, class_weights):
#     # policy and envs for sampling
#     np.random.seed(args.seed)
#     torch.manual_seed(args.seed)
#     exp_date = strftime('%Y.%m.%d', localtime(time()))
#     writer = SummaryWriter(log_dir='./exp_data/{}/{}_{}'.format(exp_date, args.env_name, time()))

#     expert_data = []
#     tensor_numeric_features = torch.tensor(numeric_features.values, dtype=torch.float, device=args.device)
#     for i,item in enumerate(tensor_numeric_features):
#         expert_data.append([])
#         expert_data[-1].append(item[0 : args.env_input_size])
#         expert_data[-1].append(item[args.env_input_size : args.env_input_size+args.env_output_size])
#         expert_data[-1].append(torch.tensor([0.09 for i in range(args.env_output_size)], dtype=torch.float, device=args.device))


#     student = Student(args)
#     print('Training student policy...')
#     time_beigin = time()
#     # train student policy

#     # import wandb
#     # wandb.init(project="ppo_student")
#     # wandb.config = {
#     #     "learning_rate": args.lr,
#     #     "epochs": args.num_student_episodes,
#     #     "batch_size": args.student_batch_size
#     #     }
    
#     for iter in count(1):
#         if args.algo == 'npg':
#             loss = student.npg_train(expert_data,class_weights)
#         elif args.algo == 'storm':
#             if iter == 1:
#                 loss, prev_params, prev_grad, direction = student.storm_train(None, None, None, expert_data, iter)
#             else:
#                 loss, prev_params, prev_grad, direction = student.storm_train(prev_params, prev_grad, direction, expert_data, iter)
#         else:
#             loss = student.train(expert_data, tensor_numeric_features ,class_weights)
#         # wandb.log({"loss": loss})
#         writer.add_scalar('{} loss'.format(args.loss_metric), loss.data, iter)
#         print('Itr {} {} loss: {:.2f}'.format(iter, args.loss_metric, loss.data))
#         if iter > args.num_student_episodes:
#             break
#     time_train = time() - time_beigin
#     print('Training student policy finished, using time {}'.format(time_train))
#     return student, loss.data


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
