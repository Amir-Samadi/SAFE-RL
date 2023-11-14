import torch
import math
from torch.distributions import Normal, Categorical
from torch.distributions.kl import kl_divergence
from torch.nn import CrossEntropyLoss
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def normal_entropy(std):
    var = std.pow(2)
    entropy = 0.5 + 0.5 * torch.log(2 * var * math.pi)
    return entropy.sum(1, keepdim=True)

def normal_log_density(x, mean, log_std, std):
    var = std.pow(2)
    log_density = -(x - mean).pow(2) / (2 * var) - 0.5 * math.log(2 * math.pi) - log_std
    return log_density.sum(1, keepdim=True)

def get_kl(teacher_dist_info, student_dist_info, class_weights):
    # class_weights = torch.tensor(class_weights, requires_grad = False, device=teacher_dist_info[0].device.type)
    # pi = Normal(loc=teacher_dist_info[0], scale=teacher_dist_info[1])
    # pi_new = Normal(loc=student_dist_info[0], scale=student_dist_info[1])
    # kl = torch.mean(class_weights * kl_divergence(pi, pi_new)) #+ torch.mean(class_weights*teacher_dist_info[0]) 
    # return kl
    class_weights = torch.tensor(class_weights, requires_grad = False, device=teacher_dist_info[0].device.type)
    pi = Categorical(probs=teacher_dist_info[0])
    pi_new = Categorical(logits=student_dist_info[0])
    kl = torch.mean(kl_divergence(pi, pi_new)) #+ torch.mean(class_weights*teacher_dist_info[0]) 
    return kl

def get_wasserstein(teacher_dist_info, student_dist_info, class_weights):
    class_weights = torch.tensor(class_weights, requires_grad = False, device=teacher_dist_info[0].device.type)
    means_t, stds_t = teacher_dist_info
    means_s, stds_s = student_dist_info
    pi = Categorical(probs=teacher_dist_info[0])
    pi_new = Categorical(logits=student_dist_info[0])
    return torch.mean(((pi_new.logits - pi.probs)) ** 2)

def get_klAndCross(teacher_dist_info, student_dist_info, betta, temperature):
    CrossLoss = CrossEntropyLoss() 
    # # class_weights = torch.tensor(class_weights,requires_grad = True, device=teacher_dist_info[0].device.type)
    # pi = Normal(loc=teacher_dist_info[0], scale=teacher_dist_info[1])
    # pi_new = Normal(loc=student_dist_info[0], scale=student_dist_info[1])
    # kl = torch.mean(kl_divergence(pi, pi_new)) #+ torch.mean(class_weights*teacher_dist_info[0]) 
    # return kl*(1-betta) + CrossLossValue*(betta)
    pi = Categorical(probs=teacher_dist_info[0])
    pi_new = Categorical(logits=student_dist_info[0])
    
    CrossLossValue = CrossLoss (pi_new.logits, pi.probs.argmax(dim=-1))

    Kl_loss = nn.KLDivLoss(reduction="batchmean")((pi_new.logits/temperature), (pi.probs/temperature))

    KD_loss =  (Kl_loss * (betta * temperature * temperature)) + (CrossLossValue * (1. - betta))

    return KD_loss

def get_nll(teacher_dist_info, student_dist_info, class_weights):
    class_weights = torch.tensor(class_weights, requires_grad = False, device=teacher_dist_info[0].device.type)
    pi = Categorical(probs=teacher_dist_info[0])
    pi_new = Categorical(logits=student_dist_info[0])
    
    m = nn.LogSoftmax(dim=1)
    loss = nn.NLLLoss()
    
    nll_loss = loss(m(pi_new.logits), pi.probs.argmax(dim=-1))
    return nll_loss
# def get_probabilistic(teacher_dist_info, student_dist_info, betta, temperature):
def get_probabilistic(mu_student, sig_student, rho_student, x_teacher):
 
    ohr = torch.pow(1.001-torch.pow(rho_student, 2),-0.5)
    z = torch.pow(sig_student,-2)*torch.pow(x_teacher-mu_student,2) - 2*rho_student*torch.pow(sig_student,-1)*(x_teacher-mu_student)

    denom = torch.log((1/(2*math.pi))*torch.pow(sig_student,-1)*ohr)
    nll = -1*(denom - 0.5*torch.pow(ohr,2)*z)
    lossVal = torch.sum(nll)/np.prod(x_teacher.shape)
    return lossVal