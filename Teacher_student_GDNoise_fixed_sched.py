import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.autograd import Function
from torch.nn.parameter import Parameter
from torch import autograd 

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import mean_squared_error

import numpy as np
from numpy import linalg as LA
import random

# from mpltools import annotation

from scipy.io import savemat, whosmat
from scipy.optimize import fsolve
import scipy.signal

import time
import os
import sys 
from tempfile import TemporaryFile

import copy
import pathlib

import io
import signal
import noisy_sgd 

import argparse

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def calc_run_time(start, end):
    seconds = int(end - start)
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    run_time = '{:d}:{:02d}:{:02d}'.format(h, m, s)
    print('run_time =', run_time)
    return run_time

def signal_handler(sig, frame):
        global stop_me
        print('stopping')
        stop_me = True

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        
def extract_inputs_and_labels(data_set, n): 
    inds = list(range(n))
    data_set = torch.utils.data.Subset(data_set, inds)
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=n, shuffle=False, num_workers=2)
    for i, data in enumerate(data_loader, 0): 
        if i == 0: 
            inputs, labels = data  
    return inputs, labels

def reshape_and_center_inputs(inputs, n, x_pixs): 
    inputs = inputs[:n, :, :x_pixs, :x_pixs]
    dim = 3*(x_pixs**2)
    inputs = inputs.reshape(n, dim)
    inputs = inputs - torch.unsqueeze(torch.mean(inputs, axis=0), 0)
    return inputs
         

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------    
    
parser = argparse.ArgumentParser(description="teacher_student_GDNoise")
parser.add_argument('--lr_fac', type=float, default=2.0)
parser.add_argument('--num_channels', type=int, default=32)
parser.add_argument('--num_channels_teacher', type=int, default=1)
parser.add_argument('--n_train', type=int, default=1024)
parser.add_argument('--input_dim', type=int, default=64)
parser.add_argument('--train_seed', type=int, default=111)
parser.add_argument('--folder', default='./')
parser.add_argument('--lr_max', type=float, default=0.00015)
parser.add_argument('--sigma2_0', type=float, default=0.001)
parser.add_argument('--scaling', type=str,default='rich')
parser.add_argument('--device', type=str,default='cuda')
parser.add_argument('--max_epochs', type=int,default=9000000)
parser.add_argument('--save_every', type=int,default=20000)

c = parser.parse_args()
torch.set_num_threads(4)

stop_me = False
signal.signal(signal.SIGINT, signal_handler)
# Collect events until released

activation_type = 'erf' # 'erf', 'relu'
arch = 'FCN' # 'FCN', 'CNN'

train_protocol = 'Langevin' # 'SGD', 'Langevin', 'Adam'
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = c.device
max_epochs = c.max_epochs  # min((10 * t_eq + t_eq), 1000000) 
start_rec_nets = 5000

momentum = 0
dampening = 0.0
nesterov = False

num_channels = c.num_channels 
num_channels_teacher = c.num_channels_teacher
scaling = c.scaling # 'lazy', 'rich'

if arch == 'CNN': 
    kernel_size = 3
    input_dim = 3
elif arch == 'FCN': 
    kernel_size = 1   
    input_dim = c.input_dim

train_seed_type = 'consistent'  # 'consistent', 'random'
train_seed = c.train_seed
data_seed = 111
init_seed = 222

num_classes = 10 
print_every = 100
save_nets_every = c.save_every


half = False
zca = False
square_loss = True
dtype=torch.float32
loss_fn = "mse"
zca_str, aug_str, half_str = '', '', ''

    
# if train_protocol == 'Langevin':     
# GD+noise hyper-params  
sample = 'first' # 'first', 'random'
lr_fac = c.lr_fac  # np.float(sys.argv[1]) # This will the factor dividing the highest spike free learning rate as found by the algorithm 
n_train = c.n_train  
lr0 = 3.90625e-06 * (256/n_train**1.1)
n_test = n_train
sample_seed = 123 # for sampling the training and test data
batch_size = n_train # full-batch
sigma2_0 = c.sigma2_0 
sigma2 = sigma2_0 
# if scaling == 'rich':
#     sigma2 = sigma2_0 / num_channels 
# elif scaling == 'lazy':    
#     sigma2 = sigma2_0 
temperature = 2.0 * sigma2  # notice factor of 2.0
prefactor = 1/2.0  # 1/4.0 # 1/2.0
wd_input = prefactor * temperature * input_dim * kernel_size**2
wd_hidden = prefactor * temperature * num_channels * kernel_size**2
if scaling == 'rich':
    wd_output = prefactor * temperature * num_channels**2 # notice num_channels**2 is due to the MF scaling
elif scaling == 'lazy':    
    print('** lazy scaling **')
    wd_output = prefactor * temperature * num_channels    # notice num_channels is due to the GP scaling
wd_tup = (wd_input, wd_hidden, wd_output)
mse_reduction = 'sum'    # use SE loss which is the sum, not the average

print('max_epochs = {} | save_nets_every = {} | start_rec_nets = {}'.format(max_epochs, save_nets_every, start_rec_nets) )    
print('sigma2 = {} | wd_input = {} | wd_hidden = {} | wd_output = {}'.format(sigma2, wd_input, wd_hidden, wd_output))
print('zca = {} | kernel_size = {}'.format(zca, kernel_size))



# folder and file names
exp_folder = c.folder
exp_name = 'FCN_3_layers__input_dim={}__C={}__n_train={}__train_seed={}__lr_fac={}__sigma2={}__activation={}__scaling={}'.format(input_dim, num_channels, n_train, train_seed, lr_fac, sigma2_0, activation_type,scaling)        
if not os.path.exists(exp_folder):
    os.makedirs(exp_folder)
results_file_name = r'{}/{}.npz'.format(exp_folder, exp_name)    

    
# architecture    
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------     

class FCN_3_layers(nn.Module):
    def __init__(self, input_dim, num_channels, init_seed=None):
        super().__init__() 
        if init_seed is not None: 
            torch.manual_seed(init_seed)
        self.linear1 = nn.Linear(input_dim, num_channels, bias=False)
        self.linear2 = nn.Linear(num_channels, num_channels, bias=False)
        self.linear3 = nn.Linear(num_channels, out_features=1, bias=False)
    
        # Initialization
        # nn.init.kaiming_normal_(self.linear1.weight, mode='fan_in', nonlinearity='sigmoid')
        # nn.init.kaiming_normal_(self.linear2.weight, mode='fan_in', nonlinearity='sigmoid')
        # nn.init.kaiming_normal_(self.linear3.weight, mode='fan_in', nonlinearity='sigmoid')
    
    def forward(self, x):
        x = torch.erf(self.linear1(x))
        x = torch.erf(self.linear2(x))
        x = self.linear3(x)
        return x       


# Data
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ 
teacher = FCN_3_layers(input_dim, num_channels=num_channels_teacher, init_seed=init_seed)

np.random.seed(data_seed)
X_train = torch.tensor(np.random.normal(loc=0,scale=1.,size=(n_train, 1, input_dim))).to(dtype=dtype)
np.random.seed(data_seed+1)
X_test = torch.tensor(np.random.normal(loc=0,scale=1.,size=(n_train, 1, input_dim))).to(dtype=dtype)
Y_train = teacher(X_train).detach().to(dtype=dtype)
Y_test = teacher(X_test).detach().to(dtype=dtype)
print('Teacher example created')

train_data = torch.utils.data.TensorDataset(X_train.to(dtype=dtype), Y_train.to(dtype=dtype))
test_data = torch.utils.data.TensorDataset(X_test.to(dtype=dtype), Y_test.to(dtype=dtype))

train_loader = torch.utils.data.DataLoader(train_data, batch_size=n_train, shuffle=False, num_workers=1)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=n_train, shuffle=False, num_workers=1)

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ 


best_acc = 0.0

if train_seed_type == 'random':
    torch.seed()      # make sure DNN initialization is different every time 
    np.random.seed()
elif train_seed_type == 'consistent':
    torch.manual_seed(train_seed)
    np.random.seed(train_seed)    

# build net    
torch.manual_seed(train_seed)   
net = FCN_3_layers(input_dim, num_channels)

# multiply weights by sqrt(6) to get the proper initialization - one that matches our P_0
for i in [1,2]:
    list(net.modules())[i].weight = nn.Parameter(list(net.modules())[i].weight*np.sqrt(6))
if scaling == 'rich':
    list(net.modules())[3].weight = nn.Parameter(list(net.modules())[3].weight*np.sqrt(6)/num_channels**0.5)
elif scaling == 'lazy':    
    list(net.modules())[3].weight = nn.Parameter(list(net.modules())[3].weight*np.sqrt(6))    

net.to(device)

criterion = nn.MSELoss(reduction=mse_reduction)

print("n_train = {} | lr_fac = {} | num_channels = {}".format(n_train, lr_fac, num_channels))


# LR schedule ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ 

lr_div = 1.0 # 2.0
lr_decay = np.sqrt(0.5)
lr_max = c.lr_max
if activation_type == 'relu':
    activation = torch.nn.ReLU()
    lr0 = 4.882e-07 / lr_div
    lr_epoch_list = [(100, 4.882e-06 / lr_div),
                     (5000, 3.417e-06 / lr_div),
                     (5500, 2.392e-06 / lr_div),
                     (6000, 1.675e-06 / lr_div),
                     (6500, 1.172e-06 / lr_div),
                     (7000, 8.206e-07 / lr_div),
                     (7500, 5.745e-07 / lr_div),
                     (9000, 4.021e-07 / lr_div),
                     (11500, 2.815e-07 / lr_div),
                     (16000, 1.970e-07 / lr_div),
                     (24000, 1.379e-07 / lr_div)
                    ]
elif activation_type == 'erf':
#     activation = torch.special.erf
    lr0 = lr_max / 10.0
    # lr_epoch_list = [(100, 3.906e-05 / lr_div),
    #              (3000, 2.734e-05 / lr_div),
    #              (4000, 1.914e-05 / lr_div),
    #              (5000, 1.340e-05 / lr_div),
    #              (6000, 9.379e-06 / lr_div),
    #              (8000, 6.565e-06 / lr_div),
    #              (10000, 4.5956e-06 / lr_div),
    #              (12000, 3.217e-06 / lr_div),
    #              (15000, 2.252e-06 / lr_div),
    #              (20000, 6.756e-07 / lr_div)] # 2.251875390624999e-06 * 0.3
    
#     lr_epoch_list = [(100, lr_max),
#                  (10000, lr_max * lr_decay),
#                  (20000, lr_max * lr_decay**2),
#                  (30000, lr_max * lr_decay**3),
#                  (45000, lr_max * lr_decay**4),    
#                  (60000, lr_max * lr_decay**5),    
#                  (80000, lr_max * lr_decay**6),    
#                  (100000, lr_max * lr_decay**7)    
#                     ]
    lr_epoch_list = [(100, lr_max),
                 (500000, lr_max * lr_decay),
                 (3000000, lr_max * lr_decay**2),
                 (5000000, lr_max * lr_decay**3),
                 (6000000, lr_max * lr_decay**4),
                 (7000000, lr_max * lr_decay**5),
                    ]
#     lr_epoch_list = [(100, lr_max),
#                  (20000, lr_max * lr_decay),
#                  (40000, lr_max * lr_decay**2),
#                  (60000, lr_max * lr_decay**3),
#                  (80000, lr_max * lr_decay**4),    
#                  (100000, lr_max * lr_decay**5),    
#                  (120000, lr_max * lr_decay**6),    
#                  (140000, lr_max * lr_decay**7)    
#                     ]
    
optimizer = noisy_sgd.LangevinSimple3(net, lr0, wd_input, wd_hidden, wd_output, temperature) # momentum, dampening, nesterov)    
    
# main training loop
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

losses_train, losses_test = [], []
outputs_train, outputs_test = [], []
nets = []
curLr = lr0

for i, data in enumerate(train_loader, 0): # for Langevin training we use full-batch
    inputs, labels = data  
    inputs, labels = inputs.to(device), labels.to(device)
start = time.time()
lr_ind = 0
for epoch in range(max_epochs+1):
    net.train()
    running_loss, train_correct, train_total = 0.0, 0, 0
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward() 
    optimizer.step()
    
    # determine lr according to a fixed scheduler
    if epoch == lr_epoch_list[lr_ind][0]:
        curLr = lr_epoch_list[lr_ind][1]
        if lr_ind < len(lr_epoch_list) - 1: 
            lr_ind += 1
        # if lr_ind > 0: 
        #     curLr = curLr * num_channels
        optimizer = noisy_sgd.LangevinSimple3(net, curLr, wd_input, wd_hidden, wd_output, temperature) #, momentum, dampening, nesterov)
    
    running_loss += loss.item() / np.var(Y_train.numpy()) / n_train
        
    if train_protocol == 'SGD':
        lr_scheduler.step()        
        
    if epoch % print_every == 0:
        outputs_train += [outputs]
        
        net.eval()
        test_correct, test_total, test_loss = 0, 0, 0.
        with torch.no_grad():
            for inputs_t, labels_t in test_loader:
                inputs_t, labels_t = inputs_t.to(device), labels_t.to(device)
                outputs = net(inputs_t)
                outputs_test += [outputs]
                
                loss = criterion(outputs, labels_t)
                test_loss += loss.item() / np.var(Y_test.numpy()) / n_test
                
        print('Epoch: '+ str(epoch),'| TrLoss: ' + str(running_loss), '| TeLoss: ' + str(test_loss), '| curLr:' + str(curLr))

        losses_train += [running_loss]
        losses_test += [test_loss]      
        
        if (epoch % save_nets_every == 0): #  and epoch >= start_rec_nets
            print('snapshot of model kept')
            best_model_wts = copy.deepcopy(net.state_dict())
            nets += [best_model_wts]
    
    if stop_me == True:
        print('terminating early')
        break

#     if (epoch % save_nets_every == 0) and (epoch != 0):
    if (epoch % save_nets_every == 0):    
        print('tmp save')
        # save data
        # ---------
        end = time.time()        
        run_time = calc_run_time(start, end)
        cur_results_file_name = r'{}/cur__lazy{}.npz'.format(exp_folder, exp_name)
        data_dict = {
                    'epoch': epoch,
                    'nets': nets,
                    'outputs_train': outputs_train,
                    'outputs_test': outputs_test,
                    'lr_epoch_list': lr_epoch_list,
                    'losses_train': losses_train,
                    'losses_test': losses_test,
                    'run_time': run_time,
                    'num_channels': num_channels,
                    'n_train': n_train,
                    'train_seed': train_seed,
                    'lr_fac': lr_fac,
                    'sigma2': sigma2,
                    'wd_tup': wd_tup, 
                    'activation_type': activation_type,
                    'train_protocol': train_protocol,
                    'zca': zca,        
                    'kernel_size': kernel_size,
                    'start_rec_nets': start_rec_nets,
                    'scaling': scaling,
                    }
        torch.save(data_dict, cur_results_file_name)       
        

end = time.time()        
run_time = calc_run_time(start, end)   


# save data
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
data_dict = {
            'epoch': epoch,
            'nets': nets,
            'outputs_train': outputs_train,
            'outputs_test': outputs_test,
            'lr_epoch_list': lr_epoch_list,
            'losses_train': losses_train,
            'losses_test': losses_test,
            'run_time': run_time,
            'num_channels': num_channels,
            'n_train': n_train,
            'train_seed': train_seed,
            'lr_fac': lr_fac,
            'sigma2': sigma2,
            'wd_tup': wd_tup,
            'activation_type': activation_type,   
            'train_protocol': train_protocol,
            'zca': zca,        
            'kernel_size': kernel_size,
            'start_rec_nets': start_rec_nets,
            'scaling': scaling,
            }
#torch.save(data_dict, results_file_name)       
