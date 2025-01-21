import time
import torch
from numpy import linspace as linspace
import numpy.linalg as la
import numpy as np
from scipy.io import loadmat
import LaFA.utils.file_name_gens as file_name_gens 
import pandas as pd
import pickle
import tqdm
import os
import glob
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import MNIST
# from keras.datasets import mnist

# from grad_attacks_dev import *
from LaFA.utils.grad_attacks import *

if torch.cuda.is_available():
    device=torch.device('cuda:0')
else:
    device=torch.device('cpu')


def experiment(args):
    print(args)
    print("MNIST experiment")
    log_file = file_name_gens.log_name_gen(args)
    print("Log is saved to: ", log_file)
    if os.path.isfile(log_file):
        log = pd.read_pickle(log_file)
    else:
        log = pd.DataFrame()

    print("Load data")
    tf = transforms.Compose([transforms.ToTensor()]) # mnist is already normalised 0 to 1
    dataset = MNIST("~/diffusion/Conditional_Diffusion_MNIST/data", train=False, download=False, transform=tf)
    dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=1)
    for cln_data, true_label in dataloader:
        break

    # (_, _), (X, _) = mnist.load_data()
    X = cln_data.reshape((args.batch,-1))
    X = X/torch.max(X)
    
    XorigT = Tcons(X,device).float()
    print("-------------------------")
    print("* DATA STATS *")
    print("Min: ", torch.min(XorigT).cpu().detach().numpy())
    print("Max: ", torch.max(XorigT).cpu().detach().numpy())
    print("Shape: ", XorigT.shape)
    print("Data type: ", XorigT.dtype)
    print("-------------------------")

    np.random.seed(seed=args.seed)
    Winit = np.random.rand(X.shape[0],args.rank)
    Hinit = np.random.rand(args.rank,X.shape[1])
    Winit = torch.Tensor(Winit)
    Hinit = torch.Tensor(Hinit)
    WinitT = Tcons(Winit,device).float()
    HinitT = Tcons(Hinit,device).float()

    Wstay,Hstay = NMFiter_KL(XorigT,args.base_iter,WinitT,HinitT)

    # Test stationary condition of base:
    Wnext,Hnext = NMFiter_KL(XorigT,100,Wstay,Hstay)
    X0 = Wstay@Hstay
    Xnext = Wnext@Hnext
    print("-------------------------")
    print("* SETUP STATIONARY BASE STATE *")
    print("Reconstruction error Linf: ", torch.max(torch.abs(X0-XorigT)).cpu().detach().numpy())
    print("Reconstruction error L2: ", (torch.norm(X0-XorigT)/torch.norm(XorigT)).cpu().detach().numpy())
    print("Recon base error: ", torch.max(torch.abs(X0-Xnext)).cpu().detach().numpy())
    print("W stationary error: ", torch.max(torch.abs(Wstay-Wnext)).cpu().detach().numpy())
    print("H stationary error: ", torch.max(torch.abs(Hstay-Hnext)).cpu().detach().numpy())
    print("-------------------------")
    
    print("Generate attacker")

    min_eps = args.eps_min
    no_eps = args.no_eps
    max_eps = args.eps_max
    norm = args.norm
    eps_range = max_eps - min_eps
    nmf_rank = args.rank
    nmf_iter = args.nmf_iter
    
    attacker = Gradient_based_attack(XorigT, 
                     nmf_rank = nmf_rank, 
                     base_nmf_iters = nmf_iter,  
                     use_cuda = True,
                     implicit_func = args.implicit,
                     taylor = args.taylor,
                     norm = norm, 
                     rec_loss = args.recloss,
                     no_batch = args.no_batch,
                     verbose = True)

    for step in tqdm.tqdm(range(no_eps)):
        
        eps = step/no_eps * (eps_range) + min_eps
        print("Generate attack for eps = ", eps)
        
        # Attack
        start_time = time.time()
        torch.cuda.reset_peak_memory_stats()
        Xperturb = attacker.pgd_attack(eps=eps, alpha=args.alpha, iters= args.no_iter, record = False, average_grad = args.average)
        end_mem_allocated = torch.cuda.memory_stats()['active_bytes.all.peak'] 
        duration = time.time()-start_time
        memory = end_mem_allocated
        memory_Mb_ifunc = memory/1000000
        
        # Stats
        in_X_dist = (torch.norm(Xperturb - XorigT)/torch.norm(XorigT)).cpu().detach().numpy()
        Wp,Hp = NMFiter_KL(Xperturb,10000,WinitT,HinitT)
        out_X_dist = (torch.norm(Wp@Hp - XorigT)/torch.norm(XorigT)).cpu().detach().numpy()
        out_FE_dist = loss_wh(Wstay,Wp,Hstay,Hp).cpu().detach().numpy()
        out_W_dist = loss_w(Wstay,Wp).cpu().detach().numpy()
        Hstay = Hstay + 1e-10
        Hp = Hp + 1e-10
        out_H_dist = loss_w(Hstay,Hp).cpu().detach().numpy()
        # Log
        print("Input distortion: ", in_X_dist)
        print("FE distortion: ", out_FE_dist)
        print("W distortion: ", out_W_dist)
        print("H distortion: ", out_H_dist)
        print("Duration (sec): ", duration)
        print("Memory (Mb): ", memory_Mb_ifunc)
        print("----")
        
        result = {
                    'eps': eps,
                    'alpha': args.alpha,
                    'pgd iters': args.no_iter,
                    'in': in_X_dist,
                    'out rec': out_X_dist,
                    'out fe': out_FE_dist,
                    'out W': out_W_dist,
                    'out H': out_H_dist,
                    'duration': duration,
                    'memory': memory,
                    'Implicit gradient': args.implicit,
                    'norm': args.norm,
                    'NMF iters': args.nmf_iter,
                    'Seed': args.seed
                }
        log = pd.concat([log, pd.DataFrame.from_records([result])])
        with open(log_file, 'wb') as logfile:
            pickle.dump(log, logfile)
    
        Hp_np = Hp.cpu().detach().numpy()
        FEATURE_FILES = file_name_gens.feature_name_gen(args, eps = eps)       
        with open(FEATURE_FILES, 'wb') as f:
            np.save(f, Hp_np)

    return



