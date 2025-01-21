import time
import torch
from numpy import linspace as linspace
import numpy.linalg as la
import numpy as np
# import importlib
# import itertools
from  LaFA.utils import file_name_gens
import pandas as pd
import pickle
import tqdm
import os

# from matplotlib import pyplot as plt
from LaFA.utils.grad_attacks import *

if torch.cuda.is_available():
    device=torch.device('cuda:0')
else:
    device=torch.device('cpu')

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def experiment(args):
    # print(args)
    
    print("Synthetic experiment")
    log_file = file_name_gens.log_name_gen(args)
    print("Log is saved to: ", log_file)
    if os.path.isfile(log_file):
        log = pd.read_pickle(log_file)
    else:
        log = pd.DataFrame() 

    print("Generate synthetic data")
    # np.random.seed(seed=702)
    np.random.seed(seed=args.seed)
    
    H = np.random.rand(3,200)
    Winit = np.random.rand(103,3)
    Hinit = np.random.rand(3,200)
    Winit = torch.Tensor(Winit)
    Hinit = torch.Tensor(Hinit)
    
    x=np.arange(0,103)
    Worig=np.zeros((103,3))
    w1 = gaussian(x, 30, 10)
    w1=w1/np.max(w1) #la.norm(w1)
    w2 = gaussian(x,70,10)
    w2=w2/np.max(w2)  #la.norm(w2)
    w3 = w1+w2*.10
    w3=w3/np.max(w1)  #la.norm(w3)
    
    Worig[:,0] = w1
    Worig[:,1] = w2
    Worig[:,2] = w3
    Worig[100,0]=1.0
    Worig[101,1]=1.0
    Worig[102,2]=1.0
    
    Xorig = Worig@H

    XorigT = Tcons(Xorig,device).float()
    WinitT = Tcons(Winit,device).float()
    HinitT = Tcons(Hinit,device).float()
    
    Wstay,Hstay = NMFiter_KL(XorigT,10000,WinitT,HinitT)

    print("Generate attacker")
    
    min_eps = args.eps_min
    no_eps = args.no_eps
    max_eps = args.eps_max
    norm = args.norm
    eps_range = max_eps - min_eps
    nmf_rank = args.rank #3
    nmf_iter = args.nmf_iter
    
    attacker = Gradient_based_attack(XorigT, 
                     nmf_rank = nmf_rank, 
                     base_nmf_iters = nmf_iter,  
                     use_cuda = True,
                     implicit_func = args.implicit,
                     taylor = args.taylor,
                     norm = norm, 
                     rec_loss = args.recloss,
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
        Hstay = Hstay + 1e-10
        Hp = Hp + 1e-10
        Wstay = Wstay + 1e-10
        Wp = Wp + 1e-10
        out_W_dist = loss_w(Wstay,Wp).cpu().detach().numpy()
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
        
        Wp_np = Wp.cpu().detach().numpy()
        FEATURE_FILES = file_name_gens.feature_name_gen(args, eps = eps)       
        with open(FEATURE_FILES, 'wb') as f:
            np.save(f, Wp_np)
    
    return

