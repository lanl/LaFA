import sys
from pathlib import Path

# Importing the configurations and experiments modules
# 'configs' provides utilities to parse program arguments
# 'experiments' contains implementations for various dataset-specific experiments
import configs
from LaFA.experiments import experiments

# Parse command-line arguments using the 'arg_parse' function from configs
prog_args = configs.arg_parse()

# Check if the dataset argument is provided and process accordingly
if prog_args.dataset is not None:
    # If the dataset is "MNIST", run the MNIST-specific NMF experiment
    if prog_args.dataset == "MNIST":
        print("NMF on MNIST dataset")
        # Dynamically select and execute the corresponding experiment
        task = "experiments.mnist_exp"
        eval(task)(prog_args)
    
    # If the dataset is "Face", run the Face dataset NMF experiment
    elif prog_args.dataset == "Face":
        print("NMF on Face dataset")
        task = "experiments.face_exp"
        eval(task)(prog_args)
    
    # If the dataset is "FaceDev", run the development version of the Face dataset experiment
    elif prog_args.dataset == "FaceDev":
        print("NMF on Face-dev dataset")
        task = "experiments.face_exp_dev"
        eval(task)(prog_args)
        
    # If the dataset is "WTSI", run the WTSI dataset experiment
    elif prog_args.dataset == "WTSI":
        print("NMF on WTSI dataset")
        task = "experiments.wtsi_exp"
        eval(task)(prog_args)
    
    # If the dataset is "swim", run the Swimmer dataset experiment
    elif prog_args.dataset == "swim":
        print("NMF on Swimmer dataset")
        task = "experiments.swim_exp"
        eval(task)(prog_args)
    
    # If no specific dataset is matched, default to the synthetic (Gaussian) dataset experiment
    else:
        print("NMF on synthetic (Gaussian) dataset")
        task = "experiments.syn_exp"
        eval(task)(prog_args)
