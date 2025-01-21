import argparse

def arg_parse():
    """
    Parse command-line arguments for configuring NMF experiments.

    Returns:
        argparse.Namespace: Parsed arguments with their respective values.
    """
    # Create an ArgumentParser object with a description of the script
    parser = argparse.ArgumentParser(description="Experiments for NMF implicit function.")

    # Add argument for specifying the dataset name
    parser.add_argument(
            "--dataset", dest="dataset", help="Name of the dataset"
        )
    
    # Add argument for specifying the rank for NMF
    parser.add_argument("--rank", dest="rank", type=int, help="NMF rank.")
    
    # Add argument for setting the number of base iterations for NMF
    parser.add_argument("--base_iter", dest="base_iter", type=int, help="NMF iterations for NMF to reach stationary state for experiment evaluations.")
    
    # Add argument for specifying the number of iterations per gradient computation
    parser.add_argument("--nmf_iter", dest="nmf_iter", type=int, help="Number of iterations for NMF to reach stationary state per grad computation.")

    # Add flag for iterative attack configuration
    parser.add_argument(
            "--iterative",
            dest="iterative",
            action="store_const",
            const=True,
            default=False,
            help="False for one-step, or True for iterative attack",
        )
    
    # Add flag for using implicit function in computations
    parser.add_argument(
            "--implicit",
            dest="implicit",
            action="store_const",
            const=True,
            default=False,
            help="False for back-propagate, or True for implicit",
        )

    # Add flag for using reconstruction loss
    parser.add_argument(
            "--recloss",
            dest="recloss",
            action="store_const",
            const=True,
            default=False,
            help="True for reconstruction loss",
        )
    
    # Add argument for specifying the number of gradient computations per step
    parser.add_argument("--average", dest="average", type=int, help="Number of gradient computations per gradient step.")
    
    # Add argument for setting the number of iterations for iterative attacks (PGD)
    parser.add_argument("--no_iter", dest="no_iter", type=int, help="Number of iterations for iterative attack (PGD).")
    
    # Add argument for specifying Taylor's order for implicit gradient computation
    parser.add_argument("--taylor", dest="taylor", type=int, help="Taylor's order for implicit gradient computation.")
    
    # Add argument for specifying the attack norm (e.g., L2 or Linf)
    parser.add_argument("--norm", dest="norm", type=str, help="Attack norm: L2 or Linf.")
    
    # Add arguments for defining the range of epsilon for attacks
    parser.add_argument("--eps_min", dest="eps_min", type=float, help="Min epsilon for attacking.")
    parser.add_argument("--eps_max", dest="eps_max", type=float, help="Max epsilon for attacking.")
    # Uncomment the following if epsilon step intervals are needed
    # parser.add_argument("--eps_step", dest="eps_step", type=float, help="Stepping of epsilon for experiment.")
    
    # Add argument for specifying the number of epsilon values for testing
    parser.add_argument("--no_eps", dest="no_eps", type=int, help="Number of epsilon values for testing.")
    
    # Add argument for setting the alpha step for PGD
    parser.add_argument("--alpha", dest="alpha", type=float, help="Alpha step for PGD.")
    
    # Add argument for specifying the random seed for initializing W and H matrices
    parser.add_argument("--seed", dest="seed", type=int, help="Seed for W and H initialization.")
    
    # Add argument for defining the number of samples used for MNIST
    parser.add_argument("--batch", dest="batch", type=int, help="Number of samples used for MNIST.")
    
    # Add argument for setting the number of batches in MNIST
    parser.add_argument("--no_batch", dest="no_batch", type=int, help="Number of batches in MNIST.")
    
    # Set default values for all arguments
    parser.set_defaults(
        dataset="Synthetic",    # Default dataset is Synthetic
        rank=3,                 # Default rank for NMF
        base_iter=2000,         # Default base iterations for NMF
        nmf_iter=2000,          # Default iterations for grad computation
        implicit=False,         # Default to back-propagation
        iterative=False,        # Default to one-step attacks
        recloss=False,          # Default to no reconstruction loss
        average=1,              # Default 1 gradient computation per step
        no_iter=40,             # Default number of PGD iterations
        taylor=200,             # Default Taylor order
        norm="Linf",            # Default attack norm
        eps_min=0.0,            # Default minimum epsilon
        eps_max=0.03,           # Default maximum epsilon
        alpha=0.01,             # Default alpha step for PGD
        no_eps=10,              # Default number of epsilon values
        seed=1,                 # Default random seed
        batch=100,              # Default batch size for MNIST
        no_batch=10             # Default number of batches for MNIST
    )
    
    # Parse and return the command-line arguments
    return parser.parse_args()
