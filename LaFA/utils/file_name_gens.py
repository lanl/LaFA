def log_name_gen(prog_args):
    """
    Generate a log file name based on program arguments.

    Args:
        prog_args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        str: The generated log file name.
    """
    # Base folder where logs will be stored
    OUTPUT_PATH = 'log'
    
    # Dataset name from arguments
    DATASET = prog_args.dataset
    
    # Determine the attack method (Iterative or One-Step)
    METHOD = 'PGD' if prog_args.iterative else 'OneStep'
    
    # Determine the gradient computation method (Implicit or Backpropagation)
    GRAD = 'Implicit' if prog_args.implicit else 'BackProp'
    
    # Seed for reproducibility
    SEED = prog_args.seed
    
    # Format the log file name
    log_name = f'{OUTPUT_PATH}/{DATASET}_{METHOD}_{prog_args.norm}_{GRAD}_{SEED}.pkl'

    return log_name


def feature_name_gen(prog_args, eps=0):
    """
    Generate a feature file name based on program arguments and epsilon value.

    Args:
        prog_args (argparse.Namespace): Parsed command-line arguments.
        eps (float): Epsilon value used in the attack (default is 0).

    Returns:
        str: The generated feature file name.
    """
    # Base folder where feature files will be stored
    OUTPUT_PATH = 'log'
    
    # Dataset name from arguments
    DATASET = prog_args.dataset
    
    # Determine the attack method (Iterative or One-Step)
    METHOD = 'PGD' if prog_args.iterative else 'OneStep'
    
    # Determine the gradient computation method (Implicit or Backpropagation)
    GRAD = 'Implicit' if prog_args.implicit else 'BackProp'
    
    # Format the epsilon as a string for inclusion in the file name
    eps_str = str(eps)
    
    # Format the feature file name
    feature_name = f'{OUTPUT_PATH}/{DATASET}_{METHOD}_{prog_args.norm}_{GRAD}_{eps_str}.npy'

    return feature_name


def init_base_name(prog_args):
    """
    Generate a base file name for initialization based on program arguments.

    Args:
        prog_args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        str: The generated base file name.
    """
    # Base folder for initialization files
    OUTPUT_PATH = 'log'
    
    # Dataset name from arguments
    DATASET = prog_args.dataset
    
    # Generate the base file name
    base_name = f'{OUTPUT_PATH}/{DATASET}_base.npy'

    return base_name


def init_ref_name(prog_args):
    """
    Generate a reference file name for initialization based on program arguments.

    Args:
        prog_args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        str: The generated reference file name.
    """
    # Base folder for reference files
    OUTPUT_PATH = 'log'
    
    # Dataset name from arguments
    DATASET = prog_args.dataset
    
    # Generate the reference file name
    ref_name = f'{OUTPUT_PATH}/{DATASET}_ref.npy'

    return ref_name
