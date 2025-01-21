'''from . import exp_synthetic
from . import exp_wtsi
from . import exp_face
from . import exp_mnist
from . import exp_swim'''

import LaFA.experiments.exp_synthetic as exp_synthetic
import LaFA.experiments.exp_wtsi as exp_wtsi
import LaFA.experiments.exp_face as exp_face
import LaFA.experiments.exp_mnist as exp_mnist
import LaFA.experiments.exp_swim as exp_swim
#import exp_syn_onestep

def syn_exp(args):
    exp_synthetic.experiment(args)

def mnist_exp(args):
    exp_mnist.experiment(args)

def face_exp(args):
    exp_face.experiment(args)

def swim_exp(args):
    exp_swim.experiment(args)

def wtsi_exp(args):
    exp_wtsi.experiment(args)

    

    
    

    