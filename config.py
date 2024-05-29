from yacs.config import CfgNode as CN
# yacs official github page https://github.com/rbgirshick/yacs

_C = CN()
''' System '''
_C.SYSTEM = CN()
# Number of GPUS to use in the experiment
_C.SYSTEM.NUM_GPUS = 0
# Number of workers for doing things
_C.SYSTEM.NUM_WORKERS = 0
# Random seed
_C.SYSTEM.RANDOM_SEED = 0

_C.HAMILTONIAN = CN()
_C.HAMILTONIAN.TYPE = 'exact'

''' Miscellaneous '''
_C.MISC = CN()
# Functionality mode
# Available choices: ['train', 'eval', 'train_maml']
_C.MISC.MODE = ''
# Logging interval
_C.MISC.LOG_INTERVAL = 0
# Logger path
_C.MISC.DIR = ''
# Number of trials
_C.MISC.NUM_TRIALS = 0
# Compute baselines and ground truth
_C.MISC.COMPUTE_GT = False
# DDP seeds
_C.MISC.SAME_LOCAL_SEED = False

''' Debugger '''
_C.DEBUG = CN()
# Debugging job name
_C.DEBUG.CASE = ''
# Ablation study index
_C.DEBUG.ABLATION_IDX = 0

''' Training hyper-parameters '''
_C.TRAIN = CN()
# Learning rate
_C.TRAIN.LEARNING_RATE = 0.0
# Training optimizer name
# Available choices: ['adam', 'sgd', 'adadelta', 'adamax', 'adagrad']
_C.TRAIN.OPTIMIZER_NAME = ''
# Learning rate scheduler name
# Available choices: ['decay', 'cyclic', 'trap', 'const']
_C.TRAIN.SCHEDULER_NAME = ''
# Training batch size
_C.TRAIN.BATCH_SIZE = 0
# Enable unique samples
_C.TRAIN.ENABLE_UNIQS = False
# Number of training epochs
_C.TRAIN.NUM_EPOCHS = 0
# Whether using stochastic reconfiguration or not
_C.TRAIN.APPLY_SR = False
# Whether to use information regularization
_C.TRAIN.ANNEALING_COEFFICIENT = 0.0
# Inner iteration to split the batch into small sizes
_C.TRAIN.INNER_ITER = 1

''' Model hyper-parameters '''
_C.MODEL = CN()
# The name of the model
# Available choices: ['rbm', 'rbm_c', ...]
_C.MODEL.MODEL_NAME = ''
# The number of hidden layer in MADE
_C.MODEL.HIDDEN_DEPTH = 0
# Hidden layer size in MADE
_C.MODEL.HIDDEN_WIDTH = 0
# Number of MADE in MAF
_C.MODEL.NUM_BLOCKS = 0
# the total number of occupied spin-up and spin-down
_C.MODEL.NUM_SPIN_UP = -1
_C.MODEL.NUM_SPIN_DOWN = -1
# Parameter std initialization
_C.MODEL.INIT_STD = 0.1

''' Data hyper-parameters '''
_C.DATA = CN()
# Number of sites
_C.DATA.LOAD_PATH = ''
# Number of sites
_C.DATA.NUM_SITES = 0
# Whether to use previously samples as initialization for the current MCMC sampling
_C.DATA.INIT_PREV_STATE = False
# Total imagary batch size for autoregressive sampling
_C.DATA.NUM_SAMPLES = 1e12
# Molecular name
_C.DATA.MOLECULE = ''
# Basis
_C.DATA.BASIS = ''

''' Evaluation hyper-parameters '''
_C.EVAL = CN()
# Loading path of the saved model
_C.EVAL.MODEL_LOAD_PATH = ''
# Name of the results logger
_C.EVAL.RESULT_LOGGER_NAME = './results/results.txt'
# Name of the dictionary that stores the results
_C.EVAL.RESULT_DIC_NAME = ''
# Evaluation interval
_C.EVAL.EVALUATION_INTERVAL = 10000

''' ISGO Trick '''
_C.ISGO = CN()
# Update interval for sampling
_C.ISGO.UPDATE_INTERVAL = 1

''' DistributedDataParallel '''
_C.DDP = CN()
# Node number globally
_C.DDP.NODE_IDX = 0
# This is passed in via launch.py
_C.DDP.LOCAL_RANK = 0
# This needs to be explicitly passed in
_C.DDP.LOCAL_WORLD_SIZE = 1
# Total number of GPUs
_C.DDP.WORLD_SIZE = 1
# Master address for communication
_C.DDP.MASTER_ADDR = ''
# Master port for communication
_C.DDP.MASTER_PORT = 0


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for the project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()

# Alternatively, provide a way to import the defaults as
# a global singleton:
# cfg = _C  # users can `from config import cfg`
