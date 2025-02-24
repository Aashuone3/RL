#
# constants go here
#

# model constants
MIN_HIDDEN_LAYERS = 2
MAX_HIDDEN_LAYERS = 6

MIN_NODES = 2
MAX_NODES = 12

# for synchronous updates
SYNCH_STEPS = 10

DIR_PATH = './datasets/'

TEST_SPLIT_FRACTION = 0.3
RANDOM_SAMPLE_LIMIT = 5
SAMPLE_SIZE = 100
EPOCHS = 60

# Single agent constants
BATCH_SIZE = 32
TRAIN_BATCH_SIZE = 64
SMS = 4

# Agent constants
ALPHA = 0.01
BETA = 0.01
GAMMA = 0.99
TAU = 0.99
BUF_LEN = 100000 
H1_DIMS = 40
H2_DIMS = 30
N_ACTIONS = 1  # For a single action
MAX_EPISODES = 40
MIN_STEPS = 30
MAX_STEPS = 40
MAX_ACTION = [4]  # Single action range for a single agent
ACTION_SPACE = [4]  # Single action space for a single agent
