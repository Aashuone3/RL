from singleAgent import DDPG  # Assuming you have a DDPG implementation for a single agent
from environment import Environment
import constants as C
import torch.multiprocessing 

# I/O for the number of layers goes here...
# layers = input("Enter hidden layers: ")
# layers = int(layers)

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    torch.multiprocessing.freeze_support()
    layers = 3  # Set the number of hidden layers
    env = Environment(layers)  # Adjusted to single agent initialization

    # Initialize the controller for a single agent
    controller = DDPG(env, alpha=C.ALPHA, beta=C.BETA, tau=C.TAU, 
                      input_dims=[env.input_dims[0]], n_actions=C.N_ACTIONS, 
                      hd1_dims=C.H1_DIMS, hd2_dims=C.H2_DIMS, 
                      mem_size=C.BUF_LEN, gamma=C.GAMMA, 
                      batch_size=C.BATCH_SIZE)

    controller.run_episodes(max_episodes=C.MAX_EPISODES, max_steps=C.MAX_STEPS)  # Adjusted method for a single agent
