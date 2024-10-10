import torch as T
import numpy as np
from agent import DDPGAgent
import constants as C
import helper as H
from torch.multiprocessing import Process, Lock, Manager

class DDPG:
    def __init__(self, env, alpha, beta, tau, input_dims, n_actions,
                 hd1_dims=400, hd2_dims=300, mem_size=1000000,
                 gamma=0.99, batch_size=64):
        self.env = env
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        
        # Initialize a single DDPG agent
        self.agent = DDPGAgent(alpha=self.alpha, beta=self.beta, tau=self.tau, 
                               input_dims=input_dims, n_actions=n_actions, 
                               hd1_dims=hd1_dims, hd2_dims=hd2_dims, 
                               mem_size=mem_size, gamma=self.gamma,
                               batch_size=self.batch_size)

    def run(self, max_episodes, max_steps, l, l1, global_state):
        return_list = []
        for episode in range(max_episodes):
            print("Episode: {}".format(episode + 1))
            returns = 0
            state = self.env.reset()  # Reset the environment for a new episode
            agent_states = state
            self.agent.actionNoise.reset()

            for step in range(max_steps):
                action, rounded_action = self.agent.choose_action(agent_states)
                next_state, reward = self.env.step(rounded_action)
                done = reward >= 10  # Define termination condition
                
                self.agent.store_transition(agent_states, action, reward, next_state)
                self.agent.learn([agent_states], [action], [reward], [next_state])
                returns += reward
                agent_states = next_state
                
                # Debugging info
                l.acquire()
                try:
                    print("Episode: {}".format(episode + 1))
                    print("Step: {}".format(step + 1))
                    print("Action: {}".format(action))
                    print("Next State: {}".format(next_state))
                    print("Reward: {}".format(reward))
                    print("Returns: {}".format(returns))
                    print("\n-----------------------------------------------------------------\n")
                finally:
                    l.release()

            return_list.append(returns)
            means = np.mean(return_list[-20:]) if len(return_list) >= 20 else returns
            print("Score Model: {}".format(means))

    def run_parallel_episodes(self, max_episodes, max_steps):
        m = Manager()
        printlock = m.Lock()
        all_processes = []
        
        for episode in range(max_episodes):
            print("Episode: {}".format(episode + 1))
            global_state = m.list()  # Shared state across processes
            
            # Start the single agent process
            p = Process(target=self.sample_run, args=(max_steps, printlock, global_state, episode))
            all_processes.append(p)
            p.start()
        
        for p in all_processes:
            p.join()

    def sample_run(self, max_steps, l, global_state, episode):
        returns = 0
        steps = 0
        agent_states = self.env.sample_reset()  # Reset the environment for sampling
        
        while steps < max_steps:
            steps += 1
            action, rounded_action = self.agent.choose_action(agent_states)
            next_state, reward = self.env.step(rounded_action)
            self.agent.store_transition(agent_states, action, reward, next_state)
            self.agent.learn([agent_states], [action], [reward], [next_state])
            returns += reward
            agent_states = next_state
            
            # Debug info
            l.acquire()
            try:
                H.print_debug(steps, action, next_state, reward, returns)
            finally:
                l.release()
        
        print("Final Score for Episode {}: {}".format(episode, returns))

# Usage Example:
# env = YourEnvironment()  # Initialize your environment
# agent = DDPG(env, alpha=0.001, beta=0.001, tau=0.001, input_dims=(state_size,), n_actions=action_size)
# agent.run(max_episodes=100, max_steps=1000, l=Lock(), l1=Lock(), global_state=[])
