import torch as T
import numpy as np
from agent import DDPGAgent
import constants as C

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
        
        # Initialize a single agent
        self.agent = DDPGAgent(alpha=self.alpha, beta=self.beta, tau=self.tau, 
                               input_dims=input_dims, n_actions=n_actions, 
                               hd1_dims=hd1_dims, hd2_dims=hd2_dims, 
                               mem_size=mem_size, gamma=self.gamma,
                               batch_size=self.batch_size, agent_no=0)

        self.agent_state = []
    
    def run(self, max_episode, max_steps):
        returns = []

        for i in range(max_episode):
            print("Episode : {}".format(i))
            total_return = 0
            self.agent_state = self.env.reset()
            steps = 0

            # Reset action noise for the agent
            self.agent.actionNoise.reset()
            while steps < max_steps:
                steps += 1
                action, rounded_action = self.agent.choose_action(self.agent_state, 0)
                print("Step: ", steps)
                print("Action: {}, Rounded Action : {}".format(action, rounded_action))

                next_state, reward = self.env.step(rounded_action, 0)
                done = False
                if reward == 0:
                    done = True

                # Store transition
                self.agent.store_transition(self.agent_state, action, reward, next_state)
                self.agent.learn([self.agent_state], [action], [reward], [next_state])

                total_return += reward
                self.agent_state = next_state
                
                # Debug info
                print("Next state : ", next_state)
                print("Reward: ", reward)
                print("Total Return: ", total_return)
                
                if done:
                    break

                print("\n-----------------------------------------------------------------\n")

            returns.append(total_return)
            print("Score: {}".format(total_return))

# Usage example:
# env = your_environment_instance_here
# ddpg_agent = DDPG(env, alpha, beta, tau, input_dims, n_actions)
# ddpg_agent.run(max_episode, max_steps)
