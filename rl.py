# Import necessary packages
import pandas as pd
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.envs import DummyVecEnv
import gym
from gym import spaces

# Custom Environment for Batch Reinforcement Learning
class BatchMDPEnv(gym.Env):
    def __init__(self, data_path):
        super(BatchMDPEnv, self).__init__()
        print("init called")
        # Load the CSV data
        self.data = pd.read_csv(data_path)
        self.states = sorted(self.data['state'].unique())
        self.actions = sorted(self.data['action'].unique())
        
        # Define action and observation space
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Discrete(len(self.states))
        
        # Create a mapping for states and actions
        self.state_map = {s: i for i, s in enumerate(self.states)}
        self.action_map = {a: i for i, a in enumerate(self.actions)}
        
        # Initialize the current state
        self.current_state = None

    def reset(self):
        # Reset environment to an initial state
        self.current_state = self.data['state'].iloc[0]
        return self.state_map[self.current_state]

    def step(self, action):
        action_value = self.actions[action]
        possible_transitions = self.data[(self.data['state'] == self.current_state) & 
                                         (self.data['action'] == action_value)]
        
        if not possible_transitions.empty:
            next_row = possible_transitions.sample().iloc[0]
            reward = next_row['reward']
            next_state = next_row['next_state']
            done = False  # For simplicity, we assume episodes continue indefinitely in this batch learning
            
            self.current_state = next_state
            return self.state_map[next_state], reward, done, {}
        else:
            # If no transition is found, terminate
            return self.state_map[self.current_state], 0, True, {}

# Load and wrap the custom environment
data_path = './project2/data/small.csv'
env = DummyVecEnv([lambda: BatchMDPEnv(data_path)])


# Initialize the DQN model
model = DQN('MlpPolicy', env, verbose=1)

# Train the model
model.learn(total_timesteps=10000)

# Generate policy output
def export_policy(model, env, output_file):
    with open(output_file, 'w') as f:
        for state_index in range(env.observation_space.n):
            state = env.envs[0].states[state_index]
            action, _ = model.predict(state_index)
            f.write(f"{action + 1}\n")  # Convert to 1-based index for output

# Export the policy
export_policy(model, env, 'output.policy')

print("Policy file generated!")
