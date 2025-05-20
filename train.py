from DoubleDQN import DQNAgent
import gymnasium as gym 

# environment
env = gym.make("LunarLander-v2", max_episode_steps=200, render_mode="rgb_array")

# parameters
num_frames = 100000
memory_size = 1000
batch_size = 32
target_update = 200
epsilon_decay = 1 / 50000

# Initialize agent 
agent = DQNAgent(env, memory_size, batch_size, target_update, epsilon_decay)

# Training code 
agent.train(num_frames)

# Save movie in a folder
video_folder="videos/lunar_lander"
agent.test(video_folder=video_folder)