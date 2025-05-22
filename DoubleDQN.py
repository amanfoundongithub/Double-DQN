import torch 
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F 

import gymnasium as gym 
import numpy as np 
import matplotlib.pyplot as plt 

from ReplayBuffer import ReplayBuffer
from Network      import Network


# Jupyter Notebook 
# from IPython.display import clear_output


class DQNAgent:
    def __init__(
        self, env,
        memory_size, batch_size, target_update,
        epsilon_decay, max_epsilon = 1.0, min_epsilon = 0.1, gamma = 0.99,
    ):
        
        # Environment and its parameters 
        self.env   = env
        obs_dim    = env.observation_space.shape[0]
        action_dim = env.action_space.n
        
        
        # Initialize memory and params 
        self.memory = ReplayBuffer(obs_dim, memory_size, batch_size)
        self.batch_size = batch_size
        
        # Epsilon decay -> linear 
        self.epsilon = max_epsilon
        self.epsilon_decay = epsilon_decay
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        
        
        # Update parameters 
        self.target_update = target_update
        self.gamma = gamma
        
        # device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # networks: primal and target 
        self.dqn = Network(obs_dim, action_dim).to(self.device)
        self.dqn_target = Network(obs_dim, action_dim).to(self.device)
        
        # Load target and primal to be same 
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()
        
        # optimizer for training 
        self.optimizer = optim.Adam(self.dqn.parameters())

        # transition to store in memory
        self.transition = list()
        
        # train/test 
        self.is_test = False

    def select_action(self, state):
        if np.random.random() < self.epsilon:
            selected_action = self.env.action_space.sample()
        else:
            state = torch.FloatTensor(state).to(self.device)
            selected_action = self.dqn(state).argmax()
            selected_action = selected_action.detach().cpu().numpy()
        
        if not self.is_test:
            self.transition = [state, selected_action]
        
        return selected_action

    def step(self, action):
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated

        if not self.is_test:
            self.transition += [reward, next_state, done]
            self.memory.store(*self.transition)
    
        return next_state, reward, done

    def __update_model(self):
        samples = self.memory.sample_batch()
        
        # Get the SARS from the data
        state = torch.FloatTensor(samples["obs"]).to(self.device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(self.device)
        action = torch.LongTensor(samples["acts"].reshape(-1, 1)).to(self.device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(self.device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(self.device)
        
        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        curr_q_value = self.dqn(state).gather(1, action)
        #       = r                       otherwise
        next_q_value = self.dqn_target(next_state).gather(1, 
                                                          self.dqn(next_state).argmax(dim=1, keepdim=True)
                                                          ).detach()
        
        target = reward if done == True else (reward + self.gamma * next_q_value)
        target = target.to(self.device)

        # L1 Huber Loss 
        loss = F.smooth_l1_loss(curr_q_value, target)

        # Training Now 
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Return loss 
        return loss.item()
        
    def train(self, 
              num_frames, 
              plotting_interval = 200):
        
        # Training mode 
        self.is_test = False
        
        state, _ = self.env.reset()
        update_cnt = 0
        epsilons = []
        losses = []
        scores = []
        score = 0

        for frame_idx in range(1, num_frames + 1):
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward

            # if episode ends
            if done:
                state, _ = self.env.reset()
                scores.append(score)
                score = 0
                
            
            # Buffer is full 
            if len(self.memory) >= self.batch_size:
                loss = self.__update_model()
                losses.append(loss)
                update_cnt += 1
                
                # linearly decrease epsilon
                self.epsilon = max(self.min_epsilon, 
                                   self.epsilon - (self.max_epsilon - self.min_epsilon) * self.epsilon_decay)
                epsilons.append(self.epsilon)
                
                # if hard update is needed
                if update_cnt % self.target_update == 0:
                    self.dqn_target.load_state_dict(self.dqn.state_dict())

            # Olots after 200 steps 
            if frame_idx % plotting_interval == 0:
                self._plot(frame_idx, scores, losses, epsilons)
                
        self.env.close()
                
    def test(self, 
             video_folder):
        
        # Test Mode 
        self.is_test = True
        
        # for recording a video
        env = self.env
        self.env = gym.wrappers.RecordVideo(self.env, video_folder = video_folder)
        
        state, _ = self.env.reset()
        done = False
        score = 0
        
        while not done:
            action = self.select_action(state)
            next_state, reward, done = self.step(action)
            state = next_state
            score += reward
        
        print("Final Score: ", score)
        self.env.close()
        
        # reset
        self.env = env 
                
    def _plot(
        self, 
        frame_idx, 
        scores, 
        losses, 
        epsilons,
    ):
        """Plot the training progresses."""
        # clear_output(True) => Jupyter notebook uncomment this line 
        plt.figure(figsize=(20, 5))
        plt.subplot(131)
        plt.title('frame %s. score: %s' % (frame_idx, np.mean(scores[-10:])))
        plt.plot(scores)
        plt.subplot(132)
        plt.title('loss')
        plt.plot(losses)
        plt.subplot(133)
        plt.title('epsilons')
        plt.plot(epsilons)
        plt.show()