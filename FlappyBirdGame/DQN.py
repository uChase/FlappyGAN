import torch
import torch.nn as nn
import torch.optim as optim
import random

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class FlappyBirdAgent:
    def __init__(self, input_size, action_space, memory_limit, EPS_START=1.0, EPS_END=0.0001, EPS_DECAY_VALUE=0.99):
        self.model = DQN(input_size, action_space).cuda()  # Move the model to CUDA
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        self.memory = []  # Memory for experience replay
        self.memory_limit = memory_limit  # Limit on memory size
        self.gamma = 0.999  # Discount factor for future rewards
        self.epsilon = EPS_START  
        self.EPS_END = EPS_END
        self.EPS_DECAY_VALUE = EPS_DECAY_VALUE

    def remember(self, state, action, next_state, reward, done):
        if len(self.memory) >= self.memory_limit:
            self.memory.pop(0)  # Remove the oldest memory if memory limit is reached
        if done:
            done = 1
        self.memory.append((state, action, next_state, reward, done))

    def select_action(self, state):
        if random.random() < self.epsilon:
            if self.epsilon > self.EPS_END:
                self.epsilon *= self.EPS_DECAY_VALUE
            print(self.epsilon)
            return random.choice([0, 1])  # Random action (0 or 1)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).cuda()  # Move the tensor to CUDA
            with torch.no_grad():
                q_values = self.model(state_tensor)
            return torch.argmax(q_values).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            print("Not enough memory to replay")
            return

        batch = random.sample(self.memory, batch_size)
        batch_state, batch_action, batch_next_state, batch_reward, batch_done = zip(*batch)

        # Convert to tensors and move to CUDA
        batch_state = torch.FloatTensor(batch_state).cuda()
        batch_next_state = torch.FloatTensor(batch_next_state).cuda()
        batch_reward = torch.FloatTensor(batch_reward).cuda()
        batch_action = torch.LongTensor(batch_action).cuda()
        batch_done = torch.FloatTensor(batch_done).cuda()

        # Q-Learning
        current_q = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_q = self.model(batch_next_state).max(1)[0]
        expected_q = batch_reward + self.gamma * next_q * (1 - batch_done)
        
        loss = self.loss_fn(current_q, expected_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_model(self, filepath):
        torch.save(self.model.state_dict(), filepath)

    def load_model(self, filepath):
        self.model.load_state_dict(torch.load(filepath))
        self.model.eval()

def prepare_state(bird_y, bird_y_change, bird_x, pipes, screen_width, screen_height):
    # Initialize distances with default values (e.g., max distances)
    dist_to_next_pipe_left = screen_width
    dist_to_next_pipe_right = screen_width
    dist_to_bottom_pipe_top = 100
    dist_to_top_pipe_bottom = 100

    if pipes:
        next_bottom_pipe, next_top_pipe, _ = pipes[0]
        dist_to_next_pipe_left = max(0, next_bottom_pipe.x - bird_x - 30 )
        dist_to_next_pipe_right = max(0, (next_bottom_pipe.x + next_bottom_pipe.width) - bird_x - 30)
        dist_to_bottom_pipe_top = next_bottom_pipe.y - bird_y - 30
        dist_to_top_pipe_bottom = bird_y - next_top_pipe.bottom 
    # Normalize distances (optional, based on your scale)
    dist_to_next_pipe_left /= screen_width
    dist_to_next_pipe_right /= screen_width
    dist_to_bottom_pipe_top /= screen_height
    dist_to_top_pipe_bottom /= screen_height
    scale_bird_y = bird_y / screen_height
    scale_bird_y_change = bird_y_change / screen_height

    # State vector
    state = [
        scale_bird_y,
        scale_bird_y_change,
        dist_to_next_pipe_left,
        dist_to_next_pipe_right,
        dist_to_bottom_pipe_top,
        dist_to_top_pipe_bottom,
    ]
    return state

def calculate_reward(bird_y, bird_y_change, pipes, score, game_over, game_overBig, frames_survived):
    reward = 0

    # Constant Reward for Each Frame Survived
    reward += 0.1

    # Reward for Each Pipe Passed
    reward += score * .1

    if game_overBig:
        return -20
    # Penalty for Game Over
    if game_over:
        return -10

    # Reward Based on Bird's Position Relative to the Next Pipe
    if pipes:
        next_bottom_pipe, next_top_pipe, _ = pipes[0]
        next_pipe = pipes[0]
        gap_center = next_pipe[1].bottom + (next_pipe[0].y - next_pipe[1].bottom) / 2
        distance_to_center = abs(bird_y - gap_center)
        if distance_to_center < 10:
            reward += .2
        dist_to_bottom_pipe_top = next_bottom_pipe.y - bird_y - 30
        dist_to_top_pipe_bottom = bird_y - next_top_pipe.bottom 
        if dist_to_bottom_pipe_top > 0 and dist_to_top_pipe_bottom > 0:
            reward += 1
        else:
            reward -= .1

    return reward