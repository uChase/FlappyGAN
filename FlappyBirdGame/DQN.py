import torch
import torch.nn as nn
import torch.optim as optim
import random

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 248)
        self.fc2 = nn.Linear(248, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class FlappyBirdAgent:
    def __init__(self, input_size, action_space, memory_limit):
        self.model = DQN(input_size, action_space).cuda()  # Move the model to CUDA
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        self.memory = []  # Memory for experience replay
        self.memory_limit = memory_limit  # Limit on memory size
        self.gamma = 0.999  # Discount factor for future rewards

    def remember(self, state, action, next_state, reward, done):
        if len(self.memory) >= self.memory_limit:
            self.memory.pop(0)  # Remove the oldest memory if memory limit is reached
        if done:
            done = 1
        self.memory.append((state, action, next_state, reward, done))

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.choice([0, 1])  # Random action (0 or 1)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).cuda()  # Move the tensor to CUDA
            with torch.no_grad():
                q_values = self.model(state_tensor)
            return torch.argmax(q_values).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
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

def prepare_state(bird_y, bird_y_change, pipes, bird_x):
    # Flatten pipe coordinates
    pipe_data = []
    bottom_pipe = None
    for pipe_pair in pipes[:2]:  # Consider at most two pipe pairs
        bottom_pipe, top_pipe = pipe_pair[0], pipe_pair[1]
        # Extract the y coordinates of the top of the bottom pipe and the bottom of the top pipe
        bottom_pipe_top_y = bottom_pipe.y
        top_pipe_bottom_y = top_pipe.bottom
        pipe_data.extend([bottom_pipe.x, bottom_pipe.x + bottom_pipe.width, top_pipe_bottom_y, bottom_pipe_top_y])

    distance_to_center = 500
    distance_to_pipe_top = 600
    distance_to_pipe_bottom = 100
    if pipes:
        next_pipe = pipes[0]
        gap_center = next_pipe[1].bottom + (next_pipe[0].y - next_pipe[1].bottom) / 2
        distance_to_center = abs(bird_y - gap_center - 16)
        distance_to_pipe_top = bird_y - next_pipe[1].bottom - 16
        distance_to_pipe_bottom = next_pipe[0].y - bird_y + 16
    # Ensure pipe_data has enough data for two pipes
    while len(pipe_data) < 8:
        pipe_data.extend([0, 0, 0, 0])  # If less than two pipes, pad with zeros

    state = []

    if bottom_pipe:
        state = [bird_y - 16, bird_y + 14, bottom_pipe.x - bird_x + 14, bird_y_change, distance_to_center, distance_to_pipe_top, distance_to_pipe_bottom] + pipe_data
    else:
        state = [bird_y - 16, bird_y + 14, 2000, bird_y_change, distance_to_center, distance_to_pipe_top, distance_to_pipe_bottom] + pipe_data
    return state
def calculate_reward(bird_y, bird_y_change, pipes, score, game_over, game_overBig):
    reward = 0

    # Constant Reward for Each Frame Survived
    reward += 0.1

    # Reward for Each Pipe Passed
    reward += score * 10

    if game_overBig:
        return -1000
    # Penalty for Game Over
    if game_over:
        return -100

    # Reward Based on Bird's Position Relative to the Next Pipe
    if pipes:
        next_pipe = pipes[0]
        gap_center = next_pipe[1].bottom + (next_pipe[0].y - next_pipe[1].bottom) / 2
        distance_to_center = abs(bird_y - gap_center - 16)
        if distance_to_center < 10:
            reward += 10
        elif distance_to_center < 20:
            reward += 5
        elif distance_to_center < 50:
            reward += 2
        elif distance_to_center < 100:
            reward += 1

    return reward