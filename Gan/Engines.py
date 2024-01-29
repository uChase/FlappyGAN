import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicsEngine(nn.Module):
    def __init__(self, action_size, noise_size, memory_size, frame_channels, frame_height, frame_width, hidden_size, num_layers, output_size):
        super(DynamicsEngine, self).__init__()

        # Convolutional layers for frame processing
        self.conv_layers = nn.Sequential(
            nn.Conv2d(frame_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Calculate the size of the flattened CNN output
        with torch.no_grad():
            self.flattened_size = self.conv_layers(torch.zeros(1, frame_channels, frame_height, frame_width)).shape[1]

        # MLP for combining action, noise, memory, and CNN output
        self.mlp = nn.Sequential(
            nn.Linear(action_size + noise_size + memory_size + self.flattened_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        # GRU layer
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)

        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, output_size)

        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, frame, action, noise, prev_memory, prev_hidden):
        # Process frame through convolutional layers
        conv_output = self.conv_layers(frame)

        # Combine action, noise, prev_memory, and conv_output; then pass through MLP
        combined_input = torch.cat([action, noise, prev_memory], dim=1)
        mlp_output = self.mlp(combined_input)

        # Element-wise multiplication with previous hidden state
        new_hidden = prev_hidden * mlp_output

        # Reshape and pass through GRU
        gru_input = conv_output.unsqueeze(1)  # Add sequence length dimension
        out, hidden = self.gru(gru_input, new_hidden)

        # Pass the output of GRU to the fully connected layer
        out = self.fc(out[:, -1, :])

        return out, hidden

    def init_hidden(self, batch_size):
        # Initialize the hidden state
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)


class Memory(nn.Module):
    def __init__(self, action_size, hidden_size, memory_units, memory_unit_size, controller_size):
        super(Memory, self).__init__()
        self.memory_units = memory_units
        self.memory_unit_size = memory_unit_size

        # Initialize the memory
        self.memory = torch.randn(memory_units, memory_unit_size)

        # Controller
        self.controller = nn.LSTM(action_size + hidden_size, controller_size)

        # Read and write heads (for simplicity, using a single head for each)
        self.read_head = nn.Linear(controller_size, memory_unit_size)
        self.write_head = nn.Linear(controller_size, memory_unit_size)

    def _addressing(self, key, memory):
        # Content-based addressing (using cosine similarity for simplicity)
        key = key / (key.norm(p=2, dim=1, keepdim=True) + 1e-16)
        memory = memory / (memory.norm(p=2, dim=2, keepdim=True) + 1e-16)
        similarity = torch.bmm(memory, key.unsqueeze(2)).squeeze(2)
        w = F.softmax(similarity, dim=1)
        return w

    def forward(self, action, hidden_state, prev_read_vector):
        # Concatenate action and hidden state to form the input
        input = torch.cat([action, hidden_state], dim=1)

        # Controller operations
        controller_out, _ = self.controller(input.unsqueeze(0))
        controller_out = controller_out.squeeze(0)

        # Read and write operations
        read_key = self.read_head(controller_out)
        write_key = self.write_head(controller_out)

        # Addressing
        read_weight = self._addressing(read_key, self.memory)
        write_weight = self._addressing(write_key, self.memory)

        # Read from memory
        read_vector = torch.bmm(read_weight.unsqueeze(1), self.memory).squeeze(1)

        # Write to memory
        self.memory = self._write_to_memory(write_weight, write_key, self.memory)

        return read_vector

    def _write_to_memory(self, write_weight, write_vector, memory):
        # Simple write operation (replace content for demonstration)
        memory = memory * (1 - write_weight.unsqueeze(-1)) + write_vector.unsqueeze(1) * write_weight.unsqueeze(-1)
        return memory

    def reset_memory(self):
        # Reset memory
        self.memory.fill_(0)


class RenderingEngine(nn.Module):
    def __init__(self, hidden_size, memory_size, output_channels, output_height, output_width):
        super(RenderingEngine, self).__init__()
        
        # Define the input size (hidden state + memory output)
        combined_size = hidden_size + memory_size

        self.decoder = nn.Sequential(
            # Transform the combined input to a suitable shape for transposed convolutions
            nn.Linear(combined_size, 256),
            nn.ReLU(),
            nn.Unflatten(1, (256, 1, 1)),  # Reshape for transposed convolution

            # Transposed convolutional layers to upscale to the desired output size
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Tanh activation to output pixel values in the range [-1, 1]
        )

    def forward(self, hidden_state, memory_output):
        # Combine the hidden state and memory output
        combined_input = torch.cat([hidden_state, memory_output], dim=1)

        # Decode the combined input into an image
        frame = self.decoder(combined_input)
        return frame

class SingleImageDiscriminator(nn.Module):
    def __init__(self, input_channels):
        super(SingleImageDiscriminator, self).__init__()
        # Example architecture
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # Additional convolutional layers...
            nn.Conv2d(64, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


class TemporalDiscriminator(nn.Module):
    def __init__(self, input_channels, sequence_length):
        super(TemporalDiscriminator, self).__init__()
        # Example architecture (assumes input is a sequence of frames)
        self.model = nn.Sequential(
            nn.Conv3d(input_channels, 64, kernel_size=(sequence_length, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            # Additional 3D convolutional layers...
            nn.Conv3d(64, 1, kernel_size=(1, 4, 4), stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

class ActionConditionedDiscriminator(nn.Module):
    def __init__(self, input_channels, action_size):
        super(ActionConditionedDiscriminator, self).__init__()
        # Adjust the input channels to accommodate two frames and an action
        self.model = nn.Sequential(
            nn.Conv2d(input_channels * 2 + action_size, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # Additional convolutional layers...
            nn.Conv2d(64, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, frame1, frame2, action):
        # Concatenate the frames and action
        # Assuming action is a one-hot encoded vector and needs to be expanded to match frame dimensions
        action = action.view(action.size(0), action.size(1), 1, 1).expand(-1, -1, frame1.size(2), frame1.size(3))
        combined_input = torch.cat([frame1, frame2, action], dim=1)
        return self.model(combined_input)
