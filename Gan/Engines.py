import torch
import torch.nn as nn
import torch.nn.functional as F

# THE IMAGE IS GOING TO BE 75 X 50

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
            #HARD CODED CAUSE IM GOING CRAZY!!
            nn.Linear(4566, hidden_size),
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

        action = action.view(1, -1)

        noise = noise.view(1, -1)

        flattened_memory = prev_memory.view(-1)

        flattened_memory = flattened_memory.view(1, -1)

        # Process frame through convolutional layers
        conv_output = self.conv_layers(frame)

        # Combine action, noise, prev_memory, and conv_output; then pass through MLP

        conv_output_flattened = conv_output.view(1, -1)
        combined_input = torch.cat([action, noise, flattened_memory, conv_output_flattened], dim=1)
        mlp_output = self.mlp(combined_input)

        # Element-wise multiplication with previous hidden state
        new_hidden = prev_hidden * mlp_output

        # Reshape and pass through GRU
        gru_input = mlp_output.unsqueeze(1)  # Add sequence length dimension
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
        self.memory = torch.randn(memory_units, memory_unit_size).cuda()

        # Controller (using GRU instead of LSTM)
        self.controller = nn.GRU(action_size + hidden_size, controller_size)

        # Read and write heads
        self.read_head = nn.Linear(controller_size, memory_unit_size)
        self.write_head = nn.Linear(controller_size, memory_unit_size)

        # Additional vectors for writing and location-based addressing
        self.add_vector = nn.Linear(controller_size, memory_unit_size)
        self.erase_vector = nn.Linear(controller_size, memory_unit_size)
        self.shift_vector = nn.Linear(controller_size, memory_units)  # For shifting attention
        self.sharpening_factor = nn.Linear(controller_size, 1)  # For sharpening the distribution
        self.interpolation_gate = nn.Linear(controller_size, 1)  # For interpolation

    def _cosine_similarity(self, key, memory):
        # Cosine similarity for content-based addressing
        key = key.to('cuda')
        memory = memory.to('cuda')
        key = key / (key.norm(p=2, dim=1, keepdim=True) + 1e-16)
        memory = memory / (memory.norm(p=2, dim=1, keepdim=True) + 1e-16)
        similarity = torch.mm(memory, key.t())
        return F.softmax(similarity, dim=1)

    def _circular_convolution(self, w, s):
        # Circular convolution for location-based addressing
        result = torch.zeros_like(w)
        for i in range(w.size(0)):
            for j in range(s.size(0)):
                # Ensure that both operands are of the same shape
                result[i] += w[(i - j) % w.size(0), 0] * s[j, 0]
        return result


    def _sharpen(self, w, gamma):
        # Sharpening the distribution
        w = w ** gamma
        w = w / torch.sum(w, dim=0, keepdim=True)
        return w

    def forward(self, action, hidden_state, prev_read_weight, prev_write_weight, prev_controller_hidden=None):
        # Detach the memory tensor
        memory = self.memory.detach()

        # Concatenate action and hidden state to form the input
        action = action.view(1, -1)

        hidden_state_last_layer = hidden_state[-1]

        input = torch.cat([action, hidden_state_last_layer], dim=1)

        if prev_controller_hidden is None:
            prev_controller_hidden = torch.zeros(1, input.size(0), self.controller.hidden_size).detach().cuda()

        # Controller operations
        controller_out, new_controller_hidden = self.controller(input.unsqueeze(0), prev_controller_hidden)
        controller_out = controller_out.squeeze(0)
        new_controller_hidden = new_controller_hidden.detach()
        # Read and write operations
        read_key = self.read_head(controller_out)
        write_key = self.write_head(controller_out)

        # Content-based addressing
        read_weight = self._cosine_similarity(read_key, memory)
        write_weight = self._cosine_similarity(write_key, memory)

        # Interpolation Gate
        g = torch.sigmoid(self.interpolation_gate(controller_out))

        # Interpolate between previous and new weights


        read_weight = g * read_weight + (1 - g) * prev_read_weight
        write_weight = g * write_weight + (1 - g) * prev_write_weight

        # Location-based addressing (shift and sharpen)
        shift = F.softmax(self.shift_vector(controller_out), dim=1)
        gamma = F.softplus(self.sharpening_factor(controller_out))

        read_weight = self._sharpen(self._circular_convolution(read_weight, shift), gamma)
        write_weight = self._sharpen(self._circular_convolution(write_weight, shift), gamma)
        memory = memory.detach()
        # Read from memory
        read_vector = torch.mm(read_weight.transpose(0, 1), memory)


        # Write to memory
        add_vec = self.add_vector(controller_out)
        erase_vec = self.erase_vector(controller_out)
        memory = self._write_to_memory(write_weight, add_vec, erase_vec, memory)

        return read_vector, read_weight, write_weight, new_controller_hidden

    def _write_to_memory(self, write_weight, add_vector, erase_vector, memory):
        # Write operation with erase and add

        memory = memory * (1 - write_weight * erase_vector) 

        memory = memory + write_weight * add_vector
        return memory

    def reset_memory(self):
        # Reset memory
        self.memory.fill_(0).detach()
    
    def get_memory(self):
        return self.memory



class RenderingEngine(nn.Module):
    def __init__(self, output_size, read_size, output_channels):
        super(RenderingEngine, self).__init__()
        
        # Define the combined input size (hidden state + memory output)
        combined_size = output_size + read_size

        # Linear layer to transform the combined input to the latent dimension
        self.input_layer = nn.Sequential(
            nn.Linear(combined_size, 256 * 2 * 3),  # Adjust latent_dim accordingly
            nn.ReLU(inplace=True)
        )

        # Rest of the decoder as before
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (256, 2, 3)),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Upsample(size=(50, 75)),
            nn.Conv2d(32, output_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, hidden_state, memory_output):
        # Combine the hidden state and memory output and process through input_layer
        combined_input = torch.cat([hidden_state, memory_output], dim=1)
        latent_input = self.input_layer(combined_input)

        # Decode the latent input into an image
        frame = self.decoder(latent_input)
        return frame


class SingleImageDiscriminator(nn.Module):
    def __init__(self, input_channels, input_height, input_width):
        super(SingleImageDiscriminator, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten()
        )

        # Dynamically calculate the input size for the linear layer
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, input_height, input_width)
            conv_output_size = self.conv_layers(dummy_input).shape[1]
        
        self.fc = nn.Sequential(
            nn.Linear(conv_output_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        conv_out = self.conv_layers(x)
        return self.fc(conv_out)


class TemporalDiscriminator(nn.Module):
    def __init__(self, input_channels, sequence_length):
        super(TemporalDiscriminator, self).__init__()
        # Architecture with 3D convolutions
        self.model = nn.Sequential(
            nn.Conv3d(input_channels, 64, kernel_size=(sequence_length, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(64, 128, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(128, 256, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(256, 512, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(512, 1, kernel_size=(1, 4, 4), stride=(1, 1, 1), padding=(0, 1, 1)),
            # nn.AdaptiveAvgPool3d((1, 1, 1)),  # Adaptive pooling to get a fixed size output
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(6, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        conv_out = self.model(x)
        return self.fc(conv_out)

class ActionConditionedDiscriminator(nn.Module):
    def __init__(self, input_channels, action_size):
        super(ActionConditionedDiscriminator, self).__init__()
        # Define the architecture
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels * 2, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten()
        )

        # Calculate the output size of the convolutional layers dynamically


        # Fully connected layers to integrate action information
        self.fc_layers = nn.Sequential(
            #THIS IS A HARD CODED IN VALUE
            nn.Linear(6145, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, combined_frames, action):
     

        # Process the concatenated frames through convolutional layers
        conv_out = self.conv_layers(combined_frames)

        # Flatten the action and concatenate it with the conv_out
        action = action.view(action.size(0), -1)


        flattened_conv = conv_out.view(1, -1)

        combined_input = torch.cat([flattened_conv, action], dim=1)

        # Pass the combined input through fully connected layers
        return self.fc_layers(combined_input)


class GameOverDiscriminator(nn.Module):
    def __init__(self, input_channels, input_height, input_width):
        super(GameOverDiscriminator, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten()
        )

        # Dynamically calculate the input size for the linear layer
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, input_height, input_width)
            conv_output_size = self.conv_layers(dummy_input).shape[1]
        
        self.fc = nn.Sequential(
            nn.Linear(conv_output_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        conv_out = self.conv_layers(x)
        return self.fc(conv_out)