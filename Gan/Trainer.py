import torch
import torch.nn as nn
import torch.optim as optim
import Engines


#!! TRY CYCLE LOSS IF BAD

class GameGANTrainer:
    def __init__(self, batch_size, frame_channels, frame_height, frame_width, action_size, noise_size, memory_size, hidden_size, num_layers, output_size, memory_units, memory_unit_size, controller_size, output_channels, learning_rate=0.001):
        self.batch_size = batch_size

        # Initialize the models
        self.dynamics_engine = Engines.DynamicsEngine(action_size, noise_size, memory_size, frame_channels, frame_height, frame_width, hidden_size, num_layers, output_size).cuda()
        self.memory = Engines.Memory(action_size, hidden_size, memory_units, memory_unit_size, controller_size).cuda()
        self.rendering_engine = Engines.RenderingEngine(hidden_size, memory_size, output_channels).cuda()
        self.single_image_discriminator = Engines.SingleImageDiscriminator(frame_channels, frame_height, frame_width).cuda()
        self.temporal_discriminator = Engines.TemporalDiscriminator(frame_channels, batch_size).cuda()
        self.action_conditioned_discriminator = Engines.ActionConditionedDiscriminator(frame_channels, action_size).cuda()
        self.game_over_discriminator = Engines.GameOverDiscriminator(frame_channels, frame_height, frame_width).cuda()

        # Define loss function
        self.BCE = nn.BCELoss()

        # Define optimizers
        self.optimizer_de = optim.Adam(self.dynamics_engine.parameters(), lr=learning_rate)
        self.optimizer_mem = optim.Adam(self.memory.parameters(), lr=learning_rate)
        self.optimizer_re = optim.Adam(self.rendering_engine.parameters(), lr=learning_rate)
        self.optimizer_sid = optim.Adam(self.single_image_discriminator.parameters(), lr=learning_rate)
        self.optimizer_td = optim.Adam(self.temporal_discriminator.parameters(), lr=learning_rate)
        self.optimizer_acd = optim.Adam(self.action_conditioned_discriminator.parameters(), lr=learning_rate)
        self.optimizer_god = optim.Adam(self.game_over_discriminator.parameters(), lr=learning_rate)

    def train(self, num_epochs, dataset, num_batches_per_epoch, sequence_length):
        for epoch in range(num_epochs):
            prev_hidden = self.dynamics_engine.init_hidden(actions.size(0)).cuda()
            prev_memory = self.memory.reset_memory().cuda()
            prev_read_weights = torch.zeros(actions.size(0), self.memory_units).cuda()
            prev_write_weights = torch.zeros(actions.size(0), self.memory_units).cuda()
            for batch in range(num_batches_per_epoch):
                current_frames, actions, next_frames, game_overs = dataset.get_random_sample(sequence_length)
                current_frames = torch.tensor(current_frames).float().cuda()
                actions = torch.tensor(actions).float().cuda()
                next_frames = torch.tensor(next_frames).float().cuda()
                game_overs = torch.tensor(game_overs).float().cuda()

                # Reset gradients
                self.optimizer_de.zero_grad()
                self.optimizer_mem.zero_grad()
                self.optimizer_re.zero_grad()
                self.optimizer_sid.zero_grad()
                self.optimizer_td.zero_grad()
                self.optimizer_acd.zero_grad()

                # Generate noise for the batch
                noise = torch.randn(actions.size(0), self.noise_size).cuda()
                

                # Render frames for the entire batch
                fake_frames, prev_hidden, prev_memory, prev_read_weights, prev_write_weights = self.render_frames_batch(actions, current_frames, prev_hidden, prev_memory, noise, prev_read_weights, prev_write_weights)

                # Compute discriminator loss on real and fake data
                discriminator_loss = self.single_discriminator_loss(next_frames, fake_frames)
                discriminator_loss.backward()
                self.optimizer_sid.step()

                # Compute generator loss on fake data
                generator_loss = self.single_generator_loss(fake_frames)
                generator_loss.backward()
                self.optimizer_re.step()

                real_frame_trios, random_frame_trios, fakeAction_frame_trios = self.generateFrameTrios(current_frames, actions, next_frames)
                acd_loss = self.action_conditioned_discriminator_loss(real_frame_trios, random_frame_trios, fakeAction_frame_trios)
                acd_loss.backward()
                self.optimizer_acd.step()

                # Compute ACG loss
                acg_loss = self.action_conditioned_generator_loss(current_frames, fake_frames, actions)
                acg_loss.backward()
                self.optimizer_re.step()

                # Prepare sequences of frames for the temporal discriminator
                # Assume that 'current_frames' and 'fake_frames' are sequences
                real_frame_sequences = current_frames
                fake_frame_sequences = fake_frames

                # Compute temporal discriminator loss
                td_loss = self.temporal_discriminator_loss(real_frame_sequences, fake_frame_sequences)
                td_loss.backward()
                self.optimizer_td.step()

                # Compute generator loss for temporal discriminator
                tg_loss = self.temporal_generator_loss(fake_frame_sequences)
                tg_loss.backward()
                self.optimizer_re.step()

                # Compute game over discriminator loss
                game_over_frames = current_frames[game_overs == 1]
                non_game_over_frames = current_frames[game_overs == 0]

                # Compute GameOver Discriminator loss
                god_loss = self.game_over_discriminator_loss(game_over_frames, non_game_over_frames)
                god_loss.backward()
                self.optimizer_god.step()

    def generateFrameTrios(self, current_frames, actions, next_frames):
        batch_size = current_frames.size(0)

        # Real Frame Trios: [current frame, next frame, action]
        real_frame_trios = torch.cat([current_frames, next_frames, actions.unsqueeze(-1)], dim=1)

        # Random Frame Trios: [random frame, random frame, random action]
        random_indices = torch.randint(0, batch_size, (batch_size,))
        random_actions = torch.randint(0, 2, (batch_size, 1)).to(actions.dtype)  # Assuming actions are either 0 or 1
        random_frame_trios = torch.cat([current_frames[random_indices], next_frames[random_indices], random_actions], dim=1)

        # Fake Action Frame Trios: [current frame, next frame, opposite action]
        opposite_actions = 1 - actions  # Flip 0 to 1 and 1 to 0
        fakeAction_frame_trios = torch.cat([current_frames, next_frames, opposite_actions.unsqueeze(-1)], dim=1)

        return real_frame_trios, random_frame_trios, fakeAction_frame_trios



    def game_over_discriminator_loss(self, game_over_frames, non_game_over_frames):
        # Labels for game over and non-game over frames
        game_over_labels = torch.ones(game_over_frames.size(0), 1).cuda()
        non_game_over_labels = torch.zeros(non_game_over_frames.size(0), 1).cuda()

        # Compute loss for game over frames
        game_over_loss = self.BCE(self.game_over_discriminator(game_over_frames), game_over_labels)

        # Compute loss for non-game over frames
        non_game_over_loss = self.BCE(self.game_over_discriminator(non_game_over_frames), non_game_over_labels)

        # Total loss
        total_loss = game_over_loss + non_game_over_loss
        return total_loss

    def single_discriminator_loss(self, real_images, fake_images):
        # Real images label is 1, fake images label is 0
        real_labels = torch.ones(real_images.size(0), 1).cuda()
        fake_labels = torch.zeros(fake_images.size(0), 1).cuda()

        # Compute the discriminator loss on real images
        real_loss = self.BCE(self.single_image_discriminator(real_images), real_labels)

        # Compute the discriminator loss on fake images
        fake_loss = self.BCE(self.single_image_discriminator(fake_images), fake_labels)

        total_loss = (real_loss + fake_loss) 
        return total_loss

    def single_generator_loss(self, fake_images):
        # As the generator tries to generate real images, labels for fake images are 1
        labels = torch.ones(fake_images.size(0), 1).cuda()

        # Compute the generator loss
        loss = self.BCE(self.single_image_discriminator(fake_images), labels)
        return loss
    

    def action_conditioned_discriminator_loss(self, real_frame_trios, random_frame_trios, fakeAction_frame_trios):
        # Ensure all inputs are tensors
        real_frame_trios = torch.tensor(real_frame_trios).cuda()
        random_frame_trios = torch.tensor(random_frame_trios).cuda()
        fakeAction_frame_trios = torch.tensor(fakeAction_frame_trios).cuda()

        # Labels for real and fake data
        real_labels = torch.ones(real_frame_trios.size(0), 1).cuda()
        fake_labels = torch.zeros(real_frame_trios.size(0), 1).cuda()

        # Extract frames and actions from the trios
        real_frames1, real_frames2, real_actions = real_frame_trios[:, 0], real_frame_trios[:, 1], real_frame_trios[:, 2]
        random_frames1, random_frames2, random_actions = random_frame_trios[:, 0], random_frame_trios[:, 1], random_frame_trios[:, 2]
        fakeAction_frames1, fakeAction_frames2, fakeAction_actions = fakeAction_frame_trios[:, 0], fakeAction_frame_trios[:, 1], fakeAction_frame_trios[:, 2]

        # Compute the discriminator loss on real frame pairs
        real_loss = self.BCE(self.action_conditioned_discriminator(real_frames1, real_frames2, real_actions), real_labels)

        # Compute the discriminator loss on random frame pairs
        fake_loss = self.BCE(self.action_conditioned_discriminator(random_frames1, random_frames2, random_actions), fake_labels)

        # Compute the discriminator loss on fake action frame pairs
        fake_loss_2 = self.BCE(self.action_conditioned_discriminator(fakeAction_frames1, fakeAction_frames2, fakeAction_actions), fake_labels)

        # Total loss
        total_loss = real_loss + fake_loss + fake_loss_2
        return total_loss


    def action_conditioned_generator_loss(self, prev_frames, g_frames, actions):
        # The generator tries to fool the discriminator, so the labels are 1 (real)
        labels = torch.ones(g_frames.size(0), 1).cuda()

        # Create pairs of previous and fake frames
        fake_frame_pairs = torch.cat([prev_frames, g_frames], dim=1)

        # Compute the generator loss
        loss = self.BCE(self.action_conditioned_discriminator(fake_frame_pairs, actions), labels)
        return loss

    def render_frame(self, action, prev_frame, prev_hidden, prev_memory, noise, prev_read_weight, prev_write_weight):
        # Generate the hidden state and memory
        output, hidden = self.dynamics_engine(prev_frame, action, noise, prev_memory, prev_hidden)

        read_vector, read_weight, write_weight = self.memory(action, output, hidden, prev_read_weight, prev_write_weight)

        frame = self.rendering_engine(output, read_vector)

        return frame, hidden, read_vector, read_weight, write_weight
    

    def temporal_discriminator_loss(self, real_frame_sequences, fake_frame_sequences):
        # Labels for real and fake data
        real_labels = torch.ones(real_frame_sequences.size(0), 1).cuda()
        fake_labels = torch.zeros(fake_frame_sequences.size(0), 1).cuda()

        # Compute the discriminator loss on real frame sequences
        real_loss = self.BCE(self.temporal_discriminator(real_frame_sequences), real_labels)

        # Compute the discriminator loss on fake frame sequences
        fake_loss = self.BCE(self.temporal_discriminator(fake_frame_sequences), fake_labels)

        # Total loss
        total_loss = real_loss + fake_loss
        return total_loss
    
    def temporal_generator_loss(self, fake_frame_sequences):
        # The generator tries to fool the discriminator, so the labels are 1 (real)
        labels = torch.ones(fake_frame_sequences.size(0), 1).cuda()

        # Compute the generator loss
        loss = self.BCE(self.temporal_discriminator(fake_frame_sequences), labels)
        return loss

    
    def render_frames_batch(self, actions, prev_frames, prev_hidden, prev_memory, noise, prev_read_weights, prev_write_weights):
        batch_size = actions.size(0)
        seq_len = actions.size(1)
        fake_frames = []

        for t in range(seq_len):
            # Extract the action for the current time step
            current_action = actions[:, t, :]

            # Process each frame in the batch
            outputs, new_hiddens, new_read_vectors, new_read_weights, new_write_weights = [], [], [], [], []
            for b in range(batch_size):
                frame, hidden, read_vector, read_weight, write_weight = self.render_frame(
                    current_action[b].unsqueeze(0), 
                    prev_frames[b].unsqueeze(0), 
                    prev_hidden[b], 
                    prev_memory[b], 
                    noise[b].unsqueeze(0), 
                    prev_read_weights[b], 
                    prev_write_weights[b]
                )

                outputs.append(frame.squeeze(0))
                new_hiddens.append(hidden)
                new_read_vectors.append(read_vector)
                new_read_weights.append(read_weight)
                new_write_weights.append(write_weight)

            # Update the previous states for the next iteration
            prev_frames = torch.stack(outputs, dim=0)
            prev_hidden = torch.stack(new_hiddens, dim=0)
            prev_memory = torch.stack(new_read_vectors, dim=0)
            prev_read_weights = torch.stack(new_read_weights, dim=0)
            prev_write_weights = torch.stack(new_write_weights, dim=0)

            fake_frames.append(prev_frames)

        # Combine frames into a tensor
        fake_frames = torch.stack(fake_frames, dim=1)  # Shape: [batch_size, seq_len, channels, height, width]

        return fake_frames, prev_hidden, prev_memory, prev_read_weights, prev_write_weights
