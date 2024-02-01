import torch
import torch.nn as nn
import torch.optim as optim
import Engines


#!! TRY CYCLE LOSS IF BAD

class GameGANTrainer:
    def __init__(self, batch_size, frame_channels, frame_height, frame_width, action_size, noise_size, hidden_size, num_layers, output_size, memory_units, memory_unit_size, controller_size, output_channels, learning_rate=0.001):
        self.batch_size = batch_size
        self.memory_units = memory_units
        self.memory_unit_size = memory_unit_size
        self.noise_size = noise_size

        # Initialize the models
        self.dynamics_engine = Engines.DynamicsEngine(action_size, noise_size, memory_unit_size * memory_units, frame_channels, frame_height, frame_width, hidden_size, num_layers, output_size).cuda()
        self.memory = Engines.Memory(action_size, hidden_size, memory_units, memory_unit_size, controller_size).cuda()
        self.rendering_engine = Engines.RenderingEngine(output_size, memory_unit_size, output_channels).cuda()
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

        # torch.autograd.set_detect_anomaly(True)

        for epoch in range(num_epochs):
            for batch in range(num_batches_per_epoch):
                current_frames, actions, next_frames, game_overs = dataset.get_random_sample(self.batch_size)


                # Reset gradients
                self.optimizer_de.zero_grad()
                self.optimizer_mem.zero_grad()
                self.optimizer_re.zero_grad()
                self.optimizer_sid.zero_grad()
                self.optimizer_td.zero_grad()
                self.optimizer_acd.zero_grad()
                self.optimizer_god.zero_grad()

                # Initialize hidden and memory states for the sequence
                prev_hidden = self.dynamics_engine.init_hidden(1).detach().cuda()
                self.memory.reset_memory()
                prev_memory = torch.zeros(1, self.memory_unit_size).detach().cuda()
                prev_read_weights = torch.zeros(self.memory_units, 1).detach().cuda()
                prev_write_weights = torch.zeros(self.memory_units, 1).detach().cuda()

                # Generate noise for the batch
                noise = torch.randn(sequence_length, self.noise_size).cuda()
                
                # Render frames for the entire batch
                fake_frames = self.render_frames_batch(actions, current_frames, prev_hidden, prev_memory, noise, prev_read_weights, prev_write_weights, game_overs)

                # Compute all losses
                discriminator_loss = self.single_discriminator_loss(next_frames, fake_frames)
                generator_loss = self.single_generator_loss(fake_frames)
                real_frame_pairs, random_frame_pairs, fakeAction_frame_pairs = self.generateFrameTrios(current_frames, actions, next_frames)
                acd_loss = self.action_conditioned_discriminator_loss(real_frame_pairs, random_frame_pairs, fakeAction_frame_pairs)
                acg_loss = self.action_conditioned_generator_loss(current_frames, fake_frames, actions)
                real_frame_sequences = current_frames  # Assuming these are sequences
                fake_frame_sequences = fake_frames  # Assuming these are sequences
                td_loss = self.temporal_discriminator_loss(real_frame_sequences, fake_frame_sequences, self.batch_size)
                tg_loss = self.temporal_generator_loss(fake_frame_sequences, self.batch_size)
                game_over_frames = current_frames[game_overs == 1]
                non_game_over_frames = current_frames[game_overs == 0]
                god_loss = self.game_over_discriminator_loss(game_over_frames, non_game_over_frames)
                

                # Accumulate all losses
                total_loss = discriminator_loss + generator_loss + acd_loss + acg_loss + td_loss + tg_loss + god_loss
                
                # Backward pass
                total_loss.backward()

                # Step optimizers
                self.optimizer_sid.step()
                self.optimizer_re.step()
                self.optimizer_acd.step()
                self.optimizer_td.step()
                self.optimizer_god.step()


            print("Epoch: ", epoch, " Loss: ", total_loss.item())


    def generateFrameTrios(self, current_frames, actions, next_frames):
        batch_size = current_frames.size(0)

        # Real frame trios: [current frame, next frame, action]
        real_frame_pairs = [(torch.cat((current_frames[i], next_frames[i]), dim=0), actions[i]) for i in range(batch_size)]

        # Random frame trios: [random frame, another random frame, random action]
        random_frame_pairs = [(torch.cat((current_frames[rand_idx], next_frames[rand_idx]), dim=0), torch.randint(0, 2, (1,))) for rand_idx in torch.randint(0, batch_size, (batch_size,))]

        # FakeAction frame trios: [current frame, next frame, opposite action]
        opposite_actions = 1 - actions
        fakeAction_frame_pairs = [(torch.cat((current_frames[i], next_frames[i]), dim=0), opposite_actions[i]) for i in range(batch_size)]

        return real_frame_pairs, random_frame_pairs, fakeAction_frame_pairs

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
    

    def action_conditioned_discriminator_loss(self, real_frame_pairs, random_frame_pairs, fakeAction_frame_pairs):
        # Batch size for labels
        batch_size = len(real_frame_pairs)

        # Labels for real and fake data
        real_labels = torch.ones(batch_size, 1, device='cuda')
        fake_labels = torch.zeros(batch_size, 1, device='cuda')

        # Concatenate outputs for real frame pairs
        real_outputs = torch.cat([
            self.action_conditioned_discriminator(frame_pair.cuda(), action.cuda().unsqueeze(0))
            for frame_pair, action in real_frame_pairs
        ], dim=0)

        # Compute the discriminator loss on real frame pairs
        real_loss = self.BCE(real_outputs, real_labels)

        # Concatenate outputs for random frame pairs
        random_outputs = torch.cat([
            self.action_conditioned_discriminator(frame_pair.cuda(), action.cuda().unsqueeze(0))
            for frame_pair, action in random_frame_pairs
        ], dim=0)

        # Compute the discriminator loss on random frame pairs
        random_loss = self.BCE(random_outputs, fake_labels)

        # Concatenate outputs for fakeAction frame pairs
        fakeAction_outputs = torch.cat([
            self.action_conditioned_discriminator(frame_pair.cuda(), action.cuda().unsqueeze(0))
            for frame_pair, action in fakeAction_frame_pairs
        ], dim=0)

        # Compute the discriminator loss on fakeAction frame pairs
        fakeAction_loss = self.BCE(fakeAction_outputs, fake_labels)

        # Total loss is the sum of the three individual losses
        total_loss = real_loss + random_loss + fakeAction_loss

        return total_loss



    def action_conditioned_generator_loss(self, prev_frames, g_frames, actions):
        # The generator tries to fool the discriminator, so the labels are 1 (real)
        batch_size = g_frames.size(0)
        labels = torch.ones(batch_size, 1, device='cuda')

        # Process each frame pair individually and concatenate the outputs
        fake_frame_outputs = torch.cat([
            self.action_conditioned_discriminator(torch.cat([prev_frames[i].unsqueeze(0), g_frames[i].unsqueeze(0)], dim=1), actions[i].unsqueeze(0))
            for i in range(batch_size)
        ], dim=0)

        # Compute the generator loss
        loss = self.BCE(fake_frame_outputs, labels)
        return loss


    

    def temporal_discriminator_loss(self, real_frames, fake_frames, sequence_length):
        # real_frames and fake_frames are of shape [batch_size, channels, height, width]
        # Reshape to add sequence length dimension
        # Ensure batch_size is divisible by sequence_length

        batch_size = real_frames.size(0)
        if batch_size % sequence_length != 0:
            raise ValueError("Batch size must be divisible by the sequence length.")

        new_batch_size = batch_size // sequence_length

        # Reshape the frames to have a sequence dimension
        real_frame_sequences = real_frames.view(new_batch_size, sequence_length, real_frames.size(1), real_frames.size(2), real_frames.size(3)).cuda()
        fake_frame_sequences = fake_frames.view(new_batch_size, sequence_length, fake_frames.size(1), fake_frames.size(2), fake_frames.size(3)).cuda()
        
        real_frame_sequences = real_frame_sequences.permute(0, 2, 1, 3, 4)
        fake_frame_sequences = fake_frame_sequences.permute(0, 2, 1, 3, 4)

        # Labels for real and fake data
        real_labels = torch.ones(new_batch_size, 1, device='cuda')
        fake_labels = torch.zeros(new_batch_size, 1, device='cuda')

        # Compute the discriminator loss on real and fake frame sequences
        real_loss = self.BCE(self.temporal_discriminator(real_frame_sequences), real_labels)
        fake_loss = self.BCE(self.temporal_discriminator(fake_frame_sequences), fake_labels)

        # Total loss
        total_loss = real_loss + fake_loss
        return total_loss
    
    def temporal_generator_loss(self, fake_frames, sequence_length):
        # The generator tries to fool the discriminator, so the labels are 1 (real)
        batch_size = fake_frames.size(0)
        new_batch_size = batch_size // sequence_length
        labels = torch.ones(new_batch_size, 1).cuda()

        fake_frame_sequences = fake_frames.view(new_batch_size, sequence_length, fake_frames.size(1), fake_frames.size(2), fake_frames.size(3)).cuda()

        fake_frame_sequences = fake_frame_sequences.permute(0, 2, 1, 3, 4)
        # Compute the generator loss
        loss = self.BCE(self.temporal_discriminator(fake_frame_sequences), labels)
        return loss


    def render_frame(self, action, prev_frame, prev_hidden, prev_memory, noise, prev_read_weight, prev_write_weight, prev_controller_hidden=None):
        # Generate the hidden state and memory
        output, hidden = self.dynamics_engine(prev_frame, action, noise, prev_memory, prev_hidden)

        read_vector, read_weight, write_weight, new_controller_hidden = self.memory(action, hidden, prev_read_weight, prev_write_weight, prev_controller_hidden)

        frame = self.rendering_engine(output, read_vector)

        return frame, hidden, read_vector, read_weight, write_weight, new_controller_hidden


    def render_frames_batch(self, actions, og_prev_frames, og_prev_hidden, og_prev_memory, noise, og_prev_read_weights, og_prev_write_weights, game_overs):
        seq_len = actions.size(0)
        fake_frames = []

        prev_frames = og_prev_frames
        prev_hidden = og_prev_hidden
        prev_memory = og_prev_memory
        prev_read_weights = og_prev_read_weights
        prev_write_weights = og_prev_write_weights
        prev_controller_hidden = None


        # Process each frame in the sequence
        for t in range(seq_len):
            # Extract the action for the current time step
            current_action = actions[t]
            current_noise = noise[t]
            # Process the frame for the entire batch
            frame, hidden, read_vector, read_weight, write_weight, new_controller_hidden = self.render_frame(
                current_action, 
                prev_frames[t, :, :, :], 
                prev_hidden, 
                prev_memory, 
                current_noise, 
                prev_read_weights, 
                prev_write_weights,
                prev_controller_hidden
            )
            fake_frames.append(frame)

            # Update the previous states for the next iteration
            prev_hidden = hidden
            ## NOTE THIS IS NOT EXACTLY WHAT TO DO, INSTEAD INSERT THE READ VECTOR
            prev_memory = read_vector
            prev_read_weights = read_weight
            prev_write_weights = write_weight
            prev_controller_hidden = new_controller_hidden
            if(game_overs[t] == 1):
                prev_hidden = og_prev_hidden
                prev_memory = og_prev_memory
                prev_read_weights = og_prev_read_weights
                prev_write_weights = og_prev_write_weights
                prev_controller_hidden = None

        # Combine frames into a tensor
        fake_frames = torch.stack(fake_frames, dim=0)
        fake_frames = fake_frames.squeeze(1)

        return fake_frames
    
    def save(self, path):
        torch.save({
            'dynamics_engine': self.dynamics_engine.state_dict(),
            'memory': self.memory.state_dict(),
            'rendering_engine': self.rendering_engine.state_dict(),
            'single_image_discriminator': self.single_image_discriminator.state_dict(),
            'temporal_discriminator': self.temporal_discriminator.state_dict(),
            'action_conditioned_discriminator': self.action_conditioned_discriminator.state_dict(),
            'game_over_discriminator': self.game_over_discriminator.state_dict()
        }, path)

    def useModel(self, path, dataset):
        checkpoint = torch.load(path)
        self.dynamics_engine.load_state_dict(checkpoint['dynamics_engine'])
        self.memory.load_state_dict(checkpoint['memory'])
        self.rendering_engine.load_state_dict(checkpoint['rendering_engine'])
        self.single_image_discriminator.load_state_dict(checkpoint['single_image_discriminator'])
        self.temporal_discriminator.load_state_dict(checkpoint['temporal_discriminator'])
        self.action_conditioned_discriminator.load_state_dict(checkpoint['action_conditioned_discriminator'])
        self.game_over_discriminator.load_state_dict(checkpoint['game_over_discriminator'])
        self.dynamics_engine.eval()
        self.memory.eval()
        self.rendering_engine.eval()
        self.single_image_discriminator.eval()
        self.temporal_discriminator.eval()
        self.action_conditioned_discriminator.eval()
        self.game_over_discriminator.eval()
        prev_hidden = self.dynamics_engine.init_hidden(1).detach().cuda()
        self.memory.reset_memory()
        prev_memory = torch.zeros(1, self.memory_unit_size).detach().cuda()
        prev_read_weights = torch.zeros(self.memory_units, 1).detach().cuda()
        prev_write_weights = torch.zeros(self.memory_units, 1).detach().cuda()
        new_controller_hidden = None

        # Test the model
        testing = True
        image = dataset.get_random_image()
        dataset.display_image(image, 3)
        print(image)
        while testing:
            print("Press 'q' to quit, 1 to jump, 0 to do nothing")
            action = input()
            if action == 'q':
                testing = False
            else:
                action = torch.tensor([int(action)], dtype=torch.float32).cuda()
                noise = torch.randn(1, self.noise_size).cuda()
                image, hidden, read_vector, read_weight, write_weight, new_controller_hidden = self.render_frame(action, image, prev_hidden, prev_memory, noise, prev_read_weights, prev_write_weights, new_controller_hidden)
                image = image.squeeze(0).detach().cpu()
                dataset.display_image(image, 3)
                prev_hidden = hidden
                prev_memory = read_vector
                prev_read_weights = read_weight
                prev_write_weights = write_weight
                gOver = self.game_over_discriminator(image)
                print("Game Over: ", gOver.item())

