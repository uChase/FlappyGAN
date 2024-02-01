import dataProcess
import numpy as np
import torch
import Trainer

batch_size = 64
frame_channels = 3
frame_height = 50
frame_width = 75
action_space = 1
latent_dim = 10
hidden_dim = 64
output_size = 256
learning_rate = 0.0002
num_layers = 2
memory_units = 50
memory_unit_size = 75
controller_size = 128
output_channels = 3

# Load data
dataset = dataProcess.Dataset('compressed_frame_data_training.npz')
trainer = Trainer.GameGANTrainer(batch_size=batch_size, frame_channels=frame_channels, frame_height=frame_height, frame_width=frame_width, action_size=action_space, noise_size=latent_dim, hidden_size=hidden_dim, output_size=output_size, learning_rate=learning_rate, num_layers=num_layers, memory_units=memory_units, memory_unit_size=memory_unit_size, controller_size=controller_size, output_channels=output_channels)

# Train the model
# trainer.train(num_epochs=25, dataset=dataset, num_batches_per_epoch=100, sequence_length=batch_size)

# trainer.save('game_gan_model.pth')
trainer.useModel('game_gan_model.pth', dataset)