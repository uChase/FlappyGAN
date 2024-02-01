import numpy as np
from PIL import Image
import random
import torch
from torch.utils.data import Dataset, DataLoader

def load_and_preprocess_data(filename):
    with np.load(filename, allow_pickle=True) as data:
        frame_data = data['frame_data']

    # Ensure that frame_data is a list of tuples
    if frame_data.ndim == 0:
        frame_data = frame_data.tolist()

    current_frames, actions, next_frames, game_overs = zip(*frame_data)

    # Additional preprocessing steps if needed
    # ...

    return current_frames, actions, next_frames, game_overs

class Dataset(Dataset):
    def __init__(self, filename):
        self.current_frames, self.actions, self.next_frames, self.game_overs = load_and_preprocess_data(filename)
        print("Data loaded")

    def __getitem__(self, index):
        current_frame = self.current_frames[index]
        action = self.actions[index]
        next_frame = self.next_frames[index]
        game_over = self.game_overs[index]

        return current_frame, action, next_frame, game_over

    def __len__(self):
        return len(self.current_frames)
    
    def get_random_sample(self, batch_number):
        startingIndex = random.randint(0, len(self.current_frames) - batch_number)

        current_frames_np = np.array([frame.astype(np.float32) / 255.0 for frame in self.current_frames[startingIndex:startingIndex + batch_number]])
        actions_np = np.array(self.actions[startingIndex:startingIndex + batch_number])
        next_frames_np = np.array([frame.astype(np.float32) / 255.0 for frame in self.next_frames[startingIndex:startingIndex + batch_number]])
        game_overs_np = np.array(self.game_overs[startingIndex:startingIndex + batch_number])

        # Convert to PyTorch tensors and adjust shape if necessary
        current_frames = torch.tensor(current_frames_np).permute(0, 3, 1, 2).cuda()  # Shape: [batch_size, C, H, W]
        actions = torch.tensor(actions_np, dtype=torch.float32).cuda()
        next_frames = torch.tensor(next_frames_np).permute(0, 3, 1, 2).cuda()  # Shape: [batch_size, C, H, W]
        game_overs = torch.tensor(game_overs_np, dtype=torch.float32).cuda()

        return current_frames, actions, next_frames, game_overs
    
    def get_random_image(self):
        # Select a random index
        random_index = random.randint(0, len(self.current_frames) - 1)

        # Convert the selected frame to a PyTorch tensor and adjust the shape
        frame_np = self.current_frames[random_index].astype(np.float32) / 255.0
        frame_tensor = torch.tensor(frame_np).permute(2, 0, 1).cuda()  # Shape: [C, H, W]

        return frame_tensor
    
    def display_image(self, image, scale_factor):
        print(image.shape)
        # Convert the PyTorch tensor to a NumPy array
        frame_np = image.permute(1, 2, 0).cpu().numpy()

        # Swap the axes if necessary
        frame_np = np.swapaxes(frame_np, 0, 1)

        # Convert the NumPy array to a PIL image
        frame_pil = Image.fromarray((frame_np * 255).astype(np.uint8))

        # Scale up the image
        width, height = frame_pil.size
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        scaled_image = frame_pil.resize((new_width, new_height))

        # Display the scaled image
        scaled_image.show()

    def open_random_frame_as_image(self):
        random_frame = random.choice(self.current_frames)

        # Correct the orientation
        # Swap the axes if necessary
        corrected_frame = np.swapaxes(random_frame, 0, 1)

        # Convert the frame from a NumPy array to a PIL image
        image = Image.fromarray(corrected_frame.astype('uint8'), 'RGB')

        # Show the image
        image.show()

        

        

# Assuming current_frames is loaded from your previous function
# dataset = Dataset('compressed_frame_data_training.npz')
# dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# dataset.get_random_sample(32)