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
        return self.current_frames[startingIndex:startingIndex+batch_number], self.actions[startingIndex:startingIndex+batch_number], self.next_frames[startingIndex:startingIndex+batch_number], self.game_overs[startingIndex:startingIndex+batch_number]

        

        

# Assuming current_frames is loaded from your previous function
dataset = Dataset('compressed_frame_data.npz')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


# def open_random_frame_as_image(current_frames):
#     # Choose a random frame
#     random_frame = random.choice(current_frames)

#     # Correct the orientation
#     # Swap the axes if necessary
#     corrected_frame = np.swapaxes(random_frame, 0, 1)

#     # Convert the frame from a NumPy array to a PIL image
#     image = Image.fromarray(corrected_frame.astype('uint8'), 'RGB')

#     # Show the image
#     image.show()

# # Assuming current_frames is loaded from your previous function
# open_random_frame_as_image(current_frames)