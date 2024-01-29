import numpy as np
from PIL import Image
import random

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

current_frames, actions, next_frames, game_overs = load_and_preprocess_data('compressed_frame_data.npz')


def open_random_frame_as_image(current_frames):
    # Choose a random frame
    random_frame = random.choice(current_frames)

    # Correct the orientation
    # Swap the axes if necessary
    corrected_frame = np.swapaxes(random_frame, 0, 1)

    # Convert the frame from a NumPy array to a PIL image
    image = Image.fromarray(corrected_frame.astype('uint8'), 'RGB')

    # Show the image
    image.show()

# Assuming current_frames is loaded from your previous function
open_random_frame_as_image(current_frames)