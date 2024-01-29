import pygame
import random
import DQN
import numpy as np
from collections import deque


from scipy.ndimage import zoom

import matplotlib.pyplot as plt

episodes = []
frames_survived = []

# TO CHANGE FOR MORE EFFICIENCY, CHANGE THE COMPRESSION FACTOR, MAKE IT ONLY SAVE ON INTERVALS OF FRAMES, AND MAKE IT SAVE TO A DIFFERENT FILE EVERY TIME
def compress_screen_data(screen_data, compression_factor=0.5):


    # Resize using scipy's zoom function
    # compressed_data = zoom(screen_data, (compression_factor, compression_factor, 1), order=0)

    return screen_data

# Initialize Pygame
pygame.init()

# Game Variables
screen_width = 400
screen_height = 600
def save_frame_data(frame_data, filename):
    # Convert the deque or list to a numpy array
    frame_data_array = np.array(frame_data, dtype=object)
    np.savez_compressed(filename, frame_data=frame_data_array)

agent = DQN.FlappyBirdAgent(6, 2, 100000, 0)

total_frames = 0

frame_data = deque(maxlen=5000)
epochs = 100000

breakOut = False
batch_size = 4000
total_score = 0
agent.load_model('./FlappyBirdGame/model.pt')
for epoch in range(epochs):
    if breakOut:
        break
    print(f'Epoch {epoch + 1}/{epochs}')
    if epoch % 100 == 0:
        print(f'Average Score: {total_score / 100}')
        total_score = 0

    bird_x = 50
    bird_y = 300
    bird_y_change = 0
    gravity = 1.4
    jump_height = 4
    game_over = False

    frame_count = 0
    # Pipe Variables
    pipe_width = 70  # Width of the pipes
    pipe_color = (0, 255, 0)  # Green color for pipes
    space_between_pipes = 200  # Space between the top and bottom pipes
    pipe_frequency = 1500  # Milliseconds between new pipes
    last_pipe = pygame.time.get_ticks()  # Time when the last pipe was created
    pipes = []  # List to store pipes

    # Set up the display
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption('Flappy Bird')
    bird_sprite = pygame.image.load('./FlappyBirdGame/bird.png').convert_alpha()
    bird_sprite = pygame.transform.scale(bird_sprite, (60, 60))

    # Clock for delta time


    export_bird_y = 0
    export_bird_y_change = 0
    export_pipes = []
    export_game_over = False


    game_overBig = False
    # Main game loop
    running = True
    score = 0
    passed_pipe = False
    pipe_id = 0 
    last_passed_pipe_id = -1

    while running:
        # current_screen_surface = pygame.display.get_surface()
        # current_screen_data = pygame.surfarray.array3d(current_screen_surface)
        # compressed_current_screen_data = compress_screen_data(current_screen_data)

        current_state = DQN.prepare_state(bird_y, bird_y_change,  bird_x, pipes, screen_width, screen_height)
        action = agent.select_action(current_state)
        if action == 1:
            bird_y_change = -jump_height
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    action = 1
                    bird_y_change = -jump_height 
                elif event.key == pygame.K_q:
                    breakOut = True
                    break

        frame_count += 1
        total_frames += 1
        # Bird mechanics
        bird_y_change += gravity * 1/20 # Delta time
        bird_y += bird_y_change

        # Game over condition (if the bird hits the ground or goes off screen)
        if bird_y > screen_height - 32 or bird_y < 0:
            game_over = True
            game_overBig = True


        if frame_count % 150 == 0:
            pipe_height_random = random.randint(100, screen_height - 250)
            bottom_pipe = pygame.Rect(screen_width, screen_height - pipe_height_random, pipe_width, pipe_height_random)
            top_pipe = pygame.Rect(screen_width, 0, pipe_width, screen_height - pipe_height_random - space_between_pipes)
            pipes.append((bottom_pipe, top_pipe, pipe_id))
            pipe_id += 1

        # Move Pipes
        for pipe in pipes:
            pipe[0].x -= 3  # Move the bottom pipe
            pipe[1].x -= 3  # Move the top pipe
        pipes = [pipe for pipe in pipes if pipe[0].x > -pipe_width]

        # Drawing
        screen.fill((255, 255, 255))  # White background
        # pygame.draw.circle(screen, (255, 200, 0), (bird_x, int(bird_y)), 16)  # Bird

        for pipe in pipes:
            pygame.draw.rect(screen, pipe_color, pipe[0])
            pygame.draw.rect(screen, pipe_color, pipe[1])

        bird_rect = pygame.Rect(bird_x, bird_y , 30, 30) 

        # Draw the rectangle (for debugging, you can comment this out later)
        # pygame.draw.rect(screen, (255, 0, 0), bird_rect)

        # Draw the bird sprite so that it aligns with the bird_rect
        bird_sprite_rect = bird_sprite.get_rect(center=bird_rect.center)
        screen.blit(bird_sprite, bird_sprite_rect)


        for pipe in pipes:
            if bird_rect.colliderect(pipe[0]) or bird_rect.colliderect(pipe[1]):
                game_over = True
                break
        for pipe in pipes:
            if pipe[0].x + pipe_width < bird_x and pipe[2] > last_passed_pipe_id:
                score += 1
                total_score += 1
                last_passed_pipe_id = pipe[2]
        # font = pygame.font.SysFont(None, 36)
        # score_text = font.render(f'Score: {score}', True, (0, 0, 0))
        # screen.blit(score_text, (10, 10))

        # Update the display
        pygame.display.update()


        export_bird_y = bird_y
        export_bird_y_change = bird_y_change
        export_pipes = pipes
        export_game_over = game_over
        reward = DQN.calculate_reward(export_bird_y, export_bird_y_change, export_pipes, score, export_game_over, game_overBig, frame_count)

        next_state = DQN.prepare_state(export_bird_y, export_bird_y_change, bird_x, export_pipes,screen_width, screen_height)
        # print(next_state)
        # print(f'pipes: {export_pipes}')
        # print(f'Current State: {current_state}')
        # print(f'Action: {action}')
        # print(f'Next State: {next_state}')
        agent.remember(current_state, action, next_state, reward, game_over)
        # agent.replay(batch_size = batch_size)
        
        # next_screen_surface = pygame.display.get_surface()
        # next_screen_data = pygame.surfarray.array3d(next_screen_surface)
        # compressed_next_screen_data = compress_screen_data(next_screen_data)

        # Store the pair of frames
        # frame_data.append((compressed_current_screen_data, action, compressed_next_screen_data, game_over))
        # # Check for Game Over
        # if (total_frames + 1) % 1000 == 0:
        #     save_frame_data(frame_data, 'compressed_frame_data.npz')
        #     frame_data.clear()
        if game_over:
            break
    agent.replay(batch_size = batch_size)


agent.save_model('model.pt')

# Quit Pygame
pygame.quit()
plt.show()