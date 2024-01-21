import pygame
import random
import DQN

# Initialize Pygame
pygame.init()

# Game Variables
screen_width = 400
screen_height = 600


agent = DQN.FlappyBirdAgent(8, 2, 100000)

epochs = 10000
epsilon = 0.1
breakOut = False
batch_size = 64
for epoch in range(epochs):
    if breakOut:
        break
    print(f'Epoch {epoch + 1}/{epochs}')
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
    space_between_pipes = 150  # Space between the top and bottom pipes
    pipe_frequency = 1500  # Milliseconds between new pipes
    last_pipe = pygame.time.get_ticks()  # Time when the last pipe was created
    pipes = []  # List to store pipes

    # Set up the display
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption('Flappy Bird')

    # Clock for delta time


    export_bird_y = 0
    export_bird_y_change = 0
    export_pipes = []
    export_game_over = False


    # Main game loop
    running = True
    score = 0
    passed_pipe = False
    pipe_id = 0 
    last_passed_pipe_id = -1

    while running:

        current_state = DQN.prepare_state(bird_y, bird_y_change, pipes)
        action = agent.select_action(current_state, epsilon)
        epsilon *= 0.1
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
        # Bird mechanics
        bird_y_change += gravity * 1/20 # Delta time
        bird_y += bird_y_change

        # Game over condition (if the bird hits the ground or goes off screen)
        if bird_y > screen_height - 32 or bird_y < 0:
            game_over = True


        if frame_count % 150 == 0:
            pipe_height_random = random.randint(100, screen_height - 150)
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
        pygame.draw.circle(screen, (255, 200, 0), (bird_x, int(bird_y)), 16)  # Bird

        for pipe in pipes:
            pygame.draw.rect(screen, pipe_color, pipe[0])
            pygame.draw.rect(screen, pipe_color, pipe[1])

        bird_rect = pygame.Rect(bird_x - 16, bird_y - 16, 30, 30) 
        
        for pipe in pipes:
            if bird_rect.colliderect(pipe[0]) or bird_rect.colliderect(pipe[1]):
                game_over = True
                break
        for pipe in pipes:
            if pipe[0].x + pipe_width < bird_x and pipe[2] > last_passed_pipe_id:
                score += 1
                last_passed_pipe_id = pipe[2]
        font = pygame.font.SysFont(None, 36)
        score_text = font.render(f'Score: {score}', True, (0, 0, 0))
        screen.blit(score_text, (10, 10))

        # Update the display
        pygame.display.update()


        export_bird_y = bird_y
        export_bird_y_change = bird_y_change
        export_pipes = pipes
        export_game_over = game_over
        reward = frame_count / 1000 + ((score) * 10) if not game_over else -100

        next_state = DQN.prepare_state(export_bird_y, export_bird_y_change, export_pipes)
        # print(f'pipes: {export_pipes}')
        # print(f'Current State: {current_state}')
        # print(f'Action: {action}')
        # print(f'Next State: {next_state}')
        agent.remember(current_state, action, next_state, reward, game_over)

        agent.replay(batch_size = batch_size)
        # Check for Game Over
        if game_over:
            break

# Quit Pygame
pygame.quit()
