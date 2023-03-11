import pygame

pygame.init()

# Set up the game window
win = pygame.display.set_mode((500, 500))
pygame.display.set_caption("My Game")

# Game loop
run = True
while run:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

    # Draw the game elements
    win.fill((255, 255, 255))
    pygame.draw.rect(win, (0, 0, 255), (50, 50, 50, 50))

    # Update the game window
    pygame.display.update()

pygame.quit()

