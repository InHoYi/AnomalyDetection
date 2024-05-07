import pygame

pygame.init()
pygame.mixer.init()

i = 10
while i > 0:
    pygame.mixer.Sound("/Users/inho/Programming/2023 연구/코드/beep-21.wav").play()
    i -= 1