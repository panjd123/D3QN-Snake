from pygame.locals import *
import pygame
import numpy as np
from snake_norender import *

class Snake(Snake_norender):
    key2id = {K_LEFT: 1, K_RIGHT: 2, K_UP: 0,
              K_DOWN: 3, K_a: 1, K_d: 2, K_w: 0, K_s: 3}

    def __init__(self, visual_dis=1) -> None:
        super().__init__(visual_dis)
        pygame.init()
        self.screen = pygame.display.set_mode(self.settings.screen_size)
        pygame.display.set_caption("Snake")

    def draw_cell(self):
        for r in range(self.settings.row):
            pygame.draw.line(self.screen, self.settings.cell_color,
                             (0, r * self.settings.cell_height), (self.settings.screen_width, r * self.settings.cell_height))
        for c in range(self.settings.col):
            pygame.draw.line(self.screen, self.settings.cell_color,
                             (c * self.settings.cell_width, 0), (c * self.settings.cell_width, self.settings.screen_height))

    def draw_rect(self, point, color=None):
        left = point.col * self.settings.cell_width
        top = point.row * self.settings.cell_height
        if not color:
            pygame.draw.rect(self.screen, point.color, (left, top,
                                                        self.settings.cell_width, self.settings.cell_height))
        else:
            pygame.draw.rect(
                self.screen, color, (left, top, self.settings.cell_width, self.settings.cell_height))

    def runner(self):
        flag_get_key = False

        if self.is_render:
            clock = pygame.time.Clock()

            if not self.game_going:
                for event in pygame.event.get():
                    if event.type == KEYDOWN:
                        if event.key == K_SPACE:
                            self.reset()
                            self.game_going = True
                    elif event.type == QUIT:
                        exit(0)
                return

            # get direction
            for event in pygame.event.get():
                if event.type == QUIT:
                    exit(0)
                elif event.type == KEYDOWN and not flag_get_key:
                    flag_get_key = True
                    self.step(self.key2id[event.key])

        if not flag_get_key:
            self.step(self.direct_id)

        if self.is_render:
            self.render()
            clock.tick(10 * self.settings.snake_speed)

    # draw the screen
    def render(self,caption='Snake'):
        self.screen.fill(self.settings.bg_color)
        for body in self.bodys[1:]:
            self.draw_rect(body)
        self.draw_rect(self.bodys[0], self.settings.head_color)
        self.draw_rect(self.food)
        self.draw_cell()
        for event in pygame.event.get():
            pass
        pygame.display.set_caption(caption)
        pygame.display.flip()

if __name__ == '__main__':
    snake = Snake()
    snake.run_game()
