import numpy as np


class Settings:
    def __init__(self):
        # the settings of color, size, speed
        self.screen_width = 400
        self.screen_height = 400
        self.col = 13
        self.row = 13
        self.bg_color = (200, 200, 200)
        self.snake_color = (100, 100, 100)
        self.head_color = (0, 0, 0)
        self.food_color = (255, 245, 225)
        self.cell_color = (0, 180, 100)
        self.snake_speed = 1
        self.cell_width = self.screen_width / self.col
        self.cell_height = self.screen_height / self.row
        self.screen_size = (self.screen_width, self.screen_height)


class Point:
    row = 0
    col = 0
    color = (0, 0, 0)

    def __init__(self, row=0, col=0, color=(0, 0, 0)):
        self.row = row
        self.col = col
        self.color = color

    def __eq__(self, rhs):
        return self.row == rhs.row and self.col == rhs.col

    def copy(self):
        return Point(self.row, self.col, self.color)


def distance(p1, p2):
    return np.abs(p1.row - p2.row) + np.abs(p1.col - p2.col)


class Snake_norender:
    settings = Settings()
    direct_col = [0, -1, 1, 0]
    direct_row = [-1, 0, 0, 1]
    # up left right down
    direct_id = 0
    bodys = []
    score_point = 0
    game_going = False
    screen = None
    is_render = True
    silent = True
    history_act = [0, 0, 0, 0]
    vis = np.zeros((settings.row, settings.col))
    mode = '1d'
    tic = 0
    lst_eat = 0
    visual_dis = 1

    def __init__(self, visual_dis=1) -> None:
        self.visual_dis = visual_dis
        self.reset()

    def setvis(self, p: Point, t=1):
        self.vis[p.row-1][p.col-1] = t

    def get_sn(self, p):
        dis = self.visual_dis
        x = p.row-1
        y = p.col-1
        sn = []
        for j in range(-dis,dis+1):
            for i in range(-dis,dis+1):
                if (i==0 and j==0) or (abs(i)+abs(j)>dis):
                    continue
                tx = x+i
                ty = y+j
                if 0<=tx and tx<self.settings.row and 0<=ty and ty<self.settings.col:
                    sn.append(self.vis[tx][ty])
                else:
                    sn.append(1)
        # s1 = self.vis[x-1][y] if x >= 1 else 1.0
        # s2 = self.vis[x+1][y] if x < self.settings.row-1 else 1.0
        # s3 = self.vis[x][y-1] if y >= 1 else 1.0
        # s4 = self.vis[x][y+1] if y < self.settings.col-1 else 1.0
        # print('s:',sn,[s1, s2, s3, s4])
        # return [s1, s2, s3, s4]
        return sn

    def get_body_food(self):
        return [self.bodys[0].row/self.settings.row, self.bodys[0].col/self.settings.col,
                self.food.row/self.settings.row-self.bodys[0].row/self.settings.row, self.food.col/self.settings.col-self.bodys[0].col/self.settings.col]

    def reset(self):
        self.lst_eat = 0
        self.tic = 0
        self.bodys = [Point(self.settings.row // 2, self.settings.col //
                            2, self.settings.snake_color)]
        self.vis = np.zeros((self.settings.row, self.settings.col))
        self.setvis(self.bodys[0])
        self.food = self.create_food()
        self.score_point = 0
        return self.get_s(mode=self.mode), 0, 0, 0, {'score': self.score_point}

    def draw_cell(self):
        pass

    def draw_rect(self, point, color=None):
        pass

    def create_food(self):
        if len(self.bodys) < 0:
            while True:
                new_food = Point(np.random.randint(0, self.settings.row - 1),
                                 np.random.randint(0, self.settings.col - 1), self.settings.food_color)
                is_coll = False
                for body in self.bodys:
                    if body == new_food:
                        is_coll = True
                        break
                if not is_coll:
                    return new_food
        else:
            points = np.ones(self.settings.row*self.settings.col)
            for body in self.bodys:
                points[body.row*self.settings.col+body.col] = 0
            points = np.nonzero(points)[0]
            id = np.random.randint(len(points))
            ij = points[id]
            i = ij//self.settings.col
            j = ij % self.settings.col
            return Point(i, j, self.settings.food_color)

    def get_lazy(self):
        # print('lazy:', (self.tic - self.lst_eat)/(0.5*self.settings.col*self.settings.row))
        return (self.tic - self.lst_eat)/(self.settings.col*self.settings.row)

    def step(self, action):
        rew = 0
        self.tic += 1

        if action == 3-self.direct_id:
            return rew, 1, self.get_s(mode=self.mode), {'score': self.score_point}

        self.history_act.pop(0)
        self.history_act.append(action)

        self.direct_id = action

        self.bodys.insert(0, self.bodys[0].copy())

        dis_old = distance(self.bodys[0], self.food)

        coll = 0
        # snake move
        self.bodys[0].row += self.direct_row[self.direct_id]
        self.bodys[0].col += self.direct_col[self.direct_id]
        self.setvis(self.bodys[0])

        if self.bodys[0].col < 0:
            # self.bodys[0].col = self.settings.col - 1
            coll = 1
        if self.bodys[0].col > self.settings.col - 1:
            # self.bodys[0].col = 0
            coll = 1
        if self.bodys[0].row < 0:
            # self.bodys[0].row = self.settings.row - 1
            coll = 1
        if self.bodys[0].row > self.settings.row - 1:
            # self.bodys[0].row = 0
            coll = 1

        eat = (self.bodys[0] == self.food)
        if eat:
            self.food = self.create_food()
            self.score_point += 1
            rew += 1.2
            self.lst_eat = self.tic
        if not eat:
            self.setvis(self.bodys[-1], 0)
            self.bodys.pop()

            dis_new = distance(self.bodys[0], self.food)
            if dis_new > dis_old:
                rew -= 0.3
                pass
            else:
                rew += 0.1

        # judge coll
        for body in self.bodys[1:]:
            if self.bodys[0] == body:
                coll = 1
                break

        # avoid loop
        if self.get_lazy() > 1:
            coll = 1

        if coll:
            self.game_going = False
            if not self.silent:
                print("You die, and please press space to restart")
            rew -= 2.0
        return rew, coll, self.get_s(mode=self.mode), {'score': self.score_point}

    def runner(self):
        self.step(self.direct_id)

    def get_s(self, mode='1d'):
        if mode == '1d':
            return [*self.get_body_food(), *self.get_sn(self.bodys[0]), *self.history_act, self.get_lazy()]
        elif mode == '2d':
            ret = np.zeros((self.settings.row, self.settings.col))
            p = 1
            for body in self.bodys:
                ret[body.row-1, body.col-1] = p
                p *= 0.98
            ret[self.food.row-1, self.food.col-1] = -1
            return ret[np.newaxis, :, :]

    # draw the screen
    def render(self, caption='Snake'):
        pass

    def run_game(self):
        self.render()
        self.silent = False
        while True:
            self.runner()

    def random_action(self, act=-1):
        act_next = np.random.randint(3)
        if act_next >= 3-act:
            return act_next+1
        else:
            return act_next

