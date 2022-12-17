# DDQN-Snake

一个用强化学习实现的贪吃蛇AI

使用了包括经验回放，双Q学习，对决网络等技巧 [强化学习 DQN 速成](https://blog.csdn.net/qq_32461955/article/details/126040912)

项目包含了一个在CPU上训练了2个小时的模型参数

## 使用方法

### 依赖库

- `numpy`
- `pygame`
- `pytorch`

主程序是目录下的 `ddqn.py`，直接运行即可，进一步使用方法请见 `python ddqn.py -h`

```cmd
usage: ddqn.py [-h] [--step STEP] [--history HISTORY] [--norender] [--train] [--play]

optional arguments:
  -h, --help         show this help message and exit
  --step STEP        the number of step it will train
  --history HISTORY  after HISTORY generations, save model every 1000 generations
  --norender         no render while training
  --train            only train
  --play             only play
```

如果你想手动玩贪吃蛇，那么直接运行 `snake.py`，这只需要安装 `pygame`

## 训练

通常，这个模型会在几分钟内开始步入正轨，在半小时到一小时左右达到比较好的水平。而这大概需要一百万个训练步数。

## 模型的输入

- 两组归一化后的坐标表示蛇头的坐标和蛇头相对食物的坐标
- 四个布尔值表示四周是否有障碍物，如墙或者身体
- 四个整数记录最近四次移动，相当于告诉蛇的前四节身体的位置
- 一个浮点数，表示上次吃食物的间隔与地图面积一半的比值，用于让它学习避免死循环

> 具体模型的架构可以见 [强化学习 DQN 速成](https://blog.csdn.net/qq_32461955/article/details/126040912)

# DDQN-Snake

A greedy snake AI implemented by reinforcement learning algorithm.

It uses such reinforcement learning skills as experience playback, double Q learning and dueling network.

The project has included a model trained on CPU for 2 hours

## Usage

### requirements

- numpy
- pygame
- pytorch

The main program is `ddqn.py`, just run it directly. For further usage, see `python ddqn.py -h`

```cmd
usage: ddqn.py [-h] [--step STEP] [--history HISTORY] [--norender] [--train] [--play]

optional arguments:
  -h, --help         show this help message and exit
  --step STEP        the number of step it will train
  --history HISTORY  after HISTORY generations, save model every 1000 generations
  --norender         no render while training
  --train            only train
  --play             only play
```

If you want to play Snake manually, run `snake.py` directly

## Training

Usually, this model will start to get on track in a few minutes, and reach a better level in about half an hour to one hour. It takes about one million training steps.

## Input of model

- The two groups of normalized coordinates represent the coordinates of snake head and the coordinates of snake head relative to food
- Four Boolean values indicate whether there are obstacles around, such as walls or bodies
-Four integers record the last four movements, which is equivalent to telling the snake the position of the first four segments
- A floating point number, which represents the ratio of the interval between eating food last time and half of the map area, and is used for learning to avoid an endless loop