import numpy as np
import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from model import DuelingNetwork
from buffer import Buffer
from snake import Snake, Snake_norender
from tqdm import tqdm
import time
import os
import os.path as osp
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from collections import deque
import pygame

parser = ArgumentParser()
parser.add_argument("--step", type=int, default=1000,
                    help="the number of step it will train")
parser.add_argument("--history", type=int, default=0,
                    help="after HISTORY generations, save model every 1000 generations")
parser.add_argument("--render", type=int, default=2,
                    help="render level while training, 0: no render, 1: render once after die, 2[default]: render every step")
parser.add_argument("--train", action="store_true", help="only train")
parser.add_argument("--play", action="store_true", help="only play")
parser.add_argument("--visual", type=int, default=2,
                    help="the manhattan distance that snakes can see, note that this argument will affect the model's parameter size, if you plan to load a model, pay attention to the corresponding.")
parser.add_argument("--model_load", type=str,
                    default="model.pkl", help="the path of the loading model")
parser.add_argument("--model_save", type=str,
                    default="model.pkl", help="the model's output path")
parser.add_argument("--test", type=str, default="")
parser.add_argument("--epsilon", type=float, default=0.95,
                    help="probability of using random movement during training")
parser.add_argument("--log", action="store_true",
                    help="output log while training")

argument = parser.parse_args()
history_dir = 'history-'+time.strftime('%Y-%m-%d-%H-%M-%S')

if not argument.play and not argument.test:
    os.mkdir(history_dir)


class DDQN:
    def __init__(self, input_shape, num_act, env: Snake, gamma=0.99, lamda=0.05, epsilon=0.95) -> None:
        self.gamma = gamma
        self.lamda = lamda
        self.model = DuelingNetwork(input_shape, num_act)
        self.target_model = DuelingNetwork(input_shape, num_act)

        # continue
        # self.model.load_state_dict(torch.load('model40w.pkl'))

        # self.target_model.load_state_dict(self.model.state_dict())
        self.train_after = 0
        self.expl_before = 0
        self.buffer = Buffer(capcity=1)
        self.env = env
        self.epsilon = epsilon
        self.batch_size = 1
        self.log = []
        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = 'cpu'
        self.model.to(self.device)
        self.target_model.to(self.device)
        self.model_optim = optim.Adam(self.model.parameters(), lr=0.0001)

    def learning(self):
        obs, act, rew, done, obs_next = self.buffer.sample(self.batch_size)
        # 2d
        obs = torch.tensor(obs, dtype=torch.float).to(self.device)
        act = torch.tensor(act, dtype=torch.long).to(self.device)
        rew = torch.tensor(rew, dtype=torch.float).to(self.device)
        done = torch.tensor(done, dtype=torch.float).to(self.device)
        obs_next = torch.tensor(obs_next, dtype=torch.float).to(self.device)

        q_value = self.model(obs)
        q_value = q_value[:, act]

        with torch.no_grad():
            q_next_value = self.model(obs_next)
            q_value_max_arg = torch.argmax(q_next_value, dim=1)
            q_next_value = self.target_model(obs_next)
            q_next_value = q_next_value[:, q_value_max_arg]

        expected_q_value = rew + (1 - done) * self.gamma * q_next_value

        loss = F.mse_loss(q_value, expected_q_value)
        self.model_optim.zero_grad()
        loss.backward()
        self.model_optim.step()

        for model_param, target_param in zip(self.model.parameters(), self.target_model.parameters()):
            target_param.data.copy_(
                (1-self.lamda)*target_param.data + self.lamda*model_param.data)

        return loss.item()

    def training(self, max_step=1000, detail_render=False, is_log=True, queue_maxlen=50):
        try:
            epoch = 0
            writer = SummaryWriter()
            total_reward = 0
            obs, act, rew, done, info = self.env.reset()
            scores = deque(maxlen=queue_maxlen)
            rewards = deque(maxlen=queue_maxlen)
            for step in tqdm(range(max_step)):
                if step > self.expl_before:
                    act = self.select_action(obs, act)
                else:
                    act = self.env.random_action(act)

                rew, done, obs_next, info = self.env.step(act)

                if detail_render:
                    self.env.render()

                self.buffer.add(obs, act, rew, done, obs_next)

                if step > self.train_after:
                    loss = self.learning()
                    writer.add_scalar('loss', loss, step)

                total_reward += rew

                if done:
                    writer.add_scalars(
                        'score', {'reward': total_reward, 'score': info['score']}, epoch)

                    if is_log:
                        rewards.append(total_reward)
                        scores.append(info['score'])
                        mean_reward = np.mean(list(rewards))
                        mean_score = np.mean(list(scores))
                        tqdm.write(f'{epoch}: '+str(mean_reward) +
                                   ", "+str(mean_score))

                    self.env.render(f"第{epoch}代小蛇")

                    obs, act, rew, done, info = self.env.reset()
                    epoch += 1
                    if epoch > argument.history and epoch % 1000 == 0:
                        torch.save(self.model.state_dict(), osp.join(
                            history_dir, f"model_{epoch}.pkl"))
                    total_reward = 0
                else:
                    obs = obs_next
            writer.close()
            torch.save(self.model.state_dict(), osp.join(
                history_dir, f"model_{epoch}.pkl"))
        except KeyboardInterrupt:
            torch.save(self.model.state_dict(), 'model_interrupt.pkl')
            torch.save(self.target_model.state_dict(),
                       'target_model_interrupt.pkl')
            writer.close()
            raise KeyboardInterrupt

    def select_action(self, obs, act):
        if np.random.rand() > self.epsilon:
            return self.env.random_action(act)
        q_value = self.model(torch.tensor(
            obs, dtype=torch.float, device=self.device).unsqueeze(0))
        q_value_max_arg = torch.argmax(q_value, dim=1)
        # if q_value_max_arg.item() == 3-act:
        #     return self.env.random_action(act)
        # else:
        #     return q_value_max_arg.item()
        return q_value_max_arg.item()

    def play(self, max_epoch=100, delay=30, is_render=True):
        self.env.render('Snake')
        self.epsilon = 1.0
        rewards = []
        scores = []
        for epoch in range(max_epoch):
            total_reward = 0
            obs, act, rew, done, info = self.env.reset()
            while not done:
                act = self.select_action(obs, act)
                rew, done, obs, info = self.env.step(act)
                if is_render:
                    self.env.render()
                    pygame.time.delay(delay)
                total_reward += rew
                # print('total_reward:', total_reward, 'obs:', obs)
            rewards.append(total_reward)
            scores.append(info['score'])
            self.env.render()
        print('mean reward:', np.mean(rewards), 'mean score:', np.mean(scores))
        return rewards, scores


def test(ddqn: DDQN, dir='history-2022-12-18-03-28-52', epoch=100):
    print("testing", dir)
    gens = []
    scores = []
    for root, dirs, files in os.walk(dir):
        files = sorted(files, key=lambda x: int(x[6:-4]))
        for name in tqdm(files):
            ddqn.model.load_state_dict(torch.load(osp.join(root, name)))
            rews, scos = ddqn.play(max_epoch=epoch, delay=0)
            scores.append(np.mean(scos))
            gens.append(name[6:-4])
            tqdm.write(name+"\t"+str(scores[-1]))
    gens = np.array(gens)
    scores = np.array(scores)
    plt.plot(scores)
    plt.show()


if __name__ == '__main__':
    if argument.render == 0:
        env_train = Snake_norender(visual_dis=argument.visual)
    elif argument.render >= 1:
        env_train = Snake(visual_dis=argument.visual)

    obs_length = len(env_train.reset()[0])
    print('obs_length=', obs_length)

    ddqn = DDQN((obs_length,), 4, env_train, epsilon=argument.epsilon)

    if argument.test:
        test(ddqn, argument.test)
        exit(0)

    if not argument.play:  # train
        # continue to train
        try:
            ddqn.model.load_state_dict(torch.load(argument.model_load))
        except Exception as e:
            print(e)
            print('Warning: loading model fail, use initialization parameters')

        try:
            ddqn.model.load_state_dict(torch.load('target_model.pkl'))
        except Exception as e:
            print(e)
            print('Warning: loading target model fail, use initialization parameters')
        ddqn.training(max_step=argument.step,
                      detail_render=argument.render == 2, is_log=argument.log)
        torch.save(ddqn.model.state_dict(), argument.model_save)
        torch.save(ddqn.target_model.state_dict(), 'target_model.pkl')

    if not argument.train:  # play
        env_play = Snake(visual_dis=argument.visual)
        ddqn_play = DDQN((obs_length,), 4, env_train, epsilon=argument.epsilon)
        if not argument.play:  # after train
            try:
                ddqn_play.model.load_state_dict(
                    torch.load(argument.model_save))
            except Exception as e:
                print(e)
                print('Warning: loading model fail, use initialization parameters')
        else:  # without train
            try:
                ddqn_play.model.load_state_dict(
                    torch.load(argument.model_load))
            except Exception as e:
                print(e)
                print('Warning: loading model fail, use initialization parameters')
        ddqn_play.play(10)

# python ddqn.py --visual 2 --train --model_load history-2022-12-18-03-28-52/model_33300.pkl --step 1000000 --render 0 --epsilon 0.98 --log
# python ddqn.py --visual 2 --train --model_load model.pkl --step 1000000 --render 0 --epsilon 0.98 --log
# python ddqn.py --visual 2 --play --model_load model.pkl
