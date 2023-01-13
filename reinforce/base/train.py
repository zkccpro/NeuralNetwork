from collections import deque
import random
import torch
import numpy as np
import time
from core import utility as util
from post_processing import draw
from post_processing import saver
from parse import configParser as cp


class Exprience:
    def __init__(self, stat, reward, action, nxt_stat):
        self.s = stat
        self.r = reward
        self.a = action
        self.ns = nxt_stat

    def __repr__(self):
        return f'[stat]: {self.s}, [rwd]: {self.r}, [act]: {self.a}, [nxt_stat]: {self.ns}'


class Expriences:
    def __init__(self, maxsize):
        self.maxsize = maxsize
        self.queue = deque(maxlen=maxsize)
    
    def get_n_rand(self, n):
        """
        Args: n, num of return exp item (Int)
        get n expriences in the pool randomly
        Returns: list of exp item (List)
        """
        # print(self.queue) # NO!!!
        sample_num = min(n, self.maxsize, len(self))
        return random.sample(self.queue, sample_num)

    # 为了这个你需要维护一个对奖励的有序结构
    def get_n_rand_greedy(self, n, psb):
        """
        Args: n, num of return exp item (Int); psb, possibility of getting random item
        random greedy strategy to get n item from pool, 
        There's [psb] possibilities to get random item, others will choose the best perfomance expriences
        Returns: list of exp item (List)
        """
        pass

    def put(self, exp):
        self.queue.append(exp)

    def __len__(self):
        return len(self.queue)


class Trainer:
    """
    NEED to be inherited
    train strategy of reinforcement learning
    """
    def __init__(self, agent, env, trainset, valset, loss_func):
        self.agent = agent
        self.env = env
        self.trainset = trainset
        self.valset = valset
        self.loss_func = loss_func

        self.train_epoch_mean_loss = dict()
        self.train_epoch_mean_mQ = 0
        self.train_epoch_mean_rwd = 0

        self.train_epoch_losses = dict()
        self.train_epoch_mQs = []
        self.train_epoch_rwds = []

        self.val_epoch_mQs = []
        self.val_epoch_rwds = []

        self.conf_parser = cp.ConfigParser()

    def train(self, max_epoch=100, max_step=-1, backup_steps=100, log_steps=1000):
        log = False
        # epoches
        for cur_epoch in range(max_epoch):
            epoch_steps = 0
            # episodes
            for cur_episode, stream in enumerate(self.trainset):
                self.env.reset(stream)
                # iterations (steps)
                if max_step < 1:
                    cur_step = 0
                    while(True):
                        ts = time.strftime('%Y-%m-%d %H:%M:%S')
                        if cur_step % backup_steps == 0:
                            self.agent.backup()
                        if cur_step % log_steps == 0:
                            # LOG
                            print(f'\n{ts} - INFO - Epoch[{cur_epoch + 1}/{max_epoch}] Episode[{cur_episode + 1}/{len(self.trainset)}] Step[{cur_step + 1}/{len(stream)}]:')
                            log = True
                        else:
                            log = False
                        if not self.iter(log):
                            break
                        cur_step += self.env.interval
                        epoch_steps += 1
                else:  # max_step >= 1
                    epoch_steps += max_step
                    for cur_step in range(max_step):
                        if cur_step % backup_steps == 0:
                            self.agent.backup()
                        if cur_step % log_steps == 0:
                            # LOG
                            print(f'\n{ts} - INFO - Epoch[{cur_epoch + 1}/{max_epoch}] Episode[{cur_episode + 1}/{len(self.trainset)}] Step[{cur_step + 1}/{len(stream)}]:')
                            log = True
                        else:
                            log = False
                        if not self.iter(log):
                            break
            print(f'\n----------Start validating in epoch {cur_epoch + 1}---------')
            self.validation(cur_epoch, max_step)
            self.agent.save_to(self.conf_parser.conf_dict['workdir']['checkpoint_dir'] + 'epoch_' + str(cur_epoch + 1))
            # saver.ModelSaver().to_disk(self.agent.obj_Q, self.conf_parser.conf_dict['workdir']['checkpoint_dir'], 'epoch_' + str(cur_epoch + 1))

            print('total_frames =', epoch_steps)
            for key in self.train_epoch_mean_loss:
                self.train_epoch_mean_loss[key] /= epoch_steps
                self.train_epoch_losses[key].append(self.train_epoch_mean_loss[key])
                self.train_epoch_mean_loss[key] = 0

            self.train_epoch_mean_mQ /= epoch_steps
            self.train_epoch_mQs.append(self.train_epoch_mean_mQ)
            self.train_epoch_mean_mQ = 0

            self.train_epoch_mean_rwd /= epoch_steps
            self.train_epoch_rwds.append(self.train_epoch_mean_rwd)
            self.train_epoch_mean_rwd = 0

        # print(f'train_epoch_losses: {self.train_epoch_losses}')
        # print(f'train_epoch_mQs: {self.train_epoch_mQs}')
        # print(f'train_epoch_rwds: {self.train_epoch_rwds}')
        # print(f'val_epoch_mQs: {self.val_epoch_mQs}')
        # print(f'val_epoch_rwds: {self.val_epoch_rwds}')

        csv_data = [
            ['train_epoch_losses', 'train_epoch_mQs', 'train_epoch_rwds', 'val_epoch_mQs', 'val_epoch_rwds'],
            self.train_epoch_losses,
            self.train_epoch_mQs,
            self.train_epoch_rwds,
            self.val_epoch_mQs,
            self.val_epoch_rwds,
        ]

        saver.DataSaver().to_csv(csv_data, self.conf_parser.conf_dict['workdir']['result_dir'], 'train_epoch_losses')
        for key in self.train_epoch_losses:
            draw.draw_1d(self.train_epoch_losses[key], xlabel='epoch', ylabel=key, name=key)
        draw.draw_1d(self.train_epoch_mQs, xlabel='epoch', ylabel='max Q', name="train_epoch_mQs")
        draw.draw_1d(self.train_epoch_rwds, xlabel='epoch', ylabel='reward', name="train_epoch_rwds")

        draw.draw_1d(self.val_epoch_mQs, xlabel='epoch', ylabel='max Q', name="val_epoch_mQs")
        draw.draw_1d(self.val_epoch_rwds, xlabel='epoch', ylabel='reward', name="val_epoch_rwds")

    def validation(self, cur_epoch, max_step):
        interval = self.env.interval
        self.env.interval = 1
        avg_rwd = 0
        avg_mQ = 0
        valset_stats = []
        for cur_episode, stream in enumerate(self.valset):
            stream_stats = []
            epoch_steps = 0
            print(f'inferencing val stream [{cur_episode + 1}/{len(self.valset)}]: ')
            self.env.reset(stream)
            format = util.ProgressFormat(len(stream), interval=100)
            format.start()
            if max_step < 1:
                cur_step = 0
                while(True):
                    format.count(cur_step)
                    exp = self.get_exp(False, eps=1)
                    if exp == None:
                        break
                    avg_rwd += exp.r
                    avg_mQ += np.max(exp.a.q_val)
                    stream_stats.append(exp.s.get_ev())
                    cur_step += 1
                epoch_steps += cur_step
            else:  # max_step >= 1
                epoch_steps += max_step
                for cur_step in range(max_step):
                    format.count(cur_step)
                    exp = self.get_exp(False, eps=1)
                    if exp == None:
                        break
                    avg_rwd += exp.r
            format.end()
            valset_stats.append(stream_stats)
        avg_rwd /= epoch_steps
        avg_mQ /= epoch_steps

        print(f'avg rwd of valset is: {avg_rwd}')
        print(f'avg mQ of valset is: {avg_mQ}')
        self.val_epoch_rwds.append(avg_rwd)
        self.val_epoch_mQs.append(avg_mQ)

        saver.DataSaver().to_csv(valset_stats, self.conf_parser.conf_dict['workdir']['epoches_dir'], f'valset_stats_epoch{cur_epoch + 1}')
        for i, stream_stats in enumerate(valset_stats):
            draw.draw_1d(stream_stats, xlabel='frames', ylabel='status(EV)', name=f'stream_stats_epoch{cur_epoch + 1}_{i}', output_dir='epoches_dir')

        self.env.interval = interval

    def iter(self, log):
        """
        Can be overrided
        Args: None
        complate one iteration of A episode
        users should implement it for different train strategies
        Returns: True, continue can be itered or False, game over (Bool)
        """
        exp = self.get_exp()
        if exp == None:
            return False
        self.update(exp)
        return True

    def update(self, exp, log):
        """
        Args: <si, ri, ai, si+1> (Exprience)
        update agent with ONE exp
        Returns: None
        """
        target = _cal_target(exp)
        output = exp.a.to_tensor()
        loss = _cal_loss(output, target)
        self.agent.update(loss)

    def _cal_target(self, exp):
        """
        Need to be overrided
        Args: <si, ri, ai, si+1> (Exprience)
        according ONE exp to calculate the target
        Returns: target(Tensor)
        """
        pass

    def _cal_loss(self, output, target):
        """
        Args: output(float), target(Action)
        according to output and target calculate the loss to backward
        Returns: loss (Tensor)
        """
        return self.loss_func(output, target)

    def get_exp(self):
        """
        Can be overrided
        Args: None
        interact with env and get exp
        if next status is None, return None
        Returns: exp(Experence) or None
        """
        stat = self.env.get_stat()
        act = self.agent.decision(stat)
        rwd, nxt_stat = self.env.step(act)
        if nxt_stat == None:
            return None
        return Exprience(stat, rwd, act, nxt_stat)
