from collections import deque
import random
import torch
import numpy as np

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
    def __init__(self, agent, env, streamset, loss_func):
        self.agent = agent
        self.env = env
        self.streamset = streamset
        self.loss_func = loss_func
        
    def train(self, max_epoch=100, max_step=-1, backup_steps=100, log_steps=1000):
        log = False
        # epoches
        for cur_epoch in range(max_epoch):
            # episodes
            for cur_episode, stream in enumerate(self.streamset):
                self.env.reset(stream)
                # iterations (steps)
                if max_step < 1:
                    cur_step = 0
                    while(True):
                        if cur_step % backup_steps == 0:
                            self.agent.backup()
                        if cur_step % log_steps == 0:
                            # LOG
                            print(f'\nINFO: Epoch[{cur_epoch}/{max_epoch}] Episode[{cur_episode}/{len(self.streamset)}] Step[{cur_step}/{len(stream)}]:')
                            log = True
                        else:
                            log = False
                        if not self.iter(log):
                            break
                        cur_step += 1
                else:
                    for cur_step in range(max_step):
                        if cur_step % backup_steps == 0:
                            self.agent.backup()
                        if cur_step % log_steps == 0:
                            # LOG
                            print(f'\nINFO: Epoch[{cur_epoch}/{max_epoch}] Episode[{cur_episode}/{len(self.streamset)}] Step[{cur_step}/{len(stream)}]:')
                            log = True
                        else:
                            log = False
                        if not self.iter(log):
                            break

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

    def get_exp(self, greedy_prob, log):
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