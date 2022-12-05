# DoubleDQNAgent(Agent)、DoubleDQNTrainer(Trainer)
from ..base import Agent, Trainer, Exprience, Expriences
import torch
from torch import Tensor
from ..env import video
from conf import globalParam
from core import utility as util
import numpy as np
from post_processing import draw
from post_processing import saver


action_conf = dict(
    ev = 0,
    acts = [-0.6, -0.4, -0.2, 0, +0.2, +0.4, +0.6]
)


class DoubleDQNAgent(Agent):
    def __init__(self, est_network, obj_network, optimizer, model_path=''):
        self.est_Q = est_network
        self.obj_Q = obj_network
        if model_path != '':
            pretrain_model = torch.load(model_path)
            self.est_Q.load_state_dict(pretrain_model.state_dict())
            self.obj_Q.load_state_dict(pretrain_model.state_dict())
        self.est_Q_optimizer = self.gen_optimizer(self.est_Q, optimizer)

    def obj_decision(self, stat):
        """decision with object-Q net"""
        input = stat.to_tensor()
        output = self.inference(self.obj_Q, input)
        return video.EV_QAction(**action_conf).parse_from_tensor(output)

    def est_decision(self, stat, eps=1):
        """decision with estimate-Q net"""
        input = stat.to_tensor()
        output = self.inference(self.est_Q, input)
        return video.EV_QAction(**action_conf).parse_from_tensor(output, eps=eps)

    def update(self, loss):
        """update the estimate-Q net"""
        assert self.est_Q != None and self.est_Q_optimizer != None
        # backward
        self.est_Q.train()
        self.est_Q_optimizer.zero_grad()
        loss.backward() # retain_graph=True
        self.est_Q_optimizer.step()

    def backup(self):
        self.obj_Q.load_state_dict(self.est_Q.state_dict())


class DoubleDQNTrainer(Trainer):
    def __init__(self, agent, env, trainset, valset,
                loss_func, batch_size, exp_pool_size,
                gamma=0.9, eps=0.8, eps_scheduler=None):
        super().__init__(agent, env, trainset, valset, loss_func)
        self.batch_size = batch_size
        self.exprience_pool = Expriences(exp_pool_size)
        self.gamma = gamma
        self.eps = eps
        self.eps_scheduler = eps_scheduler

    def iter(self, log):
        """
        Args: None
        complate one iteration of A episode
        implyment expriment pool strategy
        Returns: True, continue can be itered or False, game over (Bool)
        """
        if self.eps_scheduler != None:
            self.eps = self.eps_scheduler(self.eps)
        exp = self.get_exp(log, self.eps)
        if exp == None:
            return False
        if log:
            print(f'cur_eps: {self.eps}')
            print(f'cur_exp: {exp}')
            self.train_epoch_mean_rwd += exp.r
        self.exprience_pool.put(exp)
        exps = self.exprience_pool.get_n_rand(self.batch_size)
        self.update(exps, log)
        return True

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

    def update(self, exps, log):
        """
        Args: list of <si, ri, ai, si+1> (list(Exprience))
        update agent with list of exps
        Returns: None
        """
        targets = self._cal_target(exps)  # torch.Size([batch])
        outputs = []
        for i, _ in enumerate(exps):
            cur_act = self.agent.est_decision(exps[i].s)
            cur_q = cur_act.to_tensor()[0][exps[i].a.act_idx]
            outputs.append(cur_q.unsqueeze(0))
        outputs = torch.cat(outputs, 0)  # torch.Size([batch])
        loss = self._cal_loss(outputs, targets.detach())  # torch.Size([batch])
        if log:
            loss_log = float(loss.mean())
            outputs_log = float(outputs.mean())
            print(f'cur_avg_loss: {loss_log}')
            print(f'cur_avg_mQ: {outputs_log}')
            self.train_epoch_mean_loss += loss_log
            self.train_epoch_mean_mQ += outputs_log
        self.agent.update(loss)

    def _cal_target(self, exps):
        """
        Args: list of <si, ri, ai, si+1> (list(Exprience))
        according list of exps to calculate the target use object Q-net
        yi = ri, if s(i+1) = None
        yi = ri + gamma * obj_Q(s(i+1), a), if s(i+1) != None
        Returns: targets(torch.Size([batch,7]))
        once only change one value(max of est-Qnet output index) of action tensor, others keep the same as est Qnet output
        """
        targets = []
        for exp in exps:
            a_est_max = self.agent.est_decision(exp.ns).act_idx  # idx of est-Qnet max Q value
            r = Tensor([exp.r]).to(globalParam.device)  # cur reward, torch.Size([1])
            if exp.ns == None:
                targets.append(r)
            else:
                # objQ(ns, a_est_max), torch.Size([1])
                objQ__ns__a_est_max = self.agent.obj_decision(exp.ns).to_tensor()[0][a_est_max].unsqueeze(0)
                target = r + self.gamma * objQ__ns__a_est_max
                targets.append(target)
        return torch.cat(targets, 0)  # torch.Size([batch])

    def get_exp(self, log, eps=0.8):
        """
        Can be overrided
        Args: None
        decision with estimate Q-net
        Returns: exp(Experence) or None
        """
        stat = self.env.get_stat()
        act = self.agent.est_decision(stat, eps=eps)
        rwd, nxt_stat = self.env.step(act, log)
        if nxt_stat == None:
            return None
        rwd = float(rwd.to('cpu'))
        act.q_val = act.tensor.detach().cpu().numpy()
        act.tensor = None
        return Exprience(stat, rwd, act, nxt_stat)
