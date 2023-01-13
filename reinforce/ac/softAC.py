from ..base import Agent, Trainer, Exprience, Expriences
import torch
from torch import Tensor
from ..env import video
from conf import globalParam
import numpy as np
from post_processing import draw
from post_processing import saver


action_conf = dict(
    ev = 0,
    acts = [-0.6, -0.4, -0.2, 0, +0.2, +0.4, +0.6]
)


class SoftACAgent(Agent):
    def __init__(self, critic, target_critic, actor, 
                critic_optimizer, actor_optimizer, 
                critic_pretrain_path='', actor_pretrain_path=''):
        self.critic = critic
        self.target_critic = target_critic
        self.actor = actor
        if critic_pretrain_path != '':
            critic_pretrain = torch.load(critic_pretrain_path)
            self.critic.load_state_dict(critic_pretrain.state_dict())
            self.target_critic.load_state_dict(critic_pretrain.state_dict())
        if actor_pretrain_path != '':
            actor_pretrain = torch.load(actor_pretrain_path)
            self.actor.load_state_dict(actor_pretrain.state_dict())
        self.critic_optimizer = self.gen_optimizer(self.critic, critic_optimizer)
        self.actor_optimizer = self.gen_optimizer(self.actor, actor_optimizer)

    def actor_decision(self, stat, eps=1):
        input = stat.to_tensor()
        action, log_pi, mean = self.inference(self.actor, input)
        return video.EV_QAction(**action_conf).parse_from_tensor(action, eps=eps), log_pi, mean

    def critic_decision(self, stat):
        input = stat.to_tensor()
        min_action, q1, q2 = self.inference(self.critic, input)
        return video.EV_QAction(**action_conf).parse_from_tensor(min_action), q1, q2

    def target_critic_decision(self, stat):
        input = stat.to_tensor()
        min_action, q1, q2 = self.inference(self.target_critic, input)
        return video.EV_QAction(**action_conf).parse_from_tensor(min_action), q1, q2

    def update(self, optimizer, loss):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def backup(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def save_to(self, path):
        saver.ModelSaver().to_disk(self.actor, path, '_actor')
        saver.ModelSaver().to_disk(self.target_critic, path, '_critic')

class SoftACTrainer(Trainer):
    def __init__(self, agent, env, trainset, valset,
                loss_func, batch_size, exp_pool_size,
                alpha=0.8, gamma=0.9, action_num=len(action_conf["acts"]),
                eps=0.8, eps_scheduler=None, entropy_tuning=True, tuning_lr=0.01):
        super().__init__(agent, env, trainset, valset, loss_func)
        self.batch_size = batch_size
        self.exprience_pool = Expriences(exp_pool_size)
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.eps_scheduler = eps_scheduler
        self.entropy_tuning = entropy_tuning
        if entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor(action_num).to(globalParam.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=globalParam.device)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=tuning_lr)
        
        # log init
        self.train_epoch_mean_loss["critic loss"] = 0
        self.train_epoch_losses["critic loss"] = []
        self.train_epoch_mean_loss["actor loss"] = 0
        self.train_epoch_losses["actor loss"] = []

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

    def update(self, exps, log):
        log_pi_batch = []
        nxt_log_pi_batch = []
        min_q_pi_batch = []
        min_q_target_batch = []
        q1_batch = []
        q2_batch = []
        rwd_batch = []
        for i, _ in enumerate(exps):
            with torch.no_grad():
                nxt_action, nxt_log_pi, _ = self.agent.actor_decision(exps[i].ns)
                min_q_target, _, _ = self.agent.target_critic_decision(exps[i].ns)

                nxt_log_pi = nxt_log_pi.squeeze(1)
                nxt_log_pi_batch.append(nxt_log_pi)
                min_q_target_batch.append(min_q_target.to_tensor()[0][nxt_action.act_idx].unsqueeze(0) - self.alpha * nxt_log_pi)

                rwd_batch.append(Tensor([exps[i].r]).to(globalParam.device))

            action, log_pi, _ = self.agent.actor_decision(exps[i].s)
            min_q_pi, q1, q2 = self.agent.critic_decision(exps[i].s)

            log_pi_batch.append(log_pi.squeeze(1))

            min_q_pi_batch.append(min_q_pi.to_tensor()[0][action.act_idx].unsqueeze(0))

            q1_batch.append(q1[0][exps[i].a.act_idx].unsqueeze(0))
            q2_batch.append(q2[0][exps[i].a.act_idx].unsqueeze(0))

        log_pi_batch = torch.cat(log_pi_batch, 0)

        nxt_log_pi_batch = torch.cat(nxt_log_pi_batch, 0)

        min_q_pi_batch = torch.cat(min_q_pi_batch, 0)
        min_q_target_batch = torch.cat(min_q_target_batch, 0)

        q1_batch = torch.cat(q1_batch, 0)
        q2_batch = torch.cat(q2_batch, 0)

        rwd_batch = torch.cat(rwd_batch, 0)

        min_q_target_batch = min_q_target_batch - self.alpha * nxt_log_pi_batch
        nxt_q_value_batch = rwd_batch + self.gamma * min_q_target_batch
        critic_loss = self._cal_loss(q1_batch, nxt_q_value_batch) + self._cal_loss(q2_batch, nxt_q_value_batch)
        self.agent.update(self.agent.critic_optimizer, critic_loss)

        actor_loss = ((self.alpha * log_pi_batch) - min_q_pi_batch.detach()).mean()
        self.agent.update(self.agent.actor_optimizer, actor_loss)

        # print(q1_batch) # critic_decision_q1
        # print(q2_batch) # critic_decision_q2
        # print(nxt_q_value_batch) # actor_decision, no_grad

        # print(log_pi_batch) # actor_decision
        # print(min_q_pi_batch) # critic_decision_q1, critic_decision_q2, no_grad
        # print("\n")

        # automatic entropy tuning
        if self.entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi_batch + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = float(self.log_alpha.exp())

        if log:
            self.train_epoch_mean_loss["critic loss"] += float(critic_loss)
            self.train_epoch_mean_loss["actor loss"] += float(actor_loss)
            self.train_epoch_mean_mQ += float(min_q_pi_batch.mean())

    def get_exp(self, log, eps=0.8):
        stat = self.env.get_stat()
        act, _, _ = self.agent.actor_decision(stat, eps=eps)
        rwd, nxt_stat = self.env.step(act, log)
        if nxt_stat == None:
            return None
        rwd = float(rwd.to('cpu'))
        act.q_val = act.tensor.detach().cpu().numpy()
        act.tensor = None
        return Exprience(stat, rwd, act, nxt_stat)
