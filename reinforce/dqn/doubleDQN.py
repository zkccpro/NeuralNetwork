# DoubleDQNAgent(Agent)、DoubleDQNTrainer(Trainer)
from ..base import Agent, Trainer, Exprience, Expriences
import torch
from torch import Tensor
from ..env import video
from conf import globalParam


action_conf = dict(
    ev = 0,
    acts = [-0.6, -0.4, -0.2, 0, +0.2, +0.4, +0.6]
)


class DoubleDQNAgent(Agent):
    def __init__(self, network, optimizer):
        self.obj_Q = network
        self.est_Q = network
        self.est_Q_optimizer = self.gen_optimizer(self.est_Q, optimizer)

    def obj_decision(self, stat):
        """decision with object-Q net"""
        input = stat.to_tensor()
        output = self.inference(self.obj_Q, input)
        return video.EV_QAction(**action_conf).parse_from_tensor(output)
    
    def est_decision(self, stat):
        """decision with estimate-Q net"""
        input = stat.to_tensor()
        output = self.inference(self.est_Q, input)
        return video.EV_QAction(**action_conf).parse_from_tensor(output)
    
    def update(self, loss):
        """update the estimate-Q net"""
        assert self.est_Q != None and self.est_Q_optimizer != None
        # backward
        self.est_Q.train()
        self.est_Q_optimizer.zero_grad()
        loss.backward() # retain_graph=True
        self.est_Q_optimizer.step()
    
    def backup(self):
        self.obj_Q = self.est_Q
        print('INFO: backup estimate Qnet to object Qnet')


class DoubleDQNTrainer(Trainer):
    def __init__(self, agent, env, streamset, loss_func, batch_size, exp_pool_size, gamma):
        super().__init__(agent, env, streamset, loss_func)
        self.batch_size = batch_size
        self.exprience_pool = Expriences(exp_pool_size)
        self.gamma = gamma

    def iter(self):
        """
        Args: None
        complate one iteration of A episode
        implyment expriment pool strategy
        Returns: True, continue can be itered or False, game over (Bool)
        """
        exp = self.get_exp()
        if exp == None:
            return False
        # print(exp) # YES!!!
        self.exprience_pool.put(exp)
        exps = self.exprience_pool.get_n_rand(self.batch_size)
        self.update(exps)
        return True
    
    def update(self, exps):
        """
        Args: list of <si, ri, ai, si+1> (list(Exprience))
        update agent with list of exps
        Returns: None
        """
        targets = self._cal_target(exps)  # torch.Size([batch,7])
        outputs = []
        for i, _ in enumerate(exps):
            # print(exp.a.to_tensor())
            cur_act = self.agent.est_decision(exps[i].s)
            cur_q = cur_act.to_tensor()[0][cur_act.act_idx]
            outputs.append(cur_q.unsqueeze(0))
            # act = exps[i].a.to_tensor()
            # outputs.append(self.agent.est_decision(exps[i].s).to_tensor().gather(1, exps[i].a.to_tensor()))
        outputs = torch.cat(outputs, 0)  # torch.Size([batch,7])
        # print(outputs, targets)
        losses = self._cal_loss(outputs, targets.detach())
        # print(losses)
        self.agent.update(losses)

    def _cal_target(self, exps):
        # 你现在这么搞，意味着你的目标tensor的1个batch中只有1个数
        # 但我们知道Qnet的输出是有7个数的（上一个函数的output.shape=torch.Size([batch, 7])）
        # 那这明显对不上，怎么办呢？我觉得应该把除了 该状态下估计网络输出最大的那个位置 的其他位置元素都置为output对应位置的值，
        # 总之目的是，【只更新 该状态下估计网络 最大输出的那个位置 的loss】？？？？？
        """
        Args: list of <si, ri, ai, si+1> (list(Exprience))
        according list of exps to calculate the target use object Q-net
        yi = ri, if s(i+1) = None
        yi = ri + gamma * obj_Q(s(i+1), a), if s(i+1) != None
        Returns: targets(torch.Size([batch,7]))
        once only change one value(max of est-Qnet output index) of action tensor, others keep the same as est Qnet output
        """
        #【可能有问题，tensor维度操作太乱了！】

        targets = []
        for exp in exps:
            a_est_max = self.agent.est_decision(exp.ns).act_idx  # idx of est-Qnet max Q value
            r = Tensor([exp.r]).to(globalParam.device)  # cur reward, torch.Size([1])
            if exp.ns == None:
                targets.append(r)
            else:
                # objQ(ns, a_est_max), torch.Size([1])
                objQ__ns__a_est_max = self.agent.obj_decision(exp.ns).to_tensor()[0][a_est_max]
                target = r + self.gamma * objQ__ns__a_est_max
                targets.append(target)
        return torch.cat(targets, 0)  # torch.Size([batch, 7])

    def get_exp(self):
        """
        Can be overrided
        Args: None
        decision with estimate Q-net
        Returns: exp(Experence) or None
        """
        stat = self.env.get_stat()
        act = self.agent.est_decision(stat)
        rwd, nxt_stat = self.env.step(act)
        if nxt_stat == None:
            return None
        # stat.feats["Img"] = None
        # stat.gray_tensor = None
        # nxt_stat.feats["Img"] = None
        # nxt_stat.gray_tensor = None
        rwd = float(rwd.to('cpu'))
        act.tensor = None
        return Exprience(stat, rwd, act, nxt_stat)
