from conf import globalParam
import reinforce.dqn as dqn
import reinforce.env as env
import reinforce.ac as ac
from reinforce.base import QuadricScheduler
import torch


"""
common conf
"""
data_root = 'data/Rein/'
train_path = data_root + 'trainset/'
val_path = data_root + 'valset/'

status_conf = dict(
    ev=0,
    img=None
)

# """
# DQN conf
# """
# eps_scheduler_conf = dict(
#     steps=500,
#     max_eps=0.95
# )

# optimizer_conf = dict(
#     type="Adam",
#     lr=5e-7
# )

# agent_conf = dict(
#     est_network=globalParam.dqn_network_est,
#     obj_network=globalParam.dqn_network_obj,
#     optimizer=optimizer_conf
#     # model_path='data/Rein/pretrain/dqn_epoch_100_stage2.pth'
# )

# env_conf = dict(
#     stat=env.EV_Status(**status_conf),
#     interval=10,
#     network=globalParam.twostage_network,
#     model_path='data/Rein/env_model/epoch_100.pth'
# )

# trainer_conf = dict(
#     agent=dqn.DoubleDQNAgent(**agent_conf),
#     env=env.SupervisedEnv(**env_conf),
#     trainset=env.Videoset(train_path),
#     valset=env.Videoset(val_path),
#     loss_func=torch.nn.MSELoss(),
#     batch_size=8,
#     exp_pool_size=40,
#     gamma=0.9,
#     eps=0.8,
#     eps_scheduler=QuadricScheduler(**eps_scheduler_conf)
# )

# train_param = dict(
#     max_epoch=100,
#     max_step=-1,
#     backup_steps=env_conf['interval'] * 5,
#     log_steps=env_conf['interval'] * 100
# )


"""
SAC conf
"""
critic_optimizer = dict(
    type="Adam",
    lr=5e-7
)

actor_optimizer = dict(
    type="Adam",
    lr=5e-7
)

agent_conf = dict(
    critic=globalParam.critic_net,
    target_critic=globalParam.target_critic_net,
    actor=globalParam.actor_net,
    critic_optimizer=critic_optimizer,
    actor_optimizer=actor_optimizer
)

env_conf = dict(
    stat=env.EV_Status(**status_conf),
    interval=10,
    network=globalParam.twostage_network,
    model_path='data/Rein/env_model/epoch_100.pth'
)

trainer_conf = dict(
    agent=ac.SoftACAgent(**agent_conf),
    env=env.SupervisedEnv(**env_conf),
    trainset=env.Videoset(train_path),
    valset=env.Videoset(val_path),
    loss_func=torch.nn.MSELoss(),
    batch_size=8,
    exp_pool_size=40,
    alpha=0.2,
    gamma=0.99,
    entropy_tuning=True,
    tuning_lr=0.01
)

train_param = dict(
    max_epoch=100,
    checkpoint_epoch=20,
    max_step=-1,
    backup_steps=env_conf['interval'] * 5,
    log_steps=env_conf['interval'] * 100
)
