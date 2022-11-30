from conf import globalParam
import reinforce.dqn as dqn
import reinforce.env as env
import torch


root_path = 'data/Rein/streamset/'

status_conf = dict(
    ev = 0,
    img = None
)

eps_scheduler_conf = dict(
    steps=1000,
    max_eps=0.95
)

optimizer_conf = dict(
    type = "Adam",
    lr = 5e-4
)

agent_conf = dict(
    network=globalParam.dqn_network,
    optimizer=optimizer_conf,
)

env_conf = dict(
    stat=env.EV_Status(**status_conf),
    interval=10,
    network=globalParam.twostage_network,
    model_path='data/Rein/env_model/epoch_100.pth'
)

trainer_conf = dict(
    agent=dqn.DoubleDQNAgent(**agent_conf),
    env=env.SupervisedEnv(**env_conf),
    streamset=env.Videoset(root_path),
    loss_func=torch.nn.MSELoss(),
    batch_size=8,
    exp_pool_size=40,
    gamma=0.9,
    eps=0.5,
    eps_scheduler=dqn.QuadricScheduler(**eps_scheduler_conf)
)

train_param = dict(
    max_epoch=100,
    max_step=-1,
    backup_steps=5,
    log_steps=env_conf['interval'] * 100
)
