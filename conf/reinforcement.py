from conf import globalParam
import reinforce.dqn as dqn
import reinforce.env as env
import torch


root_path = 'data/Rein/streamset/'

status_conf = dict(
    ev = 0,
    img = None
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
    network=globalParam.twostage_network
)

trainer_conf = dict(
    agent=dqn.DoubleDQNAgent(**agent_conf),
    env=env.SupervisedEnv(**env_conf),
    streamset=env.Videoset(root_path),
    loss_func=torch.nn.MSELoss(),
    batch_size=128,
    exp_pool_size=10000,
    gamma=0.9
)
