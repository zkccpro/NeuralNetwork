import reinforce.dqn as dqn
from conf import reinforcement

def reinforcement_test():
    trainer = dqn.DoubleDQNTrainer(**reinforcement.trainer_conf)
    trainer.train(**reinforcement.train_param)
