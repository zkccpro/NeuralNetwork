import reinforce.dqn as dqn
from conf import reinforcement

def reinforcement_test():
    trainer = dqn.DoubleDQNTrainer(**reinforcement.trainer_conf)
    trainer.train(max_epoch=100, max_step=-1, backup_steps=50, log_steps=500)
