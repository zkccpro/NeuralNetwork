import reinforce.dqn as dqn
from parse import configParser as cp


conf_parser = cp.ConfigParser()
def reinforcement_test():
    trainer = dqn.DoubleDQNTrainer(**conf_parser.conf_dict['reinforcement']['trainer_conf'])
    trainer.train(**conf_parser.conf_dict['reinforcement']['train_param'])
