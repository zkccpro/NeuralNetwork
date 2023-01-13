import torch


class Agent:
    """
    Can to be inherited
    By given network, optimizer, action to build an agent
    optimizer conf example:
    optimizer = dict(
        type = "SGD" # or "Adam"
        lr = 0.2,
        momentum=0.9,
        weight_decay=1e-2
    )
    """
    def __init__(self, network, optimizer, action):
        self.network = network
        self.optimizer = self.gen_optimizer(network, optimizer)
        self.act = action

    def decision(self, stat):
        """
        Can be overrided
        Args: cur status
        network forward process with stat input
        Returns: output action parse from network ouput tensor (Action)
        """
        input = stat.to_tensor()
        output = self.inference(self.network, input)
        return self.act.parse_from_tensor(ouput)

    def inference(self, network, input):
        assert network != None
        network.eval()
        return network(input)

    def update(self, loss):
        """
        Can be overrided
        Args: loss to backword (Tensor)
        network backward and update params
        Returns: None
        """
        assert self.network != None and self.optimizer != None
        # backward
        self.network.train()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def backup(self):
        """
        Can be overrided
        Args: None
        use for some kind of agent strategy to backup itself
        such as Double DQN, need to copy est-Qnet to obj-Qnet every c steps
        Returns: None
        """
        pass

    def save_to(self, path):
        pass

    def gen_optimizer(self, network, optimizer):
        assert "type" in optimizer, "optimizer must have a type(SGD or Adam)"
        opt_type = optimizer.pop("type")
        if opt_type == "SGD":
            return torch.optim.SGD(network.parameters(), **optimizer)
        else:
            return torch.optim.Adam(network.parameters(), **optimizer)
