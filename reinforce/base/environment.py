import numpy as np

class Status:
    """
    NEED to be inherited
    user status for detail tasks
    """
    def __init__(self):
        self.feats = dict()

    def to_tensor(self):
        """
        Need be overrided
        Args: None
        convert Status to tensor for network input
        Returns: tensor (Tensor)
        """
        pass
    
    def np_nml(self, data):
        """
        Args: data (np.array)
        arr normalization use numpy, output range [0,1]
        Returns: normalized numpy arr
        """
        range = np.max(data) - np.min(data)
        if range != 0:
            return (data - np.min(data)) / range
        else:
            return data

    def np_std(self, data):
        """
        Args: data (np.array)
        arr standardization use numpy
        Returns: standardized numpy arr
        """
        stdvar = np.std(data)
        if stdvar != 0:
            return (data - np.mean(data)) / stdvar
        else:
            return data


class Action:
    """
    NEED to be inherited
    user action for detail tasks
    """

    def __init__(self):
        self.vals = dict()
        self.tensor = None
        self.dims = 1  # dim num of action space
    
    def parse_from_tensor(self, tensor):
        """
        Need be overrided
        Args: tensor to parse (Tensor)
        parse network output tensor to self
        Returns: self (Action)
        """
        self.tensor = tensor
    
    def to_tensor(self):
        """
        Can be overrided
        Args: None
        convert action to outout tensor
        Returns: ouput (Tensor)
        """
        assert self.tensor != None, 'to_tensor() should NOT be call before parse_from_tensor()'
        return self.tensor


class Env:
    """
    NEED to be inherited
    supply a base Env for users to design their environments
    """

    def __init__(self, stat, interval=1):
        self.stat = stat
        self.last_stat = stat
        self.interval = interval
    
    def get_stat(self):
        return self.stat
    
    def get_last_stat(self):
        return self.last_stat
    
    def step(self, action, log):
        self._update_stat(action)
        if self.stat == None:
            return 0, None
        rwd = self._cal_reward(log)
        return rwd, self.stat

    def reset(self, stream):
        self.stream = stream
        self.stat = stream.fst_frame()
        self.last_stat = self.stat
        return self.stat
    
    def _cal_reward(self, log):
        """
        Need be overrided
        Args: None
        calulate reward of cur stat, for users design reward function
        Returns: reward (float)
        """
        pass
    
    def _update_stat(self, action):
        self.last_stat = self.stat
        self.stat = self.stream.nxt_frame(action, self.interval)
        return self.stat
