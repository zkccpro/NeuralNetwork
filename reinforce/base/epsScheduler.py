class EpsScheduler:
    def __init__(self, steps, max_eps):
        """
        Args: 
        steps, how many steps to reach the max_eps(may difference in subs)
        max_eps, max eps can be reached
        Returns: None
        """
        self.steps = steps
        self.max_eps = max_eps

    def __call__(self, eps):
        """
        Args: 
        eps, cur eps to be caculated
        Returns: eps after caculating
        """
        pass


class LinearScheduler(EpsScheduler):
    def __init__(self, steps, max_eps):
        super().__init__(steps, max_eps)
        self.init_eps = -1
        self.k = -1
        
    def __call__(self, eps):
        if self.init_eps == -1:
            self.init_eps = eps
            self.k = (self.max_eps - eps) / self.steps
        eps += self.k
        return eps if eps < self.max_eps else self.max_eps


class QuadricScheduler(EpsScheduler):
    def __init__(self, steps, max_eps):
        super().__init__(steps, max_eps)
        
    def __call__(self, eps):
        k = (self.max_eps - eps) / self.steps
        eps += k
        return eps if eps < self.max_eps else self.max_eps
