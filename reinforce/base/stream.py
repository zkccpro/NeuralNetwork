

class Stream:
    """
    NEED to be inherited
    A REAL stream type of data while training use, holden by Env
    """
    def __init__(self):
        pass

    def fst_frame(self):
        """
        NEED to be overrided
        Args: None
        get first frame of the stream data
        Returns: first status (Status), return None if there is no game!
        """
        pass

    def nxt_frame(self, action, interval=1):
        """
        NEED to be overrided
        Args:
        action, to be executed (Action)
        interval, every n frame get one to train(Int)
        execute action to approch next status
        Returns: next status (Status), return None if game is over!
        """
        pass
        
    def __len__(self):
        """
        Can to be overrided
        Args: None
        get the total frame num in the game (sometimes may not have a certain num)
        Returns: num of total frames, not certain then return -1 (Int)
        """
        return -1


class Streamset:
    """
    NEED to be inherited
    Streamset provides a method to fetch one of all streams
    I DO NOT want the Streamset to hold all the streams...
    ONLY hold more than one stream in any time
    """
    def __init__(self):
        self.streams = []

    def __getitem__(self, index):
        """
        NEED to be overrided
        Args: index
        fetch and return cur index of stream (in streams)
        Returns: stream (Stream)
        """
        pass

    def __len__(self):
        """
        Can to be overrided
        Args: None
        return the stream number in streamset
        Returns: Int
        """
        pass
