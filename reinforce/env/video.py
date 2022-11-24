# SupervisedEnv(Env), Video(Stream), Videoset(Streamset)、EV_Stat(Status)、EV_Action(Action)
from ..base import Status, Action, Env, Stream, Streamset
import numpy as np
import torch, cv2, os, random
import os.path as osp
from conf import globalParam


class EV_Status(Status):
    """
    Exposure Value status
    EV status: [-2.0, +2.0], step = 0.2
    """
    def __init__(self, ev, img):
        super().__init__()
        self.feats["EV"] = ev
        self.feats["Img"] = img  # cv2 Image

    def to_tensor(self, gray=False):
        """
        Args: None
        convert cv2 Image to Tensor
        Returns: tensor (torch.Size([b,c,h,w]))
        """
        # if self.tensor != None and not gray:
        #     return self.tensor
        # if self.gray_tensor != None and gray:
        #     return self.gray_tensor

        img = cv2.resize(self.feats["Img"], dsize=(240, 240))
        if gray:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = np.transpose(img, (0, 1))
            img = self.np_std(img)
            return torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).to(globalParam.device)  # raise dim to torch.Size([1,1,h,w]
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.transpose(img, (2, 0, 1))
            img = self.np_std(img)
            return torch.from_numpy(img).float().unsqueeze(0).to(globalParam.device)  # raise dim to torch.Size([1,3,h,w]

    def __repr__(self):
        return f'{self.feats["EV"]}'


class EV_QAction(Action):
    """
    Exposure Value Q function action
    EV action: [-0.6, -0.4, -0.2, 0, +0.2, +0.4, +0.6]
    """
    def __init__(self, ev, acts):
        super().__init__()
        self.vals["EV"] = ev
        self.act_idx = 0  # the index in self.acts of cur action choise 
        self.acts = acts
        # self.tensor is Q-function

    def parse_from_tensor(self, tensor):
        """
        Args: tensor to parse (torch.Size([1,7])), batch_size must be 1
        parse network output([estimate] Q-Function) to vals.[EV]
        use random greedy as strategy
        Returns: self (Action)
        """
        super().parse_from_tensor(tensor)
        # 我甚至觉得根本不需要升维，torch库应该是自动处理batch的？
        # 就是说，哪怕只有batch_size=1，网络输出的tensor应该也是考虑好升维的
        # if self.tensor.dim() == self.dims:
        #     self.tensor.unsqueeze(0)  # raise dim for batch
        # elif self.tensor.dim() < self.dims or self.tensor.dim() > self.dims + 1:
        #     print("ERROR: dim wrong of output tensor")
        # else: # self.tensor.dim() == self.dims + 1, Nothing tobe done
        #     pass
        max_idx = int(torch.argmax(tensor))
        self.act_idx = max_idx
        self.vals["EV"] = self.acts[max_idx]
        return self
    
    def __repr__(self):
        return f'vals: {self.vals}; tensor: {self.tensor}'


class SupervisedEnv(Env):
    """Using an well-trained network to modeling the environment"""
    def __init__(self, stat, network):
        super().__init__(stat)
        self.model = network
    
    def _cal_reward(self):
        """
        Args: None
        calulate reward of cur stat with model
        Returns: reward (torch.Size([1]))
        """
        scores = self.inference()  # torch.Size([2, 1])
        assert scores.dim() >= 2
        last_score = 2 - 2 * torch.abs(scores[0])  # torch.Size([1])
        score = 2 - 2 * torch.abs(scores[1]) # torch.Size([1])

        if score < last_score:
            rwd = score - 3 * last_score
        elif score >= last_score and score < 1.8:
            rwd = 2 * score - last_score
        else:  # score >= 1.8
            rwd = 2 * score - last_score + 1

        return rwd

    def inference(self):
        """
        Args: None
        use last stat and cur stat as input (concat them in one batch)
        inference the model, get output scores(torch.Size([2, 1])),then parse the output to list
        Returns: env model output((torch.Size([2, 1]))
        """
        assert self.model != None

        img = self.stat.to_tensor(gray=True)
        last_img = self.last_stat.to_tensor(gray=True)
        input = torch.cat([last_img, img], 0)
        self.model.eval()
        output = self.model(input)
        return output


class EV_VideoSpace(Stream):
    """
    EV VideoSpace constitutes by multi videos
    num of videos = size of status space
    """
    def __init__(self, root_path, max_ev=2.0, min_ev=-2.0):
        """
        root_path: root path of ONE total video datas
        load all videos from root_path to self.videos
        """
        self.videos = dict()  # {ev : video}
        self.cur_frame = 0
        self.cur_ev = 0
        self.max_ev = max_ev
        self.min_ev = min_ev
        root = os.listdir(root_path)
        assert root, f'ERROR: root path of video data: "{root_path}" is invalid!'
        for name in root:
            video_path = osp.join(root_path, name)
            ev = float(name[len(name) - 8 : len(name) - 4])
            self.videos[ev] = cv2.VideoCapture(video_path)
        assert self.videos, 'ERROR: EV_VideoSpace.videos is empty!'

    def fst_frame(self):
        """
        Args: None
        get first frame of the video by random select one
        Returns: first status (Status), return None if video load fail!
        """
        select_ev = random.sample(list(self.videos), 1)[0]
        self.cur_ev = select_ev
        select_video = self.videos[select_ev]
        select_video.set(cv2.CAP_PROP_POS_FRAMES, 0)  # start with the first frame
        ok, frame = select_video.read()
        if not ok:
            return None
        else:
            return EV_Status(self.cur_ev, frame)

    def nxt_frame(self, action):
        """
        Args: action to be executed (Action)
        execute action to approch next frame
        Returns: next status (Status), return None if video is ended.
        """
        self.cur_ev = round(self.cur_ev + action.vals["EV"], 1)
        if self.cur_ev > self.max_ev:
            self.cur_ev = self.max_ev
        if self.cur_ev < self.min_ev:
            self.cur_ev = self.min_ev
        self.cur_frame += 1
        select_video = self.videos[self.cur_ev]
        select_video.set(cv2.CAP_PROP_POS_FRAMES, self.cur_frame)
        ok, frame = select_video.read()
        if not ok:
            return None
        else:
            return EV_Status(self.cur_ev, frame)

    def __len__(self):
        """
        Args: None
        get frame num of the video
        select a video randomly in self.videos and return its frame num
        Returns: num of total frames (Int)
        """
        return int(self.videos[random.sample(list(self.videos), 1)[0]].get(cv2.CAP_PROP_FRAME_COUNT))


class Videoset(Streamset):
    """
    Videoset provides a method to fetch ONE of all videos
    I DO NOT want the Streamset to hold all the streams...
    ONLY hold more than one stream in any time
    """
    def __init__(self, root_path):
        """
        root_path: root path of all video data
        get every Video pathes and put them into self.streams
        """
        super().__init__()
        root = os.listdir(root_path)
        assert root, f'ERROR: root path of video set: "{root_path}" is invalid!'
        for video_rp in root:
            stream_path = osp.join(root_path, video_rp)
            if not os.path.isdir(stream_path):
                continue
            self.streams.append(stream_path)

    def __getitem__(self,index):
        return EV_VideoSpace(self.streams[index])

    def __len__(self):
        return len(self.streams)