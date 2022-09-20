from .parsers import ParserInterface
import time
from conf import workdir as wd
from core import utility as util


@util.Singleton
class ConfigParser(ParserInterface):
    def __init__(self):
        self.conf_dict = {}

    def parse(self):
        self._parse_workdir()  # 确保最后解路径
        return self.conf_dict

    def _parse_workdir(self):
        if wd.ts == '':
            wd.ts = wd.work_dir + time.strftime('%Y%m%d_%H%M/', time.localtime())
        wd.checkpoint_dir = wd.ts + wd.checkpoint_dir
        wd.log_dir = wd.ts + wd.log_dir
        wd.result_dir = wd.ts + wd.result_dir
        self.conf_dict['workdir'] = vars(wd)  # todo: 这tm有问题啊，，内置变量太多了...

