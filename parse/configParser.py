from .parsers import ParserInterface
import time
from conf import workdir as wd
from core import utility as util
from conf import reinforcement as rf

@util.Singleton
class ConfigParser(ParserInterface):
    def __init__(self):
        self.conf_dict = {}

    def parse(self):
        self._parse_rf()
        self._parse_workdir()  # 确保最后解路径
        return self.conf_dict

    def _gen_dict_key(self, key_name, conf_file):
        """_summary_
        Args:
            key_name (String): 要添加到self.conf_dict的目标key名字
            conf_file (Py File Type): 要添加的配置文件
        把配置文件中的所有用户内容添加到self.conf_dict指定名字的键中
        """
        self.conf_dict[key_name] = dict()
        for var in vars(conf_file):
            if not var.startswith('__'):
                self.conf_dict[key_name][var] = vars(conf_file)[var]

    def _parse_workdir(self):
        if wd.ts == '':
            wd.ts = wd.work_dir + time.strftime('%Y%m%d_%H%M/', time.localtime())
        for var in vars(wd):
            if not var.startswith('__') and var != 'ts' and var != 'work_dir':
                vars(wd)[var] = wd.ts + vars(wd)[var]
        self._gen_dict_key('workdir', wd)

    def _parse_rf(self):
        self._gen_dict_key('reinforcement', rf)
