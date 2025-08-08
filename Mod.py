# -*- coding: utf-8 -*-

from collections import OrderedDict
from importlib import import_module

class ModHandler(object):
    def __init__(self):
        self._env = None
        self._mod_list = list()
        self._mod_dict = OrderedDict()

    def set_env(self, environment):
        self._env = environment

        config = environment.config

        for mod_name in config['mod']:
            if config['mod'][mod_name]['enabled'] == False:
                continue
            self._mod_list.append(f"jqdata.{mod_name}")

    def start_up(self):
        for mod_name in self._mod_list:
            #动态加载模块
            mod = import_module(mod_name)
            for api in mod.__all__:
                api(self._env.event_bus)

