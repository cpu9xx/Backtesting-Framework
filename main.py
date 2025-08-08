# -*- coding: utf-8 -*-
# from Env import Env, config
# from Mod import ModHandler
# from events import EVENT, Event
# from globalVars import GlobalVars
# from CodeLoader import CodeLoader
# from strategy_context import StrategyContext
# import api
# from api import Scheduler
# import datetime
from . import logger
from .Env import Env, config
from .Mod import ModHandler
from .events import EVENT, Event
from .globalVars import GlobalVars
from .CodeLoader import CodeLoader
from .strategy_context import StrategyContext
from .recorder import Recorder
from .scheduler import Scheduler
from .broker import Broker, UserTrade
from .strategy import Strategy
from .setting import Setting
from .data import Data
from . import api
from userConfig import userconfig

import datetime
import time
import os
import json

class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        print(f"Running time: {self.interval:.4f} s")

def run_file(strategy_file_path, **kwargs):
    backtest = userconfig.get('backtest', True)
    if not backtest:
        if 'last_trade_day' not in kwargs or 'today' not in kwargs:
            raise ValueError("'last_trade_day' and 'today' must be provided when in Live mode")
        last_trade_day = kwargs['last_trade_day']
        today = kwargs['today']
        if isinstance(last_trade_day, str):
            last_trade_day = datetime.datetime.strptime(last_trade_day, "%Y%m%d")
        if isinstance(today, str):
            today = datetime.datetime.strptime(today, "%Y%m%d")

    with Timer():
        env = Env(config)
        #启动加载模块
        mod = ModHandler()
        #加载模块中注入config
        mod.set_env(env)
        #启动加载
        mod.start_up()
        api.create_get_data_by_code_funcs(env.extra_code_db)
        api.create_get_data_by_date_funcs(env.extra_date_db)
        api.create_get_basic_data_funcs(env.basic_db)
        loader = CodeLoader(strategy_file_path)
        
        # scope = {}
        scope = {'__name__': 'userStrategy'}
        scope = loader.load(scope)
        context = StrategyContext(start_cash=env.usercfg['start_cash'])
        context.set_run_params(env.usercfg['start'], env.usercfg['end'], None, None)
        env.set_global_vars(GlobalVars())

        logger.log.set_env(env)

        scope.update({
            "g": env.global_vars,
            "log": logger.log,

        })
        
        env.current_dt =  datetime.datetime.strptime(env.usercfg['start'], "%Y%m%d")
        # context.current_dt = env.current_dt

        env.load_data()

        recorder = Recorder()
        strategy = Strategy()
        broker = Broker()
        scheduler = Scheduler()
        data = Data()

        recorder.set_user_context(context)
        api.setting.set_user_context(context)
        strategy.set_user_context(context)
        broker.set_user_context(context)
        scheduler.set_user_context(context)
        data.set_user_context(context)
        

        api._strategy = strategy
        api._broker = broker
        api._scheduler = scheduler
        api._data = data
        api._recorder = recorder

        if backtest:
            # backtest 模式
            initialize = scope.get('initialize', None)
            on_strategy_end = scope.get('on_strategy_end', None)
            process_initialize = scope.get('process_initialize', None)
            if not callable(initialize):
                raise ValueError("initialize function not found in strategy file")
            initialize(context)

            if callable(process_initialize):
                process_initialize(context)

            scheduler.start_event_src()

            if callable(on_strategy_end):
                on_strategy_end(context)
                
            recorder.plot(show=True)

        else:
            # live 模式
            from .order import UserOrder

            load_memory_path = f"./state/Memory_{last_trade_day.day}.json"
            save_memory_path = f"./state/Memory_{today.day}.json"
            
            initialize = scope.get('initialize', None)
            on_strategy_end = scope.get('on_strategy_end', None)
            process_initialize = scope.get('process_initialize', None)
            if not callable(initialize):
                print("initialize function not found in strategy file")
                raise ValueError("initialize function not found in strategy file")
            initialize(context)

            if not os.path.isfile(load_memory_path):
                # 记忆不存在, 是初次运行程序, 进行初始化
                # initialize = scope.get('initialize', None)
                # initialize(context)
                pass
            else:
                print(f"Loading memory from {load_memory_path}")
                # 记忆存在, 恢复 g, context, broker
                with open(load_memory_path, 'rb') as json_file:
                    memory = json.load(json_file)

                context.set_state(memory['context'])    
                env.global_vars.set_state(memory['g'])
                broker.set_state(memory['broker'])
                recorder.set_state(memory['recorder'])
                UserOrder.cls_set_state(memory['UserOrder'])
                UserTrade.cls_set_state(memory['UserTrade'])

            # live 模式, 检测是否有 after_code_changed 函数, 有则运行
            after_code_changed = scope.get('after_code_changed', None)
            if callable(after_code_changed):
                print("暂不支持 live 模式下自定义after_code_changed")
                raise
                after_code_changed(context)

            if callable(process_initialize):
                process_initialize(context)

            scheduler.start_event_src()

            if callable(on_strategy_end):
                on_strategy_end(context)

            # live 模式, 保存记忆
            memory = {
                'context': context.get_state(),
                'g': env.global_vars.get_state(),
                'broker': broker.get_state(),
                'recorder': recorder.get_state(),
                'UserOrder': UserOrder.cls_get_state(),
                'UserTrade': UserTrade.cls_get_state(),
            }
            print(f"Saving memory to {save_memory_path}")
            with open(save_memory_path, 'w') as json_file:
                json.dump(memory, json_file, indent=4)
            recorder.plot(show=False, save_path=f'./pnl/pnl{today.day}.png')

    # from userConfig import userconfig
    # if userconfig['backtest']:
    #     recorder.plot()

