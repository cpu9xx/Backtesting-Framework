from .Env import Env
import datetime
import numpy as np
class OrderCost(object):
    def __init__(self, open_tax=0, close_tax=0.001, open_commission=0.0003, close_commission=0.0003, min_commission=5):
        self.open_tax = open_tax
        self.close_tax = close_tax
        self.open_commission = open_commission
        self.close_commission = close_commission
        self.min_commission = min_commission

    def __str__(self):
        return (f"OrderCost(open_tax={self.open_tax}, "
            f"close_tax={self.close_tax}, "
            f"open_commission={self.open_commission}, "
            f"close_commission={self.close_commission}, "
            f"min_commission={self.min_commission})")

class SubPortfolioConfig(object):
    def __init__(self, cash, type, strategy=None):
        self.cash = cash
        # self.type = type
        self.strategy = strategy

class Setting(object):
    def __init__(self):
        self._ucontext = None
        self._order_cost = dict()
        self._benchmark = None
        

    def get_order_cost(self, type: str, ref=None):
        return self._order_cost.get(type, None)

    def get_benchmark(self):
        return self._benchmark

    def set_user_context(self, ucontext):
        self._ucontext = ucontext

    def _set_benchmark(self, security: str):
        self._benchmark = security
        self._set_trade_dates(security)

    def _set_trade_dates(self, benchmark):
        env = Env()
        start = datetime.datetime.strptime(env.usercfg['start'], "%Y%m%d")
        end = datetime.datetime.strptime(env.usercfg['end'], "%Y%m%d")
        benchmark_df = env.index_data[benchmark]
        trade_dates = benchmark_df.index
        start_idx = trade_dates.get_loc(benchmark_df.loc[start:end, :].index[0])
        # trade_dates = env.index_data[benchmark].loc[start:end, :].index
        # print(trade_dates[0], trade_dates[-1])
        # trade_dates = env.index_data[benchmark].index[(env.index_data[benchmark].index > start) & (env.index_data[benchmark].index < end)]
        # print(trade_dates[0], trade_dates[-1])
        env.set_trade_dates(trade_dates, start_idx)
    
    def _set_order_cost(self, cost: OrderCost, type: str, ref=None):
        if type in ['stock', 'cb']:
            self._order_cost[type] = cost
            return True
        return False
    
    def _set_option(self, option, value):
        pass

    def _set_universe(self, security_list):
        self._ucontext.universe = security_list

    def _set_subportfolios(self, subportfolioconfig_ls):
        # self._ucontext.subportfolios.reset()
        for i, cfg in enumerate(subportfolioconfig_ls):
            if cfg.strategy is not None:
                cfg.strategy.set_pindex(i)
            if i == 0:
                self._ucontext.subportfolios.set(config=cfg, pindex=0)
                continue
            self._ucontext.subportfolios.append(cfg)
            
        import math
        if not math.isclose(self._ucontext.subportfolios.starting_cash, Env().usercfg['start_cash']):
            raise ValueError("每个仓位的资金之和，必须等于初始资金")

setting = Setting()