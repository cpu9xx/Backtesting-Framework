from collections import defaultdict
from .Env import Env
from .logger import log
from .object import TIME

import datetime
import math
class Position:
    def __init__(self):
        self.security = None
        self.avg_cost = 0
        self.init_time = None
        self.transact_time = None
        self.total_amount = 0
        self.closeable_amount = 0


    def init_position(self, security, avg_cost, init_time, amount):
        self.security = security
        self.avg_cost = avg_cost
        self.init_time = init_time
        self.transact_time = init_time
        self.total_amount = amount
        self.closeable_amount = 0
    
    def update_position(self, price, amount, time):
        self.avg_cost = (self.avg_cost * self.total_amount + price * amount) / (self.total_amount + amount)
        self.total_amount += amount
        self.transact_time = time
    
    def close_position(self, total_gain, amount, time):
        if math.isclose(self.total_amount, amount):
            self.total_amount = 0
        else:
            self.avg_cost = (self.avg_cost * self.total_amount - total_gain) / (self.total_amount - amount)
            self.total_amount -= amount
        self.transact_time = time
        
    def update_closeable_amount(self, amount=None):
        if amount is None:
            amount = self.total_amount
        self.closeable_amount = amount

    # @property
    # def next_price(self):
    #     # 访问 self.price 时会自动调用这个函数
    #     env = Env()
    #     dtime = env.current_dt
    #     if datetime.time(9, 30) <= dtime.time() < datetime.time(15, 0):
    #         field = 'close'
    #     else:
    #         field = 'open'

    #     if datetime.time(0, 0) <= dtime.time() < datetime.time(9, 30):
    #         dtime += datetime.timedelta(days=-1, hours=0, minutes=0)
    #     # print(env.data[self.security][dtime, field][0])
    #     return env.data[self.security][dtime, field][0]

    def get_state(self):
        return {
            'security': self.security,
            'avg_cost': self.avg_cost,
            'init_time': self.init_time.strftime('%Y-%m-%d %H:%M:%S'),
            'transact_time': self.transact_time.strftime('%Y-%m-%d %H:%M:%S') if self.transact_time is not None else None,
            'total_amount': self.total_amount,
            'closeable_amount': self.closeable_amount
        }
    
    def set_state(self, state):
        self.security = state['security']
        self.avg_cost = state['avg_cost']
        self.init_time = datetime.datetime.strptime(state['init_time'], '%Y-%m-%d %H:%M:%S')
        self.transact_time = datetime.datetime.strptime(state['transact_time'], '%Y-%m-%d %H:%M:%S') if state['transact_time'] is not None else None
        self.total_amount = state['total_amount']
        self.closeable_amount = state['closeable_amount']

    @property
    def price(self):
        # 访问 self.price 时会自动调用这个函数
        env = Env()
        dtime = env.current_dt
        if TIME.OPEN.value <= dtime.time() < TIME.BEFORE_CLOSE.value:
            field = 'open'
        else:
            field = 'close'

        if TIME.DAY_START.value <= dtime.time() < TIME.OPEN.value:
            # 前一天不是交易日时, object也会自动返回前一个交易日
            dtime += datetime.timedelta(days=-1, hours=0, minutes=0)
        # print(env.data[self.security][dtime, field][0])
        if self.security in env.data:
            return env.data[self.security][dtime, field][0]
        elif self.security in env.cb_data:
            return env.cb_data[self.security][dtime, field][0]
    
    @property
    def value(self):
        return self.total_amount * self.price

    def __repr__(self):
        return f"UserPosition({self.security}, avgcost={self.avg_cost}, total_amount={self.total_amount}, closeable_amount={self.closeable_amount}, init_time={self.init_time}, transact_time={self.transact_time})"

class Params():
    def __init__(self):
        self.start_date = None
        self.end_date = None
        self.type = None
        self.frequency = None

    def get_state(self):
        return {
           'start_date': self.start_date,# str
            'end_date': self.end_date,
            'type': self.type,
            'frequency': self.frequency
        }
        # return {
        #    'start_date': self.start_date.strftime('%Y-%m-%d') if self.start_date is not None else None,
        #     'end_date': self.end_date.strftime('%Y-%m-%d') if self.end_date is not None else None,
        #     'type': self.type,
        #     'frequency': self.frequency
        # }
    
    def set_state(self, state):
        self.start_date = state['start_date']
        self.end_date = state['end_date']
        self.type = state['type']
        self.frequency = state['frequency']
        # self.start_date = datetime.datetime.strptime(state['start_date'], '%Y-%m-%d').date() if state['start_date'] is not None else None
        # self.end_date = datetime.datetime.strptime(state['end_date'], '%Y-%m-%d').date() if state['end_date'] is not None else None
        # self.type = state['type']
        # self.frequency = state['frequency']

class AutoRemoveDefaultDict(defaultdict):
    def __init__(self, default_factory):
        super().__init__(default_factory)
        self.protected_keys = set()

    def __getitem__(self, key):
        value = super().__getitem__(key)
        value.security = key
        if key not in self.protected_keys:
            log.warning(f"Security(code={key}) 在 positions 中不存在, 为了保持兼容, 我们返回空的 Position 对象, amount/price/avg_cost/acc_avg_cost 都是 0")
            del self[key]
        return value

    def create_key(self, key):
        self.protected_keys.add(key)

    def remove_key(self, key):
        if key in self.protected_keys:
            self.protected_keys.remove(key)
        if key in self:
            del self[key]

    def get_state(self):
        state = {'protected_keys': list(self.protected_keys), 'positions': {}}
        for key, position in self.items():
            state['positions'][key] = position.get_state()
        return state
    
    def set_state(self, state):
        self.protected_keys = set(state['protected_keys'])
        positions = state['positions']
        for key, position_state in positions.items():
            position = Position()
            position.set_state(position_state)
            assert key in self.protected_keys
            self[key] = position

class Portfolio():
    def __init__(self, start_cash):
        self.positions = AutoRemoveDefaultDict(lambda: Position())
        self.available_cash = start_cash
        self.starting_cash = start_cash
        self.total_units = start_cash
        self._unit_value = 1.0
        # self.cum_cash_outflow = 0
        # self.cum_cash_inflow = 0

    @property
    def positions_value(self):
        pvalue = 0
        for position in self.positions.values():
            pvalue += position.value
        return pvalue
    
    @property
    def total_value(self):
        return self.positions_value + self.available_cash
    
    @property
    def unit_value(self):
        if self.total_units != 0:
            self._unit_value = self.total_value / self.total_units
        return self._unit_value

    def deposit(self, cash: float):
        """用户存入资金"""
        units = cash / self.unit_value
        self.total_units = round(self.total_units + units, 6)
        assert self.total_units >= 0
        self.available_cash = round(self.available_cash + cash, 6)

    def withdraw(self, cash: float):
        """用户取出资金"""
        units = cash / self.unit_value
        self.total_units = round(self.total_units - units)
        assert self.total_units >= 0
        self.available_cash = round(self.available_cash - cash, 6)

    @property
    def returns(self):
        # return (self.total_value) / self.starting_cash - 1
        # return ((self.total_value + self.cum_cash_outflow) / self.starting_cash) - 1
        # return ((self.total_value + self.cum_cash_outflow)/(self.starting_cash + self.cum_cash_inflow)) - 1
        return self.unit_value - 1.0
    
    def get_state(self):
        state = {'positions': self.positions.get_state(), 
                 'available_cash': self.available_cash,
                 'starting_cash': self.starting_cash,
                 'total_units': self.total_units,
                 'unit_value': self._unit_value,
                }
        return state
    
    def set_state(self, state):
        self.positions.set_state(state['positions'])
        self.available_cash = state['available_cash']
        self.starting_cash = state['starting_cash']
        self.total_units = state['total_units']
        self._unit_value = state['unit_value']

    def __repr__(self):
        return f"Portfolio(positions_value={self.positions_value}, available_cash={self.available_cash}, total_units={self.total_units}, unit_value={self.unit_value}, total_value={self.total_value}, returns={self.returns})"

class SubPortfolios(list):
    def __init__(self, start_cash):
        super().__init__([Portfolio(start_cash)])
        self.starting_cash = start_cash
        # self.cum_cash_outflow = 0
    
    def set(self, config, pindex=0):
        self.starting_cash -= self[pindex].starting_cash
        self[pindex].__init__(config.cash)
        self.starting_cash += self[pindex].starting_cash

    def get_state(self):
        state = {'starting_cash': self.starting_cash, 'portfolios': []}
        for portfolio in self:
            state['portfolios'].append(portfolio.get_state())
        return state

    def set_state(self, state):
        self.clear()
        self.starting_cash = state['starting_cash']
        for portfolio_state in state['portfolios']:
            portfolio = Portfolio(0)
            portfolio.set_state(portfolio_state)
            super().append(portfolio)

    def reset(self):
        super().__init__([])
        self.starting_cash = 0

    def append(self, portfolioconfig):
        super().append(Portfolio(portfolioconfig.cash))
        self.starting_cash += portfolioconfig.cash
    
    @property
    def available_cash(self):
        cash = 0
        for portfolio in self:
            cash += portfolio.available_cash
        return cash

    @property
    def positions_value(self):
        pvalue = 0
        for portfolio in self:
            pvalue += portfolio.positions_value
        return pvalue
    
    @property
    def total_value(self):
        tvalue = 0
        for portfolio in self:
            tvalue += portfolio.total_value
        return tvalue
    
    @property
    def total_position_count(self):
        tpcount = 0
        for portfolio in self:
            tpcount += len(portfolio.positions)
        return tpcount
    
    @property
    def returns(self):
        return self.total_value / self.starting_cash - 1

class StrategyContext():
    _context = None
    def __init__(self, start_cash):
        if StrategyContext._context is not None:
            raise RuntimeError("StrategyContext 只能初始化一次")
        StrategyContext._context = self
        self.previous_date = None
        self.current_dt = None
        self.subportfolios = SubPortfolios(start_cash)
        self.portfolio = self.subportfolios[0]
        # self.portfolio = Portfolio(start_cash)
        self.run_params = Params()
        self.universe = None
        
    def set_run_params(self, start_date, end_date, type, frequency):
        self.run_params.start_date = start_date
        self.run_params.end_date = end_date
        self.run_params.type = type
        self.run_params.frequency = frequency

    @classmethod
    def get_instance(cls):
        if cls._context is None:
            raise RuntimeError("StrategyContext 还未初始化")
        return cls._context
    
    def set_state(self, state):
        self.previous_date = datetime.datetime.strptime(state['previous_date'], '%Y-%m-%d').date() if state['previous_date'] is not None else None
        self.current_dt = datetime.datetime.strptime(state['current_dt'], '%Y-%m-%d %H:%M:%S') if state['current_dt'] is not None else None
        self.subportfolios.set_state(state['subportfolios'])
        self.portfolio = self.subportfolios[0]
        # self.portfolio.set_state(state['portfolio'])
        # self.run_params.set_state(state['run_params'])
        self.universe = state['universe']

    def get_state(self):
        return {
            'previous_date': self.previous_date.strftime('%Y-%m-%d') if self.previous_date is not None else None,
            'current_dt': self.current_dt.strftime('%Y-%m-%d %H:%M:%S') if self.current_dt is not None else None,
            'subportfolios': self.subportfolios.get_state(),
            # 'portfolio': self.portfolio.get_state(),
            # 'run_params': self.run_params.get_state(),
            'universe': self.universe
        }