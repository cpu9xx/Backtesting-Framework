from .events import Event, EVENT
from .order import UserOrder
from .Env import Env
from .logger import log
from .api import setting
from .object import OrderStatus, TIME, UserTransfer
from userConfig import userconfig

import datetime
import numpy as np

class Strategy(object):
    def __init__(self):
        self.name = self.__class__.__name__
        print(f"策略{self.name}初始化")
        self._ucontext = None
        
    def set_user_context(self, ucontext):
        self._ucontext = ucontext

    def _order(self, security, amount, style, side, pindex, close_today, record):
        env = Env()
        if userconfig['backtest']:
            if security in env.data:
                flag = env.data[security][self._ucontext.current_dt, 'close'][0]
            elif security in env.cb_data:
                flag = env.cb_data[security][self._ucontext.current_dt, 'close'][0]
            else:
                log.error(f" {security} 不在回测数据中")
            if flag == 0:
                log.warning(f" {security} 进入退市整理期或已退市, 取消下单")
                return None
        event_bus = env.event_bus
        order = UserOrder(security, add_time=self._ucontext.current_dt, amount=amount, pindex=pindex, record=record)
        event_bus.publish_event(Event(EVENT.STOCK_ORDER, order=order))
        if order.status() != OrderStatus.rejected:
            return order
        else:
            return None

    def _order_target(self, security, amount, style, side, pindex, close_today, record):
        current_amount = self._ucontext.subportfolios[pindex].positions[security].total_amount
        amount = amount - current_amount
        return self._order(security, amount, style, side, pindex, close_today, record)

    def _order_value(self, security, value, style, side, pindex, close_today, record):
        if np.isnan(value) or value == 0:
            log.error(f" {security} 下单金额为{value}")
        order_cost = setting.get_order_cost(type='stock')

        #买入时需要预留手续费
        if value > 0:
            commission= (order_cost.open_tax + order_cost.open_commission) * value
            commission = commission if commission > order_cost.min_commission else order_cost.min_commission
            value -= commission
            if value <= 0:
                log.warning(f" {security} 下单金额少于成交所需手续费, 取消下单")
                return None
        
        # env = Env()
        # dtime = env.current_dt
        # if TIME.OPEN.value < dtime.time() <= TIME.CLOSE.value:
        #     # 日频数据, 开盘后的下一个价格是收盘价
        #     field = 'close'
        # else:
        #     # 夜间下单, 按开盘价成交
        #     field = 'open'

        # if TIME.CLOSE.value < dtime.time() < TIME.DAY_END.value:
        #     # 按下一个交易日开盘价成交
        #     dtime += datetime.timedelta(days=1, hours=0, minutes=0)

        # current_price = env.data[security][dtime, field][0]
        # if current_price == 0:
        #     log.warning(f" {security} 进入退市整理期或已退市, 取消下单")
        #     return None
        # # current_price = self._ucontext.portfolio.positions[security].price
        # amount = value / current_price

        env = Env()
        if userconfig['backtest']:
            if security in env.data:
                flag = env.data[security][self._ucontext.current_dt, 'close'][0]
            elif security in env.cb_data:
                flag = env.cb_data[security][self._ucontext.current_dt, 'close'][0]
            else:
                log.error(f" {security} 不在回测数据中")
            if flag == 0:
                log.warning(f" {security} 进入退市整理期或已退市, 取消下单")
                return None
        event_bus = env.event_bus
        order = UserOrder(security, add_time=self._ucontext.current_dt, value=value, pindex=pindex, record=record)
        event_bus.publish_event(Event(EVENT.STOCK_ORDER, order=order))
        if order.status() != OrderStatus.rejected:
            return order
        else:
            return None

    def _order_target_value(self, security, value, style, side, pindex, close_today, record):
        value -= self._ucontext.subportfolios[pindex].positions[security].value
        return self._order_value(security, value, style, side, pindex, close_today, record)
    
    def _convert_bond(self, security, amount, price, style, side, pindex, close_today):
        # 未测试
        Env().need_cb_daily()
        try:
            cb_value = Env().cb_data[security][self._ucontext.current_dt, 'cb_value'][0]
        except:
            log.error(f"若要使用convert_bond(), userConfig.py 的 cb_column_tuple 中, 必须有转股价值'cb_value'.")
            return None
        
    def _transfer_cash(self, from_pindex, to_pindex, cash, record):
        if from_pindex == to_pindex:
            log.error(f"不能转移资金至同一账户")
            return None
        if cash < 0:
            log.error(f"转移资金金额必须大于0")
            return None
        
        def is_valid_index(index, max_length):
            return 0 <= index < max_length
        if not is_valid_index(from_pindex, len(self._ucontext.subportfolios)):
            log.error(f"账户[{from_pindex}] 不存在")
            return None
        if not is_valid_index(to_pindex, len(self._ucontext.subportfolios)):
            log.error(f"账户[{to_pindex}] 不存在")
            return None
        
        if cash > self._ucontext.subportfolios[from_pindex].available_cash:
            log.error(f"账户[{from_pindex}] 可用资金不足, 需要： {cash}, 可用： {self._ucontext.subportfolios[from_pindex].available_cash}")
            return None
        from_portfolio, to_portfolio = self._ucontext.subportfolios[from_pindex], self._ucontext.subportfolios[to_pindex]
        transfer = UserTransfer(cash, self._ucontext.current_dt, from_pindex, to_pindex, from_portfolio.available_cash, to_portfolio.available_cash)
        
        from_portfolio.withdraw(cash)
        # from_portfolio.available_cash -= cash
        # from_portfolio.cum_cash_outflow += cash
        
        to_portfolio.deposit(cash)
        # to_portfolio.available_cash += cash
        # to_portfolio.cum_cash_inflow += cash
        # to_portfolio.cum_cash_outflow -= cash
        
        if record:
            Env().event_bus.publish_event(Event(EVENT.RECORD_TRANSFER, transfer=transfer))
        return True