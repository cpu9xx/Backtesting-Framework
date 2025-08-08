from .Env import Env
from .events import Event, EVENT
from .logger import log
from .api import setting
from .object import TIME

import datetime
from collections import defaultdict

    

class Scheduler(object):
    def __init__(self):
        # self._registry = []
        # self._today = None
        # self._this_week = None
        # self._this_month = None
        # self._last_minute = 0
        # self._current_minute = 0
        # self._stage = None
        self._ucontext = None
        self.event_src = defaultdict(lambda: defaultdict(list))
        # self._frequency = frequency
        env = Env()
        self.start = datetime.datetime.strptime(env.usercfg['start'], "%Y%m%d")
        self.end = datetime.datetime.strptime(env.usercfg['end'], "%Y%m%d")
        event_bus = env.event_bus

        event_bus.add_listener(EVENT.TIME, self._run_at_time)
        # event_bus.add_listener(EVENT.TIME, self._run_before_open)
        # event_bus.add_listener(EVENT.TIME, self._run_open)
        # event_bus.add_listener(EVENT.TIME, self._run_close)
        # event_bus.add_listener(EVENT.TIME, self._run_after_close)

        self.finished_necessarily = False
        # event_bus.add_listener(EVENT.BEFORE_TRADING, self.before_trading_)
        # event_bus.add_listener(EVENT.BAR, self.next_bar_)

    def set_user_context(self, ucontext):
        self._ucontext = ucontext

    def _necessarily(self, trade_dates):
        # FIRST_TICK触发初始化
        first_tick_time = trade_dates[0] + datetime.timedelta(days=0, hours=TIME.OPEN_AUCTION_END.hour, minutes=TIME.OPEN_AUCTION_END.minute)
        self.event_src[trade_dates[0]][TIME.OPEN_AUCTION_END].append(Event(EVENT.FIRST_TICK, func=None, time=first_tick_time))
        for date in trade_dates:
            # 告诉 broker开盘了, 处理积压订单
            self.event_src[date][TIME.OPEN_AUCTION_END].append(Event(EVENT.OPEN_AUCTION_END, func=None, time=date + datetime.timedelta(days=0, hours=TIME.OPEN_AUCTION_END.hour, minutes=TIME.OPEN_AUCTION_END.minute)))
            # 告诉 broker开盘了, 处理积压订单
            self.event_src[date][TIME.OPEN].append(Event(EVENT.MARKET_OPEN, func=None, time=date + datetime.timedelta(days=0, hours=TIME.OPEN.hour, minutes=TIME.OPEN.minute)))
            # 告诉 broker收盘了, 处理未成交订单
            self.event_src[date][TIME.CLOSE].append(Event(EVENT.MARKET_CLOSE, func=None, time=date + datetime.timedelta(days=0, hours=TIME.CLOSE.hour, minutes=TIME.CLOSE.minute)))
            # 卖单对应的仓位结算
            self.event_src[date][TIME.DAY_END].append(Event(EVENT.DAY_END, func=None, time=date + datetime.timedelta(days=0, hours=TIME.DAY_END.hour, minutes=TIME.DAY_END.minute)))

    def _run_daily(self, func, time, reference_security):
        env = Env()
        trade_dates = env.index_data[reference_security].loc[self.start:self.end, :].index # reference_security的trade dates
        trade_dates = trade_dates.to_pydatetime().tolist()
        if not self.finished_necessarily:
            self._necessarily(trade_dates)
            self.finished_necessarily = True
        
        for date in trade_dates:
            time_type = getattr(TIME, time.upper())
            t = date + datetime.timedelta(days=0, hours=time_type.hour, minutes=time_type.minute)
            self.event_src[date][time_type].append(Event(EVENT.TIME, func=func, time=t))


    def start_event_src(self):
        env = Env()
        event_bus = env.event_bus
        # for i in range((self.end - self.start).days + 1): 
        #     date = self.start + datetime.timedelta(days=i)
        if self.start == self.end:
            trade_dates = env.trade_dates[(env.trade_dates == self.start)].to_pydatetime().tolist()
        else:
            trade_dates = env.trade_dates[(env.trade_dates >= self.start) & (env.trade_dates <= self.end)].to_pydatetime().tolist() 
        # print(trade_dates[-1])
        # print(env.trade_dates[-1])
        # trade_dates = env.trade_dates.to_pydatetime().tolist() # benchmark的trade dates
        for idx, date in enumerate(trade_dates):
            
            events_dict = self.event_src.get(date, {})
            for time_type in TIME:
                events = events_dict.get(time_type, [])
                for event in events:
                    time = event.__dict__['time']
                    assert time_type.value == time.time()
                    if self._ucontext.current_dt:
                        assert time >= self._ucontext.current_dt
                        if time.date() > self._ucontext.current_dt.date():
                            self._ucontext.previous_date = self._ucontext.current_dt.date()
                    else:
                        # 第一个事件触发前，获取回测范围外的前一个交易
                        benchmark = setting.get_benchmark()
                        start_position = env.index_data[benchmark].index.get_loc(trade_dates[0])
                        self._ucontext.previous_date = env.index_data[benchmark].index[start_position-1].date()

                    # 回测范围内的第一个交易日，first_tick
                    self._ucontext.current_dt = time
                    env.current_dt = time
                    event_bus.publish_event(event)
            env.trade_date_idx += 1
            #每个交易日最后执行
            # aft_close_time = date + datetime.timedelta(days=0, hours=TIME.AFTER_CLOSE.hour, minutes=TIME.AFTER_CLOSE.minute)
            # self._ucontext.current_dt = aft_close_time
            # env.current_dt = aft_close_time


            # event_bus.publish_event(Event(EVENT.MARKET_CLOSE))
            
    def _run_at_time(self, event):
        time = event.__dict__['time']
        if time == self._ucontext.current_dt:
            func = event.__dict__['func']
            func(self._ucontext)
            return True
        else:
            log.error("Scheduler {time} time error")

    def _run_before_open(self, event):
        time = event.__dict__['time']
        if time == self._ucontext.current_dt:
            func = event.__dict__['func']
            func(self._ucontext)
            return True
        else:
            log.error("Scheduler before_open time error")

    def _run_open(self, event):
        time = event.__dict__['time']
        if time == self._ucontext.current_dt:
            func = event.__dict__['func']
            func(self._ucontext)
            return True
        else:
            log.error("Scheduler open time error")


    def _run_close(self, event):
        time = event.__dict__['time']
        if time == self._ucontext.current_dt:
            func = event.__dict__['func']
            func(self._ucontext)
            return True
        else:
            log.error("Scheduler close time error")

    def _run_after_close(self, event):
        time = event.__dict__['time']
        if time == self._ucontext.current_dt:
            func = event.__dict__['func']
            func(self._ucontext)
            return True
        else:
            log.error("Scheduler after_close time error")