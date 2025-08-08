# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from functools import lru_cache

from .Env import Env
from .events import Event, EVENT
from .logger import log
from .setting import setting, OrderCost, SubPortfolioConfig
from .data import Data
_scheduler = None # main.py中的scheduler
_data = None
_strategy = None
_broker = None
_recorder = None
# setting = Setting()



##################################### time ################################
def run_daily(func, time, reference_security='000300.XSHG'):
    _scheduler._run_daily(func, time, reference_security)

def get_offset_trade_date(offset):
    curr_trade_date_idx = Env().trade_date_idx
    offset_trade_date_idx = curr_trade_date_idx + offset
    if offset_trade_date_idx < 0 or offset_trade_date_idx >= len(Env().trade_dates):
        log.warning(f"offset_trade_date_idx out of range[0, {len(Env().trade_dates)-1}]: {offset_trade_date_idx}, return None")
        return None
    return Env().trade_dates[offset_trade_date_idx]

##################################### time ################################




##################################### data ################################
def create_get_data_by_code_funcs(dbs):
    for db in dbs:
        def get_func(security, start_date=None, end_date=None, frequency='daily', fields=None, skip_paused=False, fq='post', count=None, panel=False, fill_paused=True, df=True, db=db):
            return getattr(_data, f"_get_{db}")(security, start_date, end_date, frequency, fields, skip_paused, fq, count, panel, fill_paused, df)
        globals()[f"get_{db}"] = get_func

def create_get_data_by_date_funcs(dbs):
    for db in dbs:
        def get_func(start_date, end_date=None, offset=0, db=db):
            return getattr(_data, f"_get_{db}")(start_date, end_date, offset)
        globals()[f"get_{db}"] = get_func

def create_get_basic_data_funcs(dbs):
    for db in dbs:
        def get_func(code=None):
            return getattr(_data, f"_get_{db}")(code)
        globals()[f"get_{db}"] = get_func

# 返回行数与count强相关，不检查时期，退市股票仍能取到数据，设置 df=False 时性能强，推荐优先使用
def get_bars(security, count, unit='1d', fields=['close'], include_now=False, end_dt=None, fq_ref_date=None, df=False):
    return _data._get_bars(security, count, unit, fields, include_now, end_dt, fq_ref_date, df)


# 取不到退市数据，性能差, 包含end_date当天数据
def get_price(security, start_date=None, end_date=None, frequency='daily', fields=None, skip_paused=False, fq='post', count=None, panel=False, fill_paused=True, df=True):
    return _data._get_price(security, start_date, end_date, frequency, fields, skip_paused, fq, count, panel, fill_paused, df)

def get_all_securities(types='stock'):
    return _data._get_all_securities(types)

def get_all_trade_days():
    return _data._get_all_trade_days()

def get_hm_detail(date):
    return _data._get_hm_detail(date)

@lru_cache(maxsize=8192)
def trans_name(name):
    if name.lower().endswith('sz'):
        return name[:-2] + 'XSHE'
    elif name.lower().endswith('sh'):
        return name[:-2] + 'XSHG'
    elif name.endswith('XSHE'):
        return name[:6] + '.SZ'
    elif name.endswith('XSHG'):
        return name[:6] + '.SH'
    return name

# 取不到退市数据，不包含当日数据,即使是在收盘后
def history(count, unit='1d', field='avg', security_list=None, df=True, skip_paused=False, fq='pre'):
    return _data._history(count, unit, field, security_list, df, skip_paused, fq)

def filter_kcbj_stock(stock_list):
    return _data._filter_kcbj_stock(stock_list)

def b_filter(stock_list):
    return _data._b_filter(stock_list)

def bj_filter(stock_list):
    return _data._bj_filter(stock_list)

def kc_filter(stock_list):
    return _data._kc_filter(stock_list)

def cy_filter(stock_list):
    return _data._cy_filter(stock_list)

def get_limit_price(code, name, pre_close=None):
    return _data._get_limit_price(code, name, pre_close)

def get_limit_up_price(code, name, pre_close=None):
    return _data._get_limit_up_price(code, name, pre_close)

def get_days_to_delist(security, date):
    return _data._get_days_to_delist(security, date)
##################################### data ################################






##################################### strategy ################################
def order(security, amount, style=None, side='long', pindex=0, close_today=False, record=True):
    return _strategy._order(security, amount, style, side, pindex, close_today, record)

def order_target(security, target, style=None, side='long', pindex=0, close_today=False, record=True):
    return _strategy._order_target(security, target, style, side, pindex, close_today, record)

def order_value(security, value, style=None, side='long', pindex=0, close_today=False, record=True):    
    return _strategy._order_value(security, value, style, side, pindex, close_today, record)

def order_target_value(security, value, style=None, side='long', pindex=0, close_today=False, record=True):    
    return _strategy._order_target_value(security, value, style, side, pindex, close_today, record)

def convert_bond(security, amount, style=None, side='long', pindex=0, close_today=False, record=True):
    return _strategy._convert_bond(security, amount, style, side, pindex, close_today)

def transfer_cash(from_pindex, to_pindex, cash, record=True):
    return _strategy._transfer_cash(from_pindex, to_pindex, cash, record)
##################################### strategy ################################



##################################### broker ################################
def get_trades():
    return _broker._get_trades()
##################################### broker ################################





##################################### setting ################################
def set_order_cost(cost, type: str, ref=None):
    setting._set_order_cost(cost, type, ref)

def set_benchmark(security):
    setting._set_benchmark(security)

def set_option(option: str, value):
    setting._set_option(option, value)

def set_subportfolios(subportfolioconfig_ls):
    setting._set_subportfolios(subportfolioconfig_ls)
##################################### setting ################################



##################################### recorder ################################
def get_trade_count_info(security=None):
    return _recorder._get_trade_count_info(security)

def get_historical_returns(count, pindex=-1):
    return _recorder._get_historical_returns(count, pindex)

def get_win_rate(pindex=-1, spin=20):
    return _recorder._get_win_rate(pindex, spin)

def get_pl_ratio(pindex=-1, spin=20):
    return _recorder._get_pl_ratio(pindex, spin)

def get_avg_hoding_days(pindex=-1, spin=20):
    return _recorder._get_avg_hoding_days(pindex, spin)
    
def get_expected_profit(pindex=-1, spin=20):
    return _recorder._get_expected_profit(pindex, spin)



