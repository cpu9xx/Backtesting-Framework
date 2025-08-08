from .Env import Env
from .logger import log
from .setting import setting
from userConfig import stock_column_tuple
try:
    from userConfig import cb_column_tuple
except:
    cb_column_tuple = (
        #'ts_code', 'date', 'open', 'close'是必需的
        ('ts_code', 'U11', 0),
        ('date', 'datetime64[s]', 1), 
        ('pre_close', 'float64', 2),
        ('open', 'float64', 3),
        ('high', 'float64', 4),
        ('low', 'float64', 5),
        ('close', 'float64', 6),
        ('change', 'float64', 7),
        ('pct_chg', 'float64', 8),
        ('volume', 'float64', 9),
        ('money', 'float64', 10),
        ('bond_value', 'float64', 11),
        ('bond_over_rate', 'float64', 12),
        ('cb_value', 'float64', 13),
        ('cb_over_rate', 'float64', 14),
    )
from .object import TIME

import pandas as pd
import numpy as np
# from pathos.multiprocessing import ProcessingPool as Pool
# from multiprocessing import Pool
from functools import partial, lru_cache
import datetime

@lru_cache(maxsize=32)
def get_dtype(fields, column_tuple):
    dtype = []
    for f in fields:
        for item in column_tuple:
            if item[0] == f:
                dtype.append((item[0], item[1]))
                break
    return dtype


def multi_get_price_by_index(security, data, fields, reshape, newindex):
    # env = Env()
    # print(security, data)
    res, _ = data[newindex[0]:newindex[-1], fields]
    trade_dates, _ = data[newindex[0]:newindex[-1], 'date']

    # 如果数据不全
    if len(res) != len(newindex):
        # 为nanarray指定dtype
        if isinstance(fields, str):
            nanarray = np.full(len(newindex), np.nan)
        else:
            dtype = []
            for f in fields:
                for item in stock_column_tuple:
                    if item[0] == f:
                        dtype.append((item[0], item[1]))
                        break

            nanarray = np.full(len(newindex), np.nan, dtype=dtype)

        # 补全数据
        if len(res) == 0:
            res = nanarray
        elif len(res) < len(newindex):
            trade_dates_i = 0
            for i, expected_date in enumerate(newindex):
                if trade_dates_i >= len(trade_dates):
                    break
                elif expected_date == trade_dates[trade_dates_i]:
                    nanarray[i] = res[trade_dates_i]
                    trade_dates_i += 1
            res = nanarray      
        elif len(res) > len(newindex):
            raise ValueError('res length is greater than newindex length')
    
    if reshape is not None:    
        return res.reshape(reshape)
    return res

def get_price_by_index(data, fields, reshape, newindex, column_tuple):
    if isinstance(fields, str):
        fields_with_date = [fields, 'date']
    else:
        fields_with_date = fields + ['date']

    # res, _ = data[security][newindex[0]:newindex[-1], fields]
    # trade_dates, _ = data[security][newindex[0]:newindex[-1], 'date']
    res_with_date, _ = data[newindex[0]:newindex[-1], fields_with_date]
    trade_dates = res_with_date['date']
    res = res_with_date[fields]
    # 如果数据不全
    if len(res) != len(newindex):
        # 为nanarray指定dtype
        if isinstance(fields, str):
            nanarray = np.full(len(newindex), np.nan)
        else:
            dtype = get_dtype(tuple(fields), column_tuple)
            nanarray = np.full(len(newindex), np.nan, dtype=dtype)

        # 补全数据
        if len(res) == 0:
            res = nanarray
        elif len(res) < len(newindex):
            """
            下一步优化：在读取数据时自动填充好停牌日期的数据，而不是在这里补全
            """
            trade_dates_i = 0
            for i, expected_date in enumerate(newindex):
                if trade_dates_i >= len(trade_dates):
                    break
                elif expected_date == trade_dates[trade_dates_i]:
                    nanarray[i] = res[trade_dates_i]
                    trade_dates_i += 1
            # positions = np.searchsorted(newindex, trade_dates)
            # # 过滤非法索引（例如重复或溢出）
            # valid = (positions >= 0) & (positions < len(newindex))
            # positions = positions[valid]

            # # 填充 nanarray 对应位置
            # nanarray[positions] = res[valid]
            res = nanarray      
        elif len(res) > len(newindex):
            raise ValueError('res length is greater than newindex length')

    if reshape is not None:    
        return res.reshape(reshape)
    return res



        
def get_bar_by_count(security, end_dt, fields, count, df):
    env = Env()
    if df:
        return pd.DataFrame(data=env.data[security][:end_dt, fields][0][-count:], columns=fields)
    else:
        return env.data[security][:end_dt, fields][0][-count:]#.to_records(index=False)


def get_index(start_date, end_date, count):
    env = Env()
    # end_dt = pd.Timestamp(env.current_dt.date())#pd.Timestamp(self._ucontext.previous_date)
    benchmark = setting.get_benchmark()
    benchmark_index = env.index_data[benchmark].index

    end_date = pd.Timestamp(end_date) if end_date is not None else pd.Timestamp(env.current_dt.date())
    if start_date and count:
        log.error('start_date and count cannot be set at the same time')
    elif start_date:
        start_date = pd.Timestamp(start_date)
        if start_date.time() != TIME.DAY_START:
            log.warning("Time accuracy current version supported in start_date of get_price() is only to the day, time will be ignored")
            start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
        
        last_index = benchmark_index.get_loc(end_date)+1
        first_index = benchmark_index.get_loc(start_date)
        newindex = benchmark_index[first_index:last_index]
    elif count:
        # log.test(benchmark_index)
        last_index = benchmark_index.get_loc(end_date)+1
        # log.test(end_date)
        # log.test(last_index)
        if count > 0:
            if count > last_index:
                log.warning(f"get_price() count {count} is larger than the available data, current last_index {last_index}, using all available data")
                newindex = benchmark_index[0:last_index]
            else:
                newindex = benchmark_index[last_index-count:last_index]
            # log.test(newindex)
        else:
            # log.warning(f"get_price() 使用了 {last_index}({end_date}+1) - {last_index-count} 的未来数据")
            newindex = benchmark_index[last_index:last_index-count]
    else:
        log.error('start_date or count should be set')
    # log.test(end_date, benchmark_index[last_index])
    return newindex

class Data(object):
    def __init__(self):
        self._ucontext = None
        self.create_get_data_by_code_funcs(Env().extra_code_db)
        self.create_get_data_by_date_funcs(Env().extra_date_db)
        self.create_get_basic_data_funcs(Env().basic_db)
        # print(hasattr(self, '_get_imp_dividend'))
        # print(hasattr(self, '_get_hm_detail'))
        # self._get_imp_dividend('20210101')
        # pass
        # self.pool = Pool(processes=25)

    def date2str(self, date):
        if not (isinstance(date, str) and len(date) == 8):
            if isinstance(date, datetime.date) or isinstance(date, datetime.datetime):
                date = date.strftime('%Y%m%d')
            else:
                raise ValueError("date should be datetime.datetime or datetime.date or str of format 'YYYYMMDD'")
        return date
     
    def str2date(self, date):
        if isinstance(date, datetime.date) or isinstance(date, datetime.datetime):
            return date
        if isinstance(date, str) and len(date) == 8:
            return datetime.datetime.strptime(date, '%Y%m%d')
        raise ValueError("date_str should be a string of format 'YYYYMMDD' or a datetime object")  


    def create_get_data_by_code_funcs(self, dbs):
        for db in dbs:
            func_name = f"_get_{db}"

            def make_func(db_name):
                def func(self, security, start_date, end_date, frequency, fields, skip_paused, fq, count, panel, fill_paused, df):
                    if isinstance(fields, str):
                        fields = [fields]

                    # column_tuple = env.__getattribute__(f'{db_name}_column_tuple')
                    # for f in fields:
                    #     if f not in column_tuple:
                    #         log.error(f'field should be one of {column_tuple}')

                    if skip_paused or not fill_paused:
                        log.warning('skip_paused = True and fill_paused = False are not supported in this version, missing data will be filled with NaN')

                    newindex = get_index(start_date, end_date, count)
                    env = Env()
                    
                    if isinstance(security, str):
                        data = env.__getattribute__(db_name).get(security, None)
                        if data is None:
                            # log.error(f"security {security} not found in env.{db_name}")
                            res = None
                        else:
                            column_tuple = env.__getattribute__(f'{db_name}_column_tuple')
                            res = get_price_by_index(data=data, fields=fields, reshape=None, newindex=newindex, column_tuple=column_tuple)
                            if df:
                                res = pd.DataFrame(res, index=newindex, columns=fields)                     
                    else:
                        if df:
                            res_ls = [None] * len(security)
                        else:
                            res = {}

                        for idx, s in enumerate(security):
                            data = env.__getattribute__(db_name).get(s, None)
                            if data is None:
                                # log.error(f"security {s} not found in env.{db_name}")
                                resarray = None
                            else:
                                column_tuple = env.__getattribute__(f'{db_name}_column_tuple')
                                resarray = get_price_by_index(data=data, fields=fields, reshape=None, newindex=newindex, column_tuple=column_tuple)

                            if df:
                                res_ls[idx] = resarray
                            else:
                                res[s] = resarray    

                        if df:
                            res = np.concatenate(res_ls, axis=0)
                            res = pd.DataFrame(res, columns=fields)#.rename(columns={'date': 'time'})
                            res.insert(0, 'time', np.tile(newindex, len(security)))
                            name_ls = []
                            for s in security:
                                name_ls += [s] * len(newindex)
                            res.insert(1, 'code', name_ls)
                    return res
                return func

            bound_method = make_func(db).__get__(self, self.__class__)
            setattr(self, func_name, bound_method)

    def create_get_data_by_date_funcs(self, dbs):
        for db in dbs:
            func_name = f"_get_{db}"

            def make_func(db_name):
                def func(self, start_date, end_date, offset):
                    env = Env()

                    if end_date is None:
                        start_date = self.date2str(start_date)
                        return env.__getattribute__(db_name).get(start_date, None, offset)
                    else:
                        if offset != 0:
                            log.error('offset is not supported when end_date is set!')
                        df_ls = []
                        start_date = self.str2date(start_date)
                        end_date = self.str2date(end_date)
                        current_date = start_date
                        while current_date <= end_date:
                            df = env.__getattribute__(db_name).get(self.date2str(current_date), None)
                            if df is not None:
                                df_ls.append(df)
                            current_date += datetime.timedelta(days=1)
                        return df_ls
                return func

            bound_method = make_func(db).__get__(self, self.__class__)
            setattr(self, func_name, bound_method)
        
    def create_get_basic_data_funcs(self, dbs):
        for db in dbs:
            func_name = f"_get_{db}"

            def make_func(db_name):
                def func(self, code):
                    env = Env()
                    basic_dict = env.basic_data.get(db_name, None)
                    if code is None:
                        return basic_dict
                    else:
                        if code not in basic_dict:
                            log.error(f"KeyError: code {code} not found in env.{db_name}")
                        return basic_dict[code]
                return func

            bound_method = make_func(db).__get__(self, self.__class__)
            setattr(self, func_name, bound_method)

    def set_user_context(self, ucontext):
        self._ucontext = ucontext

    # def _get_hm_detail(self, date):
    #     env = Env()
    #     if isinstance(date, datetime.datetime) or isinstance(date, datetime.date):
    #         date = date.strftime('%Y%m%d')
    #     else:
    #         raise ValueError('date should be datetime.datetime or datetime.date')
    #     # log.test(date)
    #     df = env.hm_detail.get(date, None)
    #     return df

    def _get_price(self, security, start_date=None, end_date=None, frequency='daily', fields=None, skip_paused=False, fq='post', count=None, panel=False, fill_paused=True, df=True):
        if isinstance(fields, str):
            fields = [fields]

        # stock_columns = Env().stock_columns
        # for f in fields:
        #     if f not in stock_columns:
        #         log.error(f'field should be one of {stock_columns}')

        if skip_paused or not fill_paused:
            log.warning('skip_paused = True and fill_paused = False are not supported in this version, missing data will be filled with NaN')

        newindex = get_index(start_date, end_date, count)
        env = Env()
        
        if isinstance(security, str):
            if security in env.data:
                data = env.data[security]
                column_tuple = stock_column_tuple
            elif security in env.cb_data:
                data = env.cb_data[security]
                column_tuple = cb_column_tuple
            else:
                log.error(f"security {security} not found")
            res = get_price_by_index(data=data, fields=fields, reshape=None, newindex=newindex, column_tuple=column_tuple)
            if df:
                res = pd.DataFrame(res, index=newindex, columns=fields)
            else:
                return res
            
        else:
            if df:
                res_ls = [None] * len(security)
            else:
                res = {}
            # log.live('getprice 1')
            for idx, s in enumerate(security):
                if s in env.data:
                    data = env.data[s]
                    column_tuple = stock_column_tuple
                elif s in env.cb_data:
                    data = env.cb_data[s]
                    column_tuple = cb_column_tuple
                else:
                    log.error(f"security {s} not found")
                resarray = get_price_by_index(data=data, fields=fields, reshape=None, newindex=newindex, column_tuple=column_tuple)
                if df:
                    res_ls[idx] = resarray
                else:
                    res[s] = resarray    

            
            # log.live('getprice 2')

            if df:
                res = np.concatenate(res_ls, axis=0)
                res = pd.DataFrame(res, columns=fields)#.rename(columns={'date': 'time'})
                res.insert(0, 'time', np.tile(newindex, len(security)))
                name_ls = []
                for s in security:
                    name_ls += [s] * len(newindex)
                res.insert(1, 'code', name_ls)
                # log.live('getprice 3')

        return res

    def _history(self, count, unit='1d', field='avg', security_list=None, df=True, skip_paused=False, fq='pre'):
        # ['open', 'close', 'high', 'low', 'volume', 'money', 'avg', 'high_limit', 'low_limit', 'pre_close', 'paused', 'factor', 'open_interest']
        if not isinstance(field, str) or field not in Env().stock_columns:
            log.error(f'field should be one of {Env().stock_columns}')

        env = Env()
        end_dt = pd.Timestamp(env.current_dt.date())#pd.Timestamp(self._ucontext.previous_date)
        benchmark = setting.get_benchmark()
        benchmark_index = env.index_data[benchmark].index

        last_index = benchmark_index.get_loc(end_dt)
        newindex = benchmark_index[last_index-count:last_index]
        if isinstance(security_list, str):
            security_list = [security_list]
        if df:
            res_ls = [None] * len(security_list)
            # log.live(1)
            # pool = Pool(processes=20)
            # multi_get_price= partial(multi_get_price_by_index, fields=field, reshape=(len(newindex), 1), newindex=newindex)
            # res_ls = self.pool.map(multi_get_price, security_list, list(map(env.data.get, security_list)))

            # pool.close()
            for idx, s in enumerate(security_list):
                resarray = get_price_by_index(security=s, fields=field, reshape=(len(newindex), 1), newindex=newindex, data=env.data)
                res_ls[idx] = resarray    
            # log.live(2)
            res = np.concatenate(res_ls, axis=1)
            res = pd.DataFrame(res, index=newindex, columns=security_list)
            # print(res)
            # log.live(3)
            pass
        else:
            res = {}
            # log.live(1)
            for s in security_list:
                res[s] = get_price_by_index(security=s, fields=field, reshape=None, newindex=newindex, data=env.data)
            # log.live(2)
        return res

    def _get_bars(self, security, count, unit='1d', fields=['close'], include_now=False, end_dt=None, fq_ref_date=None, df=False):
        end_dt = pd.to_datetime(end_dt) if end_dt else pd.to_datetime(self._ucontext.current_dt)

        if not include_now:
            end_dt -= pd.Timedelta(days=1)

        # 处理数据
        if isinstance(security, str):
            res = get_bar_by_count(security, end_dt, fields, count, df)
        else:
            res = {s: get_bar_by_count(s, end_dt, fields, count, df) for s in security}

        return res
    
    
    def _get_all_securities(self, types='stock'):
        env = Env()

        if isinstance(types, str):
            types = [types]  # 转为列表统一处理

        data_sources = {
            'stock': env.data,
            'index': env.index_data,
            'cb': env.cb_data,
        }

        all_keys = set()
        for t in types:
            if t not in data_sources:
                raise ValueError(f'Unsupported type: {t}')
            all_keys.update(data_sources[t].keys())

        return pd.DataFrame(index=list(all_keys))
    
    def _get_all_trade_days(self):
        env = Env()
        benchmark = setting.get_benchmark()
        return np.array(env.index_data[benchmark].index.date)

    def _filter_kcbj_stock(self, stock_list):
        new_list = []
        for stock in stock_list:
            if stock[0] != '4' and stock[0] != '8' and stock[0] != '92' and stock[:2] != '68':
                new_list.append(stock)
        return new_list
    
    def _b_filter(self, stock_list):
        new_list = []
        for stock in stock_list:
            if stock[0] != '2' and stock[:2] != '90':
                new_list.append(stock)
        return new_list

    def _bj_filter(self, stock_list):
        new_list = []
        for stock in stock_list:
            if stock[0] != '4' and stock[0] != '8' and stock[:2] != '92':
                new_list.append(stock)
        return new_list
    
    def _kc_filter(self, stock_list):
        new_list = []
        for stock in stock_list:
            if stock[:2] != '68':
                new_list.append(stock)
        return new_list
    
    def _cy_filter(self, stock_list):
        new_list = []
        for stock in stock_list:
            if stock[0] != '3':
                new_list.append(stock)
        return new_list
    
    def _get_limit_price(self, code, name, pre_close):
        if pre_close is None:
            pre_close = self._get_price(code, count=1, fields='pre_close', df=False)['pre_close'][0]
        if code.startswith('68'):
            limit_up = pre_close * 1.2
            limit_down = pre_close * 0.8
            # return round(pre_close * 1.2, 2), round(pre_close * 0.8, 2)
        elif code.startswith('1'):
            if self._ucontext.current_dt >= datetime.datetime(2022, 8, 1):
                limit_up = pre_close * 1.2
                limit_down = pre_close * 0.8
                # return round(pre_close * 1.2, 2), round(pre_close * 0.8, 2)
            else:
                limit_up = pre_close * 5
                limit_down = pre_close * 0.1
                # return round(pre_close * 5, 2), round(pre_close * 0.1, 2)
        elif code.startswith('3') and self._ucontext.current_dt >= datetime.datetime(2020, 8, 24):
            limit_up = pre_close * 1.2
            limit_down = pre_close * 0.8
            # return round(pre_close * 1.2, 2), round(pre_close * 0.8, 2)
        elif 'ST' in name:
            limit_up = pre_close * 1.05
            limit_down = pre_close * 0.95
            # return round(pre_close * 1.05, 2), round(pre_close * 0.95, 2)
        else:
            limit_up = pre_close * 1.1
            limit_down = pre_close * 0.9
        return round(round(limit_up,6), 2), round(round(limit_down,6), 2)

    def _get_limit_up_price(self, code, name, pre_close):
        price, _ = self._get_limit_price(code, name, pre_close)
        return price
        # if pre_close is None:
        #     pre_close = self._get_price(code, count=1, fields='pre_close', df=False)['pre_close'][0]
        # if code.startswith('68'):
        #     return round(pre_close * 1.2, 2)
        # if code.startswith('1'):
        #     if self._ucontext.current_dt >= datetime.datetime(2022, 8, 1):
        #         return round(pre_close * 1.2, 2)
        #     else:
        #         return round(pre_close * 5, 2)
        # if code.startswith('3') and self._ucontext.current_dt >= datetime.datetime(2020, 8, 24):
        #     return round(pre_close * 1.2, 2)
        # if 'ST' in name:
        #     return round(pre_close * 1.05, 2)
        
        # return round(pre_close * 1.1, 2)
    
    def _get_days_to_delist(self, security, date):
        if isinstance(date, datetime.date) or isinstance(date, datetime.datetime):
            date = pd.Timestamp(date)
        if not isinstance(date, pd.Timestamp):
            log.error('date should be a pandas.Timestamp object')
        env = Env()

        if isinstance(security, str):
            if security in env.data:
                data = env.data[security]
            elif security in env.cb_data:
                data = env.cb_data[security]
            else:
                log.error(f"security {security} not found")

            offset = (date - data.start_timestamp).days
            res = len(data.index) - offset
        else:
            res = {}
            for s in security:
                if s in env.data:
                    data = env.data[s]
                elif s in env.cb_data:
                    data = env.cb_data[s]
                else:
                    log.error(f"security {s} not found")

                offset = (date - data.start_timestamp).days
                res[s] = len(data.index) - offset 

        return res


