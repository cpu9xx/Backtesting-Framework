# -*- coding: utf-8 -*-
from userConfig import userconfig, stock_column_tuple, index_column_tuple, db_config
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
from .object import NumpyFrame, DictedList, Index#, singleton
from .events import EventBus
from .gate import MySQL_gate
import numpy as np
import pandas as pd
import pickle
# import gzip
# import joblib
import os
from functools import lru_cache
import datetime
from collections import defaultdict

config = {
    "mod": {
        "stock": {
            "enabled": True,
        },
        "future": {
            "enabled": False,
        }
    }
}

db_start_date="2015-01-01"
db_end_date="2030-01-01"
db_gate = MySQL_gate(
        start_date=db_start_date, 
        end_date=db_end_date, 
        # end_date="2017-02-01", 
        **db_config
    )


def shift_str_date(date_str: str, days_delta: int):
    date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    new_date_obj = date_obj + datetime.timedelta(days=days_delta)
    return new_date_obj.strftime("%Y-%m-%d")

# @lru_cache(maxsize=8192)
def transform_key(key):
    if key.endswith('XSHE'):
        return key[:-4] + 'sz'
    elif key.endswith('XSHG'):
        return key[:-4] + 'sh'
    return key

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


@lru_cache(maxsize=32)
def create_map_dtype(column_tuple):
    column_map = {}
    dtype = []
    for value, item in enumerate(column_tuple):
        column_map[item[0]] = value
        dtype.append((item[0], item[1]))
    return column_map, dtype

class KeyTransDict(dict):
    def __getitem__(self, key):
        key = transform_key(key)
        return super().__getitem__(key)
    
    def __contains__(self, key):
        key = transform_key(key)
        return super().__contains__(key)

def process_code(args):
    code, db, column_tuple = args
    print(f"{code} data loading...", end="\r")
    df = db_gate.read_df(code, db=db)
    
    if df.empty:
        return
    
    df_new = pd.DataFrame()
    for column in column_tuple:
        col_name, _, col_index = column
        try:
            df_new[col_name] = df.iloc[:, col_index]
        except IndexError:
            raise IndexError(f"Column index {col_index} out of bounds for df with shape {df.shape}")
    df = df_new
    df['ts_code'] = df['ts_code'].apply(trans_name)   
    df['date'] = pd.to_datetime(df['date'])


    _, dtype = create_map_dtype(column_tuple)

    array = np.array([tuple(row) for row in df.to_records(index=False)], dtype=dtype)

    date_set = set(df['date'])
    all_dates = pd.date_range(start=df['date'].min(), end=df['date'].max())
    start_timestamp = all_dates[0]
    index_array = np.full(len(all_dates), -1, dtype=Index)
    i = 0
        
    for timestamp in all_dates:
        # offset: 第几天, i: 第几个交易日; 
        offset = (timestamp - start_timestamp).days
        assert i <= offset < len(index_array)
        if timestamp in date_set:
            index_array[offset] = Index(None, i, None, timestamp)
            i += 1
        else:
            if i == 0:
                index_array[offset] = Index(None, None, i, timestamp)
            elif i < len(date_set):
                index_array[offset] = Index(i-1, None, i, timestamp)
            else:
                index_array[offset] = Index(i-1, None, None, timestamp)
    # for _ in range(60):
    #     try:
    #         env = Env.get_instance()
    #         break
    #     except:
    #         time.sleep(0.2)
    return code, NumpyFrame(array, code, index_array=index_array, start_timestamp=start_timestamp)

def process_cb(cb):
    print(f"{cb} data loading...", end="\r")
    df = db_gate.read_df(cb, db='cb_daily')
    
    if df.empty:
        return
    
    loc_ls = []
    for column in cb_column_tuple:
        loc_ls.append(column[2])
        try:
            # 修改列名
            df.columns.values[column[2]] = column[0]
        except KeyError:
            raise KeyError(f"colume loc {column[2]} not found in {df.columns}")

    df['ts_code'] = df['ts_code'].apply(trans_name)   
    df['date'] = pd.to_datetime(df['date'])

    # 丢弃无关列
    df = df.iloc[:, loc_ls]

    _, dtype = create_map_dtype(cb_column_tuple)

    array = np.array([tuple(row) for row in df.to_records(index=False)], dtype=dtype)

    date_set = set(df['date'])
    all_dates = pd.date_range(start=df['date'].min(), end=df['date'].max())
    start_timestamp = all_dates[0]
    index_array = np.full(len(all_dates), -1, dtype=Index)
    i = 0
        
    for timestamp in all_dates:
        # offset: 第几天, i: 第几个交易日; 
        offset = (timestamp - start_timestamp).days
        assert i <= offset < len(index_array)
        if timestamp in date_set:
            index_array[offset] = Index(None, i, None, timestamp)
            i += 1
        else:
            if i == 0:
                index_array[offset] = Index(None, None, i, timestamp)
            elif i < len(date_set):
                index_array[offset] = Index(i-1, None, i, timestamp)
            else:
                index_array[offset] = Index(i-1, None, None, timestamp)
    # for _ in range(60):
    #     try:
    #         env = Env.get_instance()
    #         break
    #     except:
    #         time.sleep(0.2)
    return cb, NumpyFrame(array, cb, index_array=index_array, start_timestamp=start_timestamp)

def process_stock(stock):
    print(f"{stock} data loading...", end="\r")
    df = db_gate.read_df(stock, db='stock')
    
    if df.empty:
        return None, None
    
    loc_ls = []
    for column in stock_column_tuple:
        loc_ls.append(column[2])
        try:
            # 修改列名
            df.columns.values[column[2]] = column[0]
        except KeyError:
            raise KeyError(f"colume loc {column[2]} not found in {df.columns}")

    df['ts_code'] = df['ts_code'].apply(trans_name)   
    df['date'] = pd.to_datetime(df['date'])

    # 丢弃无关列
    df = df.iloc[:, loc_ls]

    _, dtype = create_map_dtype(stock_column_tuple)

    array = np.array([tuple(row) for row in df.to_records(index=False)], dtype=dtype)

    date_set = set(df['date'])
    all_dates = pd.date_range(start=df['date'].min(), end=df['date'].max())
    start_timestamp = all_dates[0]
    index_array = np.full(len(all_dates), -1, dtype=Index)
    i = 0
        
    for timestamp in all_dates:
        # offset: 第几天, i: 第几个交易日; 
        offset = (timestamp - start_timestamp).days
        assert i <= offset < len(index_array)
        if timestamp in date_set:
            index_array[offset] = Index(None, i, None, timestamp)
            i += 1
        else:
            if i == 0:
                index_array[offset] = Index(None, None, i, timestamp)
            elif i < len(date_set):
                index_array[offset] = Index(i-1, None, i, timestamp)
            else:
                index_array[offset] = Index(i-1, None, None, timestamp)
    # for _ in range(60):
    #     try:
    #         env = Env.get_instance()
    #         break
    #     except:
    #         time.sleep(0.2)
    return stock, NumpyFrame(array, stock, index_array=index_array, start_timestamp=start_timestamp)

def process_basic_df(basic_df):
    basic_dict = {}
    for row in basic_df.itertuples(index=False):
        row_dict = row._asdict()
        basic_dict[trans_name(row.ts_code)] = row_dict
        basic_dict[trans_name(row.stk_code)] = row_dict
    return basic_dict

def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance

@singleton
class Env(object):
    # _env = None

    def __init__(self, config):
        # Env._env = self
        self.config = config
        self.event_bus = EventBus()
        self.usercfg = userconfig
        self.global_vars = None
        self.current_dt = None
        self.trade_dates = None
        self.trade_date_idx = None
        self.event_source = None

        # 用户自定义列名
        column, _ = create_map_dtype(stock_column_tuple)
        self.stock_columns = set(column.keys())
        self.stock_column_tuple = stock_column_tuple

        cb_column, _ = create_map_dtype(cb_column_tuple)
        self.cb_columns = set(cb_column.keys())
        self.cb_column_tuple = cb_column_tuple

        self.data = {}#KeyTransDict()
        self.index_data = {}#KeyTransDict()
        self.cb_data = {}
        self.basic_data = {}
        # extra data
        self.extra_date_db = self.usercfg.get('extra_date_db', [])
        for db in self.extra_date_db:
            setattr(self, db, DictedList())

        self.extra_code_db = self.usercfg.get('extra_code_db', [])
        for db in self.extra_code_db:
            setattr(self, db, {})
            exec(f'from userConfig import {db}_column_tuple', globals())
            column_tuple = globals()[f"{db}_column_tuple"]
            setattr(self, f'{db}_column_tuple', column_tuple)

        self.basic_db = self.usercfg.get('basic_db', [])

    # @classmethod
    # def get_instance(cls):
    #     """
    #     返回已经创建的 Environment 对象
    #     """
    #     # if Env._env is None:
    #     #     raise RuntimeError("策略还未初始化")
    #     return Env()

    def set_global_vars(self, global_vars):
        self.global_vars = global_vars

    def set_event_source(self, event_source):
        self.event_source = event_source

    def set_trade_dates(self, trade_dates, idx):
        self.trade_dates = trade_dates
        self.trade_date_idx = idx
        # print(trade_dates[idx])
        # print(trade_dates[idx])

    def need_cb(self, if_load):
        if hasattr(self, 'cb_data') and len(self.cb_data) != 0:
            # print(self.cb_data)
            # raise
            return True
        else:

            if if_load:
                current_folder_path = r"C:\Users\ccccc\AppData\Local\Programs\Python\Python38\Lib\site-packages\jqdata/"
                with open(current_folder_path+'cb_daily.pkl', 'rb') as f:
                    self.cb_data = pickle.load(f)
            else:
                db_start_date="2017-01-01"
                db_end_date="2040-01-01"
                db_gate = MySQL_gate(
                        start_date=db_start_date, 
                        end_date=db_end_date, 
                        # end_date="2017-02-01", 
                        **db_config
                    )
                
                for db in self.basic_db:
                    basic_df = db_gate.read_entire_df(db, db=db)
                    self.basic_data[db] = process_basic_df(basic_df) 

                # cb_basic_df = db_gate.read_entire_df('cb_basic', db='cb_basic')
                # self.basic_data['cb_basic'] = process_basic_df(cb_basic_df)                

                import multiprocessing as mp
                cb_ls = db_gate.get_tables(db='cb_daily').iloc[:, 0]
                with mp.Pool(processes=mp.cpu_count()-1) as pool:
                    results = pool.map(process_cb, cb_ls)
                print(len(results))
                for cb, result in results:
                    self.cb_data[trans_name(cb)] = result

                with open('cb_daily.pkl', 'wb') as f:
                    pickle.dump(self.cb_data, f, protocol=4)

        return False


    def load_data(self):
        env = Env()

        # 数据库中的数据时间范围
        db_start_date="2015-01-01"
        db_end_date="2040-01-01"
        db_gate = MySQL_gate(
                start_date=db_start_date, 
                end_date=db_end_date, 
                # end_date="2017-02-01", 
                **db_config
            )

        if_load = env.usercfg['if_load_data']
        if if_load:
            print(datetime.datetime.now())
            # import os
            current_folder_path = r"C:\Users\ccccc\AppData\Local\Programs\Python\Python38\Lib\site-packages\jqdata/"#os.path.dirname(__file__) + '/'
            # current_folder_path = ''

            with open(current_folder_path+'data.pkl', 'rb') as f:
                self.data = pickle.load(f)

            with open(current_folder_path+'index_data.pkl', 'rb') as f:
                self.index_data = pickle.load(f)

            with open(current_folder_path+'basic_data.pkl', 'rb') as f:
                self.basic_data = pickle.load(f)
            
            for db in self.extra_code_db:
                with open(current_folder_path+f'{db}.pkl', 'rb') as f:
                    setattr(self, db, pickle.load(f))

            for db in self.extra_date_db:
                with open(current_folder_path+f'{db}.pkl', 'rb') as f:
                    setattr(self, db, pickle.load(f))
            print(datetime.datetime.now())

            
        else:
            import multiprocessing as mp

            stock_ls = db_gate.get_tables(db='stock').iloc[:, 0]
            with mp.Pool(processes=mp.cpu_count()-1) as pool:
                results = pool.map(process_stock, stock_ls)
            print(len(results))
            for stock, result in results:
                if result is not None:
                    self.data[trans_name(stock)] = result            

            # cb_ls = db_gate.get_tables(db='cb_daily').iloc[:, 0]
            # with mp.Pool(processes=mp.cpu_count()-1) as pool:
            #     results = pool.imap_unordered(process_cb, cb_ls)
            # for cb, result in results:
            #     self.cb_data[trans_name(cb)] = result
            for db in self.extra_code_db:
                if not hasattr(self, db):
                    raise ValueError(f"extra code db: {db} must in {self.extra_code_db}")
                code_ls = db_gate.get_tables(db=db).iloc[:, 0]
                column_tuple = getattr(self, f'{db}_column_tuple')
                object_ls = [(code, db, column_tuple) for code in code_ls]
                # a, b = process_code(object_ls[0])
                # print(a, b)
                with mp.Pool(processes=mp.cpu_count()-1) as pool:
                    results = pool.imap_unordered(process_code, object_ls)

                    for code, result in results:
                        getattr(self, db)[trans_name(code)] = result


            index_ls = db_gate.get_tables(db='index').iloc[:, 0]
            for index in index_ls:
                print(f"{index} data loading...", end="\r")
                df = db_gate.read_df(index, db='index')
                if df.empty:
                    continue
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df['date'] = df['trade_date']
                df.set_index('trade_date', inplace=True)
                self.index_data[trans_name(index)] = df

            for db in self.extra_date_db:
                table_ls = db_gate.get_tables(db=db).iloc[:, 0]
                for table in table_ls:
                    print(f"{db}.{table} data loading...", end="\r")
                    df = db_gate.read_entire_df(table, db=db)
                    if df.empty:
                        continue
                    if not hasattr(self, db):
                        raise ValueError(f"extra date db: {db} must in {self.extra_date_db}")
                    getattr(self, db)[table] = df

            self._dump_data()
            pass

        if_need_cb = env.usercfg.get('if_need_cb', False)
        if if_need_cb:
            self.need_cb(if_load=if_load)

            

    def _dump_data(self):
        # joblib.dump(self.data, 'data.pkl')
        # joblib.dump(self.index_data, 'index_data.pkl')
        with open('data.pkl', 'wb') as f:
            pickle.dump(self.data, f, protocol=4)      
        with open('index_data.pkl', 'wb') as f:
            pickle.dump(self.index_data, f, protocol=4)
        with open('cb_daily.pkl', 'wb') as f:
            pickle.dump(self.cb_data, f, protocol=4)      

        with open('basic_data.pkl', 'wb') as f:
            pickle.dump(self.basic_data, f, protocol=4)


        for db in self.extra_code_db:    
            with open(f'{db}.pkl', 'wb') as f:
                pickle.dump(getattr(self, db), f, protocol=4)

        for db in self.extra_date_db:    
            with open(f'{db}.pkl', 'wb') as f:
                pickle.dump(getattr(self, db), f, protocol=4)

        # with gzip.open('data.gz', 'wb') as f:
        #     pickle.dump(self.data, f, protocol=pickle.HIGHEST_PROTOCOL)
        # with gzip.open('index_data.gz', 'wb') as f:
        #     pickle.dump(self.index_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
