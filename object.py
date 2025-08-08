from userConfig import userconfig

import numpy as np
import pandas as pd
from typing import Union
import datetime
from enum import Enum

class UserTransfer():
    _transfer_id = 0
    def __init__(self, cash, time, from_pindex, to_pindex, from_cash, to_cash):
        self.time = time
        self.info = (cash, from_pindex, to_pindex, from_cash, to_cash)
        self.transfer_id = UserTransfer._transfer_id
        UserTransfer._transfer_id += 1

    @classmethod
    def get_current_id(cls):
        return cls._transfer_id
    
    @classmethod
    def cls_get_state(cls):
        return {'transfer_id': cls._transfer_id}
    
    @classmethod
    def cls_set_state(cls, state):
        cls._transfer_id = state['transfer_id']

    def __repr__(self):
        return f"UserTransfer(cash={self.info[0]}, time={self.time}, from P[{self.info[1]}] transfer to P[{self.info[2]}])"


class UserTrade(object):
    _trade_id = 0
    def __init__(self, order, time):
        self.time = time
        self.order = order
        self.trade_id = UserTrade._trade_id
        UserTrade._trade_id += 1

    @classmethod
    def get_current_id(cls):
        return cls._trade_id
    
    @classmethod
    def cls_get_state(cls):
        return {'trade_id': cls._trade_id}
    
    @classmethod
    def cls_set_state(cls, state):
        cls._trade_id = state['trade_id']

    def __repr__(self):
        return f"UserTrade(trder_id={self.trade_id}, order_id={self.order.order_id}, time={self.time}, price={self.order.price}, amount={self.order.amount}, commission={self.order.commission})"


class OrderStatus(Enum):
    open = 0
    filled = 1
    canceled = 2
    rejected = 3
    held = 4
    expired = 5
    new = 8
    pending_cancel = 9

    def __repr__(self):
        return self.name
    
    def __str__(self):
        return self.name

class TIME(Enum):
    
    # 必须按照时间先后, 从上到下排序, scheduler要遍历
    # (OPEN, CLOSE] 按收盘价成交
    DAY_START = datetime.time(0, 0)
    BEFORE_OPEN = datetime.time(9, 0)
    OPEN_AUCTION = datetime.time(9, 15)
    OPEN_AUCTION_END = datetime.time(9, 25)
    AFTER_OPEN_AUCTION = datetime.time(9, 26)
    OPEN = datetime.time(9, 30)
    # 14:57 集合竞价开始
    BEFORE_CLOSE = datetime.time(14, 57)
    CLOSE = datetime.time(15, 0)
    AFTER_CLOSE = datetime.time(15, 30)
    DAY_END = datetime.time(23, 59)
    

    @property
    def hour(self):
        return self.value.hour

    @property
    def minute(self):
        return self.value.minute

    def __str__(self):
        return self.value.strftime("%H:%M")

class Index:
    def __init__(self, prev, this, next, timestamp):
        object.__setattr__(self, "prev", prev)
        object.__setattr__(self, "this", this)
        object.__setattr__(self, "next", next)
        # object.__setattr__(self, "timestamp", timestamp.timestamp())

    def __repr__(self) -> str:
        return f"({self.prev}, {self.this}, {self.next})"
    
    def __setattr__(self, name, value) -> None:
        raise AttributeError("Index object is immutable")
        

class NumpyFrame:
    def __init__(self, array, name, index_array=None, start_timestamp=None):
        self.data = array
        
        self.index = index_array
        # self.column = column
        self.start_timestamp = start_timestamp
        self.name = name
        
    
    # def loc(self, index, column):
    #     if self.index is not None:
    #         index = self.index[index]
    #     if self.column is not None:
    #         column = self.column[column]
    #     return self.data[index, column]
    

    # def iloc(self, index, column):
    #     return self.data[index, column]
    
    def tail(self, n):
        return self[-n:][0]
    
    def __getitem__(self, key) -> Union[np.ndarray, tuple]:
        '''
        根据日期datetime.datetime, pd.Timestamp或int获取个股数据
        返回 (ndarray数据, 数据对应的起止日期)
        '''

        # 解构 key，它是一个包含行索引和列索引的元组
        index, columns = key
        # print(index)
        # print(self.start_timestamp)
        # print(f"data: {len(self.data)}, index: {len(self.index)}")
        dt_start, dt_stop = None, None
        # 处理索引映射
        if self.index is not None:
            if isinstance(index, slice):
                # 对start映射
                if index.start is None:
                    dt_start = self.start_timestamp
                    start = index.start
                elif isinstance(index.start, int):
                    dt_start = self.data['date'][index.start]
                    # if index.start >= 0:
                    #     dt_start = self.start_timestamp + pd.Timedelta(days=index.start)
                    # else:
                    #     dt_start = self.start_timestamp + pd.Timedelta(days=len(self.index)+index.start-1)

                    start = index.start
                else:
                    offset = (index.start - self.start_timestamp).days
                    # print(offset)
                    # self.index[x] 是 Index 对象。
                    if 0 <= offset < len(self.index):
                        start = self.index[offset]
                        if start.this is not None:
                            start = start.this
                            dt_start = self.data['date'][start]
                            
                        else:
                            # start 非交易日，取下一个交易日
                            start = start.next
                            dt_start = self.data['date'][start]
                            # for offs in range(offset+1, len(self.index)):
                            #     next_valid_index = self.index[offs]
                            #     assert next_valid_index.this <= start
                            #     if next_valid_index.this == start:
                            #         dt_start = next_valid_index.timestamp
                            #         break
                    elif offset < 0:
                        start = 0
                        dt_start = self.start_timestamp
                    else:
                        start = len(self.data)
                        dt_start = index.start
                    

                # 对stop映射
                if index.stop is None:
                    dt_stop = self.start_timestamp + pd.Timedelta(days=len(self.index)-1)
                    stop = None
                elif isinstance(index.stop, int):
                    dt_stop = self.data['date'][index.stop] 
                    # if index.start >= 0:
                    # if index.stop >= 0:
                    #     dt_stop = self.start_timestamp + pd.Timedelta(days=index.stop-1) if index.stop >=0 else self.start_timestamp + pd.Timedelta(days=len(self.index)-1+index.stop)
                    # else:
                    #     dt_stop = self.start_timestamp + pd.Timedelta(days=len(self.index)+index.stop-1)
                    stop = index.stop
                else:
                    offset = (index.stop - self.start_timestamp).days
                    # print(offset)
                    # self.index[x] 是 Index 对象
                    if 0 <= offset < len(self.index):
                        stop = self.index[offset]

                        if stop.this is not None:
                            # dt_stop = stop.timestamp
                            stop = stop.this
                            dt_stop = self.data['date'][stop]  
                        else:
                            # stop 非交易日，取前一个交易日
                            stop = stop.prev# if stop.prev is not None else 0
                            dt_stop = self.data['date'][stop] 
                            # for offs in range(offset-1, -1, -1):
                            #     prev_valid_index = self.index[offs]
                            #     assert prev_valid_index.this >= stop
                            #     if prev_valid_index.this == stop:
                            #         dt_stop = prev_valid_index.timestamp
                            #         break
                    elif offset < 0:
                        stop = -1
                        dt_stop = index.stop
                    else:
                        stop = len(self.data)
                        dt_stop = self.start_timestamp + pd.Timedelta(days=len(self.index)-1)

                # 对stop进行修正, 包括右端点的值
                if stop is not None:
                    stop += 1
                slice_index = slice(start, stop, index.step)

                # 若请求日期与已有数据不重叠, 返回原始索引
                if dt_start > dt_stop:
                    dt_start, dt_stop = index.start, index.stop
                # print(slice_index)

                
                return self.data[columns][slice_index], (dt_start, dt_stop)
            else:
                if isinstance(index, datetime.date) or isinstance(index, datetime.datetime):
                    index = pd.Timestamp(index)
                    
                if isinstance(index, pd.Timestamp):
                    offset = (index - self.start_timestamp).days
                    #已退市
                    if offset >= len(self.index):
                        from .logger import log
                        log.warning(f"{self.name} 未来没有数据了, 返回 (0, {index})")
                        log.warning(f"{self.name} last_data: {self.data[-1]}, offset: {offset}, len_index: {len(self.index)}")
                        return 0, index
                    elif userconfig['backtest']:
                        if not self.name.startswith('1'):
                            if offset >= len(self.index) - 30:
                                if self.name.startswith('3'):
                                    from .logger import log
                                    log.warning(f"{self.name} 疑似进入30天退市整理期, 务必检查, 返回 (0, {index})")
                                    return 0, index
                                elif offset >= len(self.index) - 15:
                                    from .logger import log
                                    log.warning(f"{self.name} 疑似进入15天退市整理期, 务必检查, 返回 (0, {index})")
                                    return 0, index

                    int_index = self.index[offset]
                    if int_index.this is not None:
                        int_index = int_index.this
                        data_dt = Index
                    else:
                        from .logger import log
                        # log.warning(f"{self.name} Index {index} is not a valid trading day, Index: {self.index[offset]}, returning data for previous trading day")
                        int_index = int_index.prev
                        data_dt = self.data['date'][int_index] 
                    # index为int时, 先通过字段名访问columns，再通过索引访问index, 速度会快一倍, 而index为slice时速度没区别
                    
                    return self.data[columns][int_index], data_dt
                else:
                    raise IndexError("Invalid index type: {index}")

        # print(columns)
        # 处理列映射
        # if self.column is not None:
        #     if isinstance(columns, list):
        #         # 如果是列表，映射每个列名到对应的索引
        #         columns = [self.column[col] for col in columns]
        #     elif isinstance(columns, str):
        #         # 如果是字符串，映射到对应的列索引
        #         columns = self.column[columns]
        #     else:
        #         raise IndexError("Invalid columns type: {columns}")
        # assert res.shape[0] == len(index)
    
    def __str__(self) -> str:
        # return f"{self.data}\n index_map: {self.index}\n column_map: {self.column}"
        start_length = 5
        end_length = 5
        index_map_str = f"{list(self.index)[:start_length]}...{list(self.index)[-end_length:]}"
        # column_map_str = ', '.join(f"{k}: {v}" for k, v in list(self.column.items())[:max_entries_to_show])
        return f"{self.data}\n index_map.keys(): {index_map_str}\n column_map: {self.data.dtype}"
    
    def __getstate__(self):
        # Minimize the state to save; convert NumPy array to bytes to speed up serialization
        state = {
            'data': self.data,  # Convert NumPy array to bytes
            'index': self.index,
            'name': self.name,
            'start_timestamp': self.start_timestamp.timestamp()
        }
        return state

    def __setstate__(self, state):
        # Restore the state

        # data_dtype = state['data_dtype']

        # self.data = np.empty(data_shape, dtype=data_dtype)
        # print(state['data'].dtype)
        # data_buffer = np.frombuffer(state['data'], dtype=data_dtype)
        # np.copyto(self.data, data_buffer.reshape(data_shape))

        # data = np.frombuffer(state['data'], dtype=data_dtype)#.reshape(data_shape)
        # print(data.shape)
        object.__setattr__(self, "data", state['data'])
        object.__setattr__(self, "index", state['index'])
        object.__setattr__(self, "name", state['name'])
        # print(pd.to_datetime(state['start_timestamp'], unit='s'))
        object.__setattr__(self, "start_timestamp", pd.to_datetime(state['start_timestamp'], unit='s'))
        # self.index = state['index']
        # self.column = state['column']
        
class DictedList:
    def __init__(self):
        self._data = []  # 底层数据 list
        self._key2idx = {}  # key: str -> int

    def __getitem__(self, key):
        idx = self._key2idx[key]
        return self._data[idx]

    def __setitem__(self, key, value):
        if key in self._key2idx:
            idx = self._key2idx[key]
            self._data[idx] = value
        else:
            idx = len(self._data)
            self._data.append(value)
            self._key2idx[key] = idx

    def __delitem__(self, key):
        raise NotImplementedError("DictedList is undeletable")
        # idx = self._key2idx.pop(key)
        # self._data[idx] = None  # 用 None 标记，简单安全

    def get(self, key, default=None, offset=0):
        if key in self._key2idx:
            idx = self._key2idx[key] + offset
            if idx < 0:
                raise IndexError(f"Index out of range: {idx}, str_key:{key}, ori_index:{self._key2idx[key]}, offset:{offset}")
            elif idx >= len(self._data):
                print(f"\033[31mIndex >= len(self._data), return self._data[-1]\033[0m")
                return self._data[-1]
            return self._data[idx]
        else:
            return default

    def keys(self):
        return list(self._key2idx.keys())

    def values(self):
        return [self._data[idx] for idx in self._key2idx.values()]

    def items(self):
        return [(k, self._data[idx]) for k, idx in self._key2idx.items()]

    def __contains__(self, key):
        return key in self._key2idx

    def __len__(self):
        return len(self._key2idx)

    def __iter__(self):
        return iter(self._key2idx)

    def __repr__(self):
        return "{" + ", ".join(f"{k!r}: {self._data[idx]!r}" for k, idx in self._key2idx.items()) + "}"

if __name__ == '__main__':
    # 示例数据
    data = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]

    # 索引和列的映射
    index_map = {'a': 2}
    column_map = {'b': 1, 'c': 2}

    # 创建 NumpyFrame 实例
    df = NumpyFrame(data, index_map=index_map, column_map=column_map)

    # 使用 loc 索引
    result = df[:'a', ['b', 'c']]
    print(result)
