class Strategy(object):
    def __init__(self):
        # self.data = None
        self._pindex = None
        # self.buy_list = []
        # self.sell_list = []
        if not hasattr(self, 'time'):
            raise ValueError("子策略必须指定运行时间self.time, 可以为'before_open', 'open', 'after_close'等等, 具体参考 object.py 中的 Class TIME")
    
    @property
    def pindex(self):
        return self._pindex

    def set_pindex(self, pindex):
        self._pindex = pindex

    def strategy(self):
        raise NotImplementedError("子类必须实现strategy")
    
    def after_trade(self, trade):
        pass
        # order = trade.order
        # if order.is_buy():
        #     self.hold -= 1  
        # else:
        #     self.hold += 1

    def get_state(self):
        def encode_dict_keys(d):
            """递归地将字典中的 int 类型 key 加前缀"""
            encoded = {}
            for k, v in d.items():
                # 处理 key
                if isinstance(k, int):
                    k_str = f"_int_{k}"
                else:
                    k_str = k
                # 递归处理嵌套字典
                if isinstance(v, dict):
                    v = encode_dict_keys(v)
                encoded[k_str] = v
            return encoded

        state = {}
        for key, value in self.__dict__.items():
            if key.startswith('_s_'):
                state[key] = value.get_state()
            elif key.startswith('_'):
                continue
            elif isinstance(value, dict):
                state[key] = encode_dict_keys(value)
            else:
                state[key] = value
        return state

    def set_state(self, state):
        def decode_dict_keys(d):
            """递归还原 dict 中 key 是 '_int_xx' 的项"""
            decoded = {}
            for k, v in d.items():
                if isinstance(k, str) and k.startswith('_int_'):
                    k_new = int(k[5:])
                else:
                    k_new = k
                if isinstance(v, dict):
                    v = decode_dict_keys(v)
                decoded[k_new] = v
            return decoded

        for key, value in state.items():
            if key.startswith('_s_'):
                if not hasattr(self, key):
                    raise ValueError(
                        f"Invalid state key: {key}. It must be created and have a set_state() method before calling {key}.set_state()."
                    )
                getattr(self, key).set_state(value)
            elif isinstance(value, dict):
                setattr(self, key, decode_dict_keys(value))
            else:
                setattr(self, key, value)