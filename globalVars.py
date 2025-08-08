from .logger import log
import pickle
from collections import deque

class GlobalVars(object):
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
                elif isinstance(v, deque):
                    v = list(v)  
                    k_str = f"_deque_{k_str}"
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
            elif isinstance(value, deque):
                state[f'_deque_{key}'] = list(value)
            else:
                state[key] = value
        return state

    def set_state(self, state):
        def decode_dict_keys(d):
            """递归还原 dict 中 key 是 '_int_xx' 的项"""
            decoded = {}
            for k, v in d.items():
                if isinstance(k, str):
                    if k.startswith('_int_'):
                        k_new = int(k[5:])
                    elif k.startswith('_deque_'):
                        k_new = k[7:]
                        if isinstance(v, list):
                            v = deque(v)
                        else:
                            raise ValueError(f"Invalid deque value: {k_new}:{v}. Should be a list.")
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
            elif key.startswith('_deque_'):
                real_key = key[7:]
                setattr(self, real_key, deque(value))
            elif isinstance(value, dict):
                setattr(self, key, decode_dict_keys(value))
            else:
                setattr(self, key, value)
