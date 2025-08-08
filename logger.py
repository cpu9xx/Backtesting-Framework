from .Env import Env
from datetime import datetime
class Log:
    def __init__(self):
        try:
            self._env = Env.get_instance()
        except:
            self._env = None
        self.order = True

    def set_env(self, environment):
        self._env = environment

    def set_level(self, name, level):
        if hasattr(self, name):
            setattr(self, name, False)
        else:
            self.warning("目前只支持设置order相关的日志")

    def info(self, *messages):
        combined_message = ' '.join(map(str, messages))
        print(f"--{self._env.current_dt} : {combined_message}")

    def orderinfo(self, *messages):
        if self.order:
            combined_message = ' '.join(map(str, messages))
            print(f"\033[90m--{self._env.current_dt} : {combined_message}\033[0m")

    def error(self, message):
        print(f"--{self._env.current_dt} : {message}")
        raise Exception(message)
    
    def warning(self, *messages):
        combined_message = ' '.join(map(str, messages))
        print(f"\033[31m--{self._env.current_dt} : {combined_message}\033[0m")

    def test(self, *messages):
        combined_message = ' '.join(map(str, messages))
        print(f"\033[34m--{self._env.current_dt} : {combined_message}\033[0m")

    # @classmethod
    def live(self, *messages):
        combined_message = ' '.join(map(str, messages))
        print(f"\033[33m--{datetime.now()} : {combined_message}\033[0m")

log = Log()