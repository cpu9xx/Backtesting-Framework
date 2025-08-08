# -*- coding: utf-8 -*-
from .events import EVENT
from .strategy_context import StrategyContext
__all__ = [
]
def export_as_api(func):
    __all__.append(func)
    return func

@export_as_api
def startup(event_bus):
    # broker = Broker.get_instance()
    # event_bus.add_listener(EVENT.STOCK, broker._order_handler)
    # event_bus.add_listener(EVENT.STOCK, broker._order_canceler)
    print('load and register stock mod')
  
