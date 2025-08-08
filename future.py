# -*- coding: utf-8 -*-
from events import EVENT
__all__ = []
def export_as_api(func):
    __all__.append(func)
    return func

@export_as_api
def startup(event_bus):
    event_bus.add_listener(EVENT.STOCK, handler)
    print('load and register future mod')

def handler(event):
    print('do future handler success')
    pass