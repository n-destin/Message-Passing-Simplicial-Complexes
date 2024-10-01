import inspect
import torch_geometric
from collections import OrderedDict
from typing import Dict, Any, Callable
from torch_geomteric.nn.conv.utils.insepctor import Inspector 

class CellularInspector(Inspector):
    def __init__(self, class_):
        self.class_ = class_
        
    def __implements___(self, function_name):
        # checks if a class or its basis implements a a certain function. The class should not be CochainMessagPassing
        if self.class_.__name__ == "CochainMessagePassing":
            return False 
        if function_name in self.class_.__dict__.keys():
            return True
        return any(self.__implements__(class__, function_name) for class__ in self.class_.__bases__)

    def inspect(self, function : Callable, n_items : int = 0):
        parameters = inspect.signature(function).parameters
        parameters = OrderedDict(parameters)

        # removes some parameters from a class 
        for _ in range(n_items):
            parameters.popitem(last = False)
        self.parameters[function.__name__] = parameters
    


