from abc import ABC, abstractmethod


##############################################################################
# base runtime class
##############################################################################
class ExecutionEnvironment(ABC):
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def close(self):
        pass
