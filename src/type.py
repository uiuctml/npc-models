import enum

class LogLevel(enum.Enum):
    all = 6
    trace = 5
    debug = 4
    info = 3
    warn = 2
    error = 1
    fatal = 0
    off = -1

    def __ge__(self, other):
        if self.__class__ is other.__class__:
            return self.value >= other.value
        return NotImplemented

    def __gt__(self, other):
        if self.__class__ is other.__class__:
            return self.value > other.value
        return NotImplemented

    def __le__(self, other):
        if self.__class__ is other.__class__:
            return self.value <= other.value
        return NotImplemented

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented

class ModelBaseline(enum.Enum):
    resnet152 = 0
    vit_b_32 = 1

class ModelDecomposed(enum.Enum):
    mlp_set = 0
    resnet152_mtl = 1
    vit_b_32_mtl = 2

class OptimizerSPN(enum.Enum):
    cccp_composed = 0
    cccp_offline = 1
    pgd_composed = 2
    pgd_offline = 3
