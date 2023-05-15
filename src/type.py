import enum

class NetworkModelBaseline(enum.Enum):
    resnet152 = 0
    resnet152_dropout = 1
    vit_b_32 = 2

class NetworkModelDecomposed(enum.Enum):
    resnet152_mtl = 0

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