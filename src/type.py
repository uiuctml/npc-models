import enum

class NetworkModelBaseline(enum.Enum):
    resnet152 = 0
    resnet152_clip = 1
    resnet152_dropout = 2
    vit_b_32 = 3

class NetworkModelDecomposed(enum.Enum):
    resnet152_mtl = 0
    resnet152_clip_mtl = 1
    vit_b_32_mtl = 2

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