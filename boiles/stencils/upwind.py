from .base import ReconstructionStencil


class Upwind(ReconstructionStencil):
    def __init__(self):
        pass

    @staticmethod
    def apply(value):
        if isinstance(value, list):
            return value[0]
        else:
            return value
