from abc import ABC
from functools import cached_property

class Identifiable(ABC):
    _COUNTERS: dict[str, int] = dict()

    def __init__(self):
        clazz = type(self).__name__
        id = self._gen_id(clazz)
        self._uid = f'{clazz}_{id}'

    @cached_property
    def uid(self) -> str:
        return self._uid

    def __hash__(self) -> int:
        return hash(self._uid)

    @staticmethod
    def _gen_id(clazz: str) -> str:
        instances_count = Identifiable._COUNTERS.get(clazz, 1)
        Identifiable._COUNTERS[clazz] = instances_count + 1
        return instances_count