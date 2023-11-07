import abc
from typing import List


class MetricInterface(abc.ABC):

    @abc.abstractmethod
    def set_references(self, references: List[List[List[str | int]]]) -> None:
        pass
        

    @abc.abstractmethod
    def __call__(self, candidates: List[List[str | int]]):
        pass
