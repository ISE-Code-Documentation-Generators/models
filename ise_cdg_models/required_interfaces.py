import abc
from typing import List
import typing


class MetricInterface(abc.ABC):

    @abc.abstractmethod
    def set_references(self, references: List[List[List[typing.Union[str, int]]]]) -> None:
        pass
        

    @abc.abstractmethod
    def __call__(self, candidates: List[List[typing.Union[str, int]]]):
        pass
