import abc
import typing
from typing import List, Dict

import torch
from torch.utils.data import Dataset

from tqdm import tqdm

if typing.TYPE_CHECKING:
    from ise_cdg_models.a_conv_gnn.models import AConvGNN


class MetricInterface(abc.ABC):

    @abc.abstractmethod
    def set_references(self, references: List[List[List[str | int]]]) -> None:
        pass
        

    @abc.abstractmethod
    def __call__(self, candidates: List[List[str | int]]):
        pass


class AConvGNNTesterOnDataset:

    def __init__(
            self, 
            name: str,
            model: AConvGNN,
            dataset: Dataset, 
            geo_dataset: Dataset,
            printer = print,
    ) -> None:
        self.name = name
        self.model = model
        self.dataset = dataset
        self.geo_dataset = geo_dataset
        self.printer = printer
    
    def start_testing(
            self,
            metrics_with_name: Dict[str, MetricInterface],
            dataset_id_generator: typing.Generator,
            sos_ind: int, eos_ind: int,
            device: torch.device,
    ):
        self.printer(f'x---------- {self.name} Started Testing ----------x')
        self.model.eval()
        with torch.no_grad():
            candidates = []
            mds = []
            for i in dataset_id_generator:
                src, md = self.dataset[i]
                src = src.unsqueeze(1).to(device)
                data2 = self.geo_dataset[i]
                data2 = data2.to(device)
                batch = torch.zeros(data2.x.shape[0]).long().to(device)
                output = self.model.generate_one_markdown(
                    src,
                    data2.x.long(), data2.edge_index.long(), batch,
                    sos_ind, eos_ind,
                    sequence_max_length=50,
                    device=device,
                )
                candidates.append([int(ind) for ind in output.tolist()])
                mds.append(md)
        
        for metric_name, metric in metrics_with_name.items():
            self.printer(f'x-- {metric_name} --x')
            self.printer(metric(candidates))
        self.printer()
        return candidates, mds