import abc
import random
import typing
from typing import List, Dict, Callable
from sympy import im
from torch.nn import functional as F

from ise_cdg_models.required_interfaces import MetricInterface

import torch
from torch.utils.data import Dataset


if typing.TYPE_CHECKING:
    from ise_cdg_models.cnn2rnn import CNN2RNN, CNN2RNNAttention, CNN2RNNFeatures


class CNN2RNNTesterOnDataset:

    def __init__(
            self, 
            name: str,
            model: typing.Union['CNN2RNN', 'CNN2RNNAttention'],
            dataset: Dataset,
            md_vocab,
            source_max_length,
            printer = print,
    ) -> None:
        self.name = name
        self.model = model
        self.dataset = dataset
        self.md_vocab = md_vocab
        self.printer = printer
        self.source_max_length = source_max_length

    def __pad_to_length(self, raw_input):
        # raw_input.shape: (..., raw_input_length)
        raw_input_length = raw_input.shape[0]
        pad_length = self.source_max_length - raw_input_length
        # shape: (..., expected_length)
        return F.pad(raw_input, (0, 0, 0, pad_length), value=0)

    def start_testing(
            self,
            metrics_with_name: Dict[str, MetricInterface],
            dataset_id_generator: Callable,
            sos_ind: int, eos_ind: int,
            device: torch.device,
            example_ratio : typing.Union[float, None] = None,
    ):
        self.printer(f'x---------- {self.name} Started Testing ----------x')
        self.model.eval()
        examples_shown = 0
        with torch.no_grad():
            candidates = []
            mds = []
            for i in dataset_id_generator():
                src, md = self.dataset[i]
                src = self.__pad_to_length(src.unsqueeze(1).to(device))
                output = self.model.generate_one_markdown(
                    src,
                    sos_ind, eos_ind,
                    sequence_max_length=25,
                    device=device,
                )
                candidate = [int(ind) for ind in output.tolist()]
                target = [int(ind) for ind in md.tolist()]
                candidates.append(candidate)
                mds.append([target])
                if example_ratio is not None and random.random() < example_ratio:
                    examples_shown += 1
                    self.printer(f'x- example {examples_shown} -x')
                    self.printer('\tReal: ' + ' '.join([self.md_vocab.get_itos()[tok_ind] for tok_ind in target]))
                    self.printer('\tPredicted: ' + ' '.join([self.md_vocab.get_itos()[tok_ind] for tok_ind in candidate]))

        for metric_name, metric in metrics_with_name.items():
            self.printer(f'x--- {metric_name} ---x')
            metric.set_references(mds)
            self.printer(str(metric(candidates)))
        self.printer('')
        return candidates, mds
    
class CNN2RNNFeaturesTesterOnDataset:
    def __init__(
            self, 
            name: str,
            model: 'CNN2RNNFeatures',
            dataset: Dataset,
            md_vocab,
            source_max_length,
            printer = print,
    ) -> None:
        self.name = name
        self.model = model
        self.dataset = dataset
        self.md_vocab = md_vocab
        self.printer = printer
        self.source_max_length = source_max_length

    def __pad_to_length(self, raw_input):
        # raw_input.shape: (..., raw_input_length)
        raw_input_length = raw_input.shape[0]
        pad_length = self.source_max_length - raw_input_length
        # shape: (..., expected_length)
        return F.pad(raw_input, (0, 0, 0, pad_length), value=0)

    def start_testing(
            self,
            metrics_with_name: Dict[str, MetricInterface],
            dataset_id_generator: Callable,
            sos_ind: int, eos_ind: int,
            device: torch.device,
            example_ratio : typing.Union[float, None] = None,
    ):
        self.printer(f'x---------- {self.name} Started Testing ----------x')
        self.model.eval()
        examples_shown = 0
        with torch.no_grad():
            candidates = []
            mds = []
            for i in dataset_id_generator():
                src, features, md = self.dataset[i]
                src = self.__pad_to_length(src.unsqueeze(1).to(device))
                features = features.unsqueeze(0).to(device)
                output = self.model.generate_one_markdown(
                    src, features,
                    sos_ind, eos_ind,
                    sequence_max_length=25,
                    device=device,
                )
                candidate = [int(ind) for ind in output.tolist()]
                target = [int(ind) for ind in md.tolist()]
                candidates.append(candidate)
                mds.append([target])
                if example_ratio is not None and random.random() < example_ratio:
                    examples_shown += 1
                    self.printer(f'x- example {examples_shown} -x')
                    self.printer('\tReal: ' + ' '.join([self.md_vocab.get_itos()[tok_ind] for tok_ind in target]))
                    self.printer('\tPredicted: ' + ' '.join([self.md_vocab.get_itos()[tok_ind] for tok_ind in candidate]))

        for metric_name, metric in metrics_with_name.items():
            self.printer(f'x--- {metric_name} ---x')
            metric.set_references(mds)
            self.printer(str(metric(candidates)))
        self.printer('')
        return candidates, mds