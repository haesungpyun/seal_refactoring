"""Implements mean average precision using sklearn"""
from typing import List, Tuple, Union, Dict, Any, Optional
from typing import Iterable, Optional

from sklearn.metrics import average_precision_score
from allennlp.training.metrics import Metric, Average
import torch


# @Metric.register("multilabel-classification-mean-avg-precision")
class MultilabelClassificationMeanAvgPrecision(Average):

    """Docstring for MeanAvgPrecision. """

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, predictions: torch.Tensor, gold_labels: torch.Tensor) -> None:  # type: ignore
        labels, scores = [
            t.cpu().numpy()
            for t in self.detach_tensors(gold_labels, predictions)
        ]

        for single_example_labels, single_example_scores in zip(
            labels, scores
        ):
            avg_precision = average_precision_score(
                single_example_labels, single_example_scores
            )
            super().__call__(avg_precision)

    @staticmethod
    def detach_tensors(*tensors: torch.Tensor) -> Iterable[torch.Tensor]:
        """
        If you actually passed gradient-tracking Tensors to a Metric, there will be
        a huge memory leak, because it will prevent garbage collection for the computation
        graph. This method ensures the tensors are detached.
        """
        # Check if it's actually a tensor in case something else was passed.
        return (x.detach() if isinstance(x, torch.Tensor) else x for x in tensors)
