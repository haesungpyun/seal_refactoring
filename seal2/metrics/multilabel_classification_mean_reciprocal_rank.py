"""Implements Mean Reciprocal Rank using sklearn"""
from typing import Iterable, Optional

import torch
from allennlp.training.metrics import Metric, Average
from sklearn.metrics import label_ranking_average_precision_score


# @Metric.register("multilabel-mean-reciprocal-rank")
class MultilabelClassificationMeanReciprocalRank(Average):

    """Computes mean reciprocal rank for the true label, given the energy scores for the samples and the true label"""

    def __init__(self) -> None:
        super().__init__()

    def __call__(
        self, predictions: torch.Tensor, gold_labels: torch.Tensor
    ) -> None:  # type: ignore

        labels, scores = [
            t.cpu().numpy()
            for t in self.detach_tensors(gold_labels, predictions)
        ]

        mrr = label_ranking_average_precision_score(labels, scores)
        super().__call__(mrr)

    @staticmethod
    def detach_tensors(*tensors: torch.Tensor) -> Iterable[torch.Tensor]:
        """
        If you actually passed gradient-tracking Tensors to a Metric, there will be
        a huge memory leak, because it will prevent garbage collection for the computation
        graph. This method ensures the tensors are detached.
        """
        # Check if it's actually a tensor in case something else was passed.
        return (x.detach() if isinstance(x, torch.Tensor) else x for x in tensors)