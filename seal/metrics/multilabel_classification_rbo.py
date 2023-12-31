from .rbo import RankingSimilarity
from typing import Iterable, Optional

import torch
from allennlp.training.metrics import Metric, Average
import numpy as np


@Metric.register("multilabel-rank-biased-overlap")
class MultilabelClassificationRankBiasedOverlap(Average):

    """
    Computes Rank Biased Overlap
    """

    def __init__(self) -> None:
        super().__init__()

    def __call__(
        self, predicted_scores: torch.Tensor, true_scores: torch.Tensor
    ) -> None:  # type: ignore

        true_scores, predicted_scores = [
            t.cpu().numpy()
            for t in self.detach_tensors(true_scores, predicted_scores)
        ]
        for single_example_true_scores, single_example_pred_scores in zip(
            true_scores, predicted_scores
        ):
            pred_rank = np.argsort(-single_example_pred_scores)
            true_rank = np.argsort(-single_example_true_scores)
            rbo = RankingSimilarity(true_rank, pred_rank).rbo(p=0.8)
            super().__call__(rbo)
    
    @staticmethod
    def detach_tensors(*tensors: torch.Tensor) -> Iterable[torch.Tensor]:
        """
        If you actually passed gradient-tracking Tensors to a Metric, there will be
        a huge memory leak, because it will prevent garbage collection for the computation
        graph. This method ensures the tensors are detached.
        """
        # Check if it's actually a tensor in case something else was passed.
        return (x.detach() if isinstance(x, torch.Tensor) else x for x in tensors)