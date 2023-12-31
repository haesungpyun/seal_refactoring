"""Implements Normalized Discounted Cumulative Gain using sklearn's ndcg_score"""
from typing import Iterable, Optional

import torch
from allennlp.training.metrics import Metric, Average
from sklearn.metrics import ndcg_score


@Metric.register("multilabel-norm-discounted-cumulative-gain")
class MultilabelClassificationNormalizedDiscountedCumulativeGain(Average):

    """Computes normalized discounted cumulative gain"""

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
            ndcg = ndcg_score(single_example_true_scores.reshape(1, -1), single_example_pred_scores.reshape(1, -1))
            super().__call__(ndcg)

    @staticmethod
    def detach_tensors(*tensors: torch.Tensor) -> Iterable[torch.Tensor]:
        """
        If you actually passed gradient-tracking Tensors to a Metric, there will be
        a huge memory leak, because it will prevent garbage collection for the computation
        graph. This method ensures the tensors are detached.
        """
        # Check if it's actually a tensor in case something else was passed.
        return (x.detach() if isinstance(x, torch.Tensor) else x for x in tensors)