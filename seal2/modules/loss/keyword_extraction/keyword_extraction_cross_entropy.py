from typing import List, Tuple, Union, Dict, Any, Optional
import torch
from seal2.modules.loss import Loss
import numpy as np
import logging
from transformers.models import bert
Bert = bert.modeling_bert.BertPooler

logger = logging.getLogger(__name__)


@Loss.register("keyword-extraction-ce")
class KeywordExtractionCELoss(Loss):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        self._loss_values = []

    def _forward(
        self,
        x: Any,
        labels: Optional[torch.Tensor],  # (batch, 1, num_labels)
        y_hat: torch.Tensor,  # (batch, 1, num_labels)
        y_hat_extra: Optional[torch.Tensor],
        buffer: Optional[Dict] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        assert labels is not None       
        loss = self.loss_fn(y_hat, labels.to(dtype=y_hat.dtype)).sum(
            dim=-1
        )  # (batch, 1,)
        self._loss_values.append(float(torch.mean(loss)))

        return loss

    def get_metrics(self, reset: bool = False):
        metrics = {}

        if self._loss_values:
            metrics = {"cross_entropy_loss": np.mean(self._loss_values)}

        if reset:
            self._loss_values = []

        return metrics


@Loss.register("keyword-extraction-ce-unreduced")
class KeywordExtractionBCELoss(Loss):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")

    def _forward(
        self,
        x: Any,
        labels: Optional[torch.Tensor],  # (batch, 1, num_labels)
        y_hat: torch.Tensor,  # (batch, 1, num_labels)
        y_hat_extra: Optional[torch.Tensor],
        buffer: Optional[Dict] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        assert labels is not None

        return self.loss_fn(
            y_hat, labels.to(dtype=y_hat.dtype)
        )  # (batch, label_size,)
