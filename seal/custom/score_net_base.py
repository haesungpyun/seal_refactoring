from typing import List, Tuple, Union, Dict, Any, Optional
from allennlp.common.registrable import Registrable
import torch
from .task_net_base import TaskNet


class ScoreNet(torch.nn.Module, Registrable):
    """Concrete base class for creating feature representation for any task."""

    def __init__(
        self,
        global_score:torch.nn.Module = None,
        local_score: torch.nn.Module = None,
        **kwargs: Any,
    ):
        super().__init__()  # type:ignore
        self.global_score = global_score
        self.local_score = local_score
        self._dtype = self.compute_input_dtype()

    def compute_input_dtype(self) -> Optional[torch.dtype]:
        if self.global_score is not None:
            for p in self.global_score.parameters():
                return p.dtype
        else:
            return None

        return None

    def compute_local_score(
        self, x: Any, y: Any, buffer: Dict, **kwargs: Any
    ) -> Optional[torch.Tensor]:
        if self.local_score is not None:
            return self.local_score(x, y, buffer, **kwargs)
        else:
            return None

    def compute_global_score(
        self, y: Any, buffer: Dict, **kwargs: Any  #: (batch, num_samples, ...)
    ) -> Optional[torch.Tensor]:
        if self.global_score is not None:
            return self.global_score(y, buffer, **kwargs)
        else:
            return None

    @property
    def input_dtype(self) -> Optional[torch.dtype]:
        return self._dtype

    def forward(
        self,
        x: Any,
        y: torch.Tensor,  # (batch, num_samples or 1, ...).
        buffer: Dict,
        **kwargs: Any,
    ) -> Optional[torch.Tensor]:
        score = None
        local_score = self.compute_local_score(x, y, buffer, **kwargs)

        if local_score is not None:
            score = local_score

        global_score = self.compute_global_score(y, buffer, **kwargs)  # type: ignore

        if global_score is not None:
            if score is not None:
                score = score + global_score
            else:
                score = global_score

        return score  # (batch, num_samples)
