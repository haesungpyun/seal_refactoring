from typing import List, Tuple, Union, Dict, Any, Optional, overload
from seal2.modules.sampler import (
    Sampler,
    BasicSampler,
)
import torch
from seal2.modules.score_nn import ScoreNN
from seal2.modules.oracle_value_function import (
    OracleValueFunction,
)
from seal2.modules.loss import Loss

from seal2.modules.multilabel_classification_task_nn import (
    MultilabelTaskNN,
)


# @Sampler.register("keyword-extraction-basic")
class KeywordExtractionBasicSampler(BasicSampler):
    @property
    def is_normalized(self) -> bool:
        return True

    @overload
    def normalize(self, y: None) -> None:
        ...

    @overload
    def normalize(self, y: torch.Tensor) -> torch.Tensor:
        ...

    def normalize(self, y: Optional[torch.Tensor]) -> Optional[torch.Tensor]:

        if y is not None:
            return torch.sigmoid(y)
        else:
            return None

    @property
    def different_training_and_eval(self) -> bool:
        return False
