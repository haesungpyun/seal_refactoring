from typing import List, Tuple, Union, Dict, Any, Optional
from seal2.modules.loss import DVNScoreLoss, Loss
import torch

def _normalize(y: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(y)


@Loss.register("keyword-extraction-score-loss")
class KeywordExtractionScoreLoss(DVNScoreLoss):
    """
    Non-DVN setup where score is not bounded in [0,1], 
    however the only thing we need is score from scoreNN, 
    so it's better to share with DVNScoreLoss.
    """
    def normalize(self, y: torch.Tensor) -> torch.Tensor:
        return _normalize(y)

    def compute_loss(
        self,
        predicted_score: torch.Tensor,  # logits of shape (batch, num_samples)
    ) -> torch.Tensor:
        return -predicted_score   
