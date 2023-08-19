import torch
from torch import nn
from typing import Optional, Dict, Any

class MultilabelClassificationLocalScore(nn.Module):
    def forward(        self,
        x: torch.Tensor,  #: (batch, features_size)
        y: torch.Tensor,  #: (batch, num_samples, num_labels)
        label_scores: torch.Tensor, #: (batch, num_samples, num_labels)
        buffer: Dict,
        **kwargs: Any,
    ) -> Optional[torch.Tensor]:
        local_energy = torch.sum(
            label_scores.unsqueeze(1) * y, dim=-1
        )  #: (batch, num_samples)

        return local_energy