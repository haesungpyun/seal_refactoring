import torch
from typing import List, Tuple, Union, Dict, Any, Optional, Callable
from collections import defaultdict
from seal2.modules.task_nn import TaskNN, CostAugmentedLayer, TextEncoder
import torch


class FeatureNetwork(torch.nn.Module):
    def __init__(
        self,
        image_backbone = None, 
        text_field_embedder = None,
        seq2vec_encoder = None,
        seq2seq_encoder = None,
        feedforward: torch.nn.Module = None,
        dropout: float = None,
        **kwargs: Any,
    ):
        super().__init__()
        
        self.image_backbone = image_backbone
        self.text_field_embedder = text_field_embedder
        self.seq2seq_encoder = seq2seq_encoder
        self.seq2vec_encoder = seq2vec_encoder
        self.feedforward = feedforward
    
        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = None
        
    def normalize(self, y: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if y is not None:
            return torch.sigmoid(y)
        else:
            return None

    def forward(
        self,
        x: torch.Tensor,
        buffer: Optional[Dict] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        
        if self.image_backbone:
            return self.image_backbone(x)
        
        mask = buffer.get("mask")

        if mask is None:
            masks = []
            for indexer_name, indexer_tensors in x.items():
                if "mask" in indexer_tensors:
                    masks.append(indexer_tensors["mask"].bool())
            mask = masks[0]
        
        buffer["mask"] = mask                

        if self.text_field_embedder:
            embedded_input = self.text_field_embedder(x)
        
        if self.seq2seq_encoder:
            embedded_input = self.seq2seq_encoder(embedded_input, mask=mask)

        if self.seq2vec_encoder:
            encoded_input = self.seq2vec_encoder(embedded_input, mask)
        else:
            encoded_input = embedded_input

        if self.dropout:
            encoded_input = self.dropout(encoded_input)

        if self.feedforward:
            encoded_input = self.feedforward(encoded_input)

        return encoded_input


class TaskNet(torch.nn.Module):
    def __init__(
        self, 
        feature_network: FeatureNetwork,
        projection: torch.nn.Linear,
        **kwargs: Any
    ):
        """
        projection can be vary by the task
            Timedistributed for SRL
            Label embedding for MLC ...
        """
        super().__init__()
        assert feature_network is not None
        self.feature_network = feature_network
        self.projection  = projection 

    def forward(
        self,
        x: torch.Tensor,
        buffer: Optional[Dict] = None,
        normalize: bool = True,
        **kwargs: Any,
    ) -> torch.Tensor:
        
        encoded_input = self.feature_network(x)
        
        if self.projection:
            y_hat_unormalized = self.projection(encoded_input)
        else:
            y_hat_unormalized = encoded_input

        return y_hat_unormalized

"""
forward path에 있는 layer들

input -> inference_nn -> tasknet (unnormalized logit) -> loss fn (loss) -> normalize logit  

normalize 해줄 수 잇는 inference_nn 같은 class 필요?
sampler는 하는 일이 없는데 그대로? 아니면 빼 버려?

"""
 