from typing import List, Tuple, Union, Dict, Any, Optional, Literal
from seal.modules.loss import (
    Loss,
    NCELoss,
    NCERankingLoss,
)
import torch
import logging

logger = logging.getLogger(__name__)


def _normalize(y: torch.Tensor) -> torch.Tensor:
    return torch.softmax(y,dim=-1)


class KeywordExtractionNCERankingLoss(NCERankingLoss):
    def __init__(self, 
                sign: Literal["-", "+"] = "-", 
                use_scorenn: bool = True,
                use_distance: bool = True,
                **kwargs: Any):
        super().__init__(use_scorenn, **kwargs)
        self.sign = sign
        self.mul = -1 if sign == "-" else 1
        self.bce = torch.nn.BCELoss(reduction="none")
        self.use_distance = use_distance
        # when self.use_scorenn=False, the sign should always be +,
        # as we want to have P_0/\sum(P_i) rather than (1/P_0) /\sum(1/P_i)
        assert (sign == "+" if not self.use_scorenn else True)  

    def normalize(self, y: torch.Tensor) -> torch.Tensor:
        return _normalize(y)

    def distance(
        self,
        samples: torch.Tensor,  # (batch, num_samples, num_labels)
        probs: torch.Tensor,  # (batch, num_samples, num_labels)
    ) -> torch.Tensor:  # (batch, num_samples)
        """
        mul*BCE(inp=probs, target=samples). Here mul is 1 or -1. If mul = 1 the ranking loss will
        use adjusted_score of score - BCE. (mul=-1 corresponds to standard NCE)

        Note:
            Remember that BCE = -y ln(x) - (1-y) ln(1-x). Hence if samples are discrete, then BCE = -ln Pn.
            So in that case sign of + in this class will result in adjusted_score = score - (- ln Pn) = score + ln Pn.
        """
        if not self.use_distance: # if not using distance then skip the bce computation.
            return torch.zeros([samples.shape[0], samples.shape[1]], dtype=torch.long, device=probs.device) # (batch,sample)

        return self.mul * torch.sum(
            self.bce(probs, samples), dim=-1
        )  # (batch, num_samples)


@Loss.register("keyword-extraction-discrete-sampling")
class KeywordExtractionNCERankingLossWithDiscreteSamples(KeywordExtractionNCERankingLoss):
    
    def __init__():
        super.__init__()
    
    def distance(
        self,
        samples: torch.Tensor,  # (batch, num_samples, num_labels)
        probs: torch.Tensor,  # (batch, num_samples, num_labels)
    ) -> torch.Tensor:  # (batch, num_samples)
        """
        mul*BCE(inp=probs, target=samples). Here mul is 1 or -1. If mul = 1 the ranking loss will
        use adjusted_score of score - BCE. (mul=-1 corresponds to standard NCE)

        Note:
            Remember that BCE = -y ln(x) - (1-y) ln(1-x). Hence if samples are discrete, then BCE = -ln Pn.
            So in that case sign of + in this class will result in adjusted_score = score - (- ln Pn) = score + ln Pn.
        """
        if not self.use_distance: # if not using distance then skip the bce computation.
            return torch.zeros([samples.shape[0], samples.shape[1]], dtype=torch.long, device=probs.device) # (batch,sample)
        
        return self.mul* torch.sum(
            -samples*torch.log(probs), dim=-1
        ) # (batch, num_samples)
        

    def sample(
        self,
        probs: torch.Tensor,  # (batch, 1, num_labels)
    ) -> torch.Tensor:  # (batch, num_samples, num_labels)
        """
        Discrete sampling from the Bernoulli distribution.
        """
        assert probs.dim() == 3
        p = probs.squeeze(1)  # (batch, num_labels)
        
        # when multi-labeled noise is permitted
        # samples = torch.transpose(
        #     torch.distributions.Bernoulli(probs=p).sample(  # type: ignore
        #         [self.num_samples]  # (num_samples, batch, num_labels)
        #     ),
        #     0,
        #     1,
        # )  # (batch, num_samples, num_labels)

        # when single-labeled noise is permitted
        samples = torch.transpose(
            torch.distributions.categorical.Categorical(probs=p).sample(
                [self.num_samples]
            ),
            0,
            1,
        )   # (bath, num_samples)
        samples = torch.nn.functional.one_hot(samples,probs.shape[-1])  #(batch, num_samples, num_labels)
        
        return samples
    
    def compute_loss(
        self,
        x: torch.Tensor,
        y_hat: torch.Tensor,  # (batch, 1, ...)
        labels: torch.Tensor,  # (batch, 1, ...)
        buffer: Dict,
    ) -> torch.Tensor:
        samples = self.sample(y_hat).to(
            dtype=y_hat.dtype
        )  # (batch, num_samples, ...)
        y = torch.cat(
            (labels.to(dtype=samples.dtype), samples), dim=1
        )  # (batch, 1+num_samples, ...)
        distance = self.distance(
            y, y_hat.expand_as(y)
        )  # (batch, 1+num_samples) # does the job of Pn
        if self.use_scorenn:
            score = self.score_nn(
                x, y, buffer
            )  # type:ignore # (batch, 1+num_samples)
            assert not distance.requires_grad
        else:
            score = 0 
                    
        new_score = score - distance  # (batch, 1+num_samples)
        
        # def nce_loss(pred):
        #     noisy_smpls = torch.logsumexp(pred.float()[:, 1:], -1, keepdim=True)
        #     loss = (pred - noisy_smpls)[torch.arange(pred.shape[0]), 0]        
        #     return loss     
           
        # ranking_loss = nce_loss(new_score)
        
        ranking_loss = self.cross_entropy(
            new_score,
            torch.zeros(
                new_score.shape[0], dtype=torch.long, device=new_score.device
            ),  # (batch,)
        )

        return ranking_loss
    
    def reduce(self, loss_unreduced: torch.Tensor) -> torch.Tensor:
        if self.reduction == "sum":
            return torch.sum(loss_unreduced)
        elif self.reduction == "mean":
            return torch.mean(loss_unreduced)
        elif self.reduction == "none":
            return loss_unreduced
        else:
            raise ValueError
    
    def _forward(
        self,
        x: Any,
        labels: Optional[torch.Tensor],  # (batch, 1, ...)
        y_hat: torch.Tensor,  # (batch, 1, ...) normalized
        y_hat_extra: Optional[torch.Tensor],  # (batch, 1, ...)
        buffer: Dict,
        **kwargs: Any,
    ) -> torch.Tensor:
        assert y_hat.shape[1] == 1
        assert labels is not None
        loss_unreduced = self.compute_loss(
            x, y_hat, labels, buffer
        )  # (batch, num_samples)

        return loss_unreduced

    def forward(
        self,
        x: Any,
        labels: Optional[torch.Tensor],  #: shape (batch, 1, ...)
        y_hat: torch.Tensor,  #: shape (batch, num_samples, ...), assumed to be unnormalized logits
        y_hat_extra: Optional[
            torch.Tensor
        ],  # y_hat_probabilities or y_hat_cost_augmented, assumed to be unnormalized logits
        buffer: Dict,
        **kwargs: Any,
    ) -> torch.Tensor:

        if self.normalize_y:
            y_hat = self.normalize(y_hat)
            if y_hat_extra is not None:
                y_hat_extra = self.normalize(y_hat_extra)

        loss_unreduced = self._forward(
            x, labels, y_hat, y_hat_extra, buffer, **kwargs
        )

        loss = self.reduce(loss_unreduced)
        self.log("", loss.detach().mean().item())

        return loss    

    
def inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """
    x should be in [0, 1]
    """

    return -torch.log((1.0 / (x + 1e-13)) - 1.0 + 1e-35)
