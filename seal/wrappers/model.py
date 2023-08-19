import torch
from torch import nn

from seal.common import ModelMode

class SEALBase(nn.Module):
    def __init__(
        self,
        score_net,
        task_net,
        score_net_loss_fn,
        task_net_loss_fn
    ) -> None:
        self.score_net = score_net  
        self.task_net = task_net
        
        self.score_net_loss_fn = score_net_loss_fn
        self.task_net_loss_fn = task_net_loss_fn
        
    def normalize(self, y):
        return y        

    def unsqueeze_labels(self, labels: torch.Tensor) -> torch.Tensor:
        return labels.unsqueeze(1)

    def forward(self,**kwargs):
        return self._forward(**self.construct_args_for_forward(**kwargs)) 

    def construct_args_for_forward(self, **kwargs):
        kwargs["buffer"] = self.initialize_buffer(**kwargs)
        return kwargs
    
    def _forward(
        self,
        x,
        labels,
        mode,
        **kwargs
    ):
        if mode == ModelMode.UPDATE_TASK_NN:
            results = self.forward_on_tasknn(x, labels, **kwargs)
        elif mode == ModelMode.UPDATE_SCORE_NN:
            results = self.forward_on_scorenn(x, labels, **kwargs)
        else:
            raise ValueError

        return results

    def forward_on_tasknn(
        self,
        x,
        labels,
        buffer,
        meta = None,
        **kwargs
    ):
        results = {}
        
        if labels is not None and "labeled" in buffer.get('task'):
            labels = self.convert_to_one_hot(labels)
        
        y_pred = self.task_net(
            x, labels, buffer
        ).unsqueeze(1)
        
        loss = self.task_net_loss_fn(
                x,
                labels.unsqueeze(1),  # (batch, num_samples or 1, ...)
                y_pred,
                y_pred,
                buffer,
        )
        
        y_pred, loss = self.normalize(y_pred), self.normalize(loss)
        
        results["loss"] = loss
        results["y_pred"] = self.squeeze_y(y_pred)
        
        return results
        
    def forward_on_scorenn(
        self,
        x,
        labels,
        buffer,
        **kwargs
    ):
        results = {}
        
        if labels is not None and "labeled" in buffer.get('task'):
            labels = self.convert_to_one_hot(labels)
            
        y_hat = self.task_net(
            x, labels, buffer
        ).unsqueeze(1)
        
        loss = self.task_net_loss_fn(
                x,
                labels.unsqueeze(1),  # (batch, num_samples or 1, ...)
                y_pred,
                buffer,
        )
        
        y_pred, loss = self.normalize(y_pred), self.normalize(loss)
        
        loss = self.score_net_loss_fn(
                x, self.unsqueeze_labels(labels), y_hat, y_hat, buffer
        )
    
        results["y_hat"] = y_hat
        results["y_hat_extra"] = y_hat
        results["loss"] = loss
    
        return results
