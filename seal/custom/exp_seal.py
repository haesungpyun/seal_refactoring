import difflib
import torch
from .exp_architecture import MLP, train, test
from typing import List, Tuple, Union, Dict, Any, Optional
from seal.modules.loss import Loss
import transformers

# datasets = DataWrapper.make_data_reader()
# dataloader = DataWrapper.make_data_loader(dataset = datasets)


# class TaskNNLoss(Loss):
#     def __init__(self, **kwargs: Any):
#         super().__init__(**kwargs)
#         self.energy_local 
#         self.enegry_global
#         self.energy = self.energy_local + self.energy_global
#         self.bceloss_fn = torch.nn.BCEWithLogitsLoss(reduction = "sum")

#     def energy_local(self, y, b, y_hat):
#         self.y = y
        
#     def energy_global(self):
#         self.act = torch.nn.Softplus()

#     def forward(
#         self,
#         x: Any,
#         y: Optional[torch.Tensor],  # (batch, 1, num_labels)
#         y_hat: torch.Tensor,  # (batch, 1, num_labels)
#         lamda1: Any, lamda2: Any,
#         y_hat_extra: Optional[torch.Tensor],
#         buffer: Optional[Dict] = None,
#         **kwargs: Any,
#     ) -> torch.Tensor:
#         assert y is not None

#         loss = lamda1 * self.energy + lamda2 * self.bceloss_fn(y_hat, y.to(dtype=y_hat.dtype))

#         return loss
    
    
    
class MyTaskNN(torch.nn.Module):
    def __init__(
        self, 
        feature_net
    ):  
        super().__init__()
        self.feature_net = feature_net
    
    def forward(self, x):
        y_pred = self.feature_net(x)
        
        return y_pred
    

# class MultilabelClassificationScoreNN(ScoreNN):
#     def compute_local_score(
#         self,
#         x: torch.Tensor,  #: (batch, features_size)
#         y: torch.Tensor,  #: (batch, num_samples, num_labels)
#         buffer: Dict,
#         **kwargs: Any,
#     ) -> Optional[torch.Tensor]:
#         label_scores = self.task_nn(
#             x, buffer
#         )  # unormalized logit of shape (batch, num_labels)
#         local_energy = torch.sum(
#             label_scores.unsqueeze(1) * y, dim=-1
#         )  #: (batch, num_samples)

#         return local_energy
    
    
class MyScoreNN(torch.nn.Module):
    def __init__(
        self, 
        task_net : Optional[torch.nn.Module],
        bilinear:torch.tensor,
        projection: torch.nn.Module,
    ):
        super().__init__()
        self.task_net = task_net
        self.bilinear = bilinear
        self.projection = projection
    
    def forward(self, x, y_pred):
        y_hat = self.task_net(x)
        score = self.projection(self.bilinear(y_hat) * y_pred)
        # score = self.bert.encoder(proj.unsqueeze(2).expand(-1, -1, 2, -1))
        
        return score.mean()
        # "bert":{
        #     "module_dot_path": "transformers.BertModel",
        #     "object_name": "BertModel",
        #     "constructor":"from_pretrained",
        #     "args":{
        #         "pretrained_model_name_or_path": "bert-base-uncased"
        #     }
        # }

class Infernece_module(torch.nn.Module):
    def __init__(
        self,
        task_net,
        score_net,
        loss_fn
    ):
        super().__init__()
        self.task_net = task_net
        self.score_net = score_net
        self.loss_fn = loss_fn    
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x, y):
        y_pred = self.task_net(x)
        
        score_loss = self.score_net(x, y_pred)
        ce_loss = self.loss_fn(y_pred, y)
        
        y_pred = self.softmax(y_pred)

        return y_pred, score_loss + ce_loss
    

class RankingLoss(torch.nn.Module):
    def __init__(
        self,
        score_net,
        loss_fn
    ):
        super().__init__()
        self.score_net = score_net
        self.loss_fn = loss_fn
        self.bce = torch.nn.BCELoss(reduction="none")

    def forward(self, x, labels, y_pred):
        samples = torch.nn.functional.one_hot(
            torch.transpose(
                torch.distributions.Categorical(probs=y_pred).sample(
                    [4]
                ),
                0, 
                1
            ),
            19
        ).to(dtype=y_pred.dtype)
        y = torch.cat((labels.to(dtype=samples.dtype), samples), dim=1)
        score = self.score_net(x, y_pred)
        distance = - torch.sum(
            self.bce(y_pred.unsqueeze(1).expand_as(y), y), dim=-1
        )
        new_score = score - distance
        loss = self.loss_fn(
            new_score, 
            torch.zeros(
                new_score.shape[0], dtype=torch.long, device=new_score.device
            )
        )
        return loss

class SEAL(torch.nn.Module):
    def __init__(
        self,
        inference_module,
        loss_fn
    ):
        super().__init__()
        self.inference_module = inference_module
        self.loss_fn = loss_fn

    def forward(self, x, y, mode='score'):
        result = {}
        mode = difflib.get_close_matches(mode, ['task', 'score'])[0]

        if mode == 'task':
            y_pred, loss = self.inference_module(x, y)
            result['y_pred'] = y_pred
            result['loss'] = loss
            return result
        
        if mode == 'score':
            with torch.no_grad():
                y_hat, _ = self.inference_module(x, y)
            loss = self.loss_fn(x, y.unsqueeze(1), y_hat)
            result['y_hat'] = y_hat
            result['loss'] = loss
            return result