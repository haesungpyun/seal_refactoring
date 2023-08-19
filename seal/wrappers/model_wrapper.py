import difflib
from typing_extensions import override
from .wrapper import Wrapper


class ModelWrapper(Wrapper): # type: ignore
    def __init__(self, config_dict):
        super().__init__()
        self.model_config = config_dict['model']
    
    def make_shared_nn_seal(self):
        task_net = self.construct_module(self.model_config['args']['task_net'], 'task_net')
        score_net = self.construct_module(self.model_config['args']['score_net'], 'score_net')
        inference_module = self.construct_module(
            self.model_config['args']['inference_module'], 
            'inference_module',
            task_net=task_net,
            score_net=score_net
        )
        loss_fn = self.construct_module(
            self.model_config['args']['loss_fn'], 
            'loss_fn',
            score_net=score_net
        )
        seal = self.construct_module(
            self.model_config, 
            'model',
            inference_module=inference_module,
            loss_fn=loss_fn
        )

        return seal
